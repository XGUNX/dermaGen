"""
Small helpers shared by every script: seeds, checkpoints, timing, logging,
image conversion and image grids.

Nothing here knows about a specific model. Run the self-test with:
    python -m src.utils
"""

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from src import config

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix every random number generator so a run can be repeated exactly.

    Python, NumPy and PyTorch each have their own generator, so all three are
    seeded. deterministic=True also forces cuDNN to pick reproducible
    algorithms: slightly slower, use it for final runs if results must match
    to the last digit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible seed.

    Pass it as DataLoader(..., worker_init_fn=seed_worker). Without it,
    random augmentations in worker processes are not reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_run_name(model: str, seed: int, tag: str | None = None) -> str:
    """Standard run name, e.g. make_run_name("cvae", 0, "beta2") -> "cvae_s0_beta2"."""
    return f"{model}_s{seed}" + (f"_{tag}" if tag else "")


# --------------------------------------------------------------------------
# Model size and GPU memory
# --------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters (reported in the comparison table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_peak_memory() -> None:
    """Start measuring peak GPU memory from now."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    """Highest GPU memory used since the last reset, in MB (0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return 0.0


class Timer:
    """Measure how long a block takes, in seconds.

        with Timer() as t:
            train_one_epoch(...)
        print(t.seconds)

    GPU work runs asynchronously, so the timer waits for the GPU to finish
    before reading the clock; otherwise the times would be too short.
    """

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start
        return False


# --------------------------------------------------------------------------
# Checkpoints (every run, to resume after a cut) and weights (final models)
# --------------------------------------------------------------------------


def save_checkpoint(path, epoch: int, models: dict, optimizers: dict | None = None,
                    run_config: dict | None = None, **extra) -> Path:
    """Save everything needed to resume training.

    models and optimizers are dicts so one function works for every model:
        save_checkpoint(path, epoch, {"model": vae}, {"opt": opt}, cfg)
        save_checkpoint(path, epoch, {"G": G, "D": D}, {"opt_G": oG, "opt_D": oD}, cfg)

    run_config and extra must hold only numbers, strings, lists and dicts
    (no NumPy arrays), so the file can be reloaded safely with weights_only=True.

    The file is written to a temporary name first, then renamed: if the
    session is cut in the middle, the previous checkpoint stays intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "config": run_config or {},
        "models": {name: m.state_dict() for name, m in models.items()},
        "optimizers": {name: o.state_dict() for name, o in (optimizers or {}).items()},
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path, models: dict, optimizers: dict | None = None,
                    device="cpu") -> dict:
    """Load a checkpoint written by save_checkpoint into existing objects.

    Returns the whole saved dict, so you can read state["epoch"] to resume
    and state["config"] to check the hyperparameters.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    for name, model in models.items():
        model.load_state_dict(state["models"][name])
    for name, opt in (optimizers or {}).items():
        opt.load_state_dict(state["optimizers"][name])
    return state


def export_weights(model: torch.nn.Module, path, model_config: dict,
                   half: bool = True) -> Path:
    """Save a final model for weights/ and the demo: weights + the settings
    needed to rebuild it, without the optimizer.

    half=True stores the weights in float16, which halves the file size
    (GitHub refuses files over 100 MB).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k: (v.half() if half and v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    torch.save({"config": model_config, "state_dict": state_dict}, path)
    return path


def load_weights(path, device="cpu") -> tuple[dict, dict]:
    """Read a file written by export_weights.

    Returns (state_dict, model_config). The weights are converted back to
    float32. Typical use:
        sd, cfg = load_weights("weights/cvae.pt")
        model = CVAE(**cfg); model.load_state_dict(sd); model.eval()
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    state_dict = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in saved["state_dict"].items()
    }
    return state_dict, saved["config"]


# --------------------------------------------------------------------------
# Training log: one CSV per run in results/logs/
# --------------------------------------------------------------------------


class RunLogger:
    """Write one CSV row per epoch, plus a JSON file with the run settings.

        logger = RunLogger("cvae_s0_beta2", run_config=vars(args))
        logger.log(epoch=1, loss=0.42, val_loss=0.45)
        ...
        logger.close(best_val_loss=0.40)

    The CSV feeds the training curves in the report; the JSON feeds the
    compute log (total time, peak GPU memory, device).
    """

    def __init__(self, run_name: str, run_config: dict | None = None, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else config.LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.log_dir / f"{run_name}.csv"
        self.json_path = self.log_dir / f"{run_name}.json"
        self._fields = None
        self._start = time.time()
        if self.csv_path.exists():
            self.csv_path.unlink()  # a new run with the same name replaces the old log
        self.info = {
            "run_name": run_name,
            "config": run_config or {},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": "kaggle" if config.ON_KAGGLE else platform.node(),
        }
        self._write_json()

    def log(self, **values) -> None:
        """Add one row. The columns are fixed by the first call."""
        values["elapsed_s"] = round(time.time() - self._start, 1)
        if self._fields is None:
            self._fields = list(values.keys())
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(self._fields) + "\n")
        row = [_fmt(values.get(k, "")) for k in self._fields]
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")
        print(" | ".join(f"{k} {_fmt(v)}" for k, v in values.items()))

    def close(self, **summary) -> None:
        """Record the totals at the end of the run."""
        self.info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.info["total_minutes"] = round((time.time() - self._start) / 60, 2)
        self.info["peak_gpu_memory_mb"] = round(peak_memory_mb(), 1)
        self.info["summary"] = summary
        self._write_json()

    def _write_json(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2, default=str)


def _fmt(v) -> str:
    """Short, readable number formatting for logs."""
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


# --------------------------------------------------------------------------
# Image conversion: our convention is uint8 (N, H, W, 3) on disk and
# float tensors (N, 3, H, W) in [-1, 1] inside PyTorch.
# --------------------------------------------------------------------------


def uint8_to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 images (N, H, W, 3) or (H, W, 3) -> float tensor in [-1, 1],
    channels first: (N, 3, H, W) or (3, H, W)."""
    # np.array(..., copy=True) also handles read-only arrays (e.g. from pandas)
    x = torch.from_numpy(np.array(images, dtype=np.uint8, copy=True)).float()
    if x.dim() == 4:
        x = x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x.permute(2, 0, 1)
    else:
        raise ValueError(f"expected 3 or 4 dimensions, got shape {tuple(x.shape)}")
    return x / 127.5 - 1.0


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Model output in [-1, 1], (N, 3, H, W) or (3, H, W) -> uint8 (N, H, W, 3)
    or (H, W, 3), ready to display or save.

    Values outside [-1, 1] are clipped first (a generator can overshoot).
    """
    x = images.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) * 127.5).round().to(torch.uint8)
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).numpy()
    if x.dim() == 3:
        return x.permute(1, 2, 0).numpy()
    raise ValueError(f"expected 3 or 4 dimensions, got shape {tuple(x.shape)}")


# --------------------------------------------------------------------------
# Image grids
# --------------------------------------------------------------------------


def make_grid(images, nrow: int = 8, pad: int = 2, pad_value: int = 255) -> np.ndarray:
    """Tile images into one picture, nrow images per row.

    Accepts a tensor in [-1, 1] or a uint8 array (N, H, W, 3).
    Returns a uint8 array (H, W, 3), e.g. for st.image in the demo.
    """
    arr = to_uint8(images) if isinstance(images, torch.Tensor) else np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None]
    n, h, w, c = arr.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    grid = np.full((nrows * (h + pad) + pad, ncol * (w + pad) + pad, c),
                   pad_value, dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, ncol)
        y, x = pad + r * (h + pad), pad + col * (w + pad)
        grid[y:y + h, x:x + w] = arr[i]
    return grid


def save_grid(images, path, nrow: int = 8, scale: int = 4) -> Path:
    """Save an image grid as PNG, enlarged scale times so 28×28 images are
    visible. Nearest-neighbour enlarging keeps the real pixels (no blur)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow)
    img = Image.fromarray(grid)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)
    return path


def plot_class_grid(images: np.ndarray, labels: np.ndarray, n_per_class: int = 8,
                    path=None, title: str | None = None):
    """One row per class, the first n_per_class images of each, with the class
    name and its image count on the left. Used for exploration and for showing
    generated samples. Returns the matplotlib figure; also saves it if path is given.
    """
    import matplotlib.pyplot as plt

    images = to_uint8(images) if isinstance(images, torch.Tensor) else np.asarray(images)
    labels = np.asarray(labels)
    fig, axes = plt.subplots(config.NUM_CLASSES, n_per_class,
                             figsize=(n_per_class * 1.1, config.NUM_CLASSES * 1.2))
    for c, name in enumerate(config.CLASSES):
        idx = np.flatnonzero(labels == c)[:n_per_class]
        for j in range(n_per_class):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(images[idx[j]], interpolation="nearest")
        axes[c, 0].text(-0.15, 0.5, f"{name}\n({np.sum(labels == c)})",
                        transform=axes[c, 0].transAxes, ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Self-test: python -m src.utils
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())

    set_seed(0); a = torch.randn(3)
    set_seed(0); b = torch.randn(3)
    print("1. Same seed gives same numbers  :", torch.equal(a, b))

    fake = np.random.randint(0, 256, size=(10, 28, 28, 3), dtype=np.uint8)
    back = to_uint8(uint8_to_tensor(fake))
    print("2. uint8 -> tensor -> uint8 exact:", np.array_equal(fake, back))
    t = uint8_to_tensor(fake)
    print(f"   tensor shape {tuple(t.shape)}, range [{t.min():.2f}, {t.max():.2f}]")

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters())
    save_checkpoint(tmp / "ck.pt", 3, {"model": model}, {"opt": opt}, {"lr": 1e-3})
    model2 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    state = load_checkpoint(tmp / "ck.pt", {"model": model2})
    same = all(torch.equal(p, q) for p, q in zip(model.parameters(), model2.parameters()))
    print("3. Checkpoint reload             :", same, f"(epoch {state['epoch']})")

    export_weights(model, tmp / "w.pt", {"hidden": 8})
    sd, cfg = load_weights(tmp / "w.pt")
    print("4. Exported weights (fp16)       :", cfg, f"{count_parameters(model)} parameters")

    with Timer() as timer:
        time.sleep(0.05)
    print(f"5. Timer                         : {timer.seconds:.3f} s")

    logger = RunLogger("selftest", {"lr": 1e-3}, log_dir=tmp)
    logger.log(epoch=1, loss=0.5123)
    logger.log(epoch=2, loss=0.4011)
    logger.close(best=0.4011)
    print("6. Log file                      :", logger.csv_path.read_text().splitlines())

    save_grid(fake, tmp / "grid.png")
    plot_class_grid(fake, np.arange(10) % 7, n_per_class=2, path=tmp / "classes.png")
    print("7. Grids saved                   :", sorted(p.name for p in tmp.glob("*.png")))
    print(f"\nAll files written to {tmp}")
