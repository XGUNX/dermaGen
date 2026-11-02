"""
Generate images with any generator of the project and save them to disk.

    python -m src.sample --model cvae                          # weights/cvae.pt, same class mix as the test split
    python -m src.sample --model cvae --checkpoint checkpoints/cvae_s0_b2_z16_best.pt
    python -m src.sample --model cvae --n-per-class 1000       # 1000 images of each class
    python -m src.sample --model cvae --classes df vasc --n-per-class 500   # rare classes only (week 5)
    python -m src.sample --model ddpm --steps 50 --guidance 3  # extra options for the DDPM (week 4)

Output: results/samples/<name>_seed<k>.npz with
    X        uint8 (N, H, W, 3)   the generated images, same format as the real data
    y        int64 (N,)           the class asked for each image
    seconds  float                total generation time (cost comparison, question Q1)
and a PNG grid next to it.

Why "same class mix as the test split" by default: the FID compares two
distributions of images. If the generated set had a different proportion of
each class than the real set, the FID would partly measure that difference
instead of image quality.
"""

import argparse
import importlib
import inspect
import time
from pathlib import Path

import numpy as np
import torch

from src import config, utils
from src.data import class_counts, load_split

# Name -> class of the generator. The modules are imported only when needed,
# so this file works before cgan.py and ddpm.py exist.
GENERATORS = {
    "cvae": "src.models.cvae:CVAE",
    "cgan": "src.models.cgan:Generator",
    "ddpm": "src.models.ddpm:DDPM",
}
# Where a checkpoint stores the generator's weights, by order of preference:
# "ema" (averaged weights of the DDPM), "G" (GAN generator), "model" (others).
WEIGHT_KEYS = ("ema", "G", "model")


def model_class(name: str):
    if name not in GENERATORS:
        raise ValueError(f"unknown model {name!r}, choose from {list(GENERATORS)}")
    module, cls = GENERATORS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def load_generator(name: str, path=None, device="cpu"):
    """Rebuild a generator from weights/<name>.pt (default) or from a checkpoint.

    Both file formats store the constructor arguments of the model:
    - export_weights():  {"config": ..., "state_dict": ...}
    - save_checkpoint(): {"config": ..., "models": {"model" | "G" | "ema": ...}, ...}
    Returns the model in eval mode, on device.
    """
    path = Path(path) if path else config.WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the model first, with --export.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in saved:
        state_dict, cfg = utils.load_weights(path)
    else:
        key = next((k for k in WEIGHT_KEYS if k in saved["models"]), None)
        if key is None:
            raise KeyError(f"{path}: no generator weights among {list(saved['models'])}")
        state_dict, cfg = saved["models"][key], saved["config"]
    model = model_class(name)(**cfg)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def class_plan(match: str | None = "test", n_per_class: int | None = None,
               classes: list | None = None, size: int = config.IMG_SIZE) -> np.ndarray:
    """The list of labels to generate, sorted by class.

    n_per_class given -> that many images for each class (or each class in `classes`);
    otherwise         -> as many images per class as in the real split `match`.
    """
    wanted = [config.CLASS_TO_IDX[c] for c in classes] if classes else list(range(config.NUM_CLASSES))
    if n_per_class is not None:
        counts = {c: n_per_class for c in wanted}
    else:
        _, y_real, _ = load_split(match, size)
        real = class_counts(y_real)
        counts = {c: int(real[c]) for c in wanted}
    return np.concatenate([np.full(n, c, dtype=np.int64) for c, n in counts.items()])


@torch.no_grad()
def generate(model, labels: np.ndarray, seed: int = 0, batch_size: int = 500, device="cpu",
             **sample_kwargs) -> tuple:
    """Generate one image per label. Returns (uint8 images (N, H, W, 3), seconds).

    A single seeded torch.Generator is used for all batches, so the same
    seed always gives exactly the same images. Extra options (steps,
    guidance...) are passed only if the model's sample() accepts them.
    """
    accepted = inspect.signature(model.sample).parameters
    kwargs = {k: v for k, v in sample_kwargs.items() if v is not None and k in accepted}
    ignored = [k for k, v in sample_kwargs.items() if v is not None and k not in accepted]
    if ignored:
        print(f"Note: {type(model).__name__}.sample() ignores {ignored}")

    gen = torch.Generator().manual_seed(seed)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    out = []
    with utils.Timer() as timer:
        for i in range(0, len(labels_t), batch_size):
            batch = labels_t[i:i + batch_size].to(device)
            out.append(utils.to_uint8(model.sample(batch, generator=gen, **kwargs)))
    return np.concatenate(out), timer.seconds


def save_samples(path, images: np.ndarray, labels: np.ndarray, seconds: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=images, y=labels, seconds=np.float64(seconds))
    return path


def load_samples(path) -> tuple:
    """(images uint8 (N, H, W, 3), labels (N,)) from a file written by save_samples."""
    with np.load(path) as data:
        return data["X"], data["y"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images with a trained generator.")
    p.add_argument("--model", required=True, choices=list(GENERATORS))
    p.add_argument("--checkpoint", type=str, default=None, help="default: weights/<model>.pt")
    p.add_argument("--name", type=str, default=None, help="output name (default: from the file name)")
    p.add_argument("--match", default="test", choices=["train", "val", "test"],
                   help="copy the class counts of this real split (default: test)")
    p.add_argument("--n-per-class", type=int, default=None, help="fixed number per class instead")
    p.add_argument("--classes", nargs="+", default=None, choices=config.CLASSES, help="only these classes")
    p.add_argument("--size", type=int, default=config.IMG_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--steps", type=int, default=None, help="DDPM: number of sampling steps")
    p.add_argument("--guidance", type=float, default=None, help="DDPM: guidance scale")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    device = config.get_device()
    model = load_generator(args.model, args.checkpoint, device)

    name = args.name or (Path(args.checkpoint).stem.removesuffix("_best") if args.checkpoint else args.model)
    if args.classes:
        name += "_" + "-".join(args.classes)
    labels = class_plan(args.match, args.n_per_class, args.classes, args.size)
    print(f"Generating {len(labels)} images with {name} on {device}...")

    images, seconds = generate(model, labels, args.seed, args.batch_size, device,
                               steps=args.steps, guidance=args.guidance)
    out = save_samples(config.SAMPLES_DIR / f"{name}_seed{args.seed}.npz", images, labels, seconds)

    import matplotlib.pyplot as plt
    fig = utils.plot_class_grid(images, labels, n_per_class=10, path=out.with_suffix(".png"),
                                title=f"{name} · seed {args.seed}")
    plt.close(fig)

    counts = dict(zip(config.CLASSES, class_counts(labels).tolist()))
    print(f"Saved {out.relative_to(config.ROOT)}: {images.shape}, per class {counts}")
    print(f"Time: {seconds:.1f} s in total, {1000 * seconds / len(labels):.2f} ms per image")
    return out


if __name__ == "__main__":
    main()
