"""
Train the cVAE (src/models/cvae.py).

    python -m src.train_cvae --seed 0                       # beta 1, latent 16
    python -m src.train_cvae --beta 2 --latent 16 --seed 0  # ablation on beta
    python -m src.train_cvae --beta 1 --latent 2 --seed 0   # 2-D latent space (visual map in the demo)
    python -m src.train_cvae --seed 0 --export              # the chosen model -> weights/cvae.pt

Ablations of the project: beta in {0.5, 1, 2, 4}, latent in {2, 8, 16, 32}.
Write the hypothesis in JOURNAL.md BEFORE each run.

How a run works:
1. train on the train split only (the generator never sees val or test images
   during training);
2. after each epoch, compute the loss on val; keep the epoch with the lowest
   val loss;
3. every --preview-every epochs, save a grid of generated images
   (results/samples/progress/), to watch the model learn;
4. at the end: samples + reconstructions figures, one row in
   results/generator_runs.csv, the per-epoch log in results/logs/<run>.csv.

The test split is not used here: it serves later in evaluate.py, to compare
generated images with real images the generator has never seen.
"""

import argparse

import pandas as pd
import torch

from src import config, utils
from src.data import get_loader
from src.models.cvae import CVAE

RESULTS_CSV = config.RESULTS_DIR / "generator_runs.csv"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the conditional VAE.")
    p.add_argument("--size", type=int, default=config.IMG_SIZE, help="image size (28 or 64)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term")
    p.add_argument("--latent", type=int, default=16, help="size of the latent vector z")
    p.add_argument("--kl-warmup", type=int, default=0,
                   help="raise beta linearly from 0 over this many epochs (0 = off)")
    p.add_argument("--augment", action="store_true", help="random rotations and mirrors of the train images")
    p.add_argument("--balanced", action="store_true", help="show rare classes as often as frequent ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview-every", type=int, default=10, help="save a sample grid every N epochs")
    p.add_argument("--tag", type=str, default=None, help="extra text in the run name")
    p.add_argument("--export", action="store_true", help="save the best model to weights/cvae.pt")
    return p.parse_args(argv)


def run_name(args) -> str:
    """e.g. cvae_s0_b1_z16, cvae_s0_b2_z16_aug."""
    parts = [f"b{args.beta:g}", f"z{args.latent}"]
    parts += [name for flag, name in ((args.kl_warmup, f"warm{args.kl_warmup}"), (args.augment, "aug"),
                                      (args.balanced, "bal")) if flag]
    if args.size != config.IMG_SIZE:
        parts.append(f"{args.size}px")
    if args.tag:
        parts.append(args.tag)
    return utils.make_run_name("cvae", args.seed, "_".join(parts))


def beta_at(epoch: int, args) -> float:
    """KL warm-up: start with beta close to 0 so the model first learns to
    reconstruct, then give the KL term its full weight. Prevents the
    decoder from ignoring z ("posterior collapse")."""
    if args.kl_warmup <= 0:
        return args.beta
    return args.beta * min(1.0, epoch / args.kl_warmup)


@torch.no_grad()
def evaluate_loss(model: CVAE, loader, beta: float, device) -> dict:
    """Mean loss, reconstruction and KL over a whole split (always with the full beta)."""
    model.eval()
    totals = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
    n = 0
    for xb, yb in loader:
        out = model.loss(xb.to(device), yb.to(device), beta)
        for k in totals:
            totals[k] += out[k].item() * len(xb)
        n += len(xb)
    return {k: v / n for k, v in totals.items()}


def class_grid_labels(n_per_class: int, device) -> torch.Tensor:
    """0,0,...,1,1,...,6,6: n_per_class labels for each class, in order."""
    return torch.arange(config.NUM_CLASSES, device=device).repeat_interleave(n_per_class)


def run(args) -> dict:
    config.ensure_dirs()
    utils.set_seed(args.seed)
    device = config.get_device()
    name = run_name(args)
    print(f"Run {name} on {device}")

    # ---- Data: train for learning, val for choosing the best epoch
    train_loader = get_loader("train", size=args.size, batch_size=args.batch_size, augment=args.augment,
                              balanced=args.balanced, seed=args.seed, drop_last=True)
    val_loader = get_loader("val", size=args.size, batch_size=512)

    # ---- Model and optimiser
    model = CVAE(latent_dim=args.latent, img_size=args.size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n_params = utils.count_parameters(model)

    logger = utils.RunLogger(name, vars(args) | {"run": name, "parameters": n_params})
    best_path = config.CHECKPOINT_DIR / f"{name}_best.pt"
    progress_dir = config.SAMPLES_DIR / "progress"
    fixed_labels = class_grid_labels(8, device)
    best_val, best_epoch = float("inf"), 0
    utils.reset_peak_memory()

    # ---- Training loop
    for epoch in range(1, args.epochs + 1):
        beta = beta_at(epoch, args)
        model.train()
        sums, seen = {"loss": 0.0, "rec": 0.0, "kl": 0.0}, 0
        with utils.Timer() as timer:
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model.loss(xb, yb, beta)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                optimizer.step()
                for k in sums:
                    sums[k] += out[k].item() * len(xb)
                seen += len(xb)
        scheduler.step()

        val = evaluate_loss(model, val_loader, args.beta, device)
        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch = val["loss"], epoch
            utils.save_checkpoint(best_path, epoch, {"model": model}, run_config=model.config)

        logger.log(epoch=epoch, beta=round(beta, 4), loss=sums["loss"] / seen, rec=sums["rec"] / seen,
                   kl=sums["kl"] / seen, val_loss=val["loss"], val_rec=val["rec"], val_kl=val["kl"],
                   lr=optimizer.param_groups[0]["lr"], epoch_s=round(timer.seconds, 1),
                   best="*" if improved else "")

        if args.preview_every and (epoch % args.preview_every == 0 or epoch == 1):
            gen = model.sample(fixed_labels, torch.Generator().manual_seed(0))
            utils.save_grid(gen, progress_dir / f"{name}_ep{epoch:03d}.png", nrow=8)

    # ---- Reload the best epoch and save the final figures
    utils.load_checkpoint(best_path, {"model": model}, device=device)
    val = evaluate_loss(model, val_loader, args.beta, device)

    import matplotlib.pyplot as plt
    labels = class_grid_labels(10, device)
    gen = model.sample(labels, torch.Generator().manual_seed(0))
    fig = utils.plot_class_grid(gen, labels.cpu().numpy(), n_per_class=10,
                                path=config.FIGURES_DIR / f"{name}_samples.png",
                                title=f"{name}: generated images (best epoch {best_epoch})")
    plt.close(fig)

    xb, yb = next(iter(val_loader))
    xb, yb = xb[:16].to(device), yb[:16].to(device)
    rec = model.reconstruct(xb, yb)
    utils.save_grid(torch.cat([xb, rec]), config.FIGURES_DIR / f"{name}_reconstructions.png", nrow=16)

    logger.close(best_epoch=best_epoch, val_loss=val["loss"])

    row = {
        "run": name, "model": "cvae", "seed": args.seed, "size": args.size,
        "beta": args.beta, "latent": args.latent, "kl_warmup": args.kl_warmup,
        "augment": args.augment, "balanced": args.balanced,
        "epochs": args.epochs, "best_epoch": best_epoch,
        "val_loss": round(val["loss"], 3), "val_rec": round(val["rec"], 3), "val_kl": round(val["kl"], 3),
        "parameters": n_params, "minutes": logger.info["total_minutes"],
        "peak_gpu_mb": logger.info["peak_gpu_memory_mb"],
    }
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)

    if args.export:
        path = utils.export_weights(model, config.WEIGHTS_DIR / "cvae.pt", model.config)
        print(f"Exported to {path.relative_to(config.ROOT)} (from run {name})")

    print(f"\nBest epoch {best_epoch}: val loss {val['loss']:.2f} "
          f"(reconstruction {val['rec']:.2f}, KL {val['kl']:.2f})")
    print(f"Figures: results/figures/{name}_samples.png and {name}_reconstructions.png")
    print(f"Row added to {RESULTS_CSV.relative_to(config.ROOT)}")
    return row


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
