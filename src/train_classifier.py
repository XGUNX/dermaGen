"""
Train the referee CNN (src/models/classifier.py) and evaluate it.

The same script runs every classification baseline of question Q3:
    python -m src.train_classifier --seed 0                        # real images only
    python -m src.train_classifier --weighted --seed 0             # weighted loss
    python -m src.train_classifier --balanced --augment --seed 0   # oversampling + rotations/mirrors
    python -m src.train_classifier --synthetic results/samples/xxx.npz --seed 0   # + generated images

The referee used by the metrics and the demo (trained on real images only):
    python -m src.train_classifier --weighted --augment --seed 0 --export

How a run works:
1. train on the train split; after each epoch, measure balanced accuracy on val;
2. keep the epoch with the best val balanced accuracy (stop early if it no
   longer improves for --patience epochs);
3. reload that best model and evaluate it ONCE on the test split;
4. write one row to results/classifier_runs.csv, the per-epoch log to
   results/logs/<run>.csv and the confusion matrix to results/logs/<run>_confusion.csv.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config, utils
from src.data import class_weights, get_loader
from src.models.classifier import Classifier, evaluate

RESULTS_CSV = config.RESULTS_DIR / "classifier_runs.csv"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate the lesion classifier.")
    p.add_argument("--size", type=int, default=config.IMG_SIZE, help="image size (28 or 64)")
    p.add_argument("--epochs", type=int, default=40, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=10,
                   help="stop if val balanced accuracy has not improved for this many epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32, help="channels of the first block (32 -> 64 -> 128)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weighted", action="store_true", help="weighted loss against class imbalance")
    p.add_argument("--balanced", action="store_true", help="oversample rare classes")
    p.add_argument("--augment", action="store_true", help="random rotations and mirrors")
    p.add_argument("--synthetic", type=str, default=None,
                   help=".npz file with generated images (keys X, y) added to the train set")
    p.add_argument("--tag", type=str, default=None, help="extra text in the run name")
    p.add_argument("--export", action="store_true",
                   help="save the best model to weights/classifier.pt (the referee)")
    return p.parse_args(argv)


def run_name(args) -> str:
    """e.g. clf_s0_weighted_aug, or clf_s0_real when no option is used."""
    parts = [name for flag, name in ((args.weighted, "weighted"), (args.balanced, "balanced"),
                                     (args.augment, "aug"), (args.synthetic, "syn")) if flag]
    if args.size != config.IMG_SIZE:
        parts.append(f"{args.size}px")
    if args.tag:
        parts.append(args.tag)
    return utils.make_run_name("clf", args.seed, "_".join(parts) or "real")


def load_synthetic(path):
    """Read generated images saved as an .npz with keys X (uint8 N,H,W,3) and y."""
    with np.load(path) as data:
        return data["X"].astype(np.uint8), data["y"].astype(np.int64)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """One pass over the train set. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        seen += len(yb)
    return total_loss / seen, correct / seen


def run(args) -> dict:
    """Full run: train, select the best epoch on val, evaluate on test, save everything.

    Returns the result row (also appended to results/classifier_runs.csv).
    augment.py calls this function directly.
    """
    config.ensure_dirs()
    utils.set_seed(args.seed)
    device = config.get_device()
    name = run_name(args)
    print(f"Run {name} on {device}")

    # ---- Data
    extra = load_synthetic(args.synthetic) if args.synthetic else None
    train_loader = get_loader("train", size=args.size, batch_size=args.batch_size, augment=args.augment,
                              balanced=args.balanced, extra=extra, seed=args.seed)
    val_loader = get_loader("val", size=args.size, batch_size=512)
    test_loader = get_loader("test", size=args.size, batch_size=512)
    y_train = train_loader.dataset.labels.numpy()
    print(f"Train images: {len(y_train)}" + (f" (of which {len(extra[1])} synthetic)" if extra else ""))

    # ---- Model, loss, optimiser
    model = Classifier(width=args.width).to(device)
    weight = class_weights(y_train).to(device) if args.weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # The learning rate decreases smoothly to almost 0 at the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger = utils.RunLogger(name, vars(args) | {"run": name, "parameters": utils.count_parameters(model)})
    best_path = config.CHECKPOINT_DIR / f"{name}_best.pt"
    best_score, best_epoch, waited = -1.0, 0, 0
    utils.reset_peak_memory()

    # ---- Training loop
    for epoch in range(1, args.epochs + 1):
        with utils.Timer() as timer:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, device)
        scheduler.step()

        score = val_metrics["balanced_accuracy"]
        improved = score > best_score
        if improved:
            best_score, best_epoch, waited = score, epoch, 0
            utils.save_checkpoint(best_path, epoch, {"model": model}, run_config=model.config)
        else:
            waited += 1

        logger.log(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_bal_acc=score, val_macro_f1=val_metrics["macro_f1"],
                   lr=optimizer.param_groups[0]["lr"], epoch_s=round(timer.seconds, 1),
                   best="*" if improved else "")
        if waited >= args.patience:
            print(f"No improvement for {args.patience} epochs: stopping at epoch {epoch}.")
            break

    # ---- Final evaluation of the best epoch (the only time the test set is used)
    utils.load_checkpoint(best_path, {"model": model}, device=device)
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, cm = evaluate(model, test_loader, device)

    pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES).to_csv(
        config.LOGS_DIR / f"{name}_confusion.csv")
    logger.close(best_epoch=best_epoch, val_balanced_accuracy=val_metrics["balanced_accuracy"],
                 test_balanced_accuracy=test_metrics["balanced_accuracy"])

    row = {
        "run": name, "seed": args.seed, "size": args.size,
        "weighted": args.weighted, "balanced": args.balanced, "augment": args.augment,
        "synthetic": args.synthetic or "", "n_train": len(y_train),
        "best_epoch": best_epoch, "minutes": logger.info["total_minutes"],
        **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
    }
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)

    if args.export:
        path = utils.export_weights(model, config.WEIGHTS_DIR / "classifier.pt", model.config)
        print(f"Referee exported to {path.relative_to(config.ROOT)} (from run {name})")

    # ---- Summary
    print(f"\nBest epoch {best_epoch}. Results of the best model:")
    print(f"{'':<20}{'val':>8}{'test':>8}")
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        print(f"{key:<20}{val_metrics[key]:>8.3f}{test_metrics[key]:>8.3f}")
    print("Test recall per class: " + ", ".join(f"{c} {test_metrics[f'recall_{c}']:.2f}" for c in config.CLASSES))
    print(f"Row added to {RESULTS_CSV.relative_to(config.ROOT)}")
    return row


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
