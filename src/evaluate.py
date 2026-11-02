"""
Evaluate generated images against real ones: fills results/metrics.csv.

    python -m src.evaluate                      # every sample file in results/samples/
    python -m src.evaluate --pattern "cvae_*"   # only the cVAE runs
    python -m src.evaluate --cas                # also the CAS (trains one classifier per file)
    python -m src.evaluate --no-inception       # skip the Inception FID (e.g. offline)

For each file results/samples/<run>_seed<k>.npz (made by src/sample.py with
the default --match test), it computes:
    fid_inception, kid_inception      standard metrics, InceptionV3 features
    fid_domain, kid_domain            same formulas, features of our referee CNN (question Q2)
    precision, recall                 quality and coverage (referee features)
    fidelity, fidelity_<class>        does the referee recognise the requested class?
    nn_ratio, share_suspicious        memorisation check against the train set
    cas_*                             with --cas: classifier trained on generated images only
    ms_per_image                      generation cost (question Q1)

All comparisons are made against the REAL TEST split, which no generator has
seen. A reference row "real_train" measures a subset of real TRAIN images
with exactly the same class counts as the test split: it is the floor a
perfect generator could reach (two different sets of real images are never
at distance 0).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, metrics
from src.data import class_counts, load_split
from src.sample import load_samples

METRICS_CSV = config.RESULTS_DIR / "metrics.csv"
FEATURE_CACHE = config.CHECKPOINT_DIR / "features"
MAIN_COLUMNS = ["fid_domain", "fid_inception", "precision", "recall", "fidelity", "nn_ratio",
                "cas_balanced_accuracy", "ms_per_image"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute the metrics of generated images.")
    p.add_argument("--pattern", default="*_seed*.npz", help="which files of results/samples/ (glob)")
    p.add_argument("--samples", nargs="+", default=None, help="explicit list of .npz files instead")
    p.add_argument("--size", type=int, default=config.IMG_SIZE)
    p.add_argument("--no-inception", action="store_true", help="skip the Inception features")
    p.add_argument("--cas", action="store_true", help="also compute the CAS (slower)")
    p.add_argument("--cas-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="seed of the real reference subset and of the CAS")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Real reference data
# --------------------------------------------------------------------------


def train_subset_like_test(X_train, y_train, y_test, seed: int):
    """Real train images with exactly the class counts of the test split."""
    rng = np.random.default_rng(seed)
    wanted = class_counts(y_test)
    idx = np.concatenate([rng.choice(np.flatnonzero(y_train == c), n, replace=False)
                          for c, n in enumerate(wanted) if n > 0])
    return X_train[idx], y_train[idx]


def cached_inception(name: str, images: np.ndarray, device) -> np.ndarray:
    """Inception features of a REAL image set, computed once and kept on disk
    (checkpoints/features/, not in Git). Generated sets are never cached."""
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    path = FEATURE_CACHE / f"inception_{name}.npy"
    if path.exists():
        feats = np.load(path)
        if len(feats) == len(images):
            return feats
    feats = metrics.inception_features(images, device)
    np.save(path, feats)
    return feats


def describe(path: Path) -> dict:
    """run name, model and seed from a file name like cvae_s0_b1_z16_seed0.npz."""
    stem = path.stem
    run, _, seed = stem.rpartition("_seed")
    return {"samples": stem, "run": run or stem, "model": stem.split("_")[0],
            "seed": int(seed) if seed.isdigit() else -1}


# --------------------------------------------------------------------------
# Metrics of one image set
# --------------------------------------------------------------------------


def evaluate_set(X, y, ref: dict, referee, device, args, check_memorization: bool = True) -> dict:
    """All metrics of one set of images (generated, or the real reference)."""
    row = {"n": len(y)}

    f_dom = metrics.domain_features(X, referee, device)
    row["fid_domain"] = metrics.fid(f_dom, ref["dom_test"])
    row["kid_domain"], row["kid_domain_std"] = metrics.kid(f_dom, ref["dom_test"], seed=args.seed)
    row["precision"], row["recall"] = metrics.precision_recall(ref["dom_test"], f_dom)

    if not args.no_inception:
        f_inc = metrics.inception_features(X, device)
        row["fid_inception"] = metrics.fid(f_inc, ref["inc_test"])
        row["kid_inception"], row["kid_inception_std"] = metrics.kid(f_inc, ref["inc_test"], seed=args.seed)
        row["precision_inception"], row["recall_inception"] = metrics.precision_recall(ref["inc_test"], f_inc)

    row.update(metrics.conditional_fidelity(referee, X, y, device))
    if check_memorization:
        row.update(metrics.memorization_report(X, ref["X_train"], ref["X_test"]))
    if args.cas:
        row.update(metrics.classification_accuracy_score(X, y, device, size=args.size,
                                                         epochs=args.cas_epochs, seed=args.seed))
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    config.ensure_dirs()
    device = config.get_device()

    files = [Path(p) for p in args.samples] if args.samples else sorted(config.SAMPLES_DIR.glob(args.pattern))
    if not files:
        print(f"No sample file found in {config.SAMPLES_DIR} for {args.pattern!r}. Run src/sample.py first.")
        return pd.DataFrame()

    # ---- Real data and its features (computed once)
    X_test, y_test, _ = load_split("test", args.size)
    X_train, y_train, _ = load_split("train", args.size)
    X_sub, y_sub = train_subset_like_test(X_train, y_train, y_test, args.seed)
    referee = metrics.load_referee(device)
    ref = {"X_test": X_test, "X_train": X_train,
           "dom_test": metrics.domain_features(X_test, referee, device)}
    if not args.no_inception:
        print("Inception features of the real images (cached after the first time)...")
        ref["inc_test"] = cached_inception(f"test_{args.size}", X_test, device)

    rows = []

    # ---- Reference: real train images vs real test images
    print("Reference: real train subset vs real test...")
    real = evaluate_set(X_sub, y_sub, ref, referee, device, args, check_memorization=False)
    # The referee was trained on these train images: its fidelity on them is
    # optimistic. Use its accuracy on the real TEST images instead.
    real.update(metrics.conditional_fidelity(referee, X_test, y_test, device))
    rows.append({"samples": "real_train", "run": "real_train", "model": "real", "seed": args.seed,
                 "ms_per_image": 0.0, **real})

    # ---- Each generated set
    test_mix = class_counts(y_test)
    for path in files:
        info = describe(path)
        X, y = load_samples(path)
        if X.shape[1:] != X_test.shape[1:]:
            print(f"Skip {path.name}: image shape {X.shape[1:]} differs from the test set {X_test.shape[1:]}")
            continue
        if not np.array_equal(class_counts(y), test_mix):
            print(f"Skip {path.name}: its class counts differ from the test split "
                  "(generate it with the default --match test)")
            continue
        print(f"Evaluating {path.name} ({len(y)} images)...")
        with np.load(path) as data:
            seconds = float(data["seconds"]) if "seconds" in data.files else float("nan")
        row = {**info, "ms_per_image": 1000 * seconds / len(y)}
        row.update(evaluate_set(X, y, ref, referee, device, args))
        rows.append(row)

    # ---- Save: replace earlier rows of the same files, keep the others
    new = pd.DataFrame(rows)
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        new = pd.concat([old[~old["samples"].isin(new["samples"])], new], ignore_index=True)
    new = new.round(5)
    new.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved {METRICS_CSV.relative_to(config.ROOT)} ({len(new)} rows)")

    # ---- Summary: mean ± std over the seeds of each run
    cols = [c for c in MAIN_COLUMNS if c in new.columns and new[c].notna().any()]
    summary = new.groupby("run")[cols].agg(["mean", "std"])
    with pd.option_context("display.width", 160, "display.max_columns", 30, "display.precision", 3):
        print(summary)
    print("\nReading guide: fid / kid lower is better; precision, recall, fidelity, cas higher is better; "
          "nn_ratio well below 1 means copying. Compare each run with the real_train row.")
    return new


if __name__ == "__main__":
    main()
