"""
Build the dataset used by every experiment: raw Kaggle files -> data/processed/.

What it does, in order:
1. reads HAM10000_metadata.csv and turns each diagnosis (dx) into a label 0-6;
2. splits train / val / test BY LESION: all photos of one lesion go to the
   same split, so the test set never contains a lesion seen in training;
3. crops each photo to a centred square, then shrinks it to 28×28 (and 64×64
   if asked), keeping the 3 colour channels;
4. saves the arrays in data/processed/ham_<size>.npz and the split of every
   image in data/processed/splits.csv.

Run it once, from the project root:
    python -m src.prepare_data                  # 28×28
    python -m src.prepare_data --sizes 28 64    # also 64×64
    python -m src.prepare_data --force          # rebuild even if files exist

Contents of ham_<size>.npz:
    X_train, X_val, X_test      uint8, shape (N, size, size, 3)
    y_train, y_val, y_test      int64, labels 0-6 in the order of config.CLASSES
    ids_train, ids_val, ids_test  image_id of each row (link to splits.csv)
    classes                     the class names, in label order
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config, utils

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# 1. Metadata and image files
# --------------------------------------------------------------------------


def load_metadata() -> pd.DataFrame:
    """Read the metadata and add the numeric label column."""
    if not config.METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"{config.METADATA_CSV} not found. Put the Kaggle download in data/raw/ "
            "and check with: python -m src.config"
        )
    meta = pd.read_csv(config.METADATA_CSV)
    meta["label"] = meta["dx"].map(config.CLASS_TO_IDX)
    unknown = meta.loc[meta["label"].isna(), "dx"].unique()
    if len(unknown):
        raise ValueError(f"unknown dx codes in the metadata: {list(unknown)}")
    meta["label"] = meta["label"].astype(int)
    return meta


def index_images(raw_dir: Path) -> dict:
    """Map each image_id to its JPG file, across the two image folders."""
    paths = {}
    for folder in sorted(raw_dir.glob(config.IMAGE_DIR_PATTERN)):
        for p in folder.glob("*.jpg"):
            paths.setdefault(p.stem, p)
    return paths


# --------------------------------------------------------------------------
# 2. Split by lesion
# --------------------------------------------------------------------------


def split_by_lesion(meta: pd.DataFrame, ratios=config.SPLIT_RATIOS,
                    seed: int = config.SPLIT_SEED) -> pd.Series:
    """Give each image a split name ("train", "val" or "test").

    The split is made on LESIONS, not on images: we draw lesions, and every
    photo of a drawn lesion follows it. It is stratified by class, so each
    split keeps about the same class proportions.

    Two successive draws give 70 / 10 / 20:
        all lesions -> 20 % test, 80 % rest
        rest        -> 12.5 % of it to val (= 10 % of the total), the rest to train
    The ratios are exact in lesions and very close in images (some classes
    have more photos per lesion than others).
    """
    _, r_val, r_test = ratios

    # One row per lesion, with its class. All photos of a lesion normally have
    # the same diagnosis; if not, the most frequent one is used.
    lesion_dx = meta.groupby("lesion_id")["dx"].agg(lambda s: s.mode().iloc[0])
    ids = lesion_dx.index.to_numpy()

    rest, test = train_test_split(ids, test_size=r_test,
                                  stratify=lesion_dx.loc[ids], random_state=seed)
    train, val = train_test_split(rest, test_size=r_val / (1 - r_test),
                                  stratify=lesion_dx.loc[rest], random_state=seed)

    split_of_lesion = pd.Series("train", index=lesion_dx.index, dtype=object)
    split_of_lesion.loc[val] = "val"
    split_of_lesion.loc[test] = "test"
    return meta["lesion_id"].map(split_of_lesion)


def check_split(df: pd.DataFrame) -> None:
    """Stop with an error if one lesion appears in two splits (data leakage)."""
    splits_per_lesion = df.groupby("lesion_id")["split"].nunique()
    leaking = splits_per_lesion[splits_per_lesion > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} lesions are in several splits, "
                             f"e.g. {list(leaking.index[:3])}")


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Images per class and per split, with the share of each split."""
    table = (pd.crosstab(df["dx"], df["split"])
             .reindex(index=config.CLASSES, columns=list(SPLITS), fill_value=0))
    table.loc["total"] = table.sum()
    table["total"] = table[list(SPLITS)].sum(axis=1)
    for s in SPLITS:
        table[f"% {s}"] = (100 * table[s] / table["total"]).round(1)
    return table


# --------------------------------------------------------------------------
# 3. Images
# --------------------------------------------------------------------------


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Crop the largest centred square, then shrink it to size×size.

    The photos are 4:3 and the lesion is almost always centred: cropping keeps
    its proportions, where a direct resize would squash it. LANCZOS is a
    high-quality filter for shrinking (it limits aliasing).
    """
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def load_images(image_ids: list, paths: dict, sizes: list) -> dict:
    """Open each photo once and produce one uint8 array per requested size."""
    arrays = {size: np.empty((len(image_ids), size, size, 3), dtype=np.uint8) for size in sizes}
    for i, image_id in enumerate(tqdm(image_ids, desc="Images", unit="img")):
        with Image.open(paths[image_id]) as img:
            img = img.convert("RGB")
            for size in sizes:
                arrays[size][i] = np.asarray(center_crop_resize(img, size))
    return arrays


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/ from the raw HAM10000 files.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[config.IMG_SIZE],
                        help="image sizes to build (default: 28)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the .npz files already exist")
    parser.add_argument("--no-preview", action="store_true",
                        help="do not save the preview figure")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    sizes = [s for s in args.sizes if args.force or not config.processed_path(s).exists()]
    if not sizes:
        print("Nothing to do: the files already exist. Use --force to rebuild them.")
        return

    # 1. Metadata and files
    meta = load_metadata()
    paths = index_images(config.RAW_DIR)
    missing = sorted(set(meta["image_id"]) - set(paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in the metadata have no JPG file, "
                                f"e.g. {missing[:3]}. Check data/raw/.")
    print(f"{len(meta)} images, {meta['lesion_id'].nunique()} lesions, {len(paths)} JPG files found")

    # 2. Split by lesion
    df = meta.assign(split=split_by_lesion(meta))
    check_split(df)
    df["split"] = pd.Categorical(df["split"], categories=list(SPLITS), ordered=True)
    df = df.sort_values(["split", "image_id"]).reset_index(drop=True)
    df.to_csv(config.SPLITS_CSV, index=False)
    print(f"Split saved to {config.SPLITS_CSV.relative_to(config.ROOT)}")

    # 3. Images
    arrays = load_images(df["image_id"].tolist(), paths, sizes)

    # 4. One .npz file per size
    split_col = df["split"].astype(str).to_numpy()
    for size in sizes:
        content = {"classes": np.array(config.CLASSES)}
        for s in SPLITS:
            mask = split_col == s
            content[f"X_{s}"] = arrays[size][mask]
            content[f"y_{s}"] = df.loc[mask, "label"].to_numpy(dtype=np.int64)
            content[f"ids_{s}"] = df.loc[mask, "image_id"].to_numpy(dtype=str)
        path = config.processed_path(size)
        np.savez_compressed(path, **content)
        shapes = ", ".join(f"{s} {content[f'X_{s}'].shape}" for s in SPLITS)
        print(f"Saved {path.relative_to(config.ROOT)} ({path.stat().st_size / 2**20:.1f} MB): {shapes}")

        if not args.no_preview:
            import matplotlib.pyplot as plt
            fig_path = config.FIGURES_DIR / f"prepare_{size}_train_preview.png"
            fig = utils.plot_class_grid(content["X_train"], content["y_train"], n_per_class=10,
                                        path=fig_path, title=f"Train set after preparation ({size}×{size})")
            plt.close(fig)
            print(f"Preview saved to {fig_path.relative_to(config.ROOT)}")

    # Report
    lesions = df.groupby("split", observed=True)["lesion_id"].nunique()
    print("\nLesions per split:", ", ".join(f"{s} {lesions[s]}" for s in SPLITS))
    print("No lesion appears in two splits: OK\n")
    with pd.option_context("display.width", 120):
        print(split_summary(df))


if __name__ == "__main__":
    main()
