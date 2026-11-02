"""
Serve the prepared images to PyTorch.

Reads data/processed/ham_<size>.npz (built by prepare_data.py) and gives:
- HAMDataset: images as float tensors (3, H, W) in [-1, 1] + labels;
- get_loader(): a ready DataLoader, with optional classic augmentation,
  class-balanced sampling, or extra (synthetic) training images;
- class_weights(): weights for a weighted loss against class imbalance.

Typical use:
    from src.data import get_loader
    train_loader = get_loader("train", augment=True)
    val_loader = get_loader("val")

Self-test (needs python -m src.prepare_data first):
    python -m src.data
"""

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config, utils

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# Loading the arrays
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _load_npz(size: int) -> dict:
    """Read one .npz file once; later calls reuse it from memory."""
    path = config.processed_path(size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run first: python -m src.prepare_data --sizes {size}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_split(split: str, size: int = config.IMG_SIZE):
    """Return (images uint8 (N, H, W, 3), labels int64 (N,), image_ids (N,)) for one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data = _load_npz(size)
    return data[f"X_{split}"], data[f"y_{split}"], data[f"ids_{split}"]


# --------------------------------------------------------------------------
# Classic augmentation
# --------------------------------------------------------------------------


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random rotation by 0, 90, 180 or 270 degrees, then a random mirror.

    A dermoscopic photo has no "up" or "down": these 8 transformations give
    lesions that are just as realistic, without inventing any pixel.
    This is the "classic augmentation" baseline of question Q3.
    x: one image (3, H, W).
    """
    k = int(torch.randint(4, (1,)))
    x = torch.rot90(x, k, dims=(1, 2))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(2,))
    return x


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class HAMDataset(Dataset):
    """Images + labels held in memory.

    Images are kept as uint8 (4 times lighter than float) and converted to
    float in [-1, 1] only when a sample is requested.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        # (N, H, W, 3) -> (N, 3, H, W): PyTorch wants channels first
        self.images = torch.from_numpy(np.array(images, dtype=np.uint8, copy=True)).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.images[i].float() / 127.5 - 1.0
        if self.augment:
            x = random_dihedral(x)
        return x, self.labels[i]


def make_dataset(split: str, size: int = config.IMG_SIZE, augment: bool = False,
                 extra=None) -> HAMDataset:
    """Build the dataset of one split.

    extra = (images uint8 (M, H, W, 3), labels (M,)) adds images to the
    split, e.g. synthetic images for the rare classes. Only allowed for the
    train split: validation and test must stay 100 % real.
    """
    images, labels, _ = load_split(split, size)
    if extra is not None:
        if split != "train":
            raise ValueError("extra images can only be added to the train split")
        extra_images, extra_labels = extra
        if extra_images.shape[1:] != images.shape[1:]:
            raise ValueError(f"extra images have shape {extra_images.shape[1:]}, expected {images.shape[1:]}")
        images = np.concatenate([images, extra_images.astype(np.uint8)])
        labels = np.concatenate([labels, np.asarray(extra_labels, dtype=np.int64)])
    return HAMDataset(images, labels, augment=augment)


# --------------------------------------------------------------------------
# Class imbalance
# --------------------------------------------------------------------------


def class_counts(labels) -> np.ndarray:
    """Number of images per class, in the order of config.CLASSES."""
    return np.bincount(np.asarray(labels), minlength=config.NUM_CLASSES)


def class_weights(labels) -> torch.Tensor:
    """Weights for a weighted cross-entropy: weight_c = N / (K × N_c).

    A rare class gets a large weight, so each of its mistakes costs more.
    With this formula, the weights averaged over all images equal 1, so the
    loss keeps the same scale as without weights.
    Use: nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    """
    counts = class_counts(labels).astype(np.float64)
    weights = counts.sum() / (config.NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels, seed: int = 0) -> WeightedRandomSampler:
    """Draw training images so that every class appears about equally often.

    Rare images are drawn several times per epoch (oversampling); this is one
    of the baselines the synthetic images must beat.
    """
    labels = np.asarray(labels)
    per_class = 1.0 / np.maximum(class_counts(labels), 1)
    weights = torch.as_tensor(per_class[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)


# --------------------------------------------------------------------------
# DataLoader
# --------------------------------------------------------------------------


def get_loader(split: str, size: int = config.IMG_SIZE, batch_size: int = 128,
               augment: bool = False, balanced: bool = False, extra=None,
               shuffle: bool | None = None, seed: int = 0, num_workers: int = 0,
               drop_last: bool = False) -> DataLoader:
    """Ready-to-use DataLoader for one split.

    split      "train", "val" or "test"
    augment    random rotations / mirrors (train only, normally)
    balanced   oversample rare classes (replaces shuffling)
    extra      extra training images, see make_dataset()
    shuffle    default: True for train, False otherwise
    seed       fixes the shuffling order, for reproducible runs

    num_workers=0 is the right choice here: the images are already in memory
    and tiny, so extra processes would cost more than they save.
    """
    dataset = make_dataset(split, size, augment=augment, extra=extra)
    if shuffle is None:
        shuffle = split == "train"
    sampler = balanced_sampler(dataset.labels.numpy(), seed) if balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=utils.seed_worker,
        generator=torch.Generator().manual_seed(seed),
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------
# Self-test: python -m src.data
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Classes: {config.CLASSES}\n")
    print(f"{'split':<6} {'images':>7}  " + "  ".join(f"{c:>6}" for c in config.CLASSES))
    for split in SPLITS:
        X, y, ids = load_split(split)
        counts = class_counts(y)
        print(f"{split:<6} {len(y):>7}  " + "  ".join(f"{n:>6}" for n in counts))

    loader = get_loader("train", batch_size=64, augment=True)
    xb, yb = next(iter(loader))
    print(f"\nOne batch: images {tuple(xb.shape)} {xb.dtype}, range [{xb.min():.2f}, {xb.max():.2f}], "
          f"labels {tuple(yb.shape)}")

    _, y_train, _ = load_split("train")
    w = class_weights(y_train)
    print("Class weights:", {c: round(float(v), 2) for c, v in zip(config.CLASSES, w)})

    balanced = get_loader("train", batch_size=512, balanced=True)
    _, yb = next(iter(balanced))
    print("Balanced batch, images per class:", dict(zip(config.CLASSES, class_counts(yb.numpy()).tolist())))
