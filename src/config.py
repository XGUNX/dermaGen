"""
Project settings: paths, classes, image sizes, seeds.

Every other file imports its settings from here, so a value is changed in one
place only. This file does no computation and needs no GPU.

Check that everything is in place (run from the project root):
    python -m src.config
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Project root = the folder that contains src/. Computed from this file's
# location, so it works wherever the project is (laptop, Kaggle, Colab).
ROOT = Path(__file__).resolve().parents[1]

# Kaggle sets this environment variable in every notebook and mounts datasets
# read-only under /kaggle/input.
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir()

METADATA_FILE = "HAM10000_metadata.csv"
IMAGE_DIR_PATTERN = "HAM10000_images_part_*"  # the two image folders


def _find_raw_dir() -> Path:
    """Where the raw Kaggle download is.

    Order of priority:
    1. the DERMAGEN_RAW environment variable, if you set it yourself;
    2. on Kaggle, the folder under /kaggle/input that contains the metadata file
       (its exact path depends on how the dataset was attached);
    3. otherwise data/raw/ inside the project.
    """
    env_path = os.environ.get("DERMAGEN_RAW")
    if env_path:
        return Path(env_path)

    if ON_KAGGLE:
        kaggle_input = Path("/kaggle/input")
        for depth in ("*", "*/*", "*/*/*"):
            hits = sorted(kaggle_input.glob(f"{depth}/{METADATA_FILE}"))
            if hits:
                return hits[0].parent

    return ROOT / "data" / "raw"


RAW_DIR = _find_raw_dir()
METADATA_CSV = RAW_DIR / METADATA_FILE

PROCESSED_DIR = ROOT / "data" / "processed"  # created by prepare_data.py
SPLITS_CSV = PROCESSED_DIR / "splits.csv"     # image_id, lesion_id, dx, label, split

CHECKPOINT_DIR = ROOT / "checkpoints"         # every training run (not in Git)
WEIGHTS_DIR = ROOT / "weights"                # final models only (in Git)
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"               # one CSV per training run
SAMPLES_DIR = RESULTS_DIR / "samples"         # generated images
FIGURES_DIR = RESULTS_DIR / "figures"         # charts for the report and the demo


def processed_path(size: int) -> Path:
    """Path of the prepared array file for one image size, e.g. ham_28.npz."""
    return PROCESSED_DIR / f"ham_{size}.npz"


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (PROCESSED_DIR, CHECKPOINT_DIR, WEIGHTS_DIR,
                   LOGS_DIR, SAMPLES_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------

# Alphabetical order. The label of an image is its position in this list:
# akiec = 0, bcc = 1, ..., vasc = 6.
# Warning: the hmnist_*.csv files number the classes differently (mel = 6).
# We never use their labels.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

# French names, for the Streamlit demo.
CLASS_NAMES_FR = {
    "akiec": "Kératose actinique / carcinome intraépithélial",
    "bcc": "Carcinome basocellulaire",
    "bkl": "Kératose bénigne",
    "df": "Dermatofibrome",
    "mel": "Mélanome",
    "nv": "Naevus mélanocytaire",
    "vasc": "Lésion vasculaire",
}

# Classes that the augmentation experiment fills with synthetic images
# (each has fewer than 520 images in the whole dataset).
RARE_CLASSES = ["akiec", "bcc", "df", "vasc"]

# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

IMG_SIZE = 28           # working size for all experiments
IMG_SIZES = (28, 64)    # 64×64 later, for the best model only
CHANNELS = 3            # colour: it matters in dermatology

# --------------------------------------------------------------------------
# Split and reproducibility
# --------------------------------------------------------------------------

# Train / val / test by lesion_id, stratified by class (see prepare_data.py).
SPLIT_RATIOS = (0.70, 0.10, 0.20)
SPLIT_SEED = 42         # fixed: everyone gets exactly the same split

SEEDS = (0, 1, 2)       # each experiment runs 3 times; report mean ± std


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def get_device():
    """Best available device: NVIDIA GPU, then Apple GPU, then CPU.

    torch is imported here, not at the top of the file, so that scripts which
    do not need PyTorch (like reading the metadata) can still import config.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Self-check: python -m src.config
# --------------------------------------------------------------------------

if __name__ == "__main__":
    def status(ok: bool) -> str:
        return "OK" if ok else "MISSING"

    print(f"Project root     : {ROOT}")
    print(f"Running on Kaggle: {ON_KAGGLE}")
    print(f"Raw data folder  : {RAW_DIR}  [{status(RAW_DIR.is_dir())}]")
    print(f"Metadata file    : {METADATA_CSV.name}  [{status(METADATA_CSV.is_file())}]")

    image_dirs = sorted(p for p in RAW_DIR.glob(IMAGE_DIR_PATTERN) if p.is_dir())
    n_images = sum(len(list(d.glob("*.jpg"))) for d in image_dirs)
    print(f"Image folders    : {len(image_dirs)} found, {n_images} .jpg files"
          f"  [{status(len(image_dirs) >= 2 and n_images > 0)}]")

    print(f"Classes          : {CLASSES}")
    print(f"Rare classes     : {RARE_CLASSES}")
    print(f"Image size       : {IMG_SIZE}×{IMG_SIZE}×{CHANNELS}")
    print(f"Split ratios     : {SPLIT_RATIOS} (seed {SPLIT_SEED})")

    try:
        print(f"Device           : {get_device()}")
    except ImportError:
        print("Device           : torch not installed yet (pip install -r requirements.txt)")
