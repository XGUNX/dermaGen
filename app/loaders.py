"""
Everything the demo loads, cached once: data, referee, generators.

Every page imports from here, so a model is loaded a single time even if
several pages use it. The demo runs on the CPU: the models are small and
already trained.

Cache invalidation: the loaders take the file's modification time as an
argument, so exporting a new weights/cvae.pt is picked up without restarting
Streamlit.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src import config, utils

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def available_sizes() -> list:
    return [s for s in config.IMG_SIZES if config.processed_path(s).exists()]


@st.cache_resource(show_spinner="Chargement des images…")
def load_arrays(size: int) -> dict | None:
    """ham_<size>.npz in memory. cache_resource shares the arrays without copying them."""
    path = config.processed_path(size)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


@st.cache_data
def load_splits() -> pd.DataFrame | None:
    if not config.SPLITS_CSV.exists():
        return None
    return pd.read_csv(config.SPLITS_CSV)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def generator_files(prefix: str) -> list:
    """Exported generators of one family, e.g. weights/cvae.pt, weights/cvae_z2.pt."""
    return sorted(config.WEIGHTS_DIR.glob(f"{prefix}*.pt"))


@st.cache_resource(show_spinner="Chargement du modèle…")
def _load_generator(name: str, path: str, mtime: float):
    from src.sample import load_generator
    return load_generator(name, path, DEVICE)


def load_generator(name: str, path=None):
    """A trained generator, or None if its weights file does not exist yet."""
    path = Path(path) if path else config.WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        return None
    return _load_generator(name, str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner="Chargement de l'arbitre…")
def _load_referee(mtime: float):
    from src.metrics import load_referee
    return load_referee(DEVICE)


def load_referee():
    path = config.WEIGHTS_DIR / "classifier.pt"
    return _load_referee(path.stat().st_mtime) if path.exists() else None


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


def enlarge(img: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge by repeating pixels: a 28×28 image stays sharp instead of blurred."""
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def show(images, factor: int | None = None) -> np.ndarray:
    """Model output (tensor in [-1, 1]) or uint8 images -> one enlarged uint8 picture.
    A batch becomes a horizontal strip."""
    arr = utils.to_uint8(images) if isinstance(images, torch.Tensor) else np.asarray(images)
    if arr.ndim == 4:
        arr = utils.make_grid(arr, nrow=len(arr))
    size = arr.shape[0]
    return enlarge(arr, factor or max(1, 112 // min(size, 112)))


def class_label(i: int) -> str:
    code = config.CLASSES[i]
    return f"{code} · {config.CLASS_NAMES_FR[code]}"


def missing_model_message(what: str, command: str) -> None:
    st.warning(f"{what} n'est pas encore disponible. Entraîne-le puis exporte-le :")
    st.code(command, language="bash")
