"""
Page "Le jeu de données": the real images and the class imbalance.

This was the whole app in week 1; it is now one page among others.
"""

import numpy as np
import pandas as pd
import streamlit as st

import loaders
from src import config


def render() -> None:
    sizes = loaders.available_sizes()
    if not sizes:
        st.error("Aucune donnée préparée. Lance d'abord, depuis la racine du projet : "
                 "`python -m src.prepare_data`")
        return

    with st.sidebar:
        st.header("Réglages")
        size = st.selectbox("Taille des images", sizes, format_func=lambda s: f"{s} × {s}", key="pb_size")
        split = st.radio("Split", ["train", "val", "test"], horizontal=True, key="pb_split")

    data = loaders.load_arrays(size)
    X, y = data[f"X_{split}"], data[f"y_{split}"]
    splits = loaders.load_splits()

    st.title("Le jeu de données")
    st.write("HAM10000 : 10 015 images dermatoscopiques de 7 types de lésions, découpées par lésion "
             "en train, validation et test.")

    tab_data, tab_class = st.tabs(["Vue d'ensemble", "Explorer une classe"])

    with tab_data:
        counts = pd.DataFrame({
            s: np.bincount(data[f"y_{s}"], minlength=config.NUM_CLASSES) for s in ("train", "val", "test")
        }, index=config.CLASSES)
        counts["total"] = counts.sum(axis=1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images", f"{counts['total'].sum():,}".replace(",", " "))
        c2.metric("Lésions distinctes",
                  f"{splits['lesion_id'].nunique():,}".replace(",", " ") if splits is not None else "?")
        c3.metric("Classes", config.NUM_CLASSES)
        c4.metric("Déséquilibre max / min", f"{counts['total'].max() / counts['total'].min():.0f}×")

        st.subheader("Images par classe et par split")
        st.bar_chart(counts[["train", "val", "test"]], stack=True, horizontal=True)

        table = counts.copy()
        table.insert(0, "nom", [config.CLASS_NAMES_FR[c] for c in table.index])
        table["% du total"] = (100 * table["total"] / table["total"].sum()).round(1)
        table["classe rare"] = ["oui" if c in config.RARE_CLASSES else "" for c in table.index]
        st.dataframe(table)

        st.info("Le découpage est fait **par lésion** : toutes les photos d'une même lésion "
                "sont dans le même split, donc aucune lésion du test n'a été vue à l'entraînement.")

    with tab_class:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            label = st.selectbox("Classe", range(config.NUM_CLASSES), format_func=loaders.class_label,
                                 key="pb_class")
        with col_b:
            n_show = st.slider("Nombre d'images", min_value=8, max_value=48, value=24, step=8, key="pb_n")

        if "pb_draw" not in st.session_state:
            st.session_state.pb_draw = 0
        if st.button("Nouveau tirage", key="pb_new"):
            st.session_state.pb_draw += 1

        idx = np.flatnonzero(y == label)
        rng = np.random.default_rng(st.session_state.pb_draw)
        chosen = rng.choice(idx, size=min(n_show, len(idx)), replace=False)
        st.caption(f"{len(idx)} images de la classe **{config.CLASSES[label]}** dans le split {split}. "
                   f"Tirage n° {st.session_state.pb_draw}.")
        grid = loaders.utils.make_grid(X[chosen], nrow=8)
        st.image(loaders.enlarge(grid, max(1, 112 // size)), width="stretch")
