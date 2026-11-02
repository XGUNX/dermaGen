"""
Page "Espace latent (cVAE)": see what the latent vector z encodes.

Four tabs, each tied to a part of the theory:
- Curseurs:        move z by hand; the same z decoded in the 7 classes
                   (the class changes the lesion type, z keeps the "style");
- Interpolation:   walk in a straight line from z0 to z1 (a smooth path shows
                   a well-regularised latent space, thanks to the KL term);
- Reconstruction:  encode a real image, look at its code mu, decode it back;
- Carte 2D:        for a model with latent size 2, decode a whole grid of z.
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch

import loaders
from src import config


def _decode(model, z: np.ndarray, label: int) -> torch.Tensor:
    z_t = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)
    y_t = torch.full((len(z_t),), label, dtype=torch.long)
    with torch.no_grad():
        return model.decode(z_t, y_t)


def _random_z(seed: int, dim: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim)


def render() -> None:
    st.title("Espace latent du cVAE")
    st.write("Le cVAE résume une image en un petit vecteur **z**. Le décodeur reconstruit une lésion "
             "à partir de **z** et de la **classe** demandée. Cette page montre ce que chacun contrôle.")

    files = loaders.generator_files("cvae")
    if not files:
        loaders.missing_model_message("Le cVAE", "python -m src.train_cvae --seed 0 --export")
        return

    with st.sidebar:
        st.header("Réglages")
        path = st.selectbox("Modèle", files, format_func=lambda p: p.name, key="lat_model")
        label = st.selectbox("Classe", range(config.NUM_CLASSES), format_func=loaders.class_label,
                             key="lat_class")

    model = loaders.load_generator("cvae", path)
    dim = model.latent_dim
    size = model.config["img_size"]
    factor = max(1, 168 // size)

    with st.expander("Le modèle chargé"):
        st.write(f"Fichier `{path.name}` · dimension latente **{dim}** · images {size}×{size} · "
                 f"{sum(p.numel() for p in model.parameters()):,} paramètres".replace(",", " "))
        st.json(model.config)

    tabs = st.tabs(["Curseurs", "Interpolation", "Reconstruction", "Carte 2D"])

    # ------------------------------------------------------------------ sliders
    with tabs[0]:
        _sliders(model, dim, label, factor)

    # ------------------------------------------------------------- interpolation
    with tabs[1]:
        st.write("On avance en ligne droite de **z₀** à **z₁**, pour la classe choisie. Grâce au terme KL "
                 "de la perte, l'espace latent est régulier : les images intermédiaires restent des lésions "
                 "plausibles et changent progressivement.")
        c1, c2, c3 = st.columns(3)
        seed0 = c1.number_input("Graine de z₀", 0, 9999, 1, key="lat_i0")
        seed1 = c2.number_input("Graine de z₁", 0, 9999, 2, key="lat_i1")
        steps = c3.slider("Nombre d'étapes", 5, 12, 8, key="lat_isteps")
        z0, z1 = _random_z(seed0, dim), _random_z(seed1, dim)
        alphas = np.linspace(0, 1, steps)
        path_z = np.stack([(1 - a) * z0 + a * z1 for a in alphas])
        st.image(loaders.show(_decode(model, path_z, label), factor=max(1, 112 // size)), width="stretch")
        st.caption(f"De gauche (z₀) à droite (z₁), classe {config.CLASSES[label]}.")

    # ------------------------------------------------------------ reconstruction
    with tabs[2]:
        _reconstruction(model, dim, label, size, factor)

    # ------------------------------------------------------------------ 2-D map
    with tabs[3]:
        if dim != 2:
            st.info("La carte 2D demande un modèle dont la dimension latente vaut 2. Entraîne-le puis "
                    "exporte-le sous un autre nom pour le choisir dans la barre latérale :")
            st.code("python -m src.train_cvae --beta 1 --latent 2 --seed 0\n"
                    "# puis, en Python :\n"
                    "# m = load_generator('cvae', 'checkpoints/cvae_s0_b1_z2_best.pt')\n"
                    "# utils.export_weights(m, config.WEIGHTS_DIR / 'cvae_z2.pt', m.config)", language="bash")
        else:
            n = st.slider("Points par axe", 6, 16, 10, key="lat_map_n")
            span = st.slider("Étendue de z (écarts-types)", 1.0, 3.5, 2.5, 0.5, key="lat_map_span")
            vals = np.linspace(-span, span, n)
            grid_z = np.array([[x, yv] for yv in vals[::-1] for x in vals])   # top row = high z2
            imgs = loaders.utils.to_uint8(_decode(model, grid_z, label))
            st.image(loaders.enlarge(loaders.utils.make_grid(imgs, nrow=n), max(1, 64 // size)), width="stretch")
            st.caption(f"Horizontal : z₁ de −{span} à +{span}. Vertical : z₂ de +{span} (haut) à −{span} (bas). "
                       f"Classe {config.CLASSES[label]}.")


@st.fragment
def _sliders(model, dim: int, label: int, factor: int) -> None:
    """In a fragment: moving a slider reruns only this function, not the page."""
    k = min(dim, 6)
    ss = st.session_state
    if ss.get("lat_dim") != dim:                  # new model: draw a new z
        ss.lat_dim, ss.lat_seed = dim, 0
        ss.lat_base = _random_z(0, dim)
        for i in range(k):
            ss[f"lat_z{i}"] = float(np.round(ss.lat_base[i], 2))

    st.write(f"Les {k} premières composantes de **z** sont réglables ; les autres restent fixes. "
             "Chaque composante suit une loi N(0, 1) : au-delà de ±3, on sort de ce que le modèle a appris.")
    if st.button("Tirer un nouveau z", key="lat_new"):
        ss.lat_seed += 1
        ss.lat_base = _random_z(ss.lat_seed, dim)
        for i in range(k):
            ss[f"lat_z{i}"] = float(np.round(ss.lat_base[i], 2))

    col_s, col_img = st.columns([1, 1])
    with col_s:
        z = ss.lat_base.copy()
        for i in range(k):
            z[i] = st.slider(f"z{i + 1}", -3.0, 3.0, step=0.05, key=f"lat_z{i}")
    with col_img:
        st.image(loaders.show(_decode(model, z, label), factor), caption=loaders.class_label(label))

    st.markdown("**Même z, autre classe.** Seule la classe demandée change d'une image à l'autre.")
    cols = st.columns(config.NUM_CLASSES)
    for c, col in enumerate(cols):
        col.image(loaders.show(_decode(model, z, c), max(1, 84 // model.config["img_size"])),
                  caption=config.CLASSES[c])


def _reconstruction(model, dim: int, label: int, size: int, factor: int) -> None:
    data = loaders.load_arrays(size)
    if data is None:
        st.info(f"Les images réelles {size}×{size} ne sont pas préparées sur cette machine.")
        return
    st.write("On prend une **vraie** image de validation, l'encodeur calcule son code **μ** (la moyenne de "
             "q(z | x, y)), puis le décodeur reconstruit l'image à partir de ce code. Le flou de la "
             "reconstruction est typique d'un VAE entraîné avec une erreur quadratique.")
    X, y = data["X_val"], data["y_val"]
    idx = np.flatnonzero(y == label)
    if "lat_rec_draw" not in st.session_state:
        st.session_state.lat_rec_draw = 0
    if st.button("Autre image", key="lat_rec_new"):
        st.session_state.lat_rec_draw += 1
    i = idx[st.session_state.lat_rec_draw % len(idx)]

    x = loaders.utils.uint8_to_tensor(X[i:i + 1])
    yt = torch.tensor([label])
    with torch.no_grad():
        mu, logvar = model.encode(x, yt)
        rec = model.decode(mu, yt)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.image(loaders.enlarge(X[i], factor), caption="Image réelle")
    c2.image(loaders.show(rec, factor), caption="Reconstruction")
    with c3:
        code = pd.DataFrame({"μ": mu[0].numpy(), "σ": torch.exp(0.5 * logvar[0]).numpy()},
                            index=[f"z{j + 1}" for j in range(dim)])
        st.bar_chart(code["μ"])
        st.caption("Le code μ de cette image. Une composante dont σ reste proche de 1 et μ proche de 0 "
                   "pour toutes les images n'est pas utilisée par le modèle.")
