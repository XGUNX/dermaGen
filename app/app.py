"""
DermaGen demo: entry point and menu.

Run it from the project root:
    streamlit run app/app.py

Each page is a file in app/views/ with a render() function. app.py only
declares the pages; st.navigation draws the menu in the sidebar and runs
the chosen page. Shared loading (data, models) lives in app/loaders.py.

Pages added so far:
    views/problem.py   the real images and the class imbalance   (week 1)
    views/latent.py    the latent space of the cVAE               (week 2)
Coming: generate.py (week 3), denoise.py (week 4), results.py (week 5).

How Streamlit works, in 4 ideas:
1. RERUN: every interaction reruns the script (here: the chosen page).
2. CACHE: slow loading goes through @st.cache_resource / @st.cache_data.
3. SESSION STATE: st.session_state keeps values between reruns.
4. FRAGMENTS: @st.fragment reruns only part of a page (the cVAE sliders).
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(page_title="DermaGen", page_icon="🔬", layout="wide")

from views import latent, problem  # noqa: E402  (after set_page_config and sys.path)

pages = [
    st.Page(problem.render, title="Le jeu de données", icon=":material/dataset:", url_path="donnees", default=True),
    st.Page(latent.render, title="Espace latent (cVAE)", icon=":material/tune:", url_path="espace-latent"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.caption("DermaGen · génération conditionnelle de lésions cutanées. "
               "Images de recherche et d'enseignement, sans usage diagnostique.")

nav.run()
