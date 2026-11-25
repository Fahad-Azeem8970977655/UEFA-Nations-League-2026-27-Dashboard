"""Call setup() as the first line of Home.py and of every page."""
from pathlib import Path

import streamlit as st

from unl.data import load_all
from unl.style import inject_css

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def setup(title="Nations League 2026-27"):
    st.set_page_config(page_title=title, page_icon=str(ASSETS / "unl_flag.png"), layout="wide")
    try:
        st.logo(str(ASSETS / "unl_logo_tight.png"), size="large")
    except Exception:
        pass
    inject_css()
    D = load_all()
    if D is None:
        st.error("No data found. Put unl_2026_27_all.xlsx and unl_2026_27_xy.xlsx next to Home.py "
                 "(or set UNL_DATA_DIR to the folder that holds them), then reload.")
        st.stop()
    st.sidebar.caption(f"League phase, {D.m['match_id'].nunique()} matches, data from SofaScore")
    return D
