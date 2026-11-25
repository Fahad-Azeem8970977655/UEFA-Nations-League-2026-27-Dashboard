"""Small-sample helpers. With about six matches per team, many rankings are mostly noise, so tables carry a flag."""
import numpy as np
import pandas as pd
import streamlit as st

MIN_MINUTES = 270   # three full matches
MIN_MATCHES = 5


def flag(values, threshold):
    """Series of 'low sample' / 'ok' from a numeric column and a threshold."""
    v = pd.to_numeric(values, errors="coerce")
    return np.where(v < threshold, "low sample", "ok")


def add_flag(df, col, threshold, name="Sample"):
    df = df.copy()
    df[name] = flag(df[col], threshold)
    return df


def sample_note(df, name="Sample", what="rows"):
    n = int((df[name] == "low sample").sum()) if name in df else 0
    if n:
        st.caption(f"{n} of {len(df)} {what} are marked 'low sample': too few minutes, shots or matches to trust the number. "
                   "Read those rows as hints, not rankings.")
