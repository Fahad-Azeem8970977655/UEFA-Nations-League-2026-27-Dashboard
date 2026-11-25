"""Small data helpers shared by the loader and the pages."""
import numpy as np
import pandas as pd


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def parse_group(series):
    ex = series.astype(str).str.extract(r"League\s+([A-D]).*?(?:Gr\.|Group)\s*(\d+)")
    league = np.where(ex[0].notna(), "League " + ex[0].fillna(""), "Unknown")
    no = ex[1].fillna("0").astype(int)
    key = pd.Series(league, index=series.index) + ", Group " + no.astype(str)
    return pd.Series(league, index=series.index), key


def num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)
