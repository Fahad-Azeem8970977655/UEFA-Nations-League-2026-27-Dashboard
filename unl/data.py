"""Data loading: cache (.pkl) -> Excel -> CSV. Everything is returned as one namespace `D`."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from unl.advanced import attach_advanced
from unl.extras import attach_extras
from unl.ext import EXT_SHEETS, attach_ext
from unl.helpers import num, parse_group, to_bool
from unl.metrics import build_player_table, build_team_summary

# Project root = folder that holds app.py (override with UNL_DATA_DIR)
APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("UNL_DATA_DIR", APP_DIR))
CACHE_DIR = DATA_DIR / ".cache"

MAIN_SHEETS = ["matches", "standings", "team_stats_wide", "incidents", "player_stats", "shotmap", "lineups_meta"]


XY_SHEETS = ["passes_xy", "defensive_xy", "ball_carries_xy", "dribbles_xy", "heatmap_xy",
             "average_positions", "momentum"]


def newest(pattern):
    hits = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def read_table(sheet, xlsx):
    """cache (.pkl) -> Excel sheet -> CSV, in that order of preference."""
    CACHE_DIR.mkdir(exist_ok=True)
    tag = xlsx.stem if xlsx is not None else "csv"
    cache = CACHE_DIR / f"{tag}__{sheet}.pkl"
    if cache.exists() and (xlsx is None or cache.stat().st_mtime >= xlsx.stat().st_mtime):
        return pd.read_pickle(cache)
    df = None
    if xlsx is not None:
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet)
        except ValueError:
            df = None
    if df is None:
        csv = DATA_DIR / "unl_data" / f"{sheet}.csv"
        if not csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv, low_memory=False)
    df.to_pickle(cache)
    return df


def _build():
    main_src, xy_src = newest("unl_2026_27_all*.xlsx"), newest("unl_2026_27_xy*.xlsx")
    d = {s: read_table(s, main_src) for s in MAIN_SHEETS}
    d.update({s: read_table(s, xy_src) for s in XY_SHEETS})
    if d["matches"].empty or d["player_stats"].empty:
        return None

    # ---- matches
    m = d["matches"].copy()
    m["date"] = pd.to_datetime(m["date_utc"])
    m["league"], m["group_key"] = parse_group(m["group"])
    m["matchday"] = m["round"]
    m = m.sort_values("date").reset_index(drop=True)

    # ---- standings
    st_ = d["standings"].copy()
    if not st_.empty:
        st_["league"], st_["group_key"] = parse_group(st_["group"])

    # ---- team-match table (own stats + opponent stats + result)
    ts = d["team_stats_wide"].copy()
    ts = ts.drop(columns=[c for c in ["date_utc", "group", "round", "home_team", "away_team"] if c in ts.columns])
    need = ["accurate_passes", "passes", "tackles", "interceptions", "expected_goals", "total_shots",
            "shots_on_target", "big_chances", "touches_in_penalty_area", "final_third_entries",
            "ball_possession", "clearances", "recoveries", "fouls", "corner_kicks", "distance_covered"]
    for c in need:
        if c not in ts.columns:
            ts[c] = np.nan
        ts[c] = pd.to_numeric(ts[c], errors="coerce")
    ts["pass_acc"] = ts["accurate_passes"] / ts["passes"] * 100
    ocols = ["expected_goals", "total_shots", "shots_on_target", "passes", "big_chances",
             "touches_in_penalty_area", "final_third_entries"]
    o = ts[["match_id", "side"] + ocols].copy()
    o["side"] = o["side"].map({"home": "away", "away": "home"})
    o.columns = ["match_id", "side"] + ["opp_" + c for c in ocols]
    tm = ts.merge(o, on=["match_id", "side"], how="left")
    tm = tm.merge(m[["match_id", "date", "league", "group_key", "matchday", "home_team", "away_team",
                     "home_score", "away_score"]], on="match_id", how="left")
    home = tm["side"] == "home"
    tm["gf"] = np.where(home, tm["home_score"], tm["away_score"])
    tm["ga"] = np.where(home, tm["away_score"], tm["home_score"])
    tm["opponent"] = np.where(home, tm["away_team"], tm["home_team"])
    tm["result"] = np.where(tm["gf"] > tm["ga"], "W", np.where(tm["gf"] < tm["ga"], "L", "D"))
    tm["points"] = tm["result"].map({"W": 3, "D": 1, "L": 0})
    tm = tm.sort_values(["team", "date"]).reset_index(drop=True)

    # ---- player stats
    ps = d["player_stats"].copy()
    ps["substitute"] = to_bool(ps["substitute"]) if "substitute" in ps.columns else False
    ps["date"] = pd.to_datetime(ps["date_utc"])
    ps["league"], ps["group_key"] = parse_group(ps["group"])
    ps["opponent"] = np.where(ps["side"] == "home", ps["away_team"], ps["home_team"])

    # ---- coordinate tables
    xy = {}
    for name in ["passes_xy", "defensive_xy", "ball_carries_xy", "dribbles_xy"]:
        t = d[name].copy()
        if t.empty:
            xy[name] = t
            continue
        if "end_x" not in t.columns and "pass_end_coordinates_x" in t.columns:
            t["end_x"] = t["pass_end_coordinates_x"]
            t["end_y"] = t["pass_end_coordinates_y"]
        for c in ["outcome", "keypass", "is_assist", "is_long_ball"]:
            t[c] = to_bool(t[c]) if c in t.columns else False
        for c in ["x", "y", "end_x", "end_y"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        xy[name] = t
    heat = d["heatmap_xy"].copy()
    avg = d["average_positions"].copy()
    mom = d["momentum"].copy()

    inc = d["incidents"].copy()
    shots = d["shotmap"].copy()
    shots["date"] = pd.to_datetime(shots["date_utc"])
    # SofaScore lists own goals as "goal" shots with no xG, under the team that benefited but with the
    # opponent's defender as the shooter. They are not shots, so keep them apart (D.own_goals).
    own = (shots["shot_type"] == "goal") & pd.to_numeric(shots["xg"], errors="coerce").isna()
    own_goals = shots[own].copy()
    shots = shots[~own].reset_index(drop=True)
    lm = d["lineups_meta"].copy()

    teams = sorted(tm["team"].dropna().unique())
    D = SimpleNamespace(m=m, standings=st_, tm=tm, ps=ps, inc=inc, shots=shots, own_goals=own_goals, lineups=lm,
                        passes=xy["passes_xy"], defensive=xy["defensive_xy"], carries=xy["ball_carries_xy"],
                        dribbles=xy["dribbles_xy"], heat=heat, avg=avg, mom=mom, teams=teams)
    D.team_summary = build_team_summary(tm)
    D.players = build_player_table(ps)
    attach_advanced(D)
    attach_extras(D)
    # optional extra scrape (halves, referees, odds, player profiles); a problem here must not break the rest of the dashboard
    D.ext_error = None
    try:
        ext_src = newest("unl_2026_27_extra*.xlsx")
        attach_ext(D, {s: read_table(s, ext_src) for s in EXT_SHEETS})
    except Exception as e:  # noqa: BLE001
        D.halves = D.refs = D.odds = D.odds_team = None
        D.ext_error = f"{type(e).__name__}: {e}"
    return D


def _stamp():
    """Changes whenever a data file or any code file changes, so the saved analysis is rebuilt only when needed."""
    files = list(DATA_DIR.glob("unl_2026_27_*.xlsx")) + list((APP_DIR / "unl").glob("*.py"))
    return str(int(max([f.stat().st_mtime for f in files] or [0])))


@st.cache_resource(show_spinner="Loading data. The first run after a change takes about a minute; later starts take a few seconds.")
def load_all():
    import pickle
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"D_{_stamp()}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    D = _build()
    if D is not None:
        try:
            for old in CACHE_DIR.glob("D_*.pkl"):
                old.unlink()
            f.write_bytes(pickle.dumps(D, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            pass
    return D
