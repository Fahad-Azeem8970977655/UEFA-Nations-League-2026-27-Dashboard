"""Progression channels: where completed passes and carries move the ball forward, split left / centre / right.

Frame: the team attacks to the right (x 0-100).  With y drawn upwards, low y is the team's right and high y its left.
Channels are thirds of the pitch width: Right (y < 33.3), Centre, Left (y > 66.7).
A progression is a completed pass or a carry that ends at least 25% closer to the opposition goal than it started
and ends beyond 40% of the pitch.
    Build-up   progressions that start in the team's own 45%
    Attack     progressions that end in the final third (x >= 66.7)
Both are assigned to the channel where the move ends.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

CHANNELS = ["Right", "Centre", "Left"]


def channel(y):
    y = np.asarray(y, dtype=float)
    return np.where(y < 100 / 3, "Right", np.where(y > 200 / 3, "Left", "Centre"))


def _moves(D):
    p = D.passes[D.passes["event_action_type"].isin(["pass", "cross"])].dropna(subset=["x", "y", "end_x", "end_y"])
    c = D.carries.dropna(subset=["x", "y", "end_x", "end_y"])
    cols = ["match_id", "team", "player_id", "player", "x", "y", "end_x", "end_y"]
    mv = pd.concat([p[cols].assign(kind="Pass", ok=p["outcome"].values),
                    c[cols].assign(kind="Carry", ok=True)], ignore_index=True)
    sd = np.hypot((100 - mv["x"]) * 1.05, (50 - mv["y"]) * 0.68)
    ed = np.hypot((100 - mv["end_x"]) * 1.05, (50 - mv["end_y"]) * 0.68)
    mv["gain"] = sd - ed
    mv["prog"] = mv["ok"] & (ed <= 0.75 * sd) & (mv["end_x"] >= 40)
    mv["ch"] = channel(mv["end_y"])
    mv["ch_start"] = channel(mv["y"])
    mv["buildup"] = mv["prog"] & (mv["x"] < 45)
    mv["attack"] = mv["prog"] & (mv["end_x"] >= 66.7)
    return mv


def build_progress(D):
    mv = _moves(D)
    prog = mv[mv["prog"]]
    n_match = D.tm.groupby("team").size()
    out = {}
    for phase in ("buildup", "attack"):
        t = prog[prog[phase]].groupby(["team", "ch"]).size().unstack(fill_value=0).reindex(columns=CHANNELS, fill_value=0)
        out[phase] = t.div(t.sum(axis=1), axis=0) * 100
    team = pd.DataFrame(index=sorted(prog["team"].unique()))
    for ph, lab in (("buildup", "Build-up"), ("attack", "Attack")):
        for c in CHANNELS:
            team[f"{lab} {c} %"] = out[ph][c]
    team["Channel shift (pts)"] = (team[[f"Attack {c} %" for c in CHANNELS]].values
                                    - team[[f"Build-up {c} %" for c in CHANNELS]].values).__abs__().sum(axis=1) / 2
    pg = prog.groupby(["team", "kind"]).agg(n=("gain", "size"), metres=("gain", "sum")).unstack(fill_value=0)
    team["Prog. passes / match"] = pg[("n", "Pass")] / n_match
    team["Prog. carries / match"] = pg[("n", "Carry")] / n_match
    team["Carry share of progressions %"] = pg[("n", "Carry")] / (pg[("n", "Carry")] + pg[("n", "Pass")]) * 100
    team["Pass : carry ratio"] = pg[("n", "Pass")] / pg[("n", "Carry")].replace(0, np.nan)
    team["Metres gained by passes / match"] = pg[("metres", "Pass")] / n_match
    team["Metres gained by carries / match"] = pg[("metres", "Carry")] / n_match
    team["Carry share of metres %"] = pg[("metres", "Carry")] / (pg[("metres", "Carry")] + pg[("metres", "Pass")]) * 100
    team["Final-third entries / match"] = mv[mv["ok"] & (mv["end_x"] >= 66.7) & (mv["x"] < 66.7)].groupby("team").size() / n_match
    team["Matches"] = n_match
    lanes = prog.groupby(["team", "kind", "ch"]).size().unstack(["kind", "ch"], fill_value=0)
    pl = prog.groupby(["player_id", "kind"]).agg(n=("gain", "size"), metres=("gain", "sum")).unstack(fill_value=0)
    pl.columns = [f"{a}_{b}" for a, b in pl.columns]
    pl = pl.rename(columns={"n_Pass": "prog_passes", "n_Carry": "prog_carries", "metres_Pass": "metres_pass", "metres_Carry": "metres_carry"})
    for c in ("prog_passes", "prog_carries", "metres_pass", "metres_carry"):
        if c not in pl:
            pl[c] = 0.0
    meta = D.players.set_index("player_id")[["player", "team", "position", "minutesplayed"]]
    pl = pl.join(meta)
    pl["Prog. passes p90"] = pl["prog_passes"] / pl["minutesplayed"] * 90
    pl["Prog. carries p90"] = pl["prog_carries"] / pl["minutesplayed"] * 90
    pl["Carry share %"] = pl["prog_carries"] / (pl["prog_carries"] + pl["prog_passes"]).replace(0, np.nan) * 100
    pl["Metres gained p90"] = (pl["metres_pass"] + pl["metres_carry"]) / pl["minutesplayed"] * 90
    return SimpleNamespace(**{"moves": mv, "team": team, "lanes": lanes, "player": pl.reset_index()})
