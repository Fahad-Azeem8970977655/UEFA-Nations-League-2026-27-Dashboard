"""Finishing, goalkeeping and set pieces from the shot data (xG, xGOT, goal-mouth position, situation).

Shrinkage (empirical Bayes).  Over a few matches goals - xG is mostly luck.  For a shooter with total xG X_i and
G_i goals, d_i = G_i - X_i has noise variance s2_i = sum p(1-p) over his shots.  Writing true skill as a
multiplicative edge b_i ~ N(0, tau2), d_i = b_i X_i + noise, and tau2 is estimated by the method of moments across
all shooters.  The best estimate of b_i is then  tau2 X_i / (tau2 X_i^2 + s2_i) * d_i.
If tau2 comes out as zero the data cannot yet tell finishing skill from luck and every estimate shrinks to average.
"""
from math import erf, sqrt

from types import SimpleNamespace

import numpy as np
import pandas as pd

SET_PIECE = {"corner": "Corner", "set-piece": "Set piece", "free-kick": "Free kick", "throw-in-set-piece": "Throw-in",
             "penalty": "Penalty"}


def eb_shrink(d, noise_var, scale):
    """d: observed (actual - expected); noise_var: variance of d under 'no skill'; scale: X_i (expected total).
    Returns DataFrame with shrunk edge b (ratio - 1), its sd, shrink weight, probability b > 0, and tau2."""
    d, noise_var, scale = (np.asarray(v, dtype=float) for v in (d, noise_var, scale))
    ok = scale > 0
    tau2 = max(0.0, float(np.sum(d[ok] ** 2 - noise_var[ok]) / np.sum(scale[ok] ** 2))) if ok.any() else 0.0
    denom = tau2 * scale ** 2 + noise_var
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(denom > 0, tau2 * scale / denom * d, 0.0)
        w = np.where(denom > 0, tau2 * scale ** 2 / denom, 0.0)
    var = np.where(denom > 0, tau2 * noise_var / denom, 0.0)  # posterior variance of the edge b
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        pz = np.where(sd > 0, [0.5 * (1 + erf(v / (s * sqrt(2)))) if s > 0 else 0.5 for v, s in zip(b, sd)], 0.5)
    return pd.DataFrame({"edge": b, "edge_sd": sd, "weight": w, "p_better": pz}), tau2


# ------------------------------------------------------------------ finishing
def finishing_tables(D, exclude_penalties=True):
    sh = D.shots.copy()
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    sh["xgot"] = pd.to_numeric(sh["xgot"], errors="coerce")
    sh["goal"] = (sh["shot_type"] == "goal").astype(int)
    if exclude_penalties:
        sh = sh[sh["situation"] != "penalty"]
    sh["var"] = sh["xg"] * (1 - sh["xg"])
    sh["on_target"] = sh["shot_type"].isin(["goal", "save", "post"])
    # player
    g = sh.groupby("player_id")
    pl = pd.DataFrame({"shots": g.size(), "goals": g["goal"].sum(), "xG": g["xg"].sum(), "var": g["var"].sum(),
                       "on_target": g["on_target"].sum(), "xGOT": g["xgot"].sum(),
                       "xg_on_target": g.apply(lambda d: d.loc[d["on_target"], "xg"].sum(), include_groups=False)})
    pl["G - xG"] = pl["goals"] - pl["xG"]
    pl["Placement (xGOT - xG on target)"] = pl["xGOT"] - pl["xg_on_target"]
    sk, tau2_pl = eb_shrink(pl["G - xG"], pl["var"], pl["xG"])
    sk.index = pl.index
    pl = pl.join(sk)
    pl["Est. true finishing (goals per xG)"] = 1 + pl["edge"]
    meta = D.players.set_index("player_id")[["player", "team", "position", "minutesplayed"]]
    pl = pl.join(meta)
    # team, for and against (against includes goalkeeping and defending)
    tm_for = sh.groupby("team").agg(shots=("xg", "size"), goals=("goal", "sum"), xG=("xg", "sum"), var=("var", "sum"))
    opp = sh.merge(D.tm[["match_id", "team", "opponent"]].rename(columns={"team": "_t", "opponent": "opp"}),
                   left_on=["match_id", "team"], right_on=["match_id", "_t"], how="left")
    tm_ag = opp.groupby("opp").agg(shots=("xg", "size"), goals=("goal", "sum"), xG=("xg", "sum"), var=("var", "sum"))
    for t in (tm_for, tm_ag):
        t["G - xG"] = t["goals"] - t["xG"]
    sf, tau_f = eb_shrink(tm_for["G - xG"], tm_for["var"], tm_for["xG"])
    sf.index = tm_for.index
    sa, tau_a = eb_shrink(tm_ag["G - xG"], tm_ag["var"], tm_ag["xG"])
    sa.index = tm_ag.index
    tm_for, tm_ag = tm_for.join(sf), tm_ag.join(sa)
    for t in (tm_for, tm_ag):
        t["Est. goals per xG"] = 1 + t["edge"]
    return SimpleNamespace(**{"player": pl.reset_index(), "team_for": tm_for, "team_against": tm_ag, "tau_player": tau2_pl,
                          "tau_for": tau_f, "tau_against": tau_a, "shots": sh,
                          "luck_share_player": _luck_share(pl)})


def _luck_share(pl):
    """Share of the spread of G - xG that pure shot luck would produce anyway (1 = all luck)."""
    ok = pl["xG"] >= 1.0
    if ok.sum() < 5:
        return np.nan
    return float(min(1.0, pl.loc[ok, "var"].sum() / max((pl.loc[ok, "G - xG"] ** 2).sum(), 1e-9)))


# ------------------------------------------------------------------ goalkeepers
GOAL_ZONES = {"high-left": (0, 2), "high-centre": (1, 2), "high-right": (2, 2),
              "left": (0, 1), "right": (2, 1),
              "low-left": (0, 0), "low-centre": (1, 0), "low-right": (2, 0)}


def keeper_tables(D):
    """Goals prevented against xGOT for every goalkeeper, with shrinkage, plus distribution profile.
    Shots faced are credited to the keeper with most minutes for that team in the match."""
    ps = D.ps.copy()
    ps["mins"] = pd.to_numeric(ps["minutesplayed"], errors="coerce").fillna(0)
    gk = ps[(ps["position"] == "G") & (ps["mins"] > 0)]
    main = gk.sort_values("mins", ascending=False).drop_duplicates(["match_id", "team"])[["match_id", "team", "player_id"]]
    main = main.rename(columns={"team": "gk_team", "player_id": "gk_id"})
    sh = D.shots[D.shots["shot_type"].isin(["goal", "save", "post"])].copy()
    sh["xgot"] = pd.to_numeric(sh["xgot"], errors="coerce")
    sh["goal"] = (sh["shot_type"] == "goal").astype(int)
    opp = D.tm[["match_id", "team", "opponent"]].rename(columns={"team": "shoot_team", "opponent": "gk_team"})
    sh = sh.merge(opp, left_on=["match_id", "team"], right_on=["match_id", "shoot_team"], how="left")
    sh = sh.merge(main, on=["match_id", "gk_team"], how="left").dropna(subset=["gk_id", "xgot"])
    sh["var"] = sh["xgot"].clip(0, 1) * (1 - sh["xgot"].clip(0, 1))
    sh["saved"] = 1 - sh["goal"]
    g = sh.groupby("gk_id")
    k = pd.DataFrame({"on_target_faced": g.size(), "goals_conceded": g["goal"].sum(), "xGOT_faced": g["xgot"].sum(),
                      "var": g["var"].sum()})
    k["Goals prevented"] = k["xGOT_faced"] - k["goals_conceded"]
    k["Save %"] = (1 - k["goals_conceded"] / k["on_target_faced"]) * 100
    k["Expected save %"] = (1 - k["xGOT_faced"] / k["on_target_faced"]) * 100
    sk, tau2 = eb_shrink(-k["Goals prevented"], k["var"], k["xGOT_faced"])
    sk.index = k.index
    # edge here is "extra goals per xGOT conceded"; flip the sign so positive = better shot-stopping
    k["Est. goals prevented per xGOT faced"] = -sk["edge"]
    k["Shrunk goals prevented"] = -sk["edge"] * k["xGOT_faced"]
    k["p_better"] = 1 - sk["p_better"]
    k["weight"] = sk["weight"]
    gps = ps[ps["position"] == "G"].copy()
    for c in ("totalpass", "accuratepass", "totallongballs", "accuratelongballs", "totalkeepersweeper", "accuratekeepersweeper",
              "goodhighclaim", "punches", "saves", "goalsprevented", "crossnotclaimed", "touches", "keepersavevalue"):
        gps[c] = pd.to_numeric(gps[c], errors="coerce").fillna(0)
    q = gps.groupby("player_id").agg(player=("player", "last"), team=("team", "last"), minutes=("mins", "sum"),
                                     matches=("match_id", "nunique"), passes=("totalpass", "sum"), acc=("accuratepass", "sum"),
                                     long=("totallongballs", "sum"), long_acc=("accuratelongballs", "sum"),
                                     sweeps=("totalkeepersweeper", "sum"), sweeps_ok=("accuratekeepersweeper", "sum"),
                                     claims=("goodhighclaim", "sum"), punches=("punches", "sum"), saves=("saves", "sum"),
                                     touches=("touches", "sum"))
    q = q[q["minutes"] > 0]
    q["Long-ball share %"] = q["long"] / q["passes"].replace(0, np.nan) * 100
    q["Long-ball accuracy %"] = q["long_acc"] / q["long"].replace(0, np.nan) * 100
    q["Pass accuracy %"] = q["acc"] / q["passes"].replace(0, np.nan) * 100
    q["Passes per 90"] = q["passes"] / q["minutes"] * 90
    q["Sweeper actions per 90"] = q["sweeps"] / q["minutes"] * 90
    q["High claims per 90"] = q["claims"] / q["minutes"] * 90
    q["Touches per 90"] = q["touches"] / q["minutes"] * 90
    out = q.join(k, how="left")
    out["on_target_faced"] = out["on_target_faced"].fillna(0)
    return SimpleNamespace(**{"table": out.reset_index().rename(columns={"index": "player_id"}), "shots": sh, "tau2": tau2})


def goal_zone_table(sh):
    """Save map: by goal-mouth zone, shots faced, goals, save %, expected save %."""
    d = sh[sh["goal_mouth_location"].isin(GOAL_ZONES)]
    g = d.groupby("goal_mouth_location").agg(n=("goal", "size"), goals=("goal", "sum"), xgot=("xgot", "sum"))
    g["Save %"] = (1 - g["goals"] / g["n"]) * 100
    g["Expected save %"] = (1 - g["xgot"] / g["n"]) * 100
    return g


# ------------------------------------------------------------------ set pieces
def set_piece_table(D):
    sh = D.shots.copy()
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    sh["goal"] = (sh["shot_type"] == "goal").astype(int)
    sh["sp"] = sh["situation"].isin(["corner", "set-piece", "free-kick", "throw-in-set-piece"])
    sh["pen"] = sh["situation"] == "penalty"
    opp = sh.merge(D.tm[["match_id", "team", "opponent"]].rename(columns={"team": "_t", "opponent": "opp"}),
                   left_on=["match_id", "team"], right_on=["match_id", "_t"], how="left")
    n = D.tm.groupby("team").size()
    rows = {}
    for who, key in (("for", "team"), ("against", "opp")):
        g = opp.groupby(key)
        t = pd.DataFrame({"xg_all": g["xg"].sum() - opp[opp["pen"]].groupby(key)["xg"].sum().reindex(g.size().index).fillna(0),
                          "xg_sp": opp[opp["sp"]].groupby(key)["xg"].sum(), "goals_sp": opp[opp["sp"]].groupby(key)["goal"].sum(),
                          "shots_sp": opp[opp["sp"]].groupby(key).size(),
                          "goals_all": opp[~opp["pen"]].groupby(key)["goal"].sum(),
                          "xg_corner": opp[opp["situation"] == "corner"].groupby(key)["xg"].sum(),
                          "xg_fk": opp[opp["situation"] == "free-kick"].groupby(key)["xg"].sum(),
                          "xg_throw": opp[opp["situation"] == "throw-in-set-piece"].groupby(key)["xg"].sum(),
                          "xg_other_sp": opp[opp["situation"] == "set-piece"].groupby(key)["xg"].sum(),
                          "xg_pen": opp[opp["pen"]].groupby(key)["xg"].sum(),
                          "xg_open": opp[~opp["sp"] & ~opp["pen"]].groupby(key)["xg"].sum()}).fillna(0)
        t["Matches"] = n.reindex(t.index)
        t["SP xG / match"] = t["xg_sp"] / t["Matches"]
        t["SP share of xG %"] = t["xg_sp"] / t["xg_all"].replace(0, np.nan) * 100
        t["SP goals"] = t["goals_sp"]
        t["Open-play xG / match"] = t["xg_open"] / t["Matches"]
        rows[who] = t
    return rows
