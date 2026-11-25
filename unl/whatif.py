"""What-if simulator.

Match what-if: the ridge strength model of unl.predict gives the expected xG of each side.  A scenario then changes that xG
with levers whose size is estimated from this dataset:
    * shots inside / outside the box  -> xG, the average xG of a shot in that zone in this dataset
    * a red card at minute m          -> xG multipliers for the sent-off side and for the opponent for the rest of the match,
                                         measured on the matches of this league phase that had a red card
Uncertainty comes from a bootstrap: the played matches of the league are resampled, the strengths are refitted, and the scenario
is applied to every refit.

Group what-if: change the score of one or more played matches and re-rank the group; a replay of the other matches from their
shot-level xG scoreline grids gives the chance of each finishing position, with and without the change (same random draws).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from unl import predict as PR

SIGMA_BOOT = 150


# ------------------------------------------------------------------ levers estimated from the data
def estimate_levers(D):
    """Average xG of a shot inside and outside the box.  The shot table has no zone flag, so the nearest shots are taken as the
    inside-the-box ones, in the same share as the team statistics report (shots_inside_box / all shots)."""
    tm = D.tm[["shots_inside_box", "shots_outside_box"]].apply(pd.to_numeric, errors="coerce")
    share_in = float(tm["shots_inside_box"].sum() / (tm["shots_inside_box"].sum() + tm["shots_outside_box"].sum()))
    sh = D.shots[["x", "xg"]].copy()
    sh["x"] = pd.to_numeric(sh["x"], errors="coerce")
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    sh = sh.dropna(subset=["x"])
    cut = sh["x"].quantile(share_in)
    near, far = sh[sh["x"] <= cut], sh[sh["x"] > cut]
    return {"b_in": float(near["xg"].mean()), "b_out": float(far["xg"].mean()), "share_in": share_in, "n": len(sh),
            "avg_in": float(tm["shots_inside_box"].mean()), "avg_out": float(tm["shots_outside_box"].mean())}


def estimate_red_card(D, max_minute=80, boot=400, seed=3):
    """How the xG rate of the sent-off side and of its opponent changes after the first red card of a match.
    Observed xG after the card divided by what the side's season-average rate would give in the same minutes."""
    inc = D.inc
    reds = inc[(inc["type"] == "card") & inc["class"].isin(["red", "yellowRed"])].copy()
    reds["minute"] = pd.to_numeric(reds["minute"], errors="coerce")
    reds = reds.dropna(subset=["minute"]).sort_values(["match_id", "minute"]).drop_duplicates("match_id")
    reds = reds[reds["minute"] <= max_minute]
    sh = D.shots.copy()
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    sh["minute"] = pd.to_numeric(sh["minute"], errors="coerce")
    avg = D.tm.groupby("team")["expected_goals"].mean()
    m = D.m.set_index("match_id")
    rows = []
    for r in reds.itertuples():
        if r.match_id not in m.index:
            continue
        home, away = m.loc[r.match_id, "home_team"], m.loc[r.match_id, "away_team"]
        opp = away if r.team == home else home
        after = sh[(sh["match_id"] == r.match_id) & (sh["minute"] > r.minute)]
        left = max(90 - r.minute, 1)
        rows.append((after.loc[after["team"] == r.team, "xg"].sum(), avg.get(r.team, avg.mean()) * left / 90,
                     after.loc[after["team"] == opp, "xg"].sum(), avg.get(opp, avg.mean()) * left / 90, left))
    a = np.array(rows, dtype=float).reshape(-1, 5)
    pc = 0.5  # tiny pseudo-count so an empty sample gives 1 and not a division by zero

    def ratios(x):
        return (x[:, 0].sum() + pc) / (x[:, 1].sum() + pc), (x[:, 2].sum() + pc) / (x[:, 3].sum() + pc)

    c, o = ratios(a) if len(a) else (1.0, 1.0)
    rng = np.random.default_rng(seed)
    bs = np.array([ratios(a[rng.integers(0, len(a), len(a))]) for _ in range(boot)]) if len(a) else np.ones((boot, 2))
    return {"carded": float(c), "opponent": float(o), "n": int(len(a)), "minutes": float(a[:, 4].sum()) if len(a) else 0.0, "boot": bs,
            "carded_range": tuple(np.percentile(bs[:, 0], [10, 90])), "opponent_range": tuple(np.percentile(bs[:, 1], [10, 90]))}


# ------------------------------------------------------------------ match scenario
def apply_scenario(lh, la, sc, lev, red, mult=None):
    """sc: dict h_in, h_out, a_in, a_out (extra shots), red_side ('None' / 'Home' / 'Away'), red_min.  Returns new expected xG for both sides."""
    h = lh + lev["b_in"] * sc.get("h_in", 0) + lev["b_out"] * sc.get("h_out", 0)
    a = la + lev["b_in"] * sc.get("a_in", 0) + lev["b_out"] * sc.get("a_out", 0)
    h, a = max(h, 0.15), max(a, 0.15)
    side = sc.get("red_side", "None")
    if side != "None":
        f = float(np.clip(sc.get("red_min", 45), 1, 89)) / 90
        c, o = mult if mult is not None else (red["carded"], red["opponent"])
        mh, ma = (c, o) if side == "Home" else (o, c)
        h, a = h * f + h * mh * (1 - f), a * f + a * ma * (1 - f)
    return h, a


def match_probs(lh, la):
    M = PR.scoreline_probs(lh, la)
    return M, np.array(PR.outcome_probs(M))


def bootstrap_scenario(D, league, home, away, neutral, sc, lev, red, n=SIGMA_BOOT, seed=7):
    """Refit the strengths on resampled matches of the league; return home/draw/away probabilities for base and scenario in every refit."""
    mm = D.rate.matches
    sub = mm[mm["league"] == league].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    base, scn = [], []
    for it in range(n):
        pick = rng.integers(0, len(sub), len(sub))
        stub = SimpleNamespace(rate=SimpleNamespace(matches=sub.iloc[pick], lam=D.rate.lam))
        try:
            S = PR.fit_strength(stub)
            lh, la = PR.expected_goals(S, league, home, away, neutral)
        except KeyError:  # a team missing from this resample
            continue
        base.append(match_probs(lh, la)[1])
        scn.append(match_probs(*apply_scenario(lh, la, sc, lev, red, mult=tuple(red["boot"][it % len(red["boot"])])))[1])
    return np.array(base), np.array(scn)


# ------------------------------------------------------------------ group what-if
def group_table(D, gk, overrides=None):
    overrides = overrides or {}
    g = D.m[D.m["group_key"] == gk]
    rows = {}
    for r in g.itertuples():
        hg, ag = overrides.get(r.match_id, (int(r.home_score), int(r.away_score)))
        for team, gf, ga in ((r.home_team, hg, ag), (r.away_team, ag, hg)):
            t = rows.setdefault(team, {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0})
            t["P"] += 1
            t["GF"] += gf
            t["GA"] += ga
            t["W"] += gf > ga
            t["D"] += gf == ga
            t["L"] += gf < ga
    df = pd.DataFrame(rows).T.astype(int)
    df["GD"] = df["GF"] - df["GA"]
    df["Pts"] = 3 * df["W"] + df["D"]
    df = df.reset_index().rename(columns={"index": "Team"}).sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)
    df.insert(0, "Pos", np.arange(1, len(df) + 1))
    return df


def simulate_group(D, gk, overrides=None, n=4000, seed=21):
    """Finishing-position chances: every match is sampled from its xG scoreline grid, except the overridden ones which are fixed.
    The random draws do not depend on the overrides, so a with / without comparison differs only because of the change."""
    overrides = overrides or {}
    rng = np.random.default_rng(seed)
    g = D.m[D.m["group_key"] == gk].sort_values("match_id")
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    P = np.zeros((n, T))
    GD = np.zeros((n, T))
    GF = np.zeros((n, T))
    for r in g.itertuples():
        grid = D.xpts_grid[r.match_id]
        flat = (grid / grid.sum()).ravel()
        pick = rng.choice(flat.size, size=n, p=flat)
        hg, ag = pick // grid.shape[1], pick % grid.shape[1]
        if r.match_id in overrides:
            hg, ag = np.full(n, overrides[r.match_id][0]), np.full(n, overrides[r.match_id][1])
        i, j = idx[r.home_team], idx[r.away_team]
        P[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        P[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        GD[:, i] += hg - ag
        GD[:, j] += ag - hg
        GF[:, i] += hg
        GF[:, j] += ag
    noise = rng.random((n, T)) * 0.1
    rank = (-(P * 1e6 + GD * 1e3 + GF + noise)).argsort(axis=1).argsort(axis=1) + 1
    tab = pd.DataFrame({f"P{k}": (rank == k).mean(axis=0) * 100 for k in range(1, T + 1)}, index=teams)
    tab["Expected points"] = P.mean(axis=0)
    return tab
