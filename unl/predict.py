"""Match predictor and group-table simulation.

Predictor: per league, a ridge model of the xG each side creates,
    xG = mu + home_edge * (is home) + attack_team - defence_opponent
(penalty from the ratings page, chosen by cross-validation).  Expected xG for each side feeds two independent Poisson
distributions, which give win / draw / loss probabilities and scoreline odds.  Leagues never meet in the league phase,
so predictions are only offered inside one league.

Group simulation: every played match is replayed many times by sampling its scoreline from the shot-level xG
distribution (the same one behind xPts).  Counting how often each team ends in each position shows how much of the real
table was luck.
"""
from math import exp, factorial

import numpy as np
import pandas as pd

MAXG = 8


def fit_strength(D):
    out = {}
    mm = D.rate.matches
    for league, m in mm.groupby("league"):
        teams = sorted(set(m["home_team"]) | set(m["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        rows, y = [], []
        for r in m.itertuples():
            for att, dfn, home, xg in ((r.home_team, r.away_team, 1, r.hxg), (r.away_team, r.home_team, 0, r.axg)):
                v = np.zeros(2 * n + 2)
                v[idx[att]] = 1
                v[n + idx[dfn]] = -1
                v[-2], v[-1] = home, 1
                rows.append(v)
                y.append(xg)
        Z, y = np.array(rows), np.array(y)
        lam = D.rate.lam[league][0]
        P = np.eye(Z.shape[1]) * lam
        P[-2:, -2:] = 0
        b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
        out[league] = {"teams": teams, "att": dict(zip(teams, b[:n] - b[:n].mean())), "dfn": dict(zip(teams, b[n:2 * n] - b[n:2 * n].mean())),
                       "home": float(b[-2]), "mu": float(b[-1] + b[:n].mean() - b[n:2 * n].mean())}
    return out


def expected_goals(S, league, home, away, neutral=False):
    s = S[league]
    h = s["mu"] + (0 if neutral else s["home"]) + s["att"][home] - s["dfn"][away]
    a = s["mu"] + s["att"][away] - s["dfn"][home]
    return max(h, 0.15), max(a, 0.15)


def _pois(lam):
    return np.array([exp(-lam) * lam ** k / factorial(k) for k in range(MAXG + 1)])


def scoreline_probs(lh, la):
    M = np.outer(_pois(lh), _pois(la))
    return M / M.sum()


def outcome_probs(M):
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())  # home, draw, away


def top_scorelines(M, n=5):
    idx = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0][:n]
    return [(int(i), int(j), float(M[i, j])) for i, j in idx]


def backtest(D, S):
    """In-sample check: average probability given to the actual result, against always guessing 45/25/30."""
    rows = []
    for r in D.rate.matches.itertuples():
        lh, la = expected_goals(S, r.league, r.home_team, r.away_team)
        ph, pd_, pa = outcome_probs(scoreline_probs(lh, la))
        res = 0 if r.home_score > r.away_score else 1 if r.home_score == r.away_score else 2
        rows.append(((ph, pd_, pa)[res], (0.45, 0.25, 0.30)[res], res, int(np.argmax([ph, pd_, pa]))))
    a = np.array(rows)
    return {"mean_prob": float(a[:, 0].mean()), "base_prob": float(a[:, 1].mean()), "hit": float((a[:, 2] == a[:, 3]).mean()),
            "logloss": float(-np.log(a[:, 0]).mean()), "base_logloss": float(-np.log(a[:, 1]).mean()), "n": len(a)}


def simulate_groups(D, n=4000, seed=21):
    """Position probabilities for every team: rows team, columns 1..k, plus expected points and actual position."""
    rng = np.random.default_rng(seed)
    pts = {}
    mm = D.m[["match_id", "group_key", "home_team", "away_team"]]
    res = {}
    for gk, g in mm.groupby("group_key"):
        teams = sorted(set(g["home_team"]) | set(g["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        P = np.zeros((n, T)); GD = np.zeros((n, T)); GF = np.zeros((n, T))
        for r in g.itertuples():
            grid = D.xpts_grid[r.match_id]
            flat = (grid / grid.sum()).ravel()
            pick = rng.choice(flat.size, size=n, p=flat)
            hg, ag = pick // grid.shape[1], pick % grid.shape[1]
            i, j = idx[r.home_team], idx[r.away_team]
            P[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            P[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            GD[:, i] += hg - ag; GD[:, j] += ag - hg
            GF[:, i] += hg; GF[:, j] += ag
        score = P * 1e6 + GD * 1e3 + GF + rng.random((n, T)) * 0.1  # points, then goal difference, goals, coin flip
        rank = (-score).argsort(axis=1).argsort(axis=1) + 1
        tab = pd.DataFrame({f"P{k}": (rank == k).mean(axis=0) * 100 for k in range(1, T + 1)}, index=teams)
        tab["Expected points"] = P.mean(axis=0)
        tab["Group"] = gk
        res[gk] = tab
    out = pd.concat(res.values())
    if not D.standings.empty:
        st_ = D.standings.set_index("team")
        out["Actual position"] = st_["position"].reindex(out.index)
        out["Actual points"] = st_["points"].reindex(out.index)
    return out
