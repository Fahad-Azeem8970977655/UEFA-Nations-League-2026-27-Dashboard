"""Opponent-adjusted team ratings, drivers of the SofaScore rating, and a scouting index.

Team ratings (one pool per league, because league-phase teams only play inside their own league)
    Massey / Bradley-Terry style ridge model on the xG margin of every match:
        xG_home - xG_away = h + r_home - r_away + noise
    The ridge penalty is picked by leave-one-match-out cross-validation.  A split attack / defence version fits
        xG scored = mu + h*home + attack_team - defence_opponent
    Ratings are goals-per-match above an average team of the same league.  Elo (k = 20 per 1.0 xG margin step,
    start 1500) on the same xG margins is given as a second opinion.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from unl.helpers import num


def _design(matches, teams):
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(matches), len(teams) + 1))
    for j, (h, a) in enumerate(zip(matches["home_team"], matches["away_team"])):
        X[j, idx[h]] += 1
        X[j, idx[a]] -= 1
    X[:, -1] = 1  # home advantage column
    return X


def _ridge(X, y, lam, pen_last=False):
    P = np.eye(X.shape[1]) * lam
    if not pen_last:
        P[-1, -1] = 0
    A = X.T @ X + P
    return np.linalg.solve(A, X.T @ y), A


def _loo_error(X, y, lam):
    P = np.eye(X.shape[1]) * lam
    P[-1, -1] = 0
    H = X @ np.linalg.solve(X.T @ X + P, X.T)
    res = y - H @ y
    return float(np.mean((res / (1 - np.diag(H))) ** 2))


def team_ratings(D):
    xg = D.tm[["match_id", "team", "expected_goals"]].copy()
    mm = D.m[["match_id", "date", "league", "group_key", "home_team", "away_team", "home_score", "away_score"]].copy()
    h = xg.rename(columns={"team": "home_team", "expected_goals": "hxg"})
    a = xg.rename(columns={"team": "away_team", "expected_goals": "axg"})
    mm = mm.merge(h, on=["match_id", "home_team"]).merge(a, on=["match_id", "away_team"])
    mm["xg_margin"] = mm["hxg"] - mm["axg"]
    mm["goal_margin"] = mm["home_score"] - mm["away_score"]
    out, split_rows, elo_rows, sched = [], [], [], []
    lam_used = {}
    for league, m in mm.groupby("league"):
        teams = sorted(set(m["home_team"]) | set(m["away_team"]))
        X, y = _design(m, teams), m["xg_margin"].values
        grid = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        errs = [_loo_error(X, y, l) for l in grid]
        lam = grid[int(np.argmin(errs))]
        lam_used[league] = (lam, min(errs), float(np.mean((y - y.mean()) ** 2)))
        beta, A = _ridge(X, y, lam)
        sigma2 = float(np.mean((y - X @ beta) ** 2))
        cov = sigma2 * np.linalg.inv(A) @ (X.T @ X) @ np.linalg.inv(A)
        r = beta[:-1] - beta[:-1].mean()
        se = np.sqrt(np.diag(cov)[:-1])
        # split attack / defence
        rows = []
        for t in m.itertuples():
            rows.append((t.home_team, t.away_team, 1, t.hxg))
            rows.append((t.away_team, t.home_team, 0, t.axg))
        L = pd.DataFrame(rows, columns=["att", "dfn", "home", "xg"])
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        Z = np.zeros((len(L), 2 * n + 2))
        for j, (at, df, ho) in enumerate(zip(L["att"], L["dfn"], L["home"])):
            Z[j, idx[at]] = 1
            Z[j, n + idx[df]] = -1
        Z[:, -2] = L["home"]
        Z[:, -1] = 1
        P = np.eye(Z.shape[1]) * lam
        P[-2:, -2:] = 0
        b = np.linalg.solve(Z.T @ Z + P, Z.T @ L["xg"].values)
        att, dfn = b[:n] - b[:n].mean(), b[n:2 * n] - b[n:2 * n].mean()
        # Elo
        elo = {t: 1500.0 for t in teams}
        for t in m.sort_values("date").itertuples():
            exp = 1 / (1 + 10 ** (-(elo[t.home_team] + 60 - elo[t.away_team]) / 400))
            res = 0.5 + np.tanh(t.xg_margin / 1.5) / 2        # xG margin squashed into a 0-1 'result'
            d = 20 * (res - exp) * (1 + min(abs(t.xg_margin), 3) / 3)
            elo[t.home_team] += d
            elo[t.away_team] -= d
        ev = pd.DataFrame({"team": teams, "league": league, "Rating": r, "Rating SE": se, "Attack": att, "Defence": dfn,
                           "Elo": [elo[t] for t in teams]})
        out.append(ev)
        # raw xGD and strength of schedule
        rating = dict(zip(teams, r))
        for t in teams:
            home = m[m["home_team"] == t]
            away = m[m["away_team"] == t]
            opps = list(home["away_team"]) + list(away["home_team"])
            raw = (home["xg_margin"].sum() - away["xg_margin"].sum()) / max(len(home) + len(away), 1)
            sched.append({"team": t, "Raw xGD / match": raw, "Schedule strength": np.mean([rating[o] for o in opps]) if opps else np.nan,
                          "Matches": len(opps)})
    R = pd.concat(out, ignore_index=True).merge(pd.DataFrame(sched), on="team")
    R["Rank in league"] = R.groupby("league")["Rating"].rank(ascending=False, method="min").astype(int)
    R["Adjustment"] = R["Rating"] - R["Raw xGD / match"]
    grp = pd.concat([D.m[["home_team", "group_key"]].rename(columns={"home_team": "team"}),
                     D.m[["away_team", "group_key"]].rename(columns={"away_team": "team"})]).drop_duplicates("team")
    R = R.merge(grp, on="team", how="left")
    return SimpleNamespace(**{"table": R.sort_values(["league", "Rating"], ascending=[True, False]), "matches": mm, "lam": lam_used})


# ------------------------------------------------------------------ rating drivers
DRIVER_FEATURES = [
    ("Goals", "goals"), ("Assists", "goalassist"), ("xG", "expectedgoals"), ("xA", "expectedassists"),
    ("Shots", "totalshots"), ("Key passes", "keypass"), ("Big chances created", "bigchancecreated"),
    ("Big chances missed", "bigchancemissed"), ("Passes", "totalpass"), ("Pass accuracy %", "pass_acc"),
    ("Long balls completed", "accuratelongballs"), ("Tackles won", "wontackle"), ("Interceptions", "interceptionwon"),
    ("Clearances", "totalclearance"), ("Recoveries", "ballrecovery"), ("Duels won", "duelwon"), ("Duels lost", "duellost"),
    ("Dribbles won", "woncontest"), ("Dispossessed", "dispossessed"), ("Fouls", "fouls"), ("Was fouled", "wasfouled"),
    ("Prog. carries", "progressiveballcarriescount"), ("Error led to shot", "errorleadtoashot"),
    ("Error led to goal", "errorleadtoagoal"), ("Saves", "saves"), ("Goals prevented", "goalsprevented"),
    ("Aerials won", "aerialwon"), ("Own goal", "owngoals"),
]
GK_ONLY = {"Saves", "Goals prevented"}


def rating_model(D, min_minutes=30, lam=5.0):
    """Ridge regression of the match rating on the match stats, one model per position, standardised features.
    Returns coefficients, R2, and a player-match table with predicted rating and residual."""
    p = D.ps.copy()
    p["mins"] = num(p, "minutesplayed")
    p["rating_n"] = num(p, "rating")
    p = p[(p["mins"] >= min_minutes) & p["rating_n"].notna()].copy()
    for lab, col in DRIVER_FEATURES:
        if col == "pass_acc":
            continue
        p[col] = num(p, col).fillna(0)
    p["pass_acc"] = np.where(num(p, "totalpass").fillna(0) > 0, num(p, "accuratepass").fillna(0) / num(p, "totalpass").replace(0, np.nan) * 100, np.nan)
    p["pass_acc"] = p["pass_acc"].fillna(p["pass_acc"].median())
    p["won_match"] = 0
    coefs, preds, r2s = [], [], {}
    for pos, g in p.groupby("position"):
        feats = [(l, c) for l, c in DRIVER_FEATURES if (pos == "G" or l not in GK_ONLY)]
        feats = [(l, c) for l, c in feats if g[c].std() > 0]
        X = g[[c for _, c in feats]].values.astype(float)
        mu, sd = X.mean(0), X.std(0)
        Xs = (X - mu) / sd
        y = g["rating_n"].values
        ym = y.mean()
        A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
        b = np.linalg.solve(A, Xs.T @ (y - ym))
        fit = ym + Xs @ b
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - ym) ** 2).sum()
        r2s[pos] = r2
        c = pd.DataFrame({"position": pos, "driver": [l for l, _ in feats], "coef_per_sd": b, "sd": sd,
                          "coef_per_unit": b / sd})
        coefs.append(c)
        gg = g[["match_id", "player_id", "player", "team", "position", "mins", "rating_n", "opponent", "date"]].copy()
        gg["predicted"] = fit
        gg["residual"] = y - fit
        preds.append(gg)
    return SimpleNamespace(**{"coefs": pd.concat(coefs, ignore_index=True), "r2": r2s, "matches": pd.concat(preds, ignore_index=True)})


def over_under_rated(RM, min_matches=3):
    g = RM.matches.groupby("player_id")
    t = pd.DataFrame({"player": g["player"].last(), "team": g["team"].last(), "position": g["position"].last(),
                      "Matches": g.size(), "Minutes": g["mins"].sum(), "Avg rating": g["rating_n"].mean(),
                      "Stat-implied rating": g["predicted"].mean(), "Residual": g["residual"].mean(),
                      "Residual SD": g["residual"].std()})
    t["Residual SE"] = t["Residual SD"] / np.sqrt(t["Matches"])
    t["z"] = t["Residual"] / t["Residual SE"].replace(0, np.nan)
    return t[t["Matches"] >= min_matches].reset_index()


# ------------------------------------------------------------------ scouting
def stabilised_per90(players, col, prior_minutes=180.0, group_col="role"):
    """Per-90 rate shrunk towards the role average: (x + prior_minutes/90 * mean_rate) / (minutes/90 + prior_minutes/90)."""
    mins = players["minutesplayed"]
    raw = players[col] / mins * 90
    mean_rate = players.groupby(group_col)[col].transform("sum") / players.groupby(group_col)["minutesplayed"].transform("sum") * 90
    k = prior_minutes / 90.0
    return (players[col] + k * mean_rate) / (mins / 90.0 + k)
