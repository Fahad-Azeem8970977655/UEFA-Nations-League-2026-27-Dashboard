"""Defensive shape from average positions.

For every team in every match, the average positions of the outfield players (starters and anyone with enough
touches) give:
    line height   mean x of the four deepest outfield players (0 = own goal, 100 = opponent goal)
    length        mean x of the three highest outfield players minus line height (distance between lines)
    width         mean y-spread (max - min) of the six widest-spread outfield players' average y, in pitch %
    area          convex hull area of the outfield players, as % of the pitch
Average positions mix attacking and defending phases, so these describe a team's overall shape, not only the
block without the ball.
"""
import numpy as np
import pandas as pd

MIN_POINTS = 20  # touches needed for a player's average position to count


def _hull_area(pts):
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return np.nan
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    hull = half(p)[:-1] + half(p[::-1])[:-1]
    x, y = np.array(hull)[:, 0], np.array(hull)[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 100.0  # % of pitch (100x100 frame)


def build_shape(D):
    gk = D.ps.loc[D.ps["position"] == "G", ["match_id", "player_id"]].assign(is_gk=True)
    a = D.avg.merge(gk, on=["match_id", "player_id"], how="left")
    a = a[a["is_gk"].isna() & (a["points_count"] >= MIN_POINTS)]
    rows = []
    for (mid, team), g in a.groupby(["match_id", "team"]):
        if len(g) < 8:
            continue
        g = g.sort_values("average_x")
        x, y = g["average_x"].values, g["average_y"].values
        line = x[:4].mean()
        rows.append({"match_id": mid, "team": team, "n_players": len(g), "line_height": line,
                     "length": x[-3:].mean() - line, "width": np.ptp(y), "std_x": x.std(), "std_y": y.std(),
                     "area": _hull_area(np.column_stack([x, y])), "cb_gap": np.ptp(y[:4])})
    s = pd.DataFrame(rows)
    keep = ["match_id", "team", "opponent", "side", "date", "points", "expected_goals", "opp_expected_goals",
            "opp_total_shots", "opp_big_chances", "ball_possession", "gf", "ga", "opp_touches_in_penalty_area", "result"]
    tm = D.tm[[c for c in keep if c in D.tm.columns]]
    s = s.merge(tm, on=["match_id", "team"], how="left")
    s["xga"] = s["opp_expected_goals"]
    s["xga_per_shot"] = s["xga"] / s["opp_total_shots"].replace(0, np.nan)
    s["line_dev"] = s["line_height"] - s.groupby("team")["line_height"].transform("mean")
    return s


def team_shape_table(s):
    g = s.groupby("team")
    t = g[["line_height", "length", "width", "area"]].mean()
    t["line_sd"] = g["line_height"].std()
    t["xGA"] = g["xga"].mean()
    t["Shots against"] = g["opp_total_shots"].mean()
    t["xGA per shot"] = g["xga"].sum() / g["opp_total_shots"].sum()
    t["Matches"] = g.size()
    return t


def _ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def line_effect(s, ycol="xga", within=False, control_possession=False, n_boot=600, seed=3):
    """Effect of a 10-point higher defensive line on `ycol`, with a bootstrap 95% interval (resampling matches).
    within=True removes each team's own average first (does a team concede more *when* it plays higher?)."""
    d = s.dropna(subset=["line_height", ycol]).copy()
    if control_possession:
        d = d.dropna(subset=["ball_possession"])
    xcols = ["line_height"] + (["ball_possession"] if control_possession else [])
    if within:
        for c in xcols + [ycol]:
            d[c] = d[c] - d.groupby("team")[c].transform("mean")
    X, y = d[xcols].values, d[ycol].values
    if len(d) < 10:
        return None
    b = _ols(X, y)[1] * 10
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boots.append(_ols(X[i], y[i])[1] * 10)
    r = np.corrcoef(d["line_height"], d[ycol])[0, 1]
    return {"effect": b, "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5), "r": r, "n": len(d)}
