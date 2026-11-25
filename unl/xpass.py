"""Expected pass completion (xP).

A gradient-boosted classifier estimates the probability that each pass is completed, from where it starts and
ends, its length, direction, whether it is a cross or a long ball, and the pitch zone.  Probabilities are
out-of-fold (5-fold, split by match), so a pass is never scored by a model that saw it.

    pass completed over expected (PoE) = completed passes - sum of xP
A positive PoE means the player completes more than the difficulty of his passes predicts.
Raw PoE is shrunk towards zero for small samples (see shrink_poe).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover - fall back to a logistic model written in numpy
    HistGradientBoostingClassifier = None

FEATURES = ["x", "y", "end_x", "end_y", "length", "dx", "dy", "angle", "long", "cross", "dist_goal_end",
            "dist_goal_start", "centrality_start", "centrality_end", "end_in_box"]
RISKY_BELOW = 0.60  # a pass the model gives under 60% is counted as a "difficult" pass


def _features(p):
    dx, dy = (p["end_x"] - p["x"]) * 1.05, (p["end_y"] - p["y"]) * 0.68
    f = pd.DataFrame({
        "x": p["x"].values, "y": p["y"].values, "end_x": p["end_x"].values, "end_y": p["end_y"].values,
        "dx": dx.values, "dy": dy.values, "length": np.hypot(dx, dy).values, "angle": np.arctan2(dy, dx).values,
        "long": p["is_long_ball"].astype(float).values, "cross": (p["event_action_type"] == "cross").astype(float).values,
    })
    f["dist_goal_end"] = np.hypot((100 - p["end_x"]) * 1.05, (50 - p["end_y"]) * 0.68).values
    f["dist_goal_start"] = np.hypot((100 - p["x"]) * 1.05, (50 - p["y"]) * 0.68).values
    f["centrality_start"] = (p["y"] - 50).abs().values
    f["centrality_end"] = (p["end_y"] - 50).abs().values
    f["end_in_box"] = ((p["end_x"] >= 84.3) & ((p["end_y"] - 50).abs() <= 29.6)).astype(float).values
    return f[FEATURES]


def _logit_fit_predict(Xtr, ytr, Xte, l2=1.0, iters=25):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(A.shape[1])
        w += np.linalg.solve(H, A.T @ (ytr - p) - l2 * w)
    return 1 / (1 + np.exp(-B @ w))


def fit_xpass(D, folds=5, seed=11):
    p = D.passes[D.passes["event_action_type"].isin(["pass", "cross"])].dropna(subset=["x", "y", "end_x", "end_y"]).copy()
    p = p.reset_index(drop=True)
    X, y = _features(p), p["outcome"].astype(int).values
    matches = p["match_id"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(matches, rng.permutation(len(matches)) % folds))
    f = p["match_id"].map(fold_of).values
    xp = np.zeros(len(p))
    for k in range(folds):
        tr, te = f != k, f == k
        if HistGradientBoostingClassifier is not None:
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0,
                                               random_state=seed)
            m.fit(X[tr], y[tr])
            xp[te] = m.predict_proba(X[te])[:, 1]
        else:
            xp[te] = _logit_fit_predict(X[tr].values, y[tr], X[te].values)
    p["xp"] = np.clip(xp, 0.01, 0.995)
    p["ok"] = y
    p["difficult"] = p["xp"] < RISKY_BELOW
    p["gain"] = (p["end_x"] - p["x"]) * 1.05
    # model quality
    brier = float(np.mean((p["xp"] - p["ok"]) ** 2))
    base = float(np.mean((p["ok"].mean() - p["ok"]) ** 2))
    ll = float(-np.mean(p["ok"] * np.log(p["xp"]) + (1 - p["ok"]) * np.log(1 - p["xp"])))
    # reliability by decile
    p["_bin"] = pd.qcut(p["xp"], 10, duplicates="drop")
    cal = p.groupby("_bin", observed=True).agg(pred=("xp", "mean"), actual=("ok", "mean"), n=("ok", "size")).reset_index(drop=True)
    p = p.drop(columns="_bin")
    return p, SimpleNamespace(brier=brier, brier_base=base, skill=1 - brier / base, logloss=ll, calibration=cal,
                              accuracy=float(((p["xp"] > .5) == (p["ok"] == 1)).mean()), n=len(p),
                              engine="gradient boosting" if HistGradientBoostingClassifier is not None else "logistic regression")


def shrink_poe(poe, n, k=250.0):
    """Shrink raw passes-over-expected towards zero: poe * n / (n + k)."""
    return poe * n / (n + k)


def player_table(D, p):
    g = p.groupby("player_id")
    t = pd.DataFrame({"passes": g.size(), "completed": g["ok"].sum(), "expected": g["xp"].sum(),
                      "avg_xp": g["xp"].mean(), "difficult": g["difficult"].sum(), "difficult_ok": g.apply(
                          lambda d: d.loc[d["difficult"], "ok"].sum(), include_groups=False),
                      "avg_len": g.apply(lambda d: np.hypot((d["end_x"] - d["x"]) * 1.05, (d["end_y"] - d["y"]) * .68).mean(), include_groups=False),
                      "fwd_gain": g["gain"].mean()})
    t["PoE"] = t["completed"] - t["expected"]
    t["PoE per 100"] = t["PoE"] / t["passes"] * 100
    t["PoE shrunk"] = shrink_poe(t["PoE"], t["passes"])
    t["PoE shrunk per 100"] = t["PoE shrunk"] / t["passes"] * 100
    t["Risk %"] = (1 - t["avg_xp"]) * 100
    t["Difficult %"] = t["difficult"] / t["passes"] * 100
    t["Difficult acc %"] = np.where(t["difficult"] > 0, t["difficult_ok"] / t["difficult"].replace(0, np.nan) * 100, np.nan)
    t["Accuracy %"] = t["completed"] / t["passes"] * 100
    t["Expected acc %"] = t["avg_xp"] * 100
    meta = D.players.set_index("player_id")[["player", "team", "position", "minutesplayed"]]
    t = t.join(meta)
    if "role" in getattr(D, "roles", pd.DataFrame()).columns:
        t = t.join(D.roles.set_index("player_id")["role"])
    return t.reset_index()


def team_table(p):
    g = p.groupby("team")
    t = pd.DataFrame({"passes": g.size(), "completed": g["ok"].sum(), "expected": g["xp"].sum(), "avg_xp": g["xp"].mean(),
                      "difficult": g["difficult"].mean() * 100, "fwd": g["gain"].mean(),
                      "long": g["is_long_ball"].mean() * 100,
                      "len": g.apply(lambda d: np.hypot((d["end_x"] - d["x"]) * 1.05, (d["end_y"] - d["y"]) * .68).mean(), include_groups=False)})
    t["PoE"] = t["completed"] - t["expected"]
    t["PoE per 100"] = t["PoE"] / t["passes"] * 100
    t["Risk %"] = (1 - t["avg_xp"]) * 100
    t["Accuracy %"] = t["completed"] / t["passes"] * 100
    t["Expected acc %"] = t["avg_xp"] * 100
    t["Difficult %"] = t["difficult"]
    t["Forward per pass (m)"] = t["fwd"]
    t["Long ball %"] = t["long"]
    t["Pass length (m)"] = t["len"]
    return t.drop(columns=["difficult", "fwd", "long", "len"])
