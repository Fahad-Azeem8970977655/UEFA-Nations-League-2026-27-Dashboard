"""Market-value model.

Predicts the log of a player's market value from age, minutes, per-90 output, SofaScore rating, role, height, foot, contract length and the
league (A-D) of his national team.  Models are compared with 5-fold cross-validation: a simple baseline (median value by league and role), a
ridge regression and a gradient-boosting model.  A club-level feature (the average log value of his club-mates in this dataset, never including himself or the held-out players)
stands in for club size.  Every player gets an out-of-fold prediction (made by a model that never saw him), so the gap
between real and predicted value is a fair "over / under valued" signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REF_YEAR = 2026
P90 = {"xt_total": "xT", "goals": "Goals", "expectedgoals": "xG", "expectedassists": "xA", "keypass": "Key passes", "bigchancecreated": "Big chances created",
       "totalshots": "Shots", "progressiveballcarriescount": "Progressive carries", "woncontest": "Dribbles won", "totaltackle": "Tackles",
       "interceptionwon": "Interceptions", "ballrecovery": "Recoveries", "totalclearance": "Clearances", "aerialwon": "Aerial duels won",
       "saves": "Saves", "goalsprevented": "Goals prevented"}
LEAGUE_LEVEL = {"League A": 3, "League B": 2, "League C": 1, "League D": 0}


def feature_frame(D, min_minutes=90):
    p = D.players.copy()
    p = p[(p["minutesplayed"] >= min_minutes) & p["market_value_m"].notna() & (p["market_value_m"] > 0)].copy()
    lg = pd.concat([D.m[["home_team", "league"]].rename(columns={"home_team": "team"}), D.m[["away_team", "league"]].rename(columns={"away_team": "team"})]).drop_duplicates("team")
    p = p.merge(lg, on="team", how="left")
    X = pd.DataFrame(index=p.index)
    X["Age"] = p["age"]
    X["Height"] = p["height_cm"]
    X["Minutes"] = p["minutesplayed"]
    X["SofaScore rating"] = p["rating_mean"]
    X["Contract years left"] = p["contract_year"] - REF_YEAR
    X["National team league level"] = p["league"].map(LEAGUE_LEVEL)
    X["Left-footed"] = (p["preferred_foot"] == "Left").astype(float)
    for col, lab in P90.items():
        X[lab + " p90"] = p[col] / p["minutesplayed"] * 90
    for r in sorted(p["role"].dropna().unique()):
        X["Role: " + r] = (p["role"] == r).astype(float)
    return p, X


CLUB_COL = "Club level (log value of club-mates)"


def club_feature(club, y, train, apply, leave_self_out):
    """Mean log value of the other players of the same club among the training players (NaN when the club has no other known player)."""
    df = pd.DataFrame({"club": club[train], "y": y[train]}).groupby("club")["y"].agg(["sum", "count"])
    sums, cnts = pd.Series(club[apply]).map(df["sum"]).fillna(0).values, pd.Series(club[apply]).map(df["count"]).fillna(0).values
    if leave_self_out:
        sums, cnts = sums - y[apply], cnts - 1
    return np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1), np.nan)


def _models():
    return {"Median by league and role": None,
            "Ridge regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 12))),
            "Gradient boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0, min_samples_leaf=15, random_state=0)}


def fit_value_model(D, min_minutes=90, seed=0):
    p, X = feature_frame(D, min_minutes)
    y = np.log(p["market_value_m"].values)
    kf = KFold(5, shuffle=True, random_state=seed)
    oof = {k: np.zeros(len(y)) for k in _models()}
    imps = []
    club = p["club"].fillna("?").astype(str).values
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        Xtr[CLUB_COL] = club_feature(club, y, tr, tr, True)
        Xte[CLUB_COL] = club_feature(club, y, tr, te, False)
        # baseline: median log value of the same league level and role, falling back to the league level, then to everything
        keys = p["league"].astype(str) + "|" + p["role"].astype(str)
        med_k = pd.Series(y[tr], index=keys.iloc[tr].values).groupby(level=0).median()
        med_l = pd.Series(y[tr], index=p["league"].astype(str).iloc[tr].values).groupby(level=0).median()
        base = [med_k.get(k, med_l.get(k.split("|")[0], np.median(y[tr]))) for k in keys.iloc[te]]
        oof["Median by league and role"][te] = base
        for name, mdl in _models().items():
            if mdl is None:
                continue
            mdl.fit(Xtr, y[tr])
            oof[name][te] = mdl.predict(Xte)
            if name == "Gradient boosting":
                pi = permutation_importance(mdl, Xte, y[te], n_repeats=5, random_state=seed, scoring="r2")
                imps.append(pi.importances_mean)
    rows = []
    for name, pr in oof.items():
        rows.append({"Model": name, "R2 (log value)": 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum(), "Typical error %": (np.exp(np.median(np.abs(y - pr))) - 1) * 100,
                     "Rank correlation": spearmanr(y, pr)[0], "n": len(y)})
    cv = pd.DataFrame(rows)
    best = "Gradient boosting"
    imp = pd.Series(np.mean(imps, axis=0), index=list(X.columns) + [CLUB_COL]).sort_values(ascending=False)
    # group the role dummies into one "Role" driver
    imp_g = pd.concat([imp[~imp.index.str.startswith("Role: ")], pd.Series({"Role": imp[imp.index.str.startswith("Role: ")].sum()})]).sort_values(ascending=False)
    Xf = X.copy()
    Xf[CLUB_COL] = club_feature(club, y, np.arange(len(y)), np.arange(len(y)), True)
    final = _models()[best].fit(Xf, y)
    ages = np.arange(18, 39)
    pdp = []
    for a in ages:
        Xa = Xf.copy()
        Xa["Age"] = a
        pdp.append(float(np.exp(final.predict(Xa)).mean()))
    p = p.copy()
    p["model_value_m"] = np.exp(oof[best])
    p["ratio"] = p["market_value_m"] / p["model_value_m"]
    p["log_gap"] = y - oof[best]
    return {"players": p, "cv": cv, "importance": imp_g, "pdp": pd.DataFrame({"Age": ages, "Value": pdp}), "n": len(y), "n_features": Xf.shape[1]}
