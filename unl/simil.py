"""Similar-player search.

Inside the pool of one role (or its whole position group), each player is described by the role's own per-90 metrics, turned into
z-scores, and compressed with PCA (components that keep 90% of the variance, which drops noise shared across the metrics).
Similarity is measured in that PCA space:
    style           cosine of the angle between two profiles (the shape of the player's game, whatever the level)
    style and level exp(-distance^2 / (2 k)), k = number of metrics (shape and size of the profile together)
Both are scaled to 0-100.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from unl import roles as R


def role_of(D, player_id):
    r = D.roles.loc[D.roles["player_id"] == player_id, "role"]
    return r.iloc[0] if len(r) else None


def similar(D, player_id, scope="role", min_minutes=180, mode="style", var=0.9):
    """Returns (table, info).  table: every other player of the pool with a Similarity score and his PCA coordinates."""
    role = role_of(D, player_id)
    ident, raw, pct, z = R.role_frame(D.players, D.roles, role, min_minutes, by=scope)
    ident, raw, pct, z = (x.reset_index(drop=True) for x in (ident, raw, pct, z))
    pos = ident.index[ident["player_id"] == player_id]
    if len(pos) == 0:
        return None, {"role": role, "reason": "minutes"}
    pos = pos[0]
    pca = PCA(n_components=var, svd_solver="full").fit(z.values)
    E = pca.transform(z.values)
    t = E[pos]
    if mode == "style":
        den = np.linalg.norm(E, axis=1) * np.linalg.norm(t)
        sim = np.where(den > 0, E @ t / np.where(den > 0, den, 1), 0)
        score = np.clip(sim, 0, 1) * 100
    else:
        d2 = ((E - t) ** 2).sum(axis=1)
        score = np.exp(-d2 / (2 * z.shape[1])) * 100
    out = ident.copy()
    out["Similarity"] = score
    out["PC1"] = E[:, 0]
    out["PC2"] = E[:, 1] if E.shape[1] > 1 else 0.0
    prof = D.players[["player_id", "age", "club", "market_value_m", "contract_year"]] if "age" in D.players.columns else D.players[["player_id"]]
    out = out.merge(prof, on="player_id", how="left")
    info = {"role": role, "pos": pos, "metrics": list(raw.columns), "pct": pct, "raw": raw, "ident": ident, "ncomp": int(E.shape[1]),
            "explained": float(pca.explained_variance_ratio_.sum()), "pool": len(ident)}
    return out, info
