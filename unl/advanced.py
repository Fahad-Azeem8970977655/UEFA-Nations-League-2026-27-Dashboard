"""Advanced analytics: expected points, expected threat (xT), team style clusters, player percentiles.

Everything is computed once in attach_advanced(D) (called at the end of load_all) and hangs off D:
    D.xpts_match   one row per match: win / draw / loss probability from shot-level xG, plus xPts
    D.xpts_grid    {match_id: scoreline probability matrix}
    D.xpts_team    one row per team: Pts vs xPts, xG vs goals, "true" position inside the group
    D.xt           namespace: grid (the xT surface), moves (passes + carries with xT gain), player, team
    D.style        team style features, PCA coordinates and loadings
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from unl.helpers import num
from unl.pitch import shot_xy

MAXG = 12  # goals considered per team when building scoreline probabilities


# ------------------------------------------------------------------ expected points
def poisson_binomial(p):
    """Exact distribution of the number of goals given each shot's xG (independent shots)."""
    dist = np.zeros(MAXG + 1)
    dist[0] = 1.0
    for q in p:
        shifted = np.concatenate([[0.0], dist[:-1]])
        dist = dist * (1 - q) + shifted * q
    return dist


def scoreline_matrix(home_xg, away_xg):
    m = np.outer(poisson_binomial(home_xg), poisson_binomial(away_xg))
    return m / m.sum()


def build_xpts(D):
    sh = D.shots.copy()
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    by = {k: g["xg"].values for k, g in sh.groupby(["match_id", "team"])}
    rows, grids = [], {}
    for r in D.m.itertuples():
        M = scoreline_matrix(by.get((r.match_id, r.home_team), []), by.get((r.match_id, r.away_team), []))
        grids[r.match_id] = M[:7, :7]
        win, draw, loss = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
        rows.append({"match_id": r.match_id, "home_team": r.home_team, "away_team": r.away_team,
                     "p_home": win, "p_draw": draw, "p_away": loss,
                     "xpts_home": 3 * win + draw, "xpts_away": 3 * loss + draw})
    return pd.DataFrame(rows), grids


def build_xpts_team(D, xm):
    h = xm[["match_id", "home_team", "xpts_home", "p_home", "p_draw", "p_away"]].rename(
        columns={"home_team": "team", "xpts_home": "xpts"})
    a = xm[["match_id", "away_team", "xpts_away", "p_away", "p_draw", "p_home"]].copy()
    a.columns = ["match_id", "team", "xpts", "p_win", "p_draw", "p_loss"]
    h.columns = ["match_id", "team", "xpts", "p_win", "p_draw", "p_loss"]
    long = pd.concat([h, a])
    t = D.tm.merge(long, on=["match_id", "team"], how="left")
    g = t.groupby("team")
    out = pd.DataFrame({
        "league": g["league"].first(), "group_key": g["group_key"].first(), "Matches": g.size(),
        "Pts": g["points"].sum(), "xPts": g["xpts"].sum(),
        "GF": g["gf"].sum(), "xG": g["expected_goals"].sum(),
        "GA": g["ga"].sum(), "xGA": g["opp_expected_goals"].sum()})
    out["Pts - xPts"] = out["Pts"] - out["xPts"]
    out["GD"] = out["GF"] - out["GA"]
    out["xGD"] = out["xG"] - out["xGA"]
    if not D.standings.empty:
        out["Pos"] = D.standings.set_index("team")["position"].reindex(out.index)
    else:
        out["Pos"] = np.nan
    out = out.sort_values(["group_key", "xPts", "xGD"], ascending=[True, False, False])
    out["xPos"] = out.groupby("group_key").cumcount() + 1
    return out, t


# ------------------------------------------------------------------ expected threat
NX, NY = 16, 12


def zone(x, y):
    zx = np.clip((np.asarray(x) / 100 * NX).astype(int), 0, NX - 1)
    zy = np.clip((np.asarray(y) / 100 * NY).astype(int), 0, NY - 1)
    return zx * NY + zy


def fit_xt(D, iters=40, prior=3.0):
    """Karun Singh style xT fitted on this dataset.
    xT(z) = s(z) * g(z) + (1 - s(z)) * sum_z' T(z, z') xT(z')
    s = share of actions in the zone that are shots, g = average xG of a shot from the zone,
    T = probability that a pass / carry from the zone is completed into zone z'.  Penalties are left out."""
    p = D.passes[D.passes["event_action_type"].isin(["pass", "cross"])].dropna(subset=["x", "y", "end_x", "end_y"])
    c = D.carries.dropna(subset=["x", "y", "end_x", "end_y"])
    cols = ["match_id", "team", "player_id", "player", "x", "y", "end_x", "end_y"]
    mv = pd.concat([p[cols].assign(kind="pass", ok=p["outcome"].values),
                    c[cols].assign(kind="carry", ok=True)], ignore_index=True)
    z0, z1 = zone(mv["x"], mv["y"]), zone(mv["end_x"], mv["end_y"])
    Z = NX * NY
    n_moves = np.bincount(z0, minlength=Z).astype(float)
    ok = mv["ok"].values.astype(bool)
    T = np.zeros((Z, Z))
    np.add.at(T, (z0[ok], z1[ok]), 1.0)
    T /= np.maximum(n_moves, 1)[:, None]

    sh = D.shots[D.shots["situation"] != "penalty"]
    sx, sy = shot_xy(sh)
    zs = zone(sx, sy)
    xg = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03).values
    n_shots = np.bincount(zs, minlength=Z).astype(float)
    g = (np.bincount(zs, weights=xg, minlength=Z) + prior * xg.mean()) / (n_shots + prior)
    s = n_shots / np.maximum(n_shots + n_moves, 1)
    xt = np.zeros(Z)
    for _ in range(iters):
        xt = s * g + (1 - s) * (T @ xt)
    mv["xt_gain"] = np.where(ok, xt[z1] - xt[z0], 0.0)
    mv["z0"], mv["z1"] = z0, z1
    return xt.reshape(NX, NY), mv


def xt_tables(D, moves):
    mins = D.players.set_index("player_id")["minutesplayed"]
    pl = moves.pivot_table(index="player_id", columns="kind", values="xt_gain", aggfunc="sum", fill_value=0)
    pl = pl.rename(columns={"pass": "xt_pass", "carry": "xt_carry"})
    for c in ("xt_pass", "xt_carry"):
        if c not in pl:
            pl[c] = 0.0
    pl["xt_total"] = pl["xt_pass"] + pl["xt_carry"]
    pl["minutes"] = mins.reindex(pl.index)
    pl["xt_per90"] = pl["xt_total"] / pl["minutes"] * 90
    tm = moves.groupby(["team", "kind"])["xt_gain"].sum().unstack(fill_value=0)
    tm = tm.rename(columns={"pass": "Passing xT", "carry": "Carrying xT"})
    tm["Total xT"] = tm.sum(axis=1)
    n = D.tm.groupby("team").size()
    tm["xT per match"] = tm["Total xT"] / n.reindex(tm.index)
    return pl, tm.sort_values("xT per match", ascending=False)


# ------------------------------------------------------------------ team style
STYLE_FEATURES = ["Possession %", "Pass length (m)", "Long ball %", "Cross %", "Forward per pass (m)",
                  "Final third pass %", "Press intensity", "Regain height"]


def build_style(D):
    from unl.viz2 import PRESS_TYPES, pressing_table  # local import: viz2 pulls in pitch drawing

    p = D.passes[D.passes["event_action_type"].isin(["pass", "cross"])].dropna(subset=["x", "y", "end_x", "end_y"])
    dx, dy = (p["end_x"] - p["x"]) * 1.05, (p["end_y"] - p["y"]) * 0.68
    q = pd.DataFrame({"team": p["team"].values, "len": np.hypot(dx, dy).values, "fwd": dx.values,
                      "long": p["is_long_ball"].values.astype(float),
                      "cross": (p["event_action_type"] == "cross").values.astype(float),
                      "f3": (p["x"] >= 66.7).values.astype(float)})
    g = q.groupby("team")
    f = pd.DataFrame({"Pass length (m)": g["len"].mean(), "Long ball %": g["long"].mean() * 100,
                      "Cross %": g["cross"].mean() * 100, "Forward per pass (m)": g["fwd"].mean(),
                      "Final third pass %": g["f3"].mean() * 100})
    f["Possession %"] = D.team_summary["Possession %"]
    pt = pressing_table(D)
    f["Press intensity"] = 1 / pt["PPDA"]
    f["Regain height"] = pt["Avg regain height"]
    f = f[STYLE_FEATURES].dropna()
    z = (f - f.mean()) / f.std(ddof=0).replace(0, 1)
    u, sv, vt = np.linalg.svd(z.values, full_matrices=False)
    coords = pd.DataFrame(u[:, :2] * sv[:2], index=f.index, columns=["PC1", "PC2"])
    load = pd.DataFrame(vt[:2].T, index=STYLE_FEATURES, columns=["PC1", "PC2"])
    var = (sv ** 2) / (sv ** 2).sum()
    return SimpleNamespace(features=f, z=z, coords=coords, loadings=load, explained=var[:2])


def kmeans(X, k, seed=7, n_init=25, iters=100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(centers)
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1]


def cluster_teams(style, k):
    """Returns Series team -> label such as 'Possession, Final third pass %' (the two features that define it)."""
    lab = kmeans(style.z.values, k)
    s = pd.Series(lab, index=style.z.index)
    names = {}
    for j in range(k):
        cen = style.z[s == j].mean()
        top = cen.sort_values(ascending=False)
        hi = [f for f in top.index[:2] if top[f] > 0.35]
        lo = [f for f in top.index[::-1][:1] if top[f] < -0.5]
        clean = lambda f: f.replace(" %", "").replace(" (m)", "").lower()
        parts = [f"high {clean(f)}" for f in hi] + [f"low {clean(f)}" for f in lo]
        names[j] = f"{j + 1}. " + (", ".join(parts) if parts else "balanced")
    return s.map(names)


def nearest_teams(style, team, n=5):
    d = np.sqrt(((style.z - style.z.loc[team]) ** 2).sum(axis=1)).drop(team).sort_values()
    return d.head(n)


# ------------------------------------------------------------------ player percentiles and similarity
# per position: (label, column, kind)  kind: "p90" = per 90 minutes, "val" = use value as is
POS_METRICS = {
    "F": [("Goals", "goals", "p90"), ("xG", "expectedgoals", "p90"), ("Shots", "totalshots", "p90"),
          ("xA", "expectedassists", "p90"), ("Key passes", "keypass", "p90"), ("Dribbles won", "woncontest", "p90"),
          ("Prog. carries", "progressiveballcarriescount", "p90"), ("Duel win %", "duel_win", "val"),
          ("Touches", "touches", "p90"), ("xT", "xt_total", "p90")],
    "M": [("xA", "expectedassists", "p90"), ("Key passes", "keypass", "p90"), ("Pass acc. %", "pass_acc", "val"),
          ("Prog. carries", "progressiveballcarriescount", "p90"), ("Tackles", "totaltackle", "p90"),
          ("Interceptions", "interceptionwon", "p90"), ("Recoveries", "ballrecovery", "p90"),
          ("Dribbles won", "woncontest", "p90"), ("Duel win %", "duel_win", "val"), ("xT", "xt_total", "p90")],
    "D": [("Tackles", "totaltackle", "p90"), ("Interceptions", "interceptionwon", "p90"),
          ("Clearances", "totalclearance", "p90"), ("Recoveries", "ballrecovery", "p90"),
          ("Duel win %", "duel_win", "val"), ("Pass acc. %", "pass_acc", "val"), ("Passes", "totalpass", "p90"),
          ("Prog. carries", "progressiveballcarriescount", "p90"), ("xT", "xt_total", "p90")],
    "G": [("Saves", "saves", "p90"), ("Goals prevented", "goalsprevented", "p90"),
          ("Pass acc. %", "pass_acc", "val"), ("Passes", "totalpass", "p90")],
}
POS_NAMES = {"F": "Forwards", "M": "Midfielders", "D": "Defenders", "G": "Goalkeepers"}


def position_frame(players, pos, min_minutes):
    """Per-90 metric table for one position, and the percentile of every player among their position peers."""
    d = players[(players["position"] == pos) & (players["minutesplayed"] >= min_minutes)].copy()
    raw = pd.DataFrame(index=d.index)
    for label, col, kind in POS_METRICS[pos]:
        raw[label] = d[col] / d["minutesplayed"] * 90 if kind == "p90" else d[col]
    raw = raw.fillna(raw.median())
    pct = raw.rank(pct=True) * 100
    z = (raw - raw.mean()) / raw.std(ddof=0).replace(0, 1)
    ident = d[["player_id", "player", "team", "minutesplayed"]]
    return ident, raw, pct, z


def similar_players(ident, z, player_id, n=8):
    idx = ident.index[ident["player_id"] == player_id]
    if len(idx) == 0:
        return pd.DataFrame()
    v = z.loc[idx[0]].values
    zz = z.values
    cos = zz @ v / (np.linalg.norm(zz, axis=1) * np.linalg.norm(v) + 1e-9)
    out = ident.assign(Similarity=np.clip(cos, -1, 1) * 100)
    return out[out["player_id"] != player_id].sort_values("Similarity", ascending=False).head(n)


# ------------------------------------------------------------------ entry point
def attach_advanced(D):
    D.xpts_match, D.xpts_grid = build_xpts(D)
    D.xpts_team, D.xpts_long = build_xpts_team(D, D.xpts_match)
    grid, moves = fit_xt(D)
    pl, tm = xt_tables(D, moves)
    D.xt = SimpleNamespace(grid=grid, moves=moves, player=pl, team=tm)
    for c in ("xt_pass", "xt_carry", "xt_total"):
        D.players[c] = D.players["player_id"].map(pl[c]).fillna(0.0)
    D.style = build_style(D)
    return D
