"""Bayesian team-strength model with Dixon-Coles scorelines.

Per league, the expected goals of a side are
    log(lambda_home) = mu + home_edge + attack[home] + leak[away]
    log(lambda_away) = mu + attack[away] + leak[home]
The attack and leak parameters get zero-centred normal priors (precision `tau`), so teams with little evidence stay close to the
league average.  The target is a blend of goals and xG, y = w * xG + (1 - w) * goals, fitted with the Poisson likelihood.  The
posterior is approximated by a normal distribution around the posterior mode (Laplace approximation, spread scaled by the quasi-Poisson
dispersion estimated from the residuals), which gives a credible range
for every strength and, by sampling, for every match probability.

Scorelines use the Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1), with the correlation rho fitted on the played matches.

Settings tau and w are chosen by leave-one-matchday-out cross-validation, against two reference models: a flat base rate and the
ridge model of unl.predict.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from unl import predict as PR

MAXG = 8
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WS = (0.0, 0.5, 1.0)
EQUIV = 0.01  # log-loss gap treated as no difference


# ------------------------------------------------------------------ fitting
def _design(sub, idx, n, w):
    Z = np.zeros((2 * len(sub), 2 * n + 2))
    y = np.zeros(2 * len(sub))
    for k, r in enumerate(sub.itertuples()):
        i = 2 * k
        Z[i, 0], Z[i, 1] = 1, 1
        Z[i, 2 + idx[r.home_team]] = 1
        Z[i, 2 + n + idx[r.away_team]] = 1
        y[i] = w * r.hxg + (1 - w) * r.home_score
        Z[i + 1, 0] = 1
        Z[i + 1, 2 + idx[r.away_team]] = 1
        Z[i + 1, 2 + n + idx[r.home_team]] = 1
        y[i + 1] = w * r.axg + (1 - w) * r.away_score
    return Z, y


def fit_league(sub, tau, w):
    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    Z, y = _design(sub, idx, n, w)
    P = np.diag([1e-4, 1.0] + [tau] * (2 * n))
    th = np.zeros(2 * n + 2)
    th[0] = np.log(max(y.mean(), 0.1))
    for _ in range(50):
        lam = np.exp(Z @ th)
        g = Z.T @ (y - lam) - P @ th
        H = Z.T @ (lam[:, None] * Z) + P
        step = np.linalg.solve(H, g)
        th = th + step
        if np.abs(step).max() < 1e-9:
            break
    # quasi-Poisson dispersion: xG targets vary less than a Poisson count, goals about as much; scales the posterior spread
    lam = np.exp(Z @ th)
    df_eff = len(th) - np.trace(np.linalg.solve(H, P))
    phi = float(np.clip(((y - lam) ** 2 / lam).sum() / max(len(y) - df_eff, 1.0), 0.2, 3.0))
    return {"teams": teams, "idx": idx, "n": n, "th": th, "H": H, "phi": phi}


def lambdas(model, home, away, neutral=False, th=None):
    th = model["th"] if th is None else th
    n, idx = model["n"], model["idx"]
    ah = th[2 + idx[home]] if home in idx else 0.0
    aa = th[2 + idx[away]] if away in idx else 0.0
    lh_ = th[2 + n + idx[home]] if home in idx else 0.0
    la_ = th[2 + n + idx[away]] if away in idx else 0.0
    lh = np.exp(th[0] + (0 if neutral else th[1]) + ah + la_)
    la = np.exp(th[0] + aa + lh_)
    return lh, la


def draws(model, n=300, seed=11):
    """Posterior draws of the parameter vector (Laplace approximation)."""
    cov = np.linalg.inv(model["H"]) * model.get("phi", 1.0)
    L = np.linalg.cholesky((cov + cov.T) / 2 + 1e-10 * np.eye(len(cov)))
    rng = np.random.default_rng(seed)
    return model["th"] + (L @ rng.standard_normal((len(cov), n))).T


# ------------------------------------------------------------------ Dixon-Coles scorelines
def _tau(i, j, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((i == 0) & (j == 0), 1 - lh * la * rho, t)
    t = np.where((i == 0) & (j == 1), 1 + lh * rho, t)
    t = np.where((i == 1) & (j == 0), 1 + la * rho, t)
    t = np.where((i == 1) & (j == 1), 1 - rho, t)
    return t


def fit_rho(lh, la, hg, ag):
    lh, la, hg, ag = (np.asarray(v, dtype=float) for v in (lh, la, hg, ag))

    def nll(r):
        t = _tau(hg, ag, lh, la, r)
        return 1e6 if (t <= 0).any() else -np.log(t).sum()

    return float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)


def grid(lh, la, rho=0.0):
    k = np.arange(MAXG + 1)
    M = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def probs(M):
    return np.array(PR.outcome_probs(M))


def fit_all(matches, tau, w):
    leagues = {lg: fit_league(sub, tau, w) for lg, sub in matches.groupby("league")}
    lh, la = [], []
    for r in matches.itertuples():
        a, b = lambdas(leagues[r.league], r.home_team, r.away_team)
        lh.append(a)
        la.append(b)
    return {"leagues": leagues, "rho": fit_rho(lh, la, matches["home_score"], matches["away_score"]), "tau": tau, "w": w}


def predict(fit, league, home, away, neutral=False, dc=True):
    lh, la = lambdas(fit["leagues"][league], home, away, neutral)
    return lh, la, grid(lh, la, fit["rho"] if dc else 0.0)


def predict_draws(fit, league, home, away, neutral=False, n=300):
    m = fit["leagues"][league]
    out = []
    for th in draws(m, n):
        lh, la = lambdas(m, home, away, neutral, th)
        out.append(probs(grid(lh, la, fit["rho"])))
    return np.array(out)


# ------------------------------------------------------------------ cross-validation
def _scores(p, res):
    onehot = np.zeros(3)
    onehot[res] = 1
    cum = np.cumsum(p)[:2] - np.cumsum(onehot)[:2]
    return {"ll": -np.log(max(p[res], 1e-9)), "brier": float(((p - onehot) ** 2).sum()), "rps": float(0.5 * (cum ** 2).sum()),
            "hit": float(np.argmax(p) == res), "pdraw": float(p[1]), "draw": float(res == 1)}


def match_table(D):
    m = D.rate.matches.merge(D.m[["match_id", "matchday"]], on="match_id", how="left")
    m["res"] = np.where(m["home_score"] > m["away_score"], 0, np.where(m["home_score"] == m["away_score"], 1, 2))
    return m


def cross_validate(D, taus=TAUS, ws=WS):
    """Leave-one-matchday-out predictions of every played match for each model.  Returns (summary table, per-match table, best config)."""
    m = match_table(D)
    recs = {}

    def add(model, r, p):
        recs.setdefault(model, []).append({"match_id": r.match_id, **_scores(np.asarray(p, dtype=float), r.res)})

    for day in sorted(m["matchday"].dropna().unique()):
        tr, te = m[m["matchday"] != day], m[m["matchday"] == day]
        freq = np.bincount(tr["res"], minlength=3) / len(tr)
        stub = SimpleNamespace(rate=SimpleNamespace(matches=tr, lam=D.rate.lam))
        S = PR.fit_strength(stub)
        for r in te.itertuples():
            add("Flat base rate", r, freq)
            lh, la = PR.expected_goals(S, r.league, r.home_team, r.away_team)
            add("Ridge on xG (current)", r, PR.outcome_probs(PR.scoreline_probs(lh, la)))
        for tau in taus:
            for w in ws:
                fit = fit_all(tr, tau, w)
                for r in te.itertuples():
                    _, _, M = predict(fit, r.league, r.home_team, r.away_team, dc=True)
                    add(f"Bayes tau={tau:g} w={w:g}", r, probs(M))
                    _, _, M0 = predict(fit, r.league, r.home_team, r.away_team, dc=False)
                    add(f"Bayes tau={tau:g} w={w:g} (no DC)", r, probs(M0))
    per = {k: pd.DataFrame(v).set_index("match_id") for k, v in recs.items()}
    rows = []
    for k, d in per.items():
        rows.append({"Model": k, "Log loss": d["ll"].mean(), "Brier": d["brier"].mean(), "RPS": d["rps"].mean(), "Hit rate %": d["hit"].mean() * 100,
                     "Mean draw prob %": d["pdraw"].mean() * 100, "Actual draw %": d["draw"].mean() * 100, "n": len(d)})
    summ = pd.DataFrame(rows)
    cand = summ[summ["Model"].str.startswith("Bayes") & ~summ["Model"].str.contains("no DC")]
    top = cand.sort_values("Log loss").iloc[0]["Model"]
    # practical-equivalence rule: among settings within 0.01 log loss of the best (not distinguishable on 156 matches), take the strongest shrinkage
    eligible = []
    for name in cand["Model"]:
        d = float((per[name]["ll"] - per[top]["ll"]).mean())
        if d <= EQUIV:
            tau_c, w_c = (float(x.split("=")[1]) for x in name.replace("Bayes ", "").split(" "))
            eligible.append((tau_c, -d, name, w_c))
    tau_b, _, best, w_b = sorted(eligible, reverse=True)[0]
    return summ, per, {"name": best, "tau": tau_b, "w": w_b, "best_raw": top}


def paired_gap(per, a, b, n=3000, seed=5):
    """Mean log-loss of model a minus model b over the same matches, with a 90% bootstrap range.  Positive = b is better."""
    d = (per[a]["ll"] - per[b]["ll"]).values
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def strength_table(fit, league, n=300):
    """Attack and defence with 80% credible ranges (log scale, relative to the league average).  Defence is flipped so higher = better."""
    m = fit["leagues"][league]
    D_ = draws(m, n)
    k = m["n"]
    att = D_[:, 2:2 + k] - D_[:, 2:2 + k].mean(axis=1, keepdims=True)
    dfn = -(D_[:, 2 + k:] - D_[:, 2 + k:].mean(axis=1, keepdims=True))
    rows = []
    for t, i in m["idx"].items():
        rows.append({"Team": t, "Attack": np.percentile(att[:, i], 50), "Attack low": np.percentile(att[:, i], 10), "Attack high": np.percentile(att[:, i], 90),
                     "Defence": np.percentile(dfn[:, i], 50), "Defence low": np.percentile(dfn[:, i], 10), "Defence high": np.percentile(dfn[:, i], 90)})
    return pd.DataFrame(rows).sort_values("Attack", ascending=False).reset_index(drop=True)
