"""Match events on a minute-by-minute timeline: game state, substitution impact and time-of-match profile.

Clock: minute k (1-based) covers the interval [k-1, k).  Stoppage time is added to the minute
(45+3 = minute 48, 90+4 = minute 94), so the two halves overlap slightly in the first stoppage minutes.
That is a small approximation; it is good enough for 15-minute blocks and for state accounting.

Game state of a team in a minute = score difference (own - opponent, clipped to -1 / 0 / +1) and numerical
situation from red cards (own men - opponent men, clipped to -1 / 0 / +1).  A goal or card at minute m changes the
state from minute m + 1, so the shot that scores is still counted in the state before the goal.
Momentum (SofaScore) is positive for the home side; it is flipped for the away side.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

BLOCKS = ["1-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]
STATES = {-1: "Trailing", 0: "Level", 1: "Leading"}


def eff_minute(minute, added):
    return minute.astype(float) + added.fillna(0).astype(float)


def block_of(minute, added):
    """15-minute block index 0-5.  Stoppage time stays in the block of its half."""
    k = ((minute.astype(float) - 1) // 15).clip(0, 5)
    return k.astype(int)


def build_timeline(D):
    inc = D.inc[D.inc["type"] != "period"].copy()   # period rows carry a 999 sentinel in added_time
    inc["added_time"] = pd.to_numeric(inc["added_time"], errors="coerce").where(lambda s: s < 60)
    inc["minute"] = pd.to_numeric(inc["minute"], errors="coerce").clip(lower=0)
    inc["eff"] = eff_minute(inc["minute"], inc["added_time"]) if "added_time" in inc else inc["minute"]
    sh = D.shots.copy()
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.03)
    sh["eff"] = eff_minute(sh["minute"], sh["added_time"])
    sh["block"] = block_of(sh["minute"], sh["added_time"])
    mom = D.mom.copy()
    rows, arrays, goal_events, red_events, sub_events = [], {}, [], [], []
    for r in D.m.itertuples():
        mid, home, away = r.match_id, r.home_team, r.away_team
        i = inc[inc["match_id"] == mid]
        s = sh[sh["match_id"] == mid]
        m = mom[mom["match_id"] == mid]
        T = int(max(90, np.ceil(max([s["eff"].max() if len(s) else 0, i["eff"].max() if len(i) else 0, m["minute"].max() if len(m) else 0]))))
        # ---- goals (from the running score) and red cards
        g = i[i["type"] == "goal"].sort_values("eff")
        prev = (0, 0)
        diff = np.zeros(T)           # home minus away goals during minute k+1
        goals = []
        for x in g.itertuples():
            if pd.isna(x.home_score) or pd.isna(x.away_score):
                continue
            hs, as_ = int(x.home_score), int(x.away_score)
            scorer = home if hs > prev[0] else away if as_ > prev[1] else None
            prev = (hs, as_)
            goals.append((int(np.ceil(x.eff)), scorer, x))
            diff[min(int(np.ceil(x.eff)), T):] = hs - as_
        men = np.zeros(T)            # home men minus away men
        for x in i[(i["type"] == "card") & i["class"].isin(["red", "yellowRed"])].itertuples():
            k = min(int(np.ceil(x.eff)), T)
            men[k:] += -1 if x.team == home else 1
            red_events.append({"match_id": mid, "minute": int(np.ceil(x.eff)), "team": x.team, "T": T,
                               "opponent": away if x.team == home else home})
        for scorer, x in [(sc, x) for _, sc, x in goals]:
            pass
        for k, scorer, x in goals:
            if scorer is not None:
                goal_events.append({"match_id": mid, "minute": k, "scorer": scorer,
                                    "conceder": away if scorer == home else home, "T": T,
                                    "own_goal": x.__getattribute__("_2") == "ownGoal" if False else False})
        # ---- per-team minute arrays
        for team, opp, sign in ((home, away, 1), (away, home, -1)):
            xf, xa = np.zeros(T), np.zeros(T)
            for e, xg, tm_ in zip(s["eff"], s["xg"], s["team"]):
                k = min(max(int(np.ceil(e)) - 1, 0), T - 1)
                if tm_ == team:
                    xf[k] += xg
                else:
                    xa[k] += xg
            mv = np.zeros(T)
            for mi, v in zip(m["minute"], m["value"]):
                k = int(round(mi)) - 1
                if 0 <= k < T:
                    mv[k] = sign * v
            arrays[(mid, team)] = {"xgf": xf, "xga": xa, "mom": mv, "score": sign * diff, "men": sign * men, "T": T,
                                   "opp": opp, "side": "home" if sign == 1 else "away"}
            sc = np.clip(sign * diff, -1, 1)
            mn = np.clip(sign * men, -1, 1)
            rows.append(pd.DataFrame({"match_id": mid, "team": team, "minute": np.arange(1, T + 1), "score": sc, "men": mn,
                                      "xgf": xf, "xga": xa, "mom": mv}))
        # ---- substitutions (one event per team and minute)
        for (team, mn_), grp in i[i["type"] == "substitution"].groupby(["team", "minute"]):
            sub_events.append({"match_id": mid, "team": team, "minute": int(mn_), "n": len(grp), "T": T,
                               "injury": bool((grp["class"] == "injury").any()),
                               "players_in": ", ".join(grp["player_in"].dropna().astype(str)),
                               "players_out": ", ".join(grp["player_out"].dropna().astype(str))})
    minutes = pd.concat(rows, ignore_index=True)
    # goals / shots from the shot map for the 15-min profile
    return SimpleNamespace(**{"minutes": minutes, "arrays": arrays, "goals": pd.DataFrame(goal_events),
                           "reds": pd.DataFrame(red_events), "subs": pd.DataFrame(sub_events), "shots": sh})


# ------------------------------------------------------------------ game state
def state_table(TL, by="score"):
    """Per game state: minutes, xG for / against per 90, xGD per 90, and the share of xG."""
    m = TL.minutes
    g = m.groupby(["team", by]).agg(minutes=("minute", "size"), xgf=("xgf", "sum"), xga=("xga", "sum")).reset_index()
    return g


def league_state(TL, by="score"):
    g = state_table(TL, by).groupby(by).agg(minutes=("minutes", "sum"), xgf=("xgf", "sum"), xga=("xga", "sum"))
    g["xG for / 90"] = g["xgf"] / g["minutes"] * 90
    g["xG against / 90"] = g["xga"] / g["minutes"] * 90
    g["xGD / 90"] = g["xG for / 90"] - g["xG against / 90"]
    g["xG share %"] = g["xgf"] / (g["xgf"] + g["xga"]) * 100
    return g


def team_state(TL):
    g = state_table(TL, "score")
    g["xgf90"] = g["xgf"] / g["minutes"] * 90
    g["xga90"] = g["xga"] / g["minutes"] * 90
    piv = g.pivot(index="team", columns="score", values=["minutes", "xgf90", "xga90"])
    out = pd.DataFrame(index=piv.index)
    for s, name in STATES.items():
        for k, lab in (("minutes", "Minutes"), ("xgf90", "xG for / 90"), ("xga90", "xG against / 90")):
            out[f"{name}: {lab}"] = piv[(k, s)] if (k, s) in piv.columns else np.nan
    out["Trailing minus level: xG for / 90"] = out["Trailing: xG for / 90"] - out["Level: xG for / 90"]
    out["Leading minus level: xG for / 90"] = out["Leading: xG for / 90"] - out["Level: xG for / 90"]
    out["Leading minus level: xG against / 90"] = out["Leading: xG against / 90"] - out["Level: xG against / 90"]
    return out


def event_curve(TL, events, who, window=12, kind="mom"):
    """Average of `kind` (mom: momentum, xgf, xga) around events, from `who`'s point of view, relative minute -window..+window.
    events: DataFrame with match_id, minute and a team column named `who` (the team whose view we take)."""
    out = np.full((len(events), 2 * window), np.nan)
    for j, e in enumerate(events.itertuples()):
        a = TL.arrays.get((e.match_id, getattr(e, who)))
        if a is None:
            continue
        v = a[kind]
        for r in range(-window, window):
            k = e.minute + r  # index of minute (e.minute + r + 1) in the array
            if 0 <= k < a["T"]:
                out[j, r + window] = v[k]
    return pd.DataFrame({"rel": list(range(-window + 1, window + 1)), "value": np.nanmean(out, axis=0), "n": (~np.isnan(out)).sum(0)})


# ------------------------------------------------------------------ substitutions
def sub_impact(TL, win=15, min_win=10):
    """xGD and momentum in the `win` minutes after each substitution minus the `win` minutes before."""
    rows = []
    for e in TL.subs.itertuples():
        a = TL.arrays[(e.match_id, e.team)]
        m, T = e.minute, a["T"]
        b0, a1 = max(m - win, 0), min(m + win, T)
        if m - b0 < min_win or a1 - m < min_win:
            continue
        sl_b, sl_a = slice(b0, m), slice(m, a1)
        nb, na = m - b0, a1 - m
        xgd_b = (a["xgf"][sl_b].sum() - a["xga"][sl_b].sum()) / nb * 15
        xgd_a = (a["xgf"][sl_a].sum() - a["xga"][sl_a].sum()) / na * 15
        state = int(np.sign(a["score"][max(m - 1, 0)])) if m - 1 < T else 0
        rows.append({"match_id": e.match_id, "team": e.team, "opponent": a["opp"], "minute": m, "n_players": e.n,
                     "injury": e.injury, "state": STATES[state],
                     "xgd_before": xgd_b, "xgd_after": xgd_a, "d_xgd": xgd_a - xgd_b,
                     "xgf_before": a["xgf"][sl_b].sum() / nb * 15, "xgf_after": a["xgf"][sl_a].sum() / na * 15,
                     "xga_before": a["xga"][sl_b].sum() / nb * 15, "xga_after": a["xga"][sl_a].sum() / na * 15,
                     "mom_before": a["mom"][sl_b].mean(), "mom_after": a["mom"][sl_a].mean(),
                     "players_in": e.players_in, "players_out": e.players_out})
    d = pd.DataFrame(rows)
    d["d_mom"] = d["mom_after"] - d["mom_before"]
    d["d_xgf"] = d["xgf_after"] - d["xgf_before"]
    d["d_xga"] = d["xga_after"] - d["xga_before"]
    return d


def boot_ci(x, n=1000, seed=5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def bench_impact(D):
    """What substitutes (players who did not start) produced, per team."""
    p = D.ps[(pd.to_numeric(D.ps["minutesplayed"], errors="coerce").fillna(0) > 0) & D.ps["substitute"]].copy()
    for c in ("goals", "goalassist", "expectedgoals", "expectedassists", "minutesplayed", "keypass", "totalshots"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    g = p.groupby("team").agg(**{"Sub appearances": ("player_id", "size"), "Sub minutes": ("minutesplayed", "sum"),
                                  "Goals": ("goals", "sum"), "Assists": ("goalassist", "sum"), "xG": ("expectedgoals", "sum"),
                                  "xA": ("expectedassists", "sum")})
    g["Goal involvements / 90"] = (g["Goals"] + g["Assists"]) / g["Sub minutes"] * 90
    g["xG + xA / 90"] = (g["xG"] + g["xA"]) / g["Sub minutes"] * 90
    return g


# ------------------------------------------------------------------ time of match
def time_profile(D, TL):
    sh = TL.shots
    sh = sh.assign(goal=(sh["shot_type"] == "goal").astype(int))
    f = sh.groupby(["team", "block"]).agg(goals=("goal", "sum"), xg=("xg", "sum"), shots=("xg", "size")).reset_index()
    # conceded: join opponents
    opp = sh.merge(D.tm[["match_id", "team", "opponent"]].rename(columns={"team": "tm_team", "opponent": "opp_name"}),
                   left_on=["match_id", "team"], right_on=["match_id", "tm_team"], how="left")
    a = opp.groupby(["opp_name", "block"]).agg(goals_a=("goal", "sum"), xg_a=("xg", "sum"), shots_a=("xg", "size")).reset_index()
    a = a.rename(columns={"opp_name": "team"})
    t = f.merge(a, on=["team", "block"], how="outer").fillna(0)
    t["block_label"] = t["block"].map(dict(enumerate(BLOCKS)))
    # own goals: scorer team from the goal events
    return t
