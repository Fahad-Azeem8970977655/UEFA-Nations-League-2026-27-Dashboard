"""Extra scraped data (unl_2026_27_extra.xlsx), attached to D after the main build.

    D.halves      one row per team, match and half (1st / 2nd): own stats, opponent stats, goals and goals against
    D.refs        one row per match: referee, career cards, stoppage time, cards, penalties and fouls in the match
    D.odds        one row per match: bookmaker odds, implied probabilities, xG-model probabilities, result
    D.odds_team   one row per team: points versus points expected by the market and by xG
    D.players     gains age, height, foot, market value (EUR m), contract end and club (merged on player_id)

Every part is optional: when the extra file is missing the attribute is None and the pages show a note.
"""
import numpy as np
import pandas as pd

EXT_SHEETS = ["halves", "match_info", "odds_1x2", "profiles"]

# stat key -> (label, how to read it).  "sum" stats are counted per half, "rate" stats are averaged.
HALF_STATS = {
    "goals": ("Goals", "sum"), "expectedGoals": ("xG", "sum"), "totalShotsOnGoal": ("Shots", "sum"),
    "shotsOnGoal": ("Shots on target", "sum"), "bigChanceCreated": ("Big chances", "sum"),
    "ballPossession": ("Possession %", "rate"), "pass_acc": ("Pass accuracy %", "rate"), "passes": ("Passes", "sum"),
    "finalThirdEntries": ("Final third entries", "sum"), "cornerKicks": ("Corners", "sum"), "fouls": ("Fouls", "sum"),
    "yellowCards": ("Yellow cards", "sum"), "totalTackle": ("Tackles", "sum"), "interceptionWon": ("Interceptions", "sum"),
    "ballRecovery": ("Recoveries", "sum"), "totalClearance": ("Clearances", "sum"), "dispossessed": ("Dispossessed", "sum"),
    "offsides": ("Offsides", "sum"), "goalkeeperSaves": ("Saves", "sum"),
}
HALF_LABEL = {"1ST": "1st half", "2ND": "2nd half"}


def _ok(df):
    return df is not None and not df.empty


# ------------------------------------------------------------------ halves
def build_halves(D, h):
    long = pd.concat([h.rename(columns={"home": "val", "away": "opp"}).assign(side="home"),
                      h.rename(columns={"away": "val", "home": "opp"}).assign(side="away")], ignore_index=True)
    idx = ["match_id", "side", "period"]
    own = long.pivot_table(index=idx, columns="key", values="val", aggfunc="first")
    opp = long.pivot_table(index=idx, columns="key", values="opp", aggfunc="first").add_prefix("opp_")
    w = own.join(opp).reset_index()
    for p in ("", "opp_"):
        w[p + "pass_acc"] = w[p + "accuratePasses"] / w[p + "passes"].replace(0, np.nan) * 100
    m = D.m[["match_id", "date", "league", "group_key", "matchday", "home_team", "away_team", "home_score", "away_score", "home_ht", "away_ht"]]
    w = w.merge(m, on="match_id", how="left")
    home = w["side"] == "home"
    w["team"] = np.where(home, w["home_team"], w["away_team"])
    w["opponent"] = np.where(home, w["away_team"], w["home_team"])
    gf_ft, ga_ft = np.where(home, w["home_score"], w["away_score"]), np.where(home, w["away_score"], w["home_score"])
    gf_ht, ga_ht = np.where(home, w["home_ht"], w["away_ht"]), np.where(home, w["away_ht"], w["home_ht"])
    first = w["period"] == "1ST"
    w["goals"] = np.where(first, gf_ht, gf_ft - gf_ht)
    w["opp_goals"] = np.where(first, ga_ht, ga_ft - ga_ht)
    w["half"] = w["period"].map(HALF_LABEL)
    return w.drop(columns=["home_ht", "away_ht", "home_score", "away_score"]).sort_values(["team", "date", "period"]).reset_index(drop=True)


def half_league_table(H):
    """League average per team-match in each half, with the change from 1st to 2nd half."""
    rows = []
    for k, (lab, kind) in HALF_STATS.items():
        a = H.loc[H["period"] == "1ST", k].mean()
        b = H.loc[H["period"] == "2ND", k].mean()
        rows.append({"key": k, "Stat": lab, "1st half": a, "2nd half": b, "Change": b - a, "Change %": (b / a - 1) * 100 if a else np.nan})
    return pd.DataFrame(rows)


def half_team_table(H, key):
    """Per team: average of `key` in each half and the shift from the 1st to the 2nd half."""
    g = H.pivot_table(index="team", columns="period", values=key, aggfunc="mean")
    g.columns = [HALF_LABEL[c] for c in g.columns]
    g["Change"] = g["2nd half"] - g["1st half"]
    g["Matches"] = H.groupby("team")["match_id"].nunique()
    return g


def half_results_table(H):
    """Per team: goals, goals against, xG and xGA in each half, plus the points a team would have if each half were a match."""
    rows = []
    for (team, period), g in H.groupby(["team", "period"]):
        pts = np.where(g["goals"] > g["opp_goals"], 3, np.where(g["goals"] == g["opp_goals"], 1, 0)).sum()
        rows.append({"team": team, "period": period, "GF": g["goals"].sum(), "GA": g["opp_goals"].sum(),
                     "xG": g["expectedGoals"].sum(), "xGA": g["opp_expectedGoals"].sum(), "Pts": pts, "Matches": g["match_id"].nunique()})
    t = pd.DataFrame(rows)
    a, b = t[t["period"] == "1ST"].set_index("team"), t[t["period"] == "2ND"].set_index("team")
    out = pd.DataFrame({"Matches": a["Matches"], "GF 1H": a["GF"], "GA 1H": a["GA"], "GF 2H": b["GF"], "GA 2H": b["GA"],
                        "xG 1H": a["xG"], "xGA 1H": a["xGA"], "xG 2H": b["xG"], "xGA 2H": b["xGA"], "Pts 1H": a["Pts"], "Pts 2H": b["Pts"]})
    out["GD 1H"] = out["GF 1H"] - out["GA 1H"]
    out["GD 2H"] = out["GF 2H"] - out["GA 2H"]
    out["xGD 1H"] = out["xG 1H"] - out["xGA 1H"]
    out["xGD 2H"] = out["xG 2H"] - out["xGA 2H"]
    return out


# ------------------------------------------------------------------ referees
def build_refs(D, mi):
    r = mi.merge(D.m[["match_id", "date", "league", "group_key", "home_team", "away_team", "home_score", "away_score"]], on="match_id", how="left")
    inc = D.inc
    cards = inc[inc["type"] == "card"].merge(D.m[["match_id", "home_team"]], on="match_id", how="left")
    cards["is_home"] = cards["team"] == cards["home_team"]
    cards["is_red"] = cards["class"].isin(["red", "yellowRed"])
    g = cards.groupby("match_id")
    c = pd.DataFrame({"cards": g.size(), "yellow": g["class"].apply(lambda s: (s == "yellow").sum()), "red": g["is_red"].sum(),
                      "home_cards": g["is_home"].sum()})
    c["away_cards"] = c["cards"] - c["home_cards"]
    pens = inc[((inc["type"] == "goal") & (inc["class"] == "penalty")) | (inc["type"] == "inGamePenalty")].groupby("match_id").size().rename("penalties")
    fouls = D.tm.groupby("match_id")["fouls"].sum().rename("fouls")
    r = r.merge(c, on="match_id", how="left").merge(pens, on="match_id", how="left").merge(fouls, on="match_id", how="left")
    for col in ["cards", "yellow", "red", "home_cards", "away_cards", "penalties"]:
        r[col] = r[col].fillna(0).astype(int)
    r["added_time"] = r["injury_time_1"].fillna(0) + r["injury_time_2"].fillna(0)
    r["goals"] = r["home_score"] + r["away_score"]
    return r.sort_values("date").reset_index(drop=True)


def referee_table(R):
    g = R.groupby("referee")
    t = pd.DataFrame({"Country": g["referee_country"].first(), "Matches": g.size(), "Cards": g["cards"].sum(), "Yellow": g["yellow"].sum(),
                      "Red": g["red"].sum(), "Penalties": g["penalties"].sum(), "Home cards": g["home_cards"].sum(), "Away cards": g["away_cards"].sum(),
                      "Cards per match": g["cards"].mean(), "Fouls per match": g["fouls"].mean(), "Penalties per match": g["penalties"].mean(),
                      "Added time per match": g["added_time"].mean(), "Career games": g["referee_games"].max(),
                      "Career yellows": g["referee_yellow"].max(), "Career reds": g["referee_red"].max()})
    t["Career yellows per game"] = t["Career yellows"] / t["Career games"].replace(0, np.nan)
    t["Yellows per match"] = t["Yellow"] / t["Matches"]
    t["Vs career (yellows)"] = t["Yellows per match"] - t["Career yellows per game"]
    t["Home share of cards %"] = np.where(t["Cards"] > 0, t["Home cards"] / t["Cards"].replace(0, np.nan) * 100, np.nan)
    return t


# ------------------------------------------------------------------ odds
def build_odds(D, o):
    r = o.merge(D.m[["match_id", "date", "league", "group_key", "home_team", "away_team", "home_score", "away_score"]], on="match_id", how="left")
    for tag in ("open", "close"):
        raw = pd.DataFrame({k: 1 / r[f"{tag}_{k}"] for k in ("home", "draw", "away")})
        tot = raw.sum(axis=1)
        pre = "pm" if tag == "close" else "pmo"
        for k in ("home", "draw", "away"):
            r[f"{pre}_{k}"] = raw[k] / tot
        if tag == "close":
            r["margin"] = (tot - 1) * 100
    r["result"] = np.where(r["home_score"] > r["away_score"], "H", np.where(r["home_score"] < r["away_score"], "A", "D"))
    xm = D.xpts_match[["match_id", "p_home", "p_draw", "p_away"]].rename(columns={"p_home": "px_home", "p_draw": "px_draw", "p_away": "px_away"})
    r = r.merge(xm, on="match_id", how="left")
    name = {"H": "home", "D": "draw", "A": "away"}
    for pre in ("pm", "pmo", "px"):
        r[f"{pre}_actual"] = [row[f"{pre}_{name[x]}"] for x, (_, row) in zip(r["result"], r.iterrows())]
        brier = 0
        for k, code in name.items():
            brier = brier + (r[f"{pre}_{code}"] - (r["result"] == k).astype(float)) ** 2
        r[f"{pre}_brier"] = brier
    fav = r[["pm_home", "pm_draw", "pm_away"]].idxmax(axis=1).str.replace("pm_", "")
    r["favourite"] = fav.map({"home": "H", "draw": "D", "away": "A"})
    r["fav_prob"] = r[["pm_home", "pm_draw", "pm_away"]].max(axis=1)
    r["fav_team"] = np.where(r["favourite"] == "H", r["home_team"], np.where(r["favourite"] == "A", r["away_team"], "Draw"))
    r["upset"] = (r["favourite"] != "D") & (r["result"] != r["favourite"])
    r["market_xpts_home"] = 3 * r["pm_home"] + r["pm_draw"]
    r["market_xpts_away"] = 3 * r["pm_away"] + r["pm_draw"]
    r["xg_xpts_home"] = 3 * r["px_home"] + r["px_draw"]
    r["xg_xpts_away"] = 3 * r["px_away"] + r["px_draw"]
    r["pts_home"] = r["result"].map({"H": 3, "D": 1, "A": 0})
    r["pts_away"] = r["result"].map({"H": 0, "D": 1, "A": 3})
    return r.sort_values("date").reset_index(drop=True)


def odds_team_table(O):
    h = pd.DataFrame({"team": O["home_team"], "league": O["league"], "group_key": O["group_key"], "Pts": O["pts_home"], "Market xPts": O["market_xpts_home"],
                      "xG xPts": O["xg_xpts_home"], "p_win": O["pm_home"]})
    a = pd.DataFrame({"team": O["away_team"], "league": O["league"], "group_key": O["group_key"], "Pts": O["pts_away"], "Market xPts": O["market_xpts_away"],
                      "xG xPts": O["xg_xpts_away"], "p_win": O["pm_away"]})
    t = pd.concat([h, a]).groupby("team").agg(League=("league", "first"), Group=("group_key", "first"), Matches=("Pts", "size"), Pts=("Pts", "sum"),
                                              **{"Market xPts": ("Market xPts", "sum"), "xG xPts": ("xG xPts", "sum"), "Avg win prob": ("p_win", "mean")})
    t["Pts - market"] = t["Pts"] - t["Market xPts"]
    t["xG xPts - market"] = t["xG xPts"] - t["Market xPts"]
    return t


def calibration(O, bins=(0, 0.2, 0.35, 0.5, 0.65, 1.0)):
    """Predicted probability of each outcome (home, draw, away) against how often it happened, for market and xG model."""
    rows = []
    for pre, nm in (("pm", "Market"), ("px", "xG model")):
        for k, code in (("H", "home"), ("D", "draw"), ("A", "away")):
            rows.append(pd.DataFrame({"model": nm, "p": O[f"{pre}_{code}"], "hit": (O["result"] == k).astype(float)}))
    c = pd.concat(rows)
    names = [f"{int(round(lo * 100))}-{int(round(hi * 100))}%" for lo, hi in zip(bins[:-1], bins[1:])]
    c["bin"] = pd.cut(c["p"], list(bins), include_lowest=True, labels=names)
    out = c.groupby(["model", "bin"], observed=True).agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


# ------------------------------------------------------------------ player profiles
def add_profiles(D, pf):
    ref = D.m["date"].max()
    p = pf.copy()
    p["birth_date"] = pd.to_datetime(p["birth_date"], errors="coerce")
    p["contract_until"] = pd.to_datetime(p["contract_until"], errors="coerce")
    p["age"] = ((ref - p["birth_date"]).dt.days / 365.25).round(1)
    p["market_value_m"] = pd.to_numeric(p["market_value_eur"], errors="coerce") / 1e6
    p["contract_year"] = p["contract_until"].dt.year
    keep = ["player_id", "age", "height_cm", "preferred_foot", "positions_detailed", "market_value_m", "contract_year", "club"]
    keep = [c for c in keep if c == "player_id" or c not in D.players.columns]
    return D.players.merge(p[keep].drop_duplicates("player_id"), on="player_id", how="left")


# ------------------------------------------------------------------ entry point
def attach_ext(D, ext):
    D.halves = D.refs = D.odds = D.odds_team = None
    if _ok(ext.get("halves")):
        D.halves = build_halves(D, ext["halves"])
    if _ok(ext.get("match_info")):
        D.refs = build_refs(D, ext["match_info"])
    if _ok(ext.get("odds_1x2")):
        D.odds = build_odds(D, ext["odds_1x2"])
        D.odds_team = odds_team_table(D.odds)
    if _ok(ext.get("profiles")):
        D.players = add_profiles(D, ext["profiles"])
    return D
