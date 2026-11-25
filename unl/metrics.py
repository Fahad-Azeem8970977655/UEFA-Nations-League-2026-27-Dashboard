"""Team and player metric definitions and aggregation."""
import numpy as np
import pandas as pd

from unl.helpers import num

# label, column, better ("high" / "low" / "none" = volume only), decimals
TEAM_METRICS = [
    ("Goals scored", "gf", "high", 2),
    ("Goals conceded", "ga", "low", 2),
    ("xG", "expected_goals", "high", 2),
    ("xG against", "opp_expected_goals", "low", 2),
    ("Possession %", "ball_possession", "high", 1),
    ("Passes", "passes", "high", 0),
    ("Pass accuracy %", "pass_acc", "high", 1),
    ("Final third entries", "final_third_entries", "high", 1),
    ("Touches in box", "touches_in_penalty_area", "high", 1),
    ("Shots", "total_shots", "high", 1),
    ("Shots on target", "shots_on_target", "high", 1),
    ("Shots conceded", "opp_total_shots", "low", 1),
    ("Tackles", "tackles", "none", 1),
    ("Interceptions", "interceptions", "none", 1),
    ("Clearances", "clearances", "none", 1),
    ("Recoveries", "recoveries", "none", 1),
    ("Fouls", "fouls", "low", 1),
    ("Distance covered", "distance_covered", "none", 1),
]


RADAR_LABELS = ["xG", "Shots on target", "Touches in box", "Possession %", "Pass accuracy %",
                "Final third entries", "xG against", "Shots conceded", "Recoveries", "Interceptions"]


def build_team_summary(tm):
    cols = {label: col for label, col, _, _ in TEAM_METRICS}
    g = tm.groupby("team")
    out = pd.DataFrame({label: g[col].mean() for label, col in cols.items()})
    out["Matches"] = g.size()
    out["Points"] = g["points"].sum()
    return out


def team_percentiles(summary, labels):
    better = {label: b for label, _, b, _ in TEAM_METRICS}
    pct = pd.DataFrame(index=summary.index)
    for label in labels:
        r = summary[label].rank(pct=True, method="average") * 100
        pct[label] = 100 - r + (100 / len(summary)) if better[label] == "low" else r
    return pct.clip(0, 100)


def team_profile_table(summary, team):
    better = {label: b for label, _, b, _ in TEAM_METRICS}
    dec = {label: d for label, _, _, d in TEAM_METRICS}
    n = len(summary)
    rows = []
    for label, _, _, _ in TEAM_METRICS:
        s = summary[label]
        if pd.isna(s.get(team)):
            continue
        asc = better[label] == "low"
        rank = int(s.rank(ascending=asc, method="min")[team])
        rows.append({"Metric": label, "Team average": round(float(s[team]), dec[label]),
                     "All-team average": round(float(s.mean()), dec[label]),
                     "Rank": f"{rank} of {s.notna().sum()}",
                     "Better when": {"high": "higher", "low": "lower", "none": "style, not quality"}[better[label]]})
    return pd.DataFrame(rows)


SUM_COLS = ["goals", "goalassist", "expectedgoals", "expectedassists", "totalshots", "keypass",
            "bigchancecreated", "totalpass", "accuratepass", "totaltackle", "interceptionwon",
            "totalclearance", "ballrecovery", "duelwon", "duellost", "woncontest", "totalcontest",
            "progressiveballcarriescount", "ballcarriescount", "touches", "fouls", "wasfouled", "saves",
            "goalsprevented", "ontargetscoringattempt", "totalcross", "accuratecross", "aerialwon", "aeriallost",
            "totallongballs", "accuratelongballs", "dispossessed", "bigchancemissed", "totalkeepersweeper",
            "accuratekeepersweeper", "goodhighclaim", "punches", "savedshotsfrominsidethebox", "keepersavevalue"]

# label -> (kind, column)    kind: count (has per-90), pct, mean
PLAYER_METRICS = {
    "Goals": ("count", "goals"),
    "Assists": ("count", "goalassist"),
    "Expected goals (xG)": ("count", "expectedgoals"),
    "Expected assists (xA)": ("count", "expectedassists"),
    "Shots": ("count", "totalshots"),
    "Key passes": ("count", "keypass"),
    "Big chances created": ("count", "bigchancecreated"),
    "Passes": ("count", "totalpass"),
    "Pass accuracy %": ("pct", "pass_acc"),
    "Tackles": ("count", "totaltackle"),
    "Interceptions": ("count", "interceptionwon"),
    "Clearances": ("count", "totalclearance"),
    "Recoveries": ("count", "ballrecovery"),
    "Duels won": ("count", "duelwon"),
    "Duel win %": ("pct", "duel_win"),
    "Dribbles won": ("count", "woncontest"),
    "Crosses": ("count", "totalcross"),
    "Aerial duels won": ("count", "aerialwon"),
    "Progressive carries": ("count", "progressiveballcarriescount"),
    "Expected threat (xT)": ("count", "xt_total"),
    "Touches": ("count", "touches"),
    "Fouls committed": ("count", "fouls"),
    "Fouls won": ("count", "wasfouled"),
    "Saves": ("count", "saves"),
    "Goals prevented": ("count", "goalsprevented"),
    "Average rating": ("mean", "rating_mean"),
    "Distance per match (km)": ("mean", "dist_mean"),
}


def build_player_table(ps):
    p = ps[num(ps, "minutesplayed").fillna(0) > 0].copy()
    for c in SUM_COLS:
        p[c] = num(p, c).fillna(0)
    p["minutesplayed"] = num(p, "minutesplayed")
    p = p.sort_values("date")
    grp = p.groupby("player_id")
    agg = grp[SUM_COLS + ["minutesplayed"]].sum()
    agg["Matches"] = grp.size()
    agg["Starts"] = grp["substitute"].apply(lambda s: int((~s).sum()))
    agg["player"] = grp["player"].last()
    agg["team"] = grp["team"].last()
    agg["position"] = grp["position"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    agg["rating_mean"] = grp["rating"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    agg["dist_mean"] = grp["kilometerscovered"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    agg["pass_acc"] = np.where(agg["totalpass"] > 0, agg["accuratepass"] / agg["totalpass"] * 100, np.nan)
    both = agg["duelwon"] + agg["duellost"]
    agg["duel_win"] = np.where(both > 0, agg["duelwon"] / both * 100, np.nan)
    return agg.reset_index()


def metric_values(pt, label, per90):
    kind, col = PLAYER_METRICS[label]
    if kind == "count":
        if per90:
            return pt[col] / pt["minutesplayed"] * 90
        return pt[col]
    return pt[col]
