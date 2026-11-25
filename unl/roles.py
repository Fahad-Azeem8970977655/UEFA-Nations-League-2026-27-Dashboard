"""Player roles detected from where players actually play.

The coarse D / M / F / G label from SofaScore is split into finer roles using the player's touch heatmap:
    depth  = average x of all touches (0 = own goal line, 100 = opponent goal line)
    width  = average distance of touches from the centre line (0 = central, 50 = touchline)
Rules are deliberately simple so every label can be explained.  Roles are assigned to players with
at least MIN_TOUCHES heatmap points; everyone else keeps a generic label.
"""
import numpy as np
import pandas as pd

MIN_TOUCHES = 60

ROLES = ["Goalkeeper", "Centre-back", "Full-back", "Wing-back", "Holding midfielder", "Central midfielder",
         "Attacking midfielder", "Wide midfielder", "Winger", "Second striker", "Centre-forward"]
ROLE_GROUP = {"Goalkeeper": "G", "Centre-back": "D", "Full-back": "D", "Wing-back": "D", "Holding midfielder": "M",
              "Central midfielder": "M", "Attacking midfielder": "M", "Wide midfielder": "M", "Winger": "F",
              "Second striker": "F", "Centre-forward": "F"}
ROLE_COLORS = {"Goalkeeper": "#8A97A6", "Centre-back": "#16202E", "Full-back": "#0E7C86", "Wing-back": "#2FA7B0",
               "Holding midfielder": "#5B6EAE", "Central midfielder": "#8E7CC3", "Attacking midfielder": "#B5326E",
               "Wide midfielder": "#D96F9B", "Winger": "#E8A200", "Second striker": "#D9622B", "Centre-forward": "#B52B2B"}
GENERIC = {"G": "Goalkeeper", "D": "Centre-back", "M": "Central midfielder", "F": "Centre-forward"}

# rule thresholds (touch-map coordinates)
WIDE_D, WINGBACK_X = 28, 48
WIDE_M, DM_X, AM_X, WINGER_X = 31, 48, 58, 56
WIDE_F, SECOND_X = 28, 57


def classify(pos, x, w):
    if pos == "G":
        return "Goalkeeper"
    if pos == "D":
        if w >= WIDE_D:
            return "Wing-back" if x >= WINGBACK_X else "Full-back"
        return "Centre-back"
    if pos == "M":
        if w >= WIDE_M:
            return "Winger" if x >= WINGER_X else "Wide midfielder"
        if x < DM_X:
            return "Holding midfielder"
        return "Attacking midfielder" if x >= AM_X else "Central midfielder"
    if pos == "F":
        if w >= WIDE_F:
            return "Winger"
        return "Second striker" if x < SECOND_X else "Centre-forward"
    return "Unknown"


def build_roles(D):
    """One row per player: touch-map depth / width / spread, detected role, plus the SofaScore position."""
    h = D.heat.dropna(subset=["x", "y"]).copy()
    h["w"] = (h["y"] - 50).abs()
    g = h.groupby("player_id").agg(touches=("x", "size"), depth=("x", "mean"), width=("w", "mean"),\
                                   spread_x=("x", "std"), spread_y=("y", "std"), mean_y=("y", "mean"))
    a = D.avg.copy()
    a["w"] = (a["average_y"] - 50).abs()
    ga = a.groupby("player_id").apply(lambda t: pd.Series({"avg_depth": np.average(t["average_x"], weights=t["points_count"]),
                                                           "avg_width": np.average(t["w"], weights=t["points_count"])}),
                                     include_groups=False)
    p = D.players[["player_id", "player", "team", "position", "minutesplayed", "Matches"]].set_index("player_id")
    r = p.join(g).join(ga)
    ok = r["touches"].fillna(0) >= MIN_TOUCHES
    r["role"] = [classify(pos, x, w) if good else GENERIC.get(pos, "Unknown")
                  for pos, x, w, good in zip(r["position"], r["depth"], r["width"], ok)]
    r["role_detected"] = ok
    r["role_group"] = r["role"].map(ROLE_GROUP).fillna(r["position"])
    # how many different roles a player filled across matches (versatility)
    return r.reset_index()


# per role: (label, column, kind)   kind: p90 = per 90 minutes, val = value as is
_ATT = [("xG", "expectedgoals", "p90"), ("Shots", "totalshots", "p90"), ("xA", "expectedassists", "p90"),
        ("Key passes", "keypass", "p90"), ("Dribbles won", "woncontest", "p90"),
        ("Prog. carries", "progressiveballcarriescount", "p90"), ("xT", "xt_total", "p90")]
ROLE_METRICS = {
    "Goalkeeper": [("Saves", "saves", "p90"), ("Goals prevented", "goalsprevented", "p90"), ("Pass acc. %", "pass_acc", "val"),
                   ("Passes", "totalpass", "p90"), ("Long balls", "totallongballs", "p90"),
                   ("Sweeper actions", "totalkeepersweeper", "p90"), ("High claims", "goodhighclaim", "p90")],
    "Centre-back": [("Tackles", "totaltackle", "p90"), ("Interceptions", "interceptionwon", "p90"),
                    ("Clearances", "totalclearance", "p90"), ("Aerial duels won", "aerialwon", "p90"),
                    ("Duel win %", "duel_win", "val"), ("Pass acc. %", "pass_acc", "val"), ("Long balls", "totallongballs", "p90"),
                    ("Prog. carries", "progressiveballcarriescount", "p90"), ("xT", "xt_total", "p90")],
    "Full-back": [("Tackles", "totaltackle", "p90"), ("Interceptions", "interceptionwon", "p90"), ("Recoveries", "ballrecovery", "p90"),
                  ("Crosses", "totalcross", "p90"), ("Key passes", "keypass", "p90"), ("xA", "expectedassists", "p90"),
                  ("Dribbles won", "woncontest", "p90"), ("Prog. carries", "progressiveballcarriescount", "p90"),
                  ("Duel win %", "duel_win", "val"), ("xT", "xt_total", "p90")],
    "Holding midfielder": [("Tackles", "totaltackle", "p90"), ("Interceptions", "interceptionwon", "p90"),
                           ("Recoveries", "ballrecovery", "p90"), ("Duel win %", "duel_win", "val"),
                           ("Passes", "totalpass", "p90"), ("Pass acc. %", "pass_acc", "val"), ("Long balls", "totallongballs", "p90"),
                           ("Prog. carries", "progressiveballcarriescount", "p90"), ("xT", "xt_total", "p90")],
    "Central midfielder": [("xA", "expectedassists", "p90"), ("Key passes", "keypass", "p90"), ("Pass acc. %", "pass_acc", "val"),
                           ("Prog. carries", "progressiveballcarriescount", "p90"), ("Tackles", "totaltackle", "p90"),
                           ("Interceptions", "interceptionwon", "p90"), ("Recoveries", "ballrecovery", "p90"),
                           ("Dribbles won", "woncontest", "p90"), ("Duel win %", "duel_win", "val"), ("xT", "xt_total", "p90")],
    "Attacking midfielder": _ATT + [("Big chances created", "bigchancecreated", "p90"), ("Pass acc. %", "pass_acc", "val")],
    "Wide midfielder": _ATT + [("Crosses", "totalcross", "p90"), ("Tackles", "totaltackle", "p90")],
    "Winger": _ATT + [("Crosses", "totalcross", "p90"), ("Big chances created", "bigchancecreated", "p90")],
    "Second striker": _ATT + [("Goals", "goals", "p90"), ("Touches", "touches", "p90")],
    "Centre-forward": [("Goals", "goals", "p90"), ("xG", "expectedgoals", "p90"), ("Shots", "totalshots", "p90"),
                       ("xA", "expectedassists", "p90"), ("Key passes", "keypass", "p90"), ("Aerial duels won", "aerialwon", "p90"),
                       ("Duel win %", "duel_win", "val"), ("Touches", "touches", "p90"), ("xT", "xt_total", "p90")],
}
ROLE_METRICS["Wing-back"] = ROLE_METRICS["Full-back"]


def role_frame(players, roles, role, min_minutes, by="role"):
    """Per-90 table, percentiles and z-scores of one role (or of its whole position group when by='group').
    `players` is D.players, `roles` is D.roles.  Returns (ident, raw, pct, z)."""
    d = players.drop(columns=[c for c in ("role", "role_group", "depth", "width") if c in players.columns]).merge(
        roles[["player_id", "role", "role_group", "depth", "width"]], on="player_id")
    metrics = ROLE_METRICS[role]
    if by == "group":
        d = d[d["role_group"] == ROLE_GROUP[role]]
    else:
        d = d[d["role"] == role]
    d = d[d["minutesplayed"] >= min_minutes].copy()
    raw = pd.DataFrame(index=d.index)
    for label, col, kind in metrics:
        raw[label] = d[col] / d["minutesplayed"] * 90 if kind == "p90" else d[col]
    raw = raw.fillna(raw.median())
    pct = raw.rank(pct=True) * 100
    z = (raw - raw.mean()) / raw.std(ddof=0).replace(0, 1)
    ident = d[["player_id", "player", "team", "minutesplayed", "role"]]
    return ident, raw, pct, z
