import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import scatter_labeled
from unl.pitch import show

D = setup("True table | Nations League 2026-27")

st.title("True table")
st.caption("Expected points (xPts): the points a team would have earned on average from the chances it created and "
           "allowed. Every shot has an xG, so each match gives a probability of a win, draw and loss for both teams.")

X = D.xpts_team
league = st.radio("League", sorted(X["league"].unique()), horizontal=True, key="tt_league")
sub = X[X["league"] == league]

over = sub["Pts - xPts"].idxmax()
under = sub["Pts - xPts"].idxmin()
moved = int((sub["Pos"] != sub["xPos"]).sum())
kpi_row([("Most fortunate", over, f"{sub.loc[over, 'Pts - xPts']:+.1f} pts vs xPts"), ("Least fortunate", under, f"{sub.loc[under, 'Pts - xPts']:+.1f} pts vs xPts"),
         ("Teams in a different position", f"{moved} of {len(sub)}"), ("Avg |Pts - xPts|", f"{sub['Pts - xPts'].abs().mean():.1f}", "per team")], min_width=190)

tabs = st.tabs(["Group tables", "Luck chart", "Match by match"])

with tabs[0]:
    groups = sorted(sub["group_key"].unique())
    cols = st.columns(2)
    for i, gk in enumerate(groups):
        t = sub[sub["group_key"] == gk].sort_values("xPos")
        out = pd.DataFrame({"xPos": t["xPos"], "Team": t.index, "Pos": t["Pos"].astype("Int64"), "Pts": t["Pts"],
                            "xPts": t["xPts"].round(1), "Pts - xPts": t["Pts - xPts"].round(1),
                            "GD": t["GD"], "xGD": t["xGD"].round(1)})
        with cols[i % 2]:
            st.subheader(gk)
            st.dataframe(out, hide_index=True)
    st.caption("Ordered by expected points, with xG difference breaking ties. Pos is the real table position. "
               "Own goals have no xG and are left out of xPts, so a team that benefited from one has slightly fewer xPts than its real points suggest.")

with tabs[1]:
    which = st.radio("Compare", ["Points vs xPts", "Goals scored vs xG", "Goals conceded vs xGA"], horizontal=True, key="tt_which")
    pairs = {"Points vs xPts": ("xPts", "Pts"), "Goals scored vs xG": ("xG", "GF"), "Goals conceded vs xGA": ("xGA", "GA")}
    xc, yc = pairs[which]
    team = st.selectbox("Highlight a team", [""] + list(sub.index), key="tt_hl")
    show(scatter_labeled(sub[xc], sub[yc], sub.index, f"Expected ({xc})", f"Actual ({yc})", diag=True,
                         highlight=team or None, label_n=8))
    if which == "Goals conceded vs xGA":
        st.caption("Below the dashed line = conceded fewer than the chances allowed (good defence or goalkeeping, or luck). "
                   "The labelled teams are the ones furthest from the line.")
    else:
        st.caption("Above the dashed line = more than the chances suggest. The labelled teams are the ones furthest from the line. "
                   "Six matches is a small sample, so large gaps are mostly variance.")

with tabs[2]:
    xm = D.xpts_match.merge(D.m[["match_id", "date", "group_key", "league", "home_score", "away_score"]], on="match_id")
    xm = xm[xm["league"] == league].sort_values("date")
    hs, as_ = xm["home_score"].astype(int), xm["away_score"].astype(int)
    xm["res"] = np.where(hs > as_, "H", np.where(hs < as_, "A", "D"))
    xm["p_result"] = np.where(xm["res"] == "H", xm["p_home"], np.where(xm["res"] == "A", xm["p_away"], xm["p_draw"]))
    xm["Deserved"] = xm[["p_home", "p_draw", "p_away"]].values.argmax(1)
    xm["Deserved"] = xm["Deserved"].map({0: "Home win", 1: "Draw", 2: "Away win"})
    xm["Actual"] = xm["res"].map({"H": "Home win", "D": "Draw", "A": "Away win"})
    t = pd.DataFrame({"Date": xm["date"].dt.strftime("%d %b"), "Home": xm["home_team"],
                      "Score": hs.astype(str) + "-" + as_.astype(str), "Away": xm["away_team"],
                      "Home win %": (xm["p_home"] * 100).round(0), "Draw %": (xm["p_draw"] * 100).round(0),
                      "Away win %": (xm["p_away"] * 100).round(0), "Chance of this result %": (xm["p_result"] * 100).round(0),
                      "Deserved": xm["Deserved"], "Actual": xm["Actual"]})
    st.write("Least deserved results (lowest chance of the result that happened)")
    st.dataframe(t.sort_values("Chance of this result %").head(10), hide_index=True)
    st.write("All matches")
    st.dataframe(t, hide_index=True)
    st.caption("Probabilities are computed from every shot's xG, treating shots as independent, so they ignore game state "
               "and own goals. A low chance of the actual result marks an upset by the numbers, not a verdict on the team.")
