import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row

D = setup("Groups | Nations League 2026-27")

def form_string(tm, team, n=5):
    r = tm[tm["team"] == team].sort_values("date")["result"].tolist()[-n:]
    return " ".join(r)


st.title("Group stage")
leagues = sorted(D.m["league"].unique())
league = st.radio("League", leagues, horizontal=True, key="groups_league")
sub = D.m[D.m["league"] == league]
goals = int(sub["home_score"].sum() + sub["away_score"].sum())
gp = sub.merge(D.tm[D.tm["side"] == "home"][["match_id", "expected_goals"]].rename(columns={"expected_goals": "hxg"}), on="match_id", how="left")
big = sub.assign(margin=(sub["home_score"] - sub["away_score"]).abs()).sort_values("margin", ascending=False).iloc[0]
kpi_row([("Matches played", len(sub), f"{sub['group_key'].nunique()} groups"), ("Goals", goals, f"{int(sub['home_score'].sum())} home, {int(sub['away_score'].sum())} away"),
         ("Goals per match", f"{goals / max(len(sub), 1):.2f}"), ("Home wins", f"{(sub['home_score'] > sub['away_score']).mean() * 100:.0f}%", f"draws {(sub['home_score'] == sub['away_score']).mean() * 100:.0f}%"),
         ("Biggest win", f"{int(big['home_score'])}-{int(big['away_score'])}", f"{big['home_team']} v {big['away_team']}")], min_width=170)

for gk in sorted(sub["group_key"].unique()):
    st.subheader(gk)
    tbl = D.standings[D.standings["group_key"] == gk].copy() if not D.standings.empty else pd.DataFrame()
    if tbl.empty:
        st.caption("No standings found for this group.")
        continue
    tbl["GD"] = tbl["goals_for"] - tbl["goals_against"]
    tbl["Form"] = tbl["team"].apply(lambda t: form_string(D.tm, t))
    tbl = tbl.sort_values("position")[["position", "team", "played", "wins", "draws", "losses",
                                       "goals_for", "goals_against", "GD", "points", "Form"]]
    tbl.columns = ["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form (last 5)"]
    st.dataframe(tbl, hide_index=True)

st.subheader("Results")
res = sub.copy().merge(D.tm[D.tm["side"] == "home"][["match_id", "expected_goals"]].rename(
    columns={"expected_goals": "hxg"}), on="match_id", how="left").merge(
    D.tm[D.tm["side"] == "away"][["match_id", "expected_goals"]].rename(columns={"expected_goals": "axg"}),
    on="match_id", how="left")
res["Score"] = res["home_score"].astype(int).astype(str) + "-" + res["away_score"].astype(int).astype(str)
res["xG"] = res["hxg"].round(2).astype(str) + " - " + res["axg"].round(2).astype(str)
res["Date"] = res["date"].dt.strftime("%d %b")
out = res[["matchday", "Date", "group_key", "home_team", "Score", "away_team", "xG"]]
out.columns = ["Matchday", "Date", "Group", "Home", "Score", "Away", "xG (home - away)"]
st.dataframe(out, hide_index=True)
