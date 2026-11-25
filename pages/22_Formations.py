import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.charts2 import grouped_bars, scatter_trend
from unl.kpi import kpi_row
from unl.pitch import show
from unl.sample import add_flag, sample_note

D = setup("Formations | Nations League 2026-27")
st.title("Formations")
st.caption("Which shapes teams start with, how each performs, and how much teams change shape between matches.")

L = D.lineups.copy()
L["formation"] = L["formation"].fillna("unknown")
T = D.tm[["match_id", "team", "points", "gf", "ga", "expected_goals", "opp_expected_goals", "ball_possession", "opponent", "date"]]
X = L.merge(T, on=["match_id", "team"], how="left")
X = X.merge(D.shape[["match_id", "team", "line_height", "length", "width"]], on=["match_id", "team"], how="left")
by = X.groupby("formation").agg(Teams_matches=("match_id", "size"), Teams=("team", "nunique"), PPM=("points", "mean"), GF=("gf", "mean"), GA=("ga", "mean"),
                                xG=("expected_goals", "mean"), xGA=("opp_expected_goals", "mean"), Possession=("ball_possession", "mean"),
                                Line=("line_height", "mean"), Length=("length", "mean"))
by["xGD"] = by["xG"] - by["xGA"]
by = by.sort_values("Teams_matches", ascending=False)
chg = X.groupby("team")["formation"].nunique()
top = by.index[0]
kpi_row([("Formations used", X["formation"].nunique(), f"over {len(X)} team-matches"), ("Most common", top, f"{by.iloc[0]['Teams_matches'] / len(X) * 100:.0f}% of starts"),
         ("Back-three share", f"{X['formation'].str.startswith(('3', '5')).mean() * 100:.0f}%", "3-x-x and 5-x-x"),
         ("Teams that never changed", int((chg == 1).sum()), f"of {len(chg)}"), ("Most changes", chg.idxmax(), f"{chg.max()} different shapes")], min_width=180)

tab1, tab2, tab3 = st.tabs(["League view", "Team view", "Formation and shape"])
with tab1:
    mn = st.slider("Minimum team-matches per formation", 3, 30, 8, key="fm_min")
    d = by[by["Teams_matches"] >= mn]
    show(bar_chart(list(d.index), d["Teams_matches"].values, "Team-matches by formation", fmt="{:.0f}", xlabel="Team-matches"))
    c1, c2 = st.columns(2)
    with c1:
        show(grouped_bars(list(d.index), [("xG", d["xG"].values, "#0E7C86"), ("xG against", d["xGA"].values, "#B5326E")], ylabel="xG per match",
                          xlabel="Formation", title="Chances for and against", figsize=(6.4, 3.4)))
    with c2:
        show(grouped_bars(list(d.index), [("Points per match", d["PPM"].values, "#D99A00")], ylabel="Points per match", xlabel="Formation",
                          title="Results", figsize=(6.4, 3.4), fmt="{:.2f}"))
    out = add_flag(d.reset_index().rename(columns={"formation": "Formation", "Teams_matches": "Team-matches"}), "Team-matches", 15)
    st.dataframe(out.round(2), hide_index=True)
    sample_note(out, what="formations")
    st.caption("Formation quality is mixed up with team quality: strong sides pick shapes too. Treat gaps as description, not proof a shape is better.")
with tab2:
    team = st.selectbox("Team", D.teams, key="fm_team")
    t = X[X["team"] == team].sort_values("date")
    out = t[["date", "opponent", "formation", "gf", "ga", "expected_goals", "opp_expected_goals", "line_height"]].copy()
    out["date"] = out["date"].dt.strftime("%d %b")
    out.columns = ["Date", "Opponent", "Formation", "GF", "GA", "xG", "xGA", "Back line"]
    kpi_row([("Formations used", t["formation"].nunique()), ("Main shape", t["formation"].mode().iloc[0], f"{(t['formation'] == t['formation'].mode().iloc[0]).mean() * 100:.0f}% of matches"),
             ("Points per match", f"{t['points'].mean():.2f}")], min_width=190)
    st.dataframe(out.round(2), hide_index=True)
    g = t.groupby("formation").agg(Matches=("points", "size"), PPM=("points", "mean"), xGD=("expected_goals", "mean"))
    st.dataframe(g.round(2))
with tab3:
    d = by[by["Teams_matches"] >= 8]
    show(scatter_trend(d["Line"], d["xGA"], list(d.index), "Average back line height", "xG against per match", label_n=len(d),
                       title="Back line height and xGA by formation"))
    st.caption("Back-three and back-five shapes usually sit deeper. Back line comes from average positions, so it describes the whole match.")
