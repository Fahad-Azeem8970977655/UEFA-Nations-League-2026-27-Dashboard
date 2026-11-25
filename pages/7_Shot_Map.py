import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.pitch import new_figure, show
from unl.style import INK
from unl.viz2 import ON_TARGET, draw_goal_frame, draw_shot_map, goals_table, shot_kind

D = setup("Shot map | Nations League 2026-27")

st.title("Shot map")
st.caption("Where shots were taken from and where goals were scored. Gold star = goal, circle size = xG.")

S = D.shots.copy()
S["kind"] = shot_kind(S["shot_type"])
S["opponent"] = np.where(S["team"] == S["home_team"], S["away_team"], S["home_team"])

c = st.columns([1.2, 1, 1.4])
team = c[0].selectbox("Team", ["All teams"] + D.teams, key="shot_team")
role = c[1].radio("Shots", ["Taken", "Conceded"], horizontal=True, key="shot_role", disabled=team == "All teams")
result = c[2].radio("Show", ["All shots", "On target", "Goals only"], horizontal=True, key="shot_result", index=0)

if team != "All teams":
    mids = D.tm.loc[D.tm["team"] == team, ["match_id", "opponent", "side", "date"]].sort_values("date")
    labels = {r.match_id: f"{r.opponent} ({'H' if r.side == 'home' else 'A'}, {r.date:%d %b})" for r in mids.itertuples()}
    sel = st.multiselect("Matches included", list(labels), default=list(labels), format_func=lambda k: labels[k],
                         key="shot_matches")
    S = S[S["match_id"].isin(sel)]
    S = S[S["team"] == team] if role == "Taken" else S[S["team"] != team]

f = st.columns(3)
sit = f[0].multiselect("Situation", sorted(S["situation"].dropna().unique()), key="shot_sit")
body = f[1].multiselect("Body part", sorted(S["body_part"].dropna().unique()), key="shot_body")
players = f[2].multiselect("Player", sorted(S["player"].dropna().unique()), key="shot_players")
if sit:
    S = S[S["situation"].isin(sit)]
if body:
    S = S[S["body_part"].isin(body)]
if players:
    S = S[S["player"].isin(players)]
if result == "On target":
    S = S[S["kind"].isin(ON_TARGET)]
elif result == "Goals only":
    S = S[S["kind"] == "goal"]

if S.empty:
    st.info("No shots match these filters.")
    st.stop()

goals = int((S["kind"] == "goal").sum())
xg = pd.to_numeric(S["xg"], errors="coerce").sum()
xgot = pd.to_numeric(S["xgot"], errors="coerce").sum()
kpi_row([("Shots", len(S)), ("Goals", goals), ("xG", f"{xg:.1f}", f"{xg / len(S):.2f} per shot"), ("Goals minus xG", f"{goals - xg:+.1f}"),
         ("xGOT", f"{xgot:.1f}"), ("Conversion", f"{goals / len(S) * 100:.0f}%", f"on target {S['kind'].isin(ON_TARGET).mean() * 100:.0f}%")], min_width=140)

lines = st.checkbox("Draw a line from each goal to the goal mouth", value=True, key="shot_lines")
left, right = st.columns([1.3, 1])
with left:
    fig, axes = new_figure(1, w=7.0, half=True)
    draw_shot_map(axes[0], S, lines=lines)
    show(fig)
with right:
    fig, axes = new_figure(1, w=5.0)
    draw_goal_frame(axes[0], S)
    axes[0].set_title("Where shots ended up (view from the front)", fontsize=10, color=INK)
    show(fig)
st.caption("Pitch: the team attacks to the right. Goal frame: blocked shots have no end point and are left out. "
           "The crossbar height is an estimate from the data range.")

st.subheader("Goals")
G = goals_table(S, D.inc)
if G.empty:
    st.write("No goals in this selection.")
else:
    G = G.sort_values(["date", "minute"])
    t = G[["date", "team", "opponent", "player", "minute", "assist1", "situation", "body_part", "xg"]].copy()
    t["date"] = t["date"].dt.strftime("%d %b")
    t["xg"] = t["xg"].round(2)
    t.columns = ["Date", "Team", "Against", "Scorer", "Minute", "Assist", "Situation", "Body part", "xG"]
    st.dataframe(t, hide_index=True)

st.subheader("Shooters")
agg = S.assign(goal=(S["kind"] == "goal").astype(int), xg_=pd.to_numeric(S["xg"], errors="coerce"),
               xgot_=pd.to_numeric(S["xgot"], errors="coerce"),
               on_t=S["kind"].isin(ON_TARGET).astype(int)).groupby(["player", "team"]).agg(
    Shots=("goal", "size"), OnTarget=("on_t", "sum"), Goals=("goal", "sum"), xG=("xg_", "sum"), xGOT=("xgot_", "sum")
).reset_index()
agg["Goals minus xG"] = agg["Goals"] - agg["xG"]
agg = agg.sort_values(["Goals", "xG"], ascending=False).head(25).round(2)
agg.columns = ["Player", "Team", "Shots", "On target", "Goals", "xG", "xGOT", "Goals minus xG"]
st.dataframe(agg, hide_index=True)
