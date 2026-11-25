import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import radar_chart
from unl.metrics import RADAR_LABELS, team_percentiles, team_profile_table
from unl.pitch import (PASS_FILTERS, apply_pass_filter, draw_defensive, draw_heat, draw_passes, draw_shots,
                       new_figure, show)
from unl.style import AWAY, HOME, INK

D = setup("Teams | Nations League 2026-27")



st.title("Teams")
team = st.selectbox("Team", D.teams, key="team_pick")
t = D.tm[D.tm["team"] == team].sort_values("date")
srow = D.standings[D.standings["team"] == team] if not D.standings.empty else pd.DataFrame()
if len(srow):
    r = srow.iloc[0]
    st.caption(f"{r['group_key']}: position {int(r['position'])}, {int(r['points'])} points "
               f"({int(r['wins'])}W {int(r['draws'])}D {int(r['losses'])}L, "
               f"goals {int(r['goals_for'])}-{int(r['goals_against'])})")
summ = D.team_summary
rk = lambda col, asc=False: int(summ[col].rank(ascending=asc, method="min")[team])
kpi_row([("Matches", int(summ.loc[team, "Matches"]), f"{int(summ.loc[team, 'Points'])} points"),
         ("xG per match", f"{summ.loc[team, 'xG']:.2f}", f"rank {rk('xG')} of {len(summ)}"),
         ("xG against", f"{summ.loc[team, 'xG against']:.2f}", f"rank {rk('xG against', True)} (1 = fewest)"),
         ("Possession", f"{summ.loc[team, 'Possession %']:.0f}%", f"rank {rk('Possession %')}"),
         ("Pass accuracy", f"{summ.loc[team, 'Pass accuracy %']:.0f}%", f"rank {rk('Pass accuracy %')}"),
         ("Shots per match", f"{summ.loc[team, 'Shots']:.1f}", f"rank {rk('Shots')}"),
         ("Rating in league", f"{D.rate.table.set_index('team').loc[team, 'Rating']:+.2f}", "opponent-adjusted xGD")], min_width=140)

if summ.loc[team, "Matches"] < 5:
    st.warning(f"{team} has played only {int(summ.loc[team, 'Matches'])} matches: averages and ranks are unstable.")
left, right = st.columns([1, 1])
with left:
    pct = team_percentiles(summ, RADAR_LABELS)
    show(radar_chart([(team, pct.loc[team], HOME)]))
    st.caption("Each axis is the team's percentile among all teams (100 = best). "
               "Defensive axes such as xG against are inverted so that outward is always better.")
with right:
    st.dataframe(team_profile_table(summ, team), hide_index=True)

st.subheader("Match log")
log = t[["date", "matchday", "opponent", "side", "gf", "ga", "result", "expected_goals", "opp_expected_goals",
         "ball_possession", "total_shots", "shots_on_target", "pass_acc"]].copy()
log["date"] = log["date"].dt.strftime("%d %b")
log["gf"] = log["gf"].astype(int)
log["ga"] = log["ga"].astype(int)
log["pass_acc"] = log["pass_acc"].round(0)
log.columns = ["Date", "Matchday", "Opponent", "Home/away", "GF", "GA", "Result", "xG", "xG against",
               "Possession %", "Shots", "On target", "Pass accuracy %"]
st.dataframe(log, hide_index=True)

st.subheader("Maps")
labels = {r.match_id: f"{r.opponent} ({'H' if r.side == 'home' else 'A'}, {r.date:%d %b})" for r in t.itertuples()}
sel = st.multiselect("Matches included", list(labels), default=list(labels), format_func=lambda k: labels[k],
                     key="team_map_matches")
if not sel:
    st.info("Select at least one match.")
    st.stop()
tabs = st.tabs(["Shots", "Passing", "Defensive actions", "Touch heatmap"])

with tabs[0]:
    mine = D.shots[(D.shots["team"] == team) & D.shots["match_id"].isin(sel)]
    conc = D.shots[(D.shots["team"] != team) & D.shots["match_id"].isin(sel)]
    fig, axes = new_figure(2, half=True)
    draw_shots(axes[0], mine, HOME, half=True)
    axes[0].set_title("Shots taken", fontsize=10, color=INK)
    draw_shots(axes[1], conc, AWAY, half=True)
    axes[1].set_title("Shots conceded", fontsize=10, color=INK)
    show(fig)
    a, b = st.columns(2)
    a.metric("Taken: shots / goals / xG", f"{len(mine)} / {(mine['shot_type'] == 'goal').sum()} / {mine['xg'].sum():.1f}")
    b.metric("Conceded: shots / goals / xG", f"{len(conc)} / {(conc['shot_type'] == 'goal').sum()} / {conc['xg'].sum():.1f}")
    st.caption("Star = goal. Circle size = xG. Both panels face the goal on the right.")

with tabs[1]:
    kind = st.selectbox("Pass type", PASS_FILTERS, key="team_pass_kind")
    ok_only = st.checkbox("Completed passes only", value=False, key="team_pass_ok")
    d = D.passes[(D.passes["team"] == team) & D.passes["match_id"].isin(sel)]
    allp = len(d)
    d = apply_pass_filter(d, kind, ok_only)
    fig, axes = new_figure(2)
    draw_passes(axes[0], d, HOME)
    axes[0].set_title(f"{kind}: {len(d):,} of {allp:,} passes", fontsize=10, color=INK)
    draw_heat(axes[1], d)
    axes[1].set_title("Where they start", fontsize=10, color=INK)
    show(fig)
    st.caption("Team attacks to the right. Grey arrows are incomplete passes. "
               "Up to 500 arrows are drawn when there are more passes. "
               "Progressive = completed pass that gets at least 25% closer to goal and ends beyond 40% of the pitch.")

with tabs[2]:
    d = D.defensive[(D.defensive["team"] == team) & D.defensive["match_id"].isin(sel)]
    types = sorted(d["event_action_type"].unique())
    pick = st.multiselect("Actions", types, default=types, key="team_def_types")
    d = d[d["event_action_type"].isin(pick)]
    fig, axes = new_figure(1, w=7.0)
    draw_defensive(axes[0], d)
    show(fig)
    counts = d["event_action_type"].value_counts()
    st.caption("  |  ".join(f"{k}: {v}" for k, v in counts.items()) + f"   (per match: {len(d) / len(sel):.1f})")

with tabs[3]:
    d = D.heat[(D.heat["team"] == team) & D.heat["match_id"].isin(sel)]
    fig, axes = new_figure(1, w=7.0)
    draw_heat(axes[0], d)
    show(fig)
    st.caption("All touches by all players, smoothed. The team attacks to the right.")
