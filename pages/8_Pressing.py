import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.pitch import new_figure, show
from unl.style import AWAY, GREY, HOME, INK
from unl.viz2 import PRESS_TYPES, draw_press_map, pressing_table

D = setup("Pressing | Nations League 2026-27")

st.title("Pressing")
st.caption("How high a team wins the ball back and how hard it works against the opponent's build-up.")

T = pressing_table(D)

_b = T["PPDA"].idxmin()
kpi_row([("League average PPDA", f"{T['PPDA'].mean():.1f}", "lower = more pressing"), ("Most intense press", _b, f"PPDA {T.loc[_b, 'PPDA']:.1f}"),
         ("Highest regains", T["Avg regain height"].idxmax(), f"height {T['Avg regain height'].max():.0f}"),
         ("Passive end", T["PPDA"].idxmax(), f"PPDA {T['PPDA'].max():.1f}")], min_width=190)
tab1, tab2 = st.tabs(["Team pressing map", "League ranking"])

with tab1:
    team = st.selectbox("Team", D.teams, key="press_team")
    mids = D.tm[D.tm["team"] == team].sort_values("date")
    labels = {r.match_id: f"{r.opponent} ({'H' if r.side == 'home' else 'A'}, {r.date:%d %b})" for r in mids.itertuples()}
    sel = st.multiselect("Matches included", list(labels), default=list(labels), format_func=lambda k: labels[k],
                         key="press_matches")
    if not sel:
        st.info("Select at least one match.")
        st.stop()
    types = st.multiselect("Actions counted", PRESS_TYPES, default=PRESS_TYPES, key="press_types")
    d = D.defensive[(D.defensive["team"] == team) & D.defensive["match_id"].isin(sel)
                    & D.defensive["event_action_type"].isin(types)]
    r = T.loc[team]
    c = st.columns(5)
    c[0].metric("PPDA", f"{r['PPDA']:.1f}", help="Opponent passes in their own 60% per tackle or interception by the team in that zone. Lower = more pressing.")
    c[1].metric("PPDA rank", f"{int(T['PPDA'].rank().loc[team])} of {len(T)}")
    c[2].metric("High regains / match", f"{r['High regains / match']:.1f}")
    c[3].metric("Avg regain height", f"{r['Avg regain height']:.0f}")
    c[4].metric("Regains in opp half", f"{r['Regains in opp half %']:.0f}%")
    left, right = st.columns([1.5, 1])
    with left:
        fig, axes = new_figure(1, w=7.0)
        draw_press_map(axes[0], d)
        show(fig)
    with right:
        z = pd.cut(d["x"], [0, 33.3, 66.7, 100], labels=["Defensive third", "Middle third", "Final third"], include_lowest=True)
        tab = pd.crosstab(z, d["event_action_type"]).reindex(["Final third", "Middle third", "Defensive third"])
        tab["Total"] = tab.sum(axis=1)
        st.write("Ball-winning actions by third")
        st.dataframe(tab)
        pm = len(sel)
        st.caption(f"{len(d) / pm:.1f} actions per match in this selection.")
    st.caption("Heat = where the team wins the ball (tackles, interceptions, recoveries), team attacking to the right. "
               "PPDA and the ranking use all of the team's matches, not the match filter.")

    pl = d.groupby(["player", "position"]).size().rename("Actions").reset_index().sort_values("Actions", ascending=False).head(15)
    hi = d[d["x"] >= 66.7].groupby("player").size().rename("In final third")
    pl = pl.merge(hi, on="player", how="left").fillna({"In final third": 0})
    pl.columns = ["Player", "Pos", "Actions", "In final third"]
    st.write("Players with most ball-winning actions")
    st.dataframe(pl, hide_index=True)

with tab2:
    metric = st.selectbox("Rank by", ["PPDA", "High regains / match", "Avg regain height", "Press actions / match"], key="press_metric")
    asc = metric == "PPDA"
    R = T[metric].dropna().sort_values(ascending=asc)
    show_n = st.slider("Teams shown", 10, len(R), min(30, len(R)), key="press_n")
    R = R.head(show_n)[::-1]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(R) + 1))
    ax.barh(R.index, R.values, color=[AWAY if t == st.session_state.get("press_team") else HOME for t in R.index])
    ax.set_xlabel(metric + (" (lower = more pressing)" if asc else ""), fontsize=9, color=INK)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    show(fig)
    st.dataframe(T.round(2).reset_index().rename(columns={"team": "Team"}), hide_index=True)
    st.caption("PPDA = opponent passes in their own 60% of the pitch divided by the team's tackles and interceptions in that zone. "
               "The standard PPDA also counts fouls, which have no coordinates here, so values run higher than published PPDA figures. Compare teams with each other, not with outside numbers. "
               "Press actions / match is the team's tackles and interceptions in that zone. Highlighted bar = team picked on the first tab.")
