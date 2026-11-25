import numpy as np
import pandas as pd
import streamlit as st

from unl import events as E
from unl.bootstrap import setup
from unl.charts2 import forest, grouped_bars, line_chart
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, HOME

D = setup("Match flow | Nations League 2026-27")
st.title("Match flow")
st.caption("How matches change with the score, cards and substitutions, using the minute of every goal, card, substitution and shot.")

TL = D.tl
LS = E.league_state(TL, "score")
LM = E.league_state(TL, "men")
sub = E.sub_impact(TL)
prof = E.time_profile(D, TL)
late = prof[prof["block"] == 5]["goals"].sum() / max(prof["goals"].sum(), 1) * 100
lo_ci = E.boot_ci(sub["d_xgd"])
kpi_row([("Goals in the last block", f"{late:.0f}%", "minutes 76 to the end", "blue"),
         ("xG per 90 when leading", f"{LS.loc[1.0, 'xG for / 90']:.2f}", f"level {LS.loc[0.0, 'xG for / 90']:.2f}, trailing {LS.loc[-1.0, 'xG for / 90']:.2f}", "red"),
         ("Red cards", len(TL.reds), "with minute data", "green"),
         ("Substitution windows", len(sub), "with 10+ minutes either side", "gold"),
         ("xGD change after a change", f"{sub['d_xgd'].mean():+.2f}", f"per 15 min, 95% range {lo_ci[0]:+.2f} to {lo_ci[1]:+.2f}", "teal")], min_width=200)

tab1, tab2, tab3, tab4 = st.tabs(["Game state", "Goals and red cards", "Substitutions", "Time of match"])

with tab1:
    st.markdown("Teams are tagged each minute as **trailing, level or leading**, and as **a man down, equal or a man up** after red cards. "
                "Rates are per 90 minutes spent in that state.")
    c1, c2 = st.columns(2)
    names = [E.STATES[k] for k in (-1, 0, 1)]
    with c1:
        show(grouped_bars(names, [("xG for", [LS.loc[k, "xG for / 90"] for k in (-1.0, 0.0, 1.0)], HOME),
                                  ("xG against", [LS.loc[k, "xG against / 90"] for k in (-1.0, 0.0, 1.0)], AWAY)],
                          ylabel="xG per 90", title="By score", fmt="{:.2f}", figsize=(5.4, 3.2)))
    with c2:
        mn = ["A man down", "Equal", "A man up"]
        show(grouped_bars(mn, [("xG for", [LM.loc[k, "xG for / 90"] for k in (-1.0, 0.0, 1.0)], HOME),
                               ("xG against", [LM.loc[k, "xG against / 90"] for k in (-1.0, 0.0, 1.0)], AWAY)],
                          ylabel="xG per 90", title="By red cards", fmt="{:.2f}", figsize=(5.4, 3.2)))
    st.caption(f"Minutes: trailing/leading {int(LS.loc[1.0, 'minutes']):,} each, level {int(LS.loc[0.0, 'minutes']):,}. "
               f"Minutes a man up or down: {int(LM.loc[1.0, 'minutes']):,} each, so those rates are noisy.")
    ts = E.team_state(TL)
    sortc = st.selectbox("Rank teams by", ["Trailing minus level: xG for / 90", "Leading minus level: xG for / 90",
                                           "Leading minus level: xG against / 90"], key="mf_sort")
    st.dataframe(ts.sort_values(sortc, ascending=False).round(2), height=420)
    st.caption("Positive 'trailing minus level' = the team creates more when behind (chasing). Negative 'leading minus level' for xG against = it "
               "defends well when ahead. Minutes in each state are small for most teams, so treat single rows with care.")

with tab2:
    c = st.columns(2)
    win = 12
    with c[0]:
        m1 = E.event_curve(TL, TL.goals, "scorer", win, "mom")
        m2 = E.event_curve(TL, TL.goals, "conceder", win, "mom")
        show(line_chart(m1["rel"], [("Scoring team", m1["value"], HOME), ("Conceding team", m2["value"], AWAY)],
                        ylabel="Momentum (SofaScore)", xlabel="Minutes from the goal", title=f"Momentum around {len(TL.goals)} goals", vline=0))
    with c[1]:
        x1 = E.event_curve(TL, TL.goals, "scorer", win, "xgf")
        x2 = E.event_curve(TL, TL.goals, "conceder", win, "xgf")
        k = np.ones(3) / 3
        sm = lambda v: np.convolve(np.nan_to_num(v), k, mode="same")
        show(line_chart(x1["rel"], [("Scoring team", sm(x1["value"].values), HOME), ("Conceding team", sm(x2["value"].values), AWAY)],
                        ylabel="xG per minute (3-min smoothing)", xlabel="Minutes from the goal", title="Chance creation around goals", vline=0))
    st.caption("The spike at minute 0 is the goal itself. Look at the minutes after: do teams that concede push back, do scorers sit?")
    if len(TL.reds):
        c = st.columns(2)
        with c[0]:
            r1 = E.event_curve(TL, TL.reds, "team", 15, "mom")
            show(line_chart(r1["rel"], [("Team with the red card", r1["value"], AWAY)], ylabel="Momentum (SofaScore)",
                            xlabel="Minutes from the red card", title=f"Momentum around {len(TL.reds)} red cards", vline=0))
        with c[1]:
            a = E.event_curve(TL, TL.reds, "team", 15, "xgf")
            b = E.event_curve(TL, TL.reds, "team", 15, "xga")
            k = np.ones(5) / 5
            show(line_chart(a["rel"], [("Carded team xG for", np.convolve(np.nan_to_num(a["value"].values), k, mode="same"), HOME),
                                       ("Carded team xG against", np.convolve(np.nan_to_num(b["value"].values), k, mode="same"), AWAY)],
                            ylabel="xG per minute (5-min smoothing)", xlabel="Minutes from the red card", title="Chances after a red card", vline=0))
        mm = TL.minutes.merge(TL.reds[["match_id", "team", "minute"]].rename(columns={"minute": "red_min"}), on=["match_id", "team"], how="inner")
        after = mm[mm["minute"] > mm["red_min"]]
        st.caption(f"After a red card the carded team's xG for is {after['xgf'].sum():.1f} and against {after['xga'].sum():.1f} in "
                   f"{len(after):,} team-minutes ({after['xgf'].sum() / len(after) * 90:.2f} versus {after['xga'].sum() / len(after) * 90:.2f} per 90). "
                   "Numbers include later cards for the same side.")

with tab3:
    st.markdown("Each substitution minute (several changes at once count as one) is compared with the 15 minutes before and after, in xG difference "
                "per 15 minutes and in momentum. Only windows with at least 10 minutes either side are used.")
    gs = sub.groupby("state").agg(n=("d_xgd", "size"), mean=("d_xgd", "mean")).reindex(["Trailing", "Level", "Leading"])
    cis = [E.boot_ci(sub.loc[sub["state"] == s, "d_xgd"]) for s in gs.index]
    left, right = st.columns([1.1, 1])
    with left:
        show(forest([f"{s} ({int(n)})" for s, n in zip(gs.index, gs["n"])], gs["mean"].values, [c[0] for c in cis], [c[1] for c in cis],
                    title="Change in xGD per 15 min after a substitution, by game state", xlabel="after minus before"))
    with right:
        st.dataframe(pd.DataFrame({"Windows": gs["n"].astype(int), "xGD change": gs["mean"].round(2),
                                   "95% low": [round(c[0], 2) for c in cis], "95% high": [round(c[1], 2) for c in cis]}))
        st.caption("Teams trailing tend to improve after changes and teams leading tend to fall back, so game state explains most of the average. "
                   "Compare teams within the same state.")
    state = st.radio("Game state at the change", ["All", "Trailing", "Level", "Leading"], horizontal=True, key="mf_state")
    s2 = sub if state == "All" else sub[sub["state"] == state]
    g = s2.groupby("team").agg(Windows=("d_xgd", "size"), **{"Avg minute": ("minute", "mean"), "xGD change": ("d_xgd", "mean"), "xG for change": ("d_xgf", "mean"),
                                                             "xG against change": ("d_xga", "mean"), "Momentum change": ("d_mom", "mean")})
    bench = E.bench_impact(D)
    g = g.join(bench[["Sub appearances", "Goals", "Assists", "xG + xA / 90"]], how="left")
    st.dataframe(g.sort_values("xGD change", ascending=False).round(2), height=420)
    st.caption("A team is used as the proxy for its coach. With five or six windows per team a gap of 0.1 xG is noise: look for teams that are "
               "consistently positive across states rather than at the top of one list. Bench columns count goals and assists by players who did not start.")
    with st.expander("All substitution windows"):
        out = s2[["team", "opponent", "minute", "state", "xgd_before", "xgd_after", "d_xgd", "d_mom", "players_in", "players_out"]].copy()
        out.columns = ["Team", "Opponent", "Minute", "State", "xGD before", "xGD after", "Change", "Momentum change", "Players on", "Players off"]
        st.dataframe(out.round(2).sort_values("Change", ascending=False), hide_index=True)

with tab4:
    team = st.selectbox("Team", ["All teams"] + D.teams, key="mf_team")
    d = prof if team == "All teams" else prof[prof["team"] == team]
    g = d.groupby("block").agg(goals=("goals", "sum"), xg=("xg", "sum"), goals_a=("goals_a", "sum"), xg_a=("xg_a", "sum")).reindex(range(6)).fillna(0)
    div = 1 if team != "All teams" else 2
    c1, c2 = st.columns(2)
    with c1:
        show(grouped_bars(E.BLOCKS, [("Goals", g["goals"] / (1 if team != "All teams" else 1), HOME), ("xG", g["xg"], "#9FD3D8")],
                          ylabel="Scored", title=f"{team}: goals and xG for by 15-minute block", fmt="{:.1f}", figsize=(5.6, 3.2)))
    with c2:
        show(grouped_bars(E.BLOCKS, [("Goals", g["goals_a"], AWAY), ("xG", g["xg_a"], "#E3A9C3")], ylabel="Conceded",
                          title=f"{team}: goals and xG against", fmt="{:.1f}", figsize=(5.6, 3.2)))
    t = prof.groupby("team").apply(lambda x: pd.Series({
        "Goals": x["goals"].sum(), "Late goals (76+) %": x.loc[x["block"] == 5, "goals"].sum() / max(x["goals"].sum(), 1) * 100,
        "Goals conceded": x["goals_a"].sum(), "Late conceded (76+)": x.loc[x["block"] == 5, "goals_a"].sum(),
        "Conceded in first 15": x.loc[x["block"] == 0, "goals_a"].sum(), "xG first 15": x.loc[x["block"] == 0, "xg"].sum(),
        "xG last block": x.loc[x["block"] == 5, "xg"].sum(),
        "Second-half xG share %": x.loc[x["block"] >= 3, "xg"].sum() / max(x["xg"].sum(), 1e-9) * 100}), include_groups=False)
    sortc = st.selectbox("Rank teams by", list(t.columns), index=1, key="mf_tsort")
    st.dataframe(t.sort_values(sortc, ascending=False).round(1), height=420)
    st.caption("Blocks follow the clock: stoppage time sits in the last block of each half. Own goals are not in the shot data, so they are missing here.")
