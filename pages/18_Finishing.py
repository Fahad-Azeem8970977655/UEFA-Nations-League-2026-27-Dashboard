import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.sample import add_flag, sample_note, MIN_MINUTES
from unl.charts import bar_chart
from unl.charts2 import goal_zone_grid, grouped_bars, scatter_trend, stacked_share
from unl.finishing import goal_zone_table
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, HOME

D = setup("Finishing and goalkeepers | Nations League 2026-27")
st.title("Finishing, goalkeepers and set pieces")
st.caption("Goals minus xG over four to six matches is mostly luck. Here every player and team is shrunk towards the average by how much "
           "of the gap shot luck alone could produce, and goalkeepers are judged on goals prevented against xGOT.")

F, K, SP = D.fin, D.gk, D.setp
sh = F.shots
pl = F.player
tot_g, tot_x = int(sh["goal"].sum()), sh["xg"].sum()
luck = F.luck_share_player
kt = K.table[K.table["on_target_faced"] >= 8]
kpi_row([("Open-play goals vs xG", f"{tot_g} vs {tot_x:.0f}", "penalties excluded", "blue"),
         ("Finishing gap that is luck", f"{luck * 100:.0f}%" if pd.notna(luck) else "n/a", "of player goals minus xG, by variance", "red"),
         ("Estimated true skill spread", f"{np.sqrt(F.tau_player) * 100:.0f}%", "goals per xG, one standard deviation", "green"),
         ("Shots on target faced", int(K.table['on_target_faced'].sum()), f"{len(kt)} keepers with 8+", "gold"),
         ("Set-piece share of xG", f"{SP['for']['xg_sp'].sum() / SP['for']['xg_all'].sum() * 100:.0f}%", "excluding penalties", "teal")], min_width=200)

tab1, tab2, tab3 = st.tabs(["Finishing skill", "Goalkeepers", "Set pieces"])

with tab1:
    c = st.columns(3)
    mx = c[0].slider("Minimum xG", 0.5, 4.0, 1.5, 0.25, key="fin_xg")
    pos = c[1].selectbox("Position", ["All", "D", "M", "F"], key="fin_pos")
    view = c[2].radio("View", ["Players", "Teams scoring", "Teams conceding"], horizontal=True, key="fin_view")
    if view == "Players":
        d = pl[pl["xG"] >= mx]
        if pos != "All":
            d = d[d["position"] == pos]
        d = d.sort_values("G - xG", ascending=False)
        if d.empty:
            st.info("No players match.")
        else:
            top = pd.concat([d.head(10), d.tail(5)]).drop_duplicates("player_id")
            lab = [f"{r.player} ({r.team})" for r in top.itertuples()]
            c1, c2 = st.columns(2)
            with c1:
                show(bar_chart(lab, top["G - xG"].values, "Raw: goals minus xG", fmt="{:+.1f}", color=AWAY))
            with c2:
                show(bar_chart(lab, (top["edge"] * top["xG"]).values, "Shrunk: estimated true goals above xG", fmt="{:+.2f}", color=HOME))
            st.caption("Left: what happened. Right: what is probably skill after removing shot luck. "
                       "When the right panel is nearly flat, the data cannot yet separate skill from luck.")
            out = d[["player", "team", "position", "shots", "goals", "xG", "G - xG", "Est. true finishing (goals per xG)", "p_better",
                     "Placement (xGOT - xG on target)"]].copy()
            out.columns = ["Player", "Team", "Pos", "Shots", "Goals", "xG", "Goals - xG", "Est. goals per xG", "Chance better than average",
                           "Placement (xGOT - xG)"]
            out["Chance better than average"] = (out["Chance better than average"] * 100).round(0)
            out = add_flag(out, "Shots", 8)
            st.dataframe(out.round(2), hide_index=True, height=420)
            sample_note(out, what="players")
            st.caption("Placement = xGOT minus xG on the player's on-target shots: where he puts the ball compared with where he shot from. "
                       "It is less noisy than goals but only covers on-target shots.")
    else:
        t = F.team_for if view == "Teams scoring" else F.team_against
        t = t.copy()
        t["Raw goals per xG"] = t["goals"] / t["xG"]
        out = t[["shots", "goals", "xG", "G - xG", "Est. goals per xG", "p_better"]].copy()
        out["p_better"] = (out["p_better"] * 100).round(0)
        out.columns = ["Shots", "Goals", "xG", "Goals - xG", "Est. goals per xG", "Chance above average %"]
        st.dataframe(out.sort_values("Goals - xG", ascending=False).round(2), height=480)
        tau = F.tau_for if view == "Teams scoring" else F.tau_against
        st.caption(("Conceding: goals above xG is bad, and includes goalkeeping and finishing by opponents. " if view == "Teams conceding" else "")
                   + f"Estimated spread of true team differences: {np.sqrt(tau) * 100:.0f}% goals per xG.")
    with st.expander("How the shrinkage works"):
        st.markdown("A shooter's goals minus xG has a noise variance equal to the sum of p(1-p) over his shots. Writing true finishing as a "
                    "multiplicative edge with unknown spread, the spread is estimated from all shooters at once (method of moments). "
                    "Each player is then moved towards average by the share of his variance that is luck. If the estimated spread is "
                    "zero, every player shrinks to average. Penalties and own goals are excluded.")

with tab2:
    c = st.columns(2)
    mf = c[0].number_input("Minimum shots on target faced", 1, 40, 8, key="gk_min")
    show_all = c[1].checkbox("Include keepers below the minimum in the table", False, key="gk_all")
    t = K.table[K.table["on_target_faced"] >= (0 if show_all else mf)].copy()
    t = t.sort_values("Shrunk goals prevented", ascending=False)
    if t.empty:
        st.info("No keepers match.")
    else:
        b = t.iloc[0]
        kpi_row([("Keepers", len(t), None, "blue"), ("Best shot-stopper (shrunk)", b["player"], f"{b['Shrunk goals prevented']:+.2f} goals prevented, {b['team']}", "green"),
                 ("Goals prevented, all keepers", f"{t['Goals prevented'].sum():+.1f}", "xGOT faced minus goals conceded", "gold"),
                 ("Longest passer", t.sort_values("Long-ball share %").iloc[-1]["player"], f"{t['Long-ball share %'].max():.0f}% long balls", "teal")], min_width=220)
        out = t[["player", "team", "minutes", "on_target_faced", "goals_conceded", "xGOT_faced", "Goals prevented", "Shrunk goals prevented", "Save %",
                 "Expected save %"]].copy()
        out.columns = ["Keeper", "Team", "Minutes", "On target faced", "Goals conceded", "xGOT faced", "Goals prevented", "Shrunk", "Save %", "Expected save %"]
        out["Minutes"] = out["Minutes"].astype(int)
        out["On target faced"] = out["On target faced"].astype(int)
        out["Goals conceded"] = out["Goals conceded"].fillna(0).astype(int)
        out = add_flag(out, "On target faced", 15)
        st.dataframe(out.round(2), hide_index=True, height=360)
        sample_note(out, what="keepers")
        st.caption("Goals prevented = xGOT faced minus goals conceded (positive is good). Shrunk pulls it towards zero by how much shot luck "
                   "could explain it. Penalties are included.")
        st.subheader("Save map")
        names = {int(r.player_id): f"{r.player} ({r.team})" for r in t.itertuples() if r.on_target_faced > 0}
        who = st.selectbox("Keeper", ["All keepers"] + list(names), format_func=lambda k: k if k == "All keepers" else names[k], key="gk_who")
        shots = K.shots if who == "All keepers" else K.shots[K.shots["gk_id"] == who]
        z = goal_zone_table(shots)
        l, r = st.columns(2)
        with l:
            show(goal_zone_grid(z, "Save %", "Save % by goal zone (keeper's view of the goal)"))
        with r:
            show(goal_zone_grid(z, "Expected save %", "What xGOT expected"))
        st.caption("Each cell shows save percentage, shots on target and goals. The right panel is the save percentage the average keeper would "
                   "be expected to have from the same shots. Posts count as on target. Side positions are drawn as seen from the pitch.")
        st.subheader("Distribution profile")
        d2 = K.table[K.table["minutes"] >= 180]
        show(scatter_trend(d2["Long-ball share %"], d2["Pass accuracy %"], d2["player"], "Long-ball share of passes (%)", "Pass accuracy (%)",
                           title="Keepers with 180+ minutes: launch or build", label_n=10))
        prof = d2[["player", "team", "minutes", "Passes per 90", "Long-ball share %", "Long-ball accuracy %", "Pass accuracy %",
                   "Sweeper actions per 90", "High claims per 90", "Touches per 90"]].sort_values("Long-ball share %", ascending=False)
        prof.columns = ["Keeper", "Team", "Minutes"] + list(prof.columns[3:])
        prof["Minutes"] = prof["Minutes"].astype(int)
        st.dataframe(prof.round(1), hide_index=True, height=320)

with tab3:
    f, a = SP["for"], SP["against"]
    side = st.radio("View", ["For", "Against"], horizontal=True, key="sp_side")
    t = f if side == "For" else a
    hl = st.selectbox("Highlight a team", [""] + list(t.index), key="sp_hl")
    best = t["SP xG / match"].idxmax() if side == "For" else t["SP xG / match"].idxmin()
    kpi_row([("Set-piece xG per match", f"{t['xg_sp'].sum() / t['Matches'].sum():.2f}", "league average", "blue"),
             ("Most reliant team" if side == "For" else "Concedes most", t["SP share of xG %"].idxmax(), f"{t['SP share of xG %'].max():.0f}% of its xG", "red"),
             ("Best" if side == "For" else "Best defending", best, f"{t.loc[best, 'SP xG / match']:.2f} xG per match", "green"),
             ("Set-piece goals", int(t["SP goals"].sum()), "corners, free kicks, throw-ins", "gold")], min_width=210)
    c1, c2 = st.columns(2)
    with c1:
        show(scatter_trend(t["Open-play xG / match"], t["SP xG / match"], t.index, "Open-play xG per match", "Set-piece xG per match", highlight=hl or None,
                           title="Set pieces against open play"))
    with c2:
        top = t.sort_values("SP xG / match", ascending=False).head(15)
        parts = {"Corners": top["xg_corner"], "Free kicks": top["xg_fk"], "Throw-ins": top["xg_throw"], "Other set pieces": top["xg_other_sp"]}
        tot = sum(parts.values()).replace(0, np.nan)
        show(stacked_share(list(top.index), {k: (v / tot * 100).fillna(0).values for k, v in parts.items()}, ["#0E7C86", "#D99A00", "#B5326E", "#8A97A6"],
                           title="Where the top 15 set-piece teams' xG comes from"))
    out = t[["Matches", "SP xG / match", "SP share of xG %", "SP goals", "xg_corner", "xg_fk", "xg_throw", "Open-play xG / match"]].copy()
    out.columns = ["Matches", "SP xG / match", "SP share of xG %", "SP goals", "Corner xG", "Free-kick xG", "Throw-in xG", "Open-play xG / match"]
    st.dataframe(out.round(2).sort_values("SP xG / match", ascending=False), height=420)
    st.caption("Set pieces are corners, free kicks, other set-piece situations and long throw-ins as labelled by SofaScore. Penalties are separate and left out.")
