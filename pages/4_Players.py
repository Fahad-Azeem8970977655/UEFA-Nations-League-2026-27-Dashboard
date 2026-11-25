import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.sample import add_flag, sample_note, MIN_MINUTES
from unl.kpi import kpi_row
from unl.advanced import POS_NAMES, position_frame, similar_players
from unl.charts import bar_chart, radar_chart
from unl.helpers import num
from unl.metrics import PLAYER_METRICS, metric_values
from unl.pitch import (PASS_FILTERS, apply_pass_filter, draw_carries, draw_defensive, draw_heat, draw_passes,
                       draw_shots, new_figure, show)
from unl.style import HOME, INK

D = setup("Players | Nations League 2026-27")



st.title("Players")
_p = D.players[D.players["minutesplayed"] >= 90]
_g = D.players.sort_values("goals", ascending=False).iloc[0]
_a = D.players.sort_values("goalassist", ascending=False).iloc[0]
_r = _p.dropna(subset=["rating_mean"]).sort_values("rating_mean", ascending=False).iloc[0]
kpi_row([("Players used", len(D.players), f"{int(D.players['minutesplayed'].sum() / 90):,} player-90s"), ("Top scorer", f"{int(_g['goals'])}", f"{_g['player']} ({_g['team']})"),
         ("Most assists", f"{int(_a['goalassist'])}", f"{_a['player']} ({_a['team']})"), ("Best average rating", f"{_r['rating_mean']:.2f}", f"{_r['player']} ({_r['team']}), 90+ min"),
         ("Roles detected", D.roles["role"].nunique(), "see Player roles page")], min_width=190)
tabs = st.tabs(["Leaderboards", "Player profile", "Percentiles and similar players"])
pt = D.players

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    league = c1.selectbox("League", ["All leagues"] + sorted(D.m["league"].unique()), key="lb_league")
    pos = c2.selectbox("Position", ["All", "Goalkeepers", "Defenders", "Midfielders", "Forwards"], key="lb_pos")
    min_min = c3.number_input("Minimum minutes", min_value=0, max_value=1000, value=180, step=30, key="lb_min")
    metric = c4.selectbox("Metric", list(PLAYER_METRICS), key="lb_metric")
    kind, _ = PLAYER_METRICS[metric]
    per90 = False
    if kind == "count":
        per90 = st.checkbox("Per 90 minutes", value=False, key="lb_per90")
    d = pt[pt["minutesplayed"] >= min_min].copy()
    if league != "All leagues":
        teams_in = set(D.m[D.m["league"] == league][["home_team", "away_team"]].values.ravel())
        d = d[d["team"].isin(teams_in)]
    pm = {"Goalkeepers": "G", "Defenders": "D", "Midfielders": "M", "Forwards": "F"}
    if pos != "All":
        d = d[d["position"] == pm[pos]]
    d["value"] = metric_values(d, metric, per90)
    d = d.dropna(subset=["value"]).sort_values("value", ascending=False).head(25)
    if d.empty:
        st.info("No players match these filters.")
    else:
        lab = (d["player"] + " (" + d["team"] + ")").tolist()
        show(bar_chart(lab, d["value"].values, f"{metric}{' per 90' if per90 else ''}",
                       fmt="{:.2f}" if (kind != "count" or per90 or metric in ("Expected goals (xG)", "Expected assists (xA)", "Goals prevented")) else "{:.0f}"))
        out = d[["player", "team", "position", "Matches", "minutesplayed", "value"]].copy()
        out["value"] = out["value"].round(2)
        out.columns = ["Player", "Team", "Pos", "Matches", "Minutes", metric + (" per 90" if per90 else "")]
        out["Minutes"] = out["Minutes"].astype(int)
        out = add_flag(out, "Minutes", MIN_MINUTES)
        st.dataframe(out, hide_index=True)
        sample_note(out, what="players")
        if metric == "Distance per match (km)":
            st.caption("Distance and sprint data is missing for several matches, so this covers only matches that have it.")

with tabs[2]:
    c = st.columns(3)
    pos_k = c[0].selectbox("Position", list(POS_NAMES), format_func=lambda k: POS_NAMES[k], index=2, key="pc_pos")
    mm = c[1].number_input("Minimum minutes", 90, 1000, 180, step=30, key="pc_min")
    peer = st.radio("Compare with", ["Position (coarse)", "Detected role (sharper)"], horizontal=True, key="pc_peer")
    if peer.startswith("Detected"):
        st.caption("Role-based percentiles live on the Player roles page, where each role has its own metric set. "
                   "Open it from the sidebar; the coarse position view is shown below.")
    ident, raw, pct, z = position_frame(pt, pos_k, mm)
    if ident.empty:
        st.info("No players match these filters.")
    else:
        names = {int(r.player_id): f"{r.player} ({r.team})" for r in ident.sort_values("player").itertuples()}
        pid_ = c[2].selectbox("Player", list(names), format_func=lambda k: names[k], key="pc_player")
        i = ident.index[ident["player_id"] == pid_][0]
        left, right = st.columns([1, 1])
        with left:
            show(radar_chart([(names[pid_], pct.loc[i], HOME)]))
            st.caption(f"Percentile among {len(ident)} {POS_NAMES[pos_k].lower()} with at least {mm} minutes (100 = best of the group). "
                       "Counts are per 90 minutes.")
        with right:
            tbl = pd.DataFrame({"Metric": raw.columns, "Value": raw.loc[i].round(2).values,
                                "Percentile": pct.loc[i].round(0).astype(int).values})
            st.dataframe(tbl, hide_index=True)
        st.subheader("Similar players")
        sim = similar_players(ident, z, pid_, 8)
        sim_t = pd.DataFrame({"Player": sim["player"], "Team": sim["team"], "Minutes": sim["minutesplayed"].astype(int),
                              "Similarity %": sim["Similarity"].round(0).astype(int)})
        st.dataframe(sim_t, hide_index=True)
        st.caption("Cosine similarity of standardised per-90 profiles within the same position. It compares playing style, not quality.")

with tabs[1]:
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", D.teams, key="pp_team")
    roster = pt[pt["team"] == team].sort_values("minutesplayed", ascending=False)
    pmap = {int(r.player_id): f"{r.player} ({r.position})" for r in roster.itertuples()}
    pid = c2.selectbox("Player", list(pmap), format_func=lambda k: pmap[k], key="pp_player")
    row = roster[roster["player_id"] == pid].iloc[0]
    c = st.columns(6)
    c[0].metric("Matches", int(row["Matches"]))
    c[1].metric("Minutes", int(row["minutesplayed"]))
    c[2].metric("Goals", int(row["goals"]))
    c[3].metric("Assists", int(row["goalassist"]))
    c[4].metric("xG", f"{row['expectedgoals']:.2f}")
    c[5].metric("Average rating", f"{row['rating_mean']:.2f}" if pd.notna(row["rating_mean"]) else "n/a")

    logp = D.ps[(D.ps["player_id"] == pid) & (num(D.ps, "minutesplayed").fillna(0) > 0)].sort_values("date")
    lt = logp[["date", "opponent", "minutesplayed", "rating", "goals", "goalassist", "expectedgoals", "totalshots",
               "keypass", "totalpass", "accuratepass", "totaltackle", "interceptionwon", "totalclearance",
               "ballrecovery"]].copy()
    lt["date"] = lt["date"].dt.strftime("%d %b")
    for cc in ["goals", "goalassist", "totalshots", "keypass", "totalpass", "accuratepass", "totaltackle",
               "interceptionwon", "totalclearance", "ballrecovery"]:
        lt[cc] = lt[cc].fillna(0).astype(int)
    lt["expectedgoals"] = lt["expectedgoals"].round(2)
    lt.columns = ["Date", "Opponent", "Min", "Rating", "Goals", "Assists", "xG", "Shots", "Key passes", "Passes",
                  "Accurate", "Tackles", "Interceptions", "Clearances", "Recoveries"]
    st.dataframe(lt, hide_index=True)

    labels = {int(r.match_id): f"{r.opponent} ({r.date:%d %b})" for r in logp.itertuples()}
    sel = st.multiselect("Matches included in the maps", list(labels), default=list(labels),
                         format_func=lambda k: labels[k], key="pp_matches")
    if not sel:
        st.info("Select at least one match.")
        st.stop()
    mt = st.tabs(["Passing", "Defensive actions", "Carries and dribbles", "Touch heatmap", "Shots"])
    with mt[0]:
        kind = st.selectbox("Pass type", PASS_FILTERS, key="pp_pass_kind")
        ok_only = st.checkbox("Completed passes only", value=False, key="pp_pass_ok")
        d = D.passes[(D.passes["player_id"] == pid) & D.passes["match_id"].isin(sel)]
        total = len(d)
        d = apply_pass_filter(d, kind, ok_only)
        fig, axes = new_figure(1, w=7.0)
        draw_passes(axes[0], d, HOME)
        axes[0].set_title(f"{kind}: {len(d)} of {total}", fontsize=10, color=INK)
        show(fig)
    with mt[1]:
        d = D.defensive[(D.defensive["player_id"] == pid) & D.defensive["match_id"].isin(sel)]
        fig, axes = new_figure(1, w=7.0)
        draw_defensive(axes[0], d)
        show(fig)
        st.caption("  |  ".join(f"{k}: {v}" for k, v in d["event_action_type"].value_counts().items()))
    with mt[2]:
        car = D.carries[(D.carries["player_id"] == pid) & D.carries["match_id"].isin(sel)]
        dri = D.dribbles[(D.dribbles["player_id"] == pid) & D.dribbles["match_id"].isin(sel)]
        fig, axes = new_figure(1, w=7.0)
        draw_carries(axes[0], car, dri, HOME)
        show(fig)
        st.caption(f"{len(car)} carries, {len(dri)} dribbles ({int(dri['outcome'].sum()) if len(dri) else 0} won).")
    with mt[3]:
        d = D.heat[(D.heat["player_id"] == pid) & D.heat["match_id"].isin(sel)]
        fig, axes = new_figure(1, w=7.0)
        draw_heat(axes[0], d)
        show(fig)
    with mt[4]:
        d = D.shots[(D.shots["player_id"] == pid) & D.shots["match_id"].isin(sel)]
        if d.empty:
            st.info("No shots by this player in the selected matches.")
        else:
            fig, axes = new_figure(1, w=7.0, half=True)
            draw_shots(axes[0], d, HOME, half=True)
            show(fig)
            st.caption(f"{len(d)} shots, {(d['shot_type'] == 'goal').sum()} goals, {d['xg'].sum():.2f} xG.")
