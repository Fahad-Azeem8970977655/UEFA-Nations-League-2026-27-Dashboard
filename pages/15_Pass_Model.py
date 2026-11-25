import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.sample import add_flag, sample_note, MIN_MINUTES
from unl.charts import bar_chart
from unl.charts2 import reliability, scatter_trend
from unl.kpi import fmt, kpi_row
from unl.pitch import draw_passes, new_figure, show
from unl.style import AWAY, HOME, INK

D = setup("Pass model | Nations League 2026-27")
st.title("Expected pass completion")
st.caption("A model gives every pass a chance of being completed from where it starts and ends, its length, direction and type. "
           "Players and teams are then judged on passing over expectation instead of raw accuracy, and the average chance shows who takes risks.")

X = D.xpass
q = X.quality
kpi_row([("Passes modelled", f"{q.n:,}", q.engine, "blue"),
         ("Overall completion", f"{X.passes['ok'].mean() * 100:.1f}%", "actual", "red"),
         ("Difficult passes", f"{X.passes['difficult'].mean() * 100:.1f}%", "model chance under 60%", "green"),
         ("Model skill", f"{q.skill * 100:.0f}%", "Brier improvement on a flat guess", "gold"),
         ("Hit rate", f"{q.accuracy * 100:.1f}%", "right side of 50/50, out of fold", "teal")])

tab1, tab2, tab3, tab4 = st.tabs(["Players", "Teams and risk", "Pass viewer", "Model check"])

with tab1:
    pt = X.player.copy()
    c = st.columns(4)
    mp = c[0].number_input("Minimum passes", 50, 1000, 200, step=50, key="xp_minp")
    pos = c[1].selectbox("Position", ["All", "G", "D", "M", "F"], key="xp_pos")
    shrunk = c[2].checkbox("Shrink small samples", True, key="xp_shr",
                           help="Pulls passes-over-expected towards zero by passes / (passes + 250).")
    team = c[3].selectbox("Team", ["All teams"] + D.teams, key="xp_team")
    d = pt[pt["passes"] >= mp]
    if pos != "All":
        d = d[d["position"] == pos]
    if team != "All teams":
        d = d[d["team"] == team]
    col = "PoE shrunk per 100" if shrunk else "PoE per 100"
    d = d.sort_values(col, ascending=False)
    if d.empty:
        st.info("No players match these filters.")
    else:
        best, worst = d.iloc[0], d.iloc[-1]
        kpi_row([("Players listed", len(d), None, "blue"), ("Best over expectation", best["player"], f"{best[col]:+.1f} per 100 passes ({best['team']})", "green"),
                 ("Furthest below", worst["player"], f"{worst[col]:+.1f} per 100 passes ({worst['team']})", "red"),
                 ("Boldest passer", d.sort_values("Risk %").iloc[-1]["player"], f"{d['Risk %'].max():.0f}% average risk", "gold")], min_width=220)
        top = pd.concat([d.head(12), d.tail(12)]).drop_duplicates("player_id")
        lab = [f"{r.player} ({r.team})" for r in top.itertuples()]
        fig = bar_chart(lab, top[col].values, f"Completed minus expected, per 100 passes{' (shrunk)' if shrunk else ''}", fmt="{:+.1f}")
        show(fig)
        out = d[["player", "team", "position", "passes", "Accuracy %", "Expected acc %", "PoE per 100", "PoE shrunk per 100", "Risk %",
                 "Difficult %", "Difficult acc %", "avg_len"]].copy()
        out.columns = ["Player", "Team", "Pos", "Passes", "Accuracy %", "Expected %", "Over expected per 100", "Shrunk per 100", "Risk %",
                       "Difficult passes %", "Difficult pass accuracy %", "Avg length (m)"]
        out["Minutes"] = d["minutesplayed"].astype(int).values
        out = add_flag(out, "Passes", 300)
        st.dataframe(out.round(1), hide_index=True, height=420)
        sample_note(out, what="players")
        st.caption("Risk % = 100 minus the average model chance of completion: higher means the player attempts harder passes. "
                   "Over expected = completed passes minus the model's expected completions, per 100 passes. Sample size matters, so use the shrunk "
                   "version for ranking.")

with tab2:
    tt = X.team.copy()
    c = st.columns(2)
    league_of = pd.concat([D.m[["home_team", "league"]].rename(columns={"home_team": "team"}), D.m[["away_team", "league"]].rename(columns={"away_team": "team"})]).drop_duplicates("team").set_index("team")["league"]
    tt["league"] = league_of.reindex(tt.index)
    lg = c[0].selectbox("League", ["All leagues"] + sorted(tt["league"].dropna().unique()), key="xp_league")
    hl = c[1].selectbox("Highlight a team", [""] + list(tt.index), key="xp_hl")
    if lg != "All leagues":
        tt = tt[tt["league"] == lg]
    c1, c2 = st.columns(2)
    with c1:
        show(scatter_trend(tt["Risk %"], tt["Accuracy %"], tt.index, "Risk: average chance of failure per pass (%)", "Pass accuracy (%)",
                           highlight=hl or None, title="Risk taken against accuracy"))
    with c2:
        show(scatter_trend(tt["Forward per pass (m)"], tt["PoE per 100"], tt.index, "Forward distance per pass (m)", "Completed over expected per 100",
                           highlight=hl or None, title="Directness against execution"))
    st.dataframe(tt.drop(columns=["league", "completed", "expected", "avg_xp"]).round(1).sort_values("Risk %", ascending=False), height=420)
    st.caption("Teams to the right take on harder passes. Teams above the trend line complete more than the difficulty predicts.")

with tab3:
    c = st.columns(3)
    team = c[0].selectbox("Team", D.teams, key="xpv_team")
    P = X.passes[X.passes["team"] == team]
    which = c[1].radio("Show", ["Hardest completed", "Easiest failed", "Difficult passes"], key="xpv_which")
    pl = c[2].selectbox("Player", ["All players"] + sorted(P["player"].unique()), key="xpv_pl")
    if pl != "All players":
        P = P[P["player"] == pl]
    if which == "Hardest completed":
        d = P[P["ok"] == 1].sort_values("xp").head(60)
    elif which == "Easiest failed":
        d = P[P["ok"] == 0].sort_values("xp", ascending=False).head(60)
    else:
        d = P[P["difficult"]].sample(min(300, int(P["difficult"].sum())), random_state=1) if P["difficult"].sum() else P.head(0)
    fig, axes = new_figure(1, w=7.4)
    draw_passes(axes[0], d, HOME)
    axes[0].set_title(f"{which}: {len(d)} passes", fontsize=10, color=INK)
    show(fig)

with tab4:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        show(reliability(q.calibration))
    with c2:
        st.markdown(f"""
**Model.** {q.engine} on start and end coordinates, length, direction, long-ball and cross flags, distance to goal and box entry.
Probabilities are out of fold (5 folds split by match), so no pass is scored by a model that saw it.

- Brier score {q.brier:.3f} against {q.brier_base:.3f} for always guessing the average, a {q.skill * 100:.0f}% improvement.
- Passes per player are small samples: the shrunk column moves each player towards zero by passes / (passes + 250).
- The model does not know about pressure, teammates' positions or the pass's weight, so some of what looks like skill is context.
""")
