import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts2 import grouped_bars, scatter_trend, stacked_share
from unl.kpi import kpi_row
from unl.pitch import show
from unl.progress import CHANNELS

D = setup("Progression | Nations League 2026-27")
st.title("Progression channels")
st.caption("Where teams move the ball forward, left, centre or right, and whether they do it with passes or with carries. "
           "Build-up (progressions that start in the own 45%) is compared with attack (progressions that end in the final third).")

P = D.prog
T = P.team
league_of = pd.concat([D.m[["home_team", "league"]].rename(columns={"home_team": "team"}),
                       D.m[["away_team", "league"]].rename(columns={"away_team": "team"})]).drop_duplicates("team").set_index("team")["league"]
T = T.assign(league=league_of.reindex(T.index))
mv = P.moves
prog_total = int(mv["prog"].sum())
carry_share = (mv["prog"] & (mv["kind"] == "Carry")).sum() / prog_total * 100
c_hi, c_lo = T["Carry share of progressions %"].idxmax(), T["Carry share of progressions %"].idxmin()
sh_hi = T["Channel shift (pts)"].idxmax()
kpi_row([("Progressions", f"{prog_total:,}", "completed passes and carries", "blue"),
         ("Carry share", f"{carry_share:.0f}%", f"passes make up {100 - carry_share:.0f}%", "red"),
         ("Most carry-led team", c_hi, f"{T.loc[c_hi, 'Carry share of progressions %']:.0f}% of progressions", "green"),
         ("Most pass-led team", c_lo, f"{T.loc[c_lo, 'Carry share of progressions %']:.0f}% of progressions", "gold"),
         ("Biggest build-up to attack shift", sh_hi, f"{T.loc[sh_hi, 'Channel shift (pts)']:.0f} points of channel share", "teal")], min_width=200)

tab1, tab2, tab3 = st.tabs(["Channels", "Passes versus carries", "Players"])

with tab1:
    c = st.columns(2)
    team = c[0].selectbox("Team", D.teams, key="pg_team")
    phase_all = c[1].radio("League view", ["Build-up", "Attack"], horizontal=True, key="pg_phase")
    r = T.loc[team]
    lg = T[T["league"] == r["league"]]
    avg = T[[f"Build-up {x} %" for x in CHANNELS] + [f"Attack {x} %" for x in CHANNELS]].mean()
    left, right = st.columns([1, 1])
    with left:
        series = [(f"{team} build-up", [r[f"Build-up {x} %"] for x in CHANNELS], "#0E7C86"),
                  (f"{team} attack", [r[f"Attack {x} %"] for x in CHANNELS], "#B5326E"),
                  ("All teams build-up", [avg[f"Build-up {x} %"] for x in CHANNELS], "#9FD3D8"),
                  ("All teams attack", [avg[f"Attack {x} %"] for x in CHANNELS], "#E3A9C3")]
        show(grouped_bars(CHANNELS, series, ylabel="% of progressions", figsize=(6.4, 3.4), fmt="{:.0f}"))
        st.caption("Channel = third of the pitch width where the progression ends, seen from the team's own attacking direction "
                   "(low y = right). A team whose attack bars differ from its build-up bars changes lanes on the way forward.")
    with right:
        st.dataframe(pd.DataFrame({
            "Measure": ["Progressive passes / match", "Progressive carries / match", "Carry share of progressions %",
                        "Pass : carry ratio", "Metres gained by passes / match", "Metres gained by carries / match",
                        "Final-third entries / match", "Channel shift (pts)"],
            team: [r[m] for m in ["Prog. passes / match", "Prog. carries / match", "Carry share of progressions %", "Pass : carry ratio",
                                  "Metres gained by passes / match", "Metres gained by carries / match", "Final-third entries / match", "Channel shift (pts)"]],
            "League average": [T[m].mean() for m in ["Prog. passes / match", "Prog. carries / match", "Carry share of progressions %", "Pass : carry ratio",
                                                       "Metres gained by passes / match", "Metres gained by carries / match", "Final-third entries / match", "Channel shift (pts)"]]
        }).round(1), hide_index=True)
    st.subheader(f"{r['league']}: {phase_all.lower()} channels")
    lt = lg.sort_values(f"{phase_all} Left %")
    show(stacked_share(list(lt.index), {x: lt[f"{phase_all} {x} %"].values for x in CHANNELS}, ["#4C5FB0", "#D99A00", "#0E7C86"]))
    st.caption("Share of each team's progressions by channel, ordered by left-side share.")

with tab2:
    hl = st.selectbox("Highlight a team", [""] + list(T.index), key="pg_hl")
    c1, c2 = st.columns(2)
    with c1:
        show(scatter_trend(T["Prog. passes / match"], T["Prog. carries / match"], T.index, "Progressive passes per match", "Progressive carries per match",
                           highlight=hl or None, title="How teams progress"))
    with c2:
        show(scatter_trend(T["Carry share of progressions %"], T["Final-third entries / match"], T.index, "Carry share of progressions %",
                           "Final-third entries per match", highlight=hl or None, title="Carry-led or pass-led, and the volume of entries"))
    cols = ["Prog. passes / match", "Prog. carries / match", "Carry share of progressions %", "Pass : carry ratio", "Carry share of metres %",
            "Final-third entries / match"]
    st.dataframe(T[cols].round(1).sort_values("Carry share of progressions %", ascending=False), height=420)
    st.caption("Progression = completed pass, or a carry, that ends at least 25% closer to goal and beyond 40% of the pitch. "
               "Pass : carry ratio is progressive passes per progressive carry.")

with tab3:
    c = st.columns(4)
    pos = c[0].selectbox("Position", ["All", "D", "M", "F"], key="pgp_pos")
    mm = c[1].number_input("Minimum minutes", 90, 1000, 270, step=30, key="pgp_min")
    team = c[2].selectbox("Team", ["All teams"] + D.teams, key="pgp_team")
    sort = c[3].selectbox("Sort by", ["Prog. passes p90", "Prog. carries p90", "Carry share %", "Metres gained p90"], key="pgp_sort")
    d = P.player[P.player["minutesplayed"] >= mm].copy()
    d = d.merge(D.roles[["player_id", "role"]], on="player_id", how="left")
    if pos != "All":
        d = d[d["position"] == pos]
    if team != "All teams":
        d = d[d["team"] == team]
    if d.empty:
        st.info("No players match these filters.")
    else:
        kpi_row([("Players", len(d), None, "blue"), ("Top by " + sort.lower(), d.sort_values(sort, ascending=False).iloc[0]["player"],
                                                    f"{d[sort].max():.1f}", "green"),
                 ("Median carry share", f"{d['Carry share %'].median():.0f}%", None, "gold")], min_width=220)
        show(scatter_trend(d["Prog. passes p90"], d["Prog. carries p90"], d["player"], "Progressive passes per 90", "Progressive carries per 90",
                           title="Players: passers and carriers", label_n=10, trend=False))
        out = d.sort_values(sort, ascending=False)[["player", "team", "role", "minutesplayed", "Prog. passes p90", "Prog. carries p90",
                                                     "Carry share %", "Metres gained p90"]].head(40)
        out.columns = ["Player", "Team", "Role", "Minutes", "Prog. passes p90", "Prog. carries p90", "Carry share %", "Metres gained p90"]
        out["Minutes"] = out["Minutes"].astype(int)
        st.dataframe(out.round(1), hide_index=True)
