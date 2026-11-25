import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, HOME
from unl.viz2 import pressing_table

D = setup("Team report | Nations League 2026-27")
st.title("Team report card")
st.caption("One page per team: results, strengths and weaknesses written in plain sentences from rankings inside its own league, and the players who carried it.")


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


league_of = D.rate.table.set_index("team")["league"]
S = D.team_summary
items = {  # label: (series, higher is better)
    "Expected goals per match": (S["xG"], True), "Expected goals against": (S["xG against"], False), "Goals scored per match": (S["Goals scored"], True),
    "Goals conceded per match": (S["Goals conceded"], False), "Shots per match": (S["Shots"], True), "Shots conceded per match": (S["Shots conceded"], False),
    "Possession": (S["Possession %"], True), "Pass accuracy": (S["Pass accuracy %"], True), "Final-third entries": (S["Final third entries"], True),
    "Touches in the box": (S["Touches in box"], True), "Fouls committed": (S["Fouls"], False),
    "Opponent-adjusted rating": (D.rate.table.set_index("team")["Rating"], True),
    "Expected threat per match": (D.xt.team["xT per match"], True),
    "Passing over expectation": (D.xpass.team["PoE per 100"], True),
    "Set-piece xG for": (D.setp["for"]["SP xG / match"], True), "Set-piece xG against": (D.setp["against"]["SP xG / match"], False),
    "Pressing (PPDA, lower is more intense)": (pressing_table(D)["PPDA"], False),
}
fmtv = {"Possession": "{:.0f}%", "Pass accuracy": "{:.0f}%", "Opponent-adjusted rating": "{:+.2f}", "Passing over expectation": "{:+.1f} per 100"}

team = st.selectbox("Team", D.teams, key="rep_team")
lg = league_of.get(team)
peers = [t for t in D.teams if league_of.get(t) == lg]
rows = []
for label, (ser, hi) in items.items():
    ser = ser.reindex(peers).dropna()
    if team not in ser.index:
        continue
    rank = int(ser.rank(ascending=not hi, method="min")[team])
    rows.append({"Measure": label, "Value": fmtv.get(label, "{:.2f}").format(ser[team]), "Rank": rank, "Of": len(ser), "pct": 1 - (rank - 1) / max(len(ser) - 1, 1)})
R = pd.DataFrame(rows)
core = R[~R["Measure"].isin(["Fouls committed", "Possession", "Passes"])].sort_values("pct", ascending=False)
strong, weak = core.head(3), core.sort_values("pct").head(3)

tm = D.tm[D.tm["team"] == team].sort_values("date")
w, d_, l_ = int((tm["points"] == 3).sum()), int((tm["points"] == 1).sum()), int((tm["points"] == 0).sum())
xp = D.xpts_team.loc[team]
rt = D.rate.table.set_index("team").loc[team]
kpi_row([("Record", f"{w}-{d_}-{l_}", f"{int(tm['points'].sum())} points in {len(tm)} matches", "blue"),
         ("Goals", f"{int(tm['gf'].sum())}-{int(tm['ga'].sum())}", f"xG {tm['expected_goals'].sum():.1f}-{tm['opp_expected_goals'].sum():.1f}", "red"),
         ("Rating in league", f"{rt['Rank in league']} of {len(peers)}", f"{rt['Rating']:+.2f} xGD vs average", "green"),
         ("Points vs expected", f"{xp['Pts - xPts']:+.1f}", "positive = results ahead of chances", "gold"),
         ("Main shape", D.lineups[D.lineups["team"] == team]["formation"].mode().iloc[0] if (D.lineups["team"] == team).any() else "n/a", "most used formation", "teal")], min_width=170)

sh = D.shape[D.shape["team"] == team]
style_bits = []
if len(sh):
    allsh = D.shape.groupby("team")["line_height"].mean().reindex(peers)
    style_bits.append(f"back line at {sh['line_height'].mean():.0f} ({ordinal(allsh.rank(ascending=False, method='min')[team])} highest of {len(allsh)})")
pt = D.xpass.team.reindex(peers)
style_bits.append(f"pass risk {ordinal(pt['Risk %'].rank(ascending=False, method='min')[team])} highest")
pg = D.prog.team.reindex(peers)
style_bits.append(f"{pg.loc[team, 'Carry share of progressions %']:.0f}% of progressions by carrying ({ordinal(pg['Carry share of progressions %'].rank(ascending=False, method='min')[team])} most)")

left, right = st.columns(2)
with left:
    st.subheader("Strengths")
    for r in strong.itertuples():
        st.markdown(f"- **{r.Measure}**: {r.Value}, {ordinal(r.Rank)} of {r.Of} in {lg}")
with right:
    st.subheader("Weaknesses")
    for r in weak.itertuples():
        st.markdown(f"- **{r.Measure}**: {r.Value}, {ordinal(r.Rank)} of {r.Of} in {lg}")
st.markdown("**Style:** " + "; ".join(style_bits) + ".")

pl = D.players[D.players["team"] == team].copy()
pl["g_a"] = pl["goals"] + pl["goalassist"]
scorers = pl.sort_values(["g_a", "expectedgoals"], ascending=False).head(3)
rated = pl[pl["minutesplayed"] >= 180].dropna(subset=["rating_mean"]).sort_values("rating_mean", ascending=False).head(3)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Most goal involvements")
    st.dataframe(scorers[["player", "position", "goals", "goalassist", "expectedgoals"]].rename(columns={"player": "Player", "position": "Pos", "goals": "Goals", "goalassist": "Assists", "expectedgoals": "xG"}).round(2), hide_index=True)
with c2:
    st.subheader("Highest rated (180+ minutes)")
    st.dataframe(rated[["player", "position", "minutesplayed", "rating_mean"]].rename(columns={"player": "Player", "position": "Pos", "minutesplayed": "Minutes", "rating_mean": "Avg rating"}).round(2), hide_index=True)

st.subheader("All rankings inside the league")
show(bar_chart(list(core.sort_values("pct")["Measure"]), (core.sort_values("pct")["pct"] * 100).values, f"{team}: percentile among {lg} teams (100 = best)", fmt="{:.0f}", xlabel="Percentile in league"))
st.dataframe(R.drop(columns="pct"), hide_index=True)

lines = [f"# {team} report card ({lg})", f"Record {w}-{d_}-{l_}, {int(tm['points'].sum())} points; goals {int(tm['gf'].sum())}-{int(tm['ga'].sum())}; rating rank {rt['Rank in league']} of {len(peers)}.", "", "## Strengths"]
lines += [f"- {r.Measure}: {r.Value} ({ordinal(r.Rank)} of {r.Of})" for r in strong.itertuples()] + ["", "## Weaknesses"]
lines += [f"- {r.Measure}: {r.Value} ({ordinal(r.Rank)} of {r.Of})" for r in weak.itertuples()] + ["", "Style: " + "; ".join(style_bits) + "."]
st.download_button("Download report (Markdown)", "\n".join(lines).encode("utf-8"), f"{team}_report.md", "text/markdown", key="rep_dl")
st.caption("Ranks compare a team only with the others in its own league. With about six matches each, one strong or weak game can move a rank a long way.")
