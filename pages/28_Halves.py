import numpy as np
import pandas as pd
import streamlit as st

from unl import ext
from unl.bootstrap import setup
from unl.charts import bar_chart, scatter_labeled
from unl.charts2 import grouped_bars
from unl.kpi import kpi_row
from unl.pitch import show
from unl.sample import add_flag, sample_note

D = setup("Halves | Nations League 2026-27")
st.title("First half vs second half")
st.caption("Every team statistic split by half: who starts fast, who finishes strongly, and what changes after the break. "
           "Goals come from the half-time scores, everything else from the SofaScore period statistics.")

if D.halves is None:
    st.info("This page needs unl_2026_27_extra.xlsx next to Home.py (sheet 'halves'). "
            + (f"Loading it failed with: {D.ext_error}" if getattr(D, "ext_error", None) else "The file was not found."))
    st.stop()

H = D.halves.copy()
refs = D.refs
if refs is not None:
    H = H.merge(refs[["match_id", "injury_time_1", "injury_time_2"]], on="match_id", how="left")
    H["minutes"] = 45 + np.where(H["period"] == "1ST", H["injury_time_1"], H["injury_time_2"])
    inj1, inj2 = refs["injury_time_1"].mean(), refs["injury_time_2"].mean()
else:
    H["minutes"] = 45.0
    inj1 = inj2 = np.nan

adjust = st.checkbox("Adjust for minutes played (per 90 minutes)", False, key="hv_adj",
                     help="The second half is longer because of more stoppage time. Counting stats are scaled to a 90-minute rate using the minutes actually played in that half. "
                          "Possession and pass accuracy are not changed.")
if adjust:
    for k, (lab, kind) in ext.HALF_STATS.items():
        if kind == "sum":
            H[k] = H[k] / H["minutes"] * 90
            H["opp_" + k] = H["opp_" + k] / H["minutes"] * 90
if not np.isnan(inj1):
    st.caption(f"Stoppage time averages {inj1:.1f} minutes in the first half and {inj2:.1f} in the second, so some of the rise in counting stats is simply more minutes played. "
               "Tick the box above to remove that effect.")

# ---------------------------------------------------------------- key numbers
tot = H.groupby("period")[["goals", "expectedGoals", "yellowCards", "fouls"]].sum()
g1, g2 = tot.loc["1ST", "goals"], tot.loc["2ND", "goals"]
if adjust:
    note_g = "per-90 adjusted"
else:
    note_g = f"{int(g1)} first half, {int(g2)} second half"
y_share = tot.loc["2ND", "yellowCards"] / tot["yellowCards"].sum() * 100
xg_share = tot.loc["2ND", "expectedGoals"] / tot["expectedGoals"].sum() * 100
hr = ext.half_results_table(H)
best2 = (hr["GD 2H"] - hr["GD 1H"]).idxmax()
best1 = (hr["GD 1H"] - hr["GD 2H"]).idxmax()
kpi_row([("Goals in 2nd half", f"{g2 / (g1 + g2) * 100:.0f}%", note_g, "blue"),
         ("xG in 2nd half", f"{xg_share:.0f}%", "share of all xG", "teal"),
         ("Yellow cards in 2nd half", f"{y_share:.0f}%", "share of all yellow cards", "gold"),
         ("Best second-half side", best2, f"{hr.loc[best2, 'GD 2H'] - hr.loc[best2, 'GD 1H']:+.0f} goal difference vs 1st half", "green"),
         ("Best first-half side", best1, f"{hr.loc[best1, 'GD 1H'] - hr.loc[best1, 'GD 2H']:+.0f} goal difference vs 2nd half", "red")], min_width=200)

tabs = st.tabs(["League overview", "Teams", "Goals and xG by half", "Match view"])

# ---------------------------------------------------------------- league overview
with tabs[0]:
    L = ext.half_league_table(H)
    st.subheader("What changes after the break")
    pick = ["expectedGoals", "totalShotsOnGoal", "shotsOnGoal", "bigChanceCreated", "cornerKicks", "fouls", "totalTackle"]
    Lp = L.set_index("key").loc[pick]
    show(grouped_bars(list(Lp["Stat"]), [("1st half", Lp["1st half"].values, "#1D5FAF"), ("2nd half", Lp["2nd half"].values, "#D61F3D")],
                      ylabel="Per team per match" + (" (per 90)" if adjust else ""), fmt="{:.2f}", figsize=(8, 3.4)))
    c1, c2 = st.columns([3, 2])
    with c1:
        tbl = L.drop(columns=["key"]).round(2)
        st.dataframe(tbl, hide_index=True, height=420)
    with c2:
        M = L[~L["Stat"].isin(["Possession %"])].sort_values("Change %", ascending=False)
        show(bar_chart(list(M["Stat"]), M["Change %"].values, "Change from 1st to 2nd half, %", fmt="{:+.0f}%", xlabel="Percent change"))
    st.caption("Averages per team per match. Yellow cards roughly double after the break, in line with more fouls and tackles; "
               "possession is 50% in both halves by construction because it is shared between the two teams.")

# ---------------------------------------------------------------- teams
with tabs[1]:
    names = {lab: k for k, (lab, kind) in ext.HALF_STATS.items()}
    f = st.columns([2, 2, 3])
    lab = f[0].selectbox("Statistic", list(names), index=list(names).index("xG"), key="hv_stat")
    league = f[1].radio("League", ["All"] + sorted(H["league"].dropna().unique()), horizontal=True, key="hv_league")
    key = names[lab]
    Hs = H if league == "All" else H[H["league"] == league]
    T = ext.half_team_table(Hs, key)
    T = add_flag(T.reset_index(), "Matches", 5).set_index("team")
    c1, c2 = st.columns(2)
    up, down = T.sort_values("Change", ascending=False).head(10), T.sort_values("Change").head(10)
    with c1:
        show(bar_chart(list(up.index), up["Change"].values, f"Biggest rise after the break: {lab}", fmt="{:+.2f}", xlabel="2nd half minus 1st half"))
    with c2:
        show(bar_chart(list(down.index), down["Change"].values, f"Biggest fall after the break: {lab}", fmt="{:+.2f}", xlabel="2nd half minus 1st half"))
    team = st.selectbox("Highlight a team", [""] + list(T.index), key="hv_team")
    show(scatter_labeled(T["1st half"], T["2nd half"], T.index, f"{lab}, 1st half", f"{lab}, 2nd half", diag=True, highlight=team or None, label_n=8))
    st.caption("Above the dashed line = the team does more of this in the second half. Teams with fewer than 5 matches are marked 'low sample'.")
    out = T.reset_index().rename(columns={"team": "Team"}).sort_values("Change", ascending=False)
    st.dataframe(out.round(2), hide_index=True, height=360)
    sample_note(out, what="teams")

# ---------------------------------------------------------------- goals and xG by half
with tabs[2]:
    st.subheader("Goals, xG and points by half")
    hr2 = hr.copy()
    hr2["GD shift (2H - 1H)"] = hr2["GD 2H"] - hr2["GD 1H"]
    hr2["xGD shift (2H - 1H)"] = hr2["xGD 2H"] - hr2["xGD 1H"]
    team = st.selectbox("Highlight a team", [""] + list(hr2.index), key="hv_team2")
    show(scatter_labeled(hr2["xGD 1H"], hr2["xGD 2H"], hr2.index, "xG difference, 1st half", "xG difference, 2nd half", diag=True,
                         highlight=team or None, label_n=8))
    st.caption("xG difference (chances created minus chances allowed) is steadier than goals over six matches. Above the dashed line = a better second half than first.")
    order = st.radio("Sort by", ["GD shift (2H - 1H)", "xGD shift (2H - 1H)", "GD 2H", "GD 1H"], horizontal=True, key="hv_sort")
    cols = ["Matches", "GF 1H", "GA 1H", "GF 2H", "GA 2H", "GD 1H", "GD 2H", "xG 1H", "xGA 1H", "xG 2H", "xGA 2H", "xGD 1H", "xGD 2H", "Pts 1H", "Pts 2H",
            "GD shift (2H - 1H)", "xGD shift (2H - 1H)"]
    out = add_flag(hr2[cols].reset_index().rename(columns={"team": "Team"}), "Matches", 5).sort_values(order, ascending=False)
    st.dataframe(out.round(2), hide_index=True, height=420)
    sample_note(out, what="teams")
    st.caption("Pts 1H and Pts 2H are the points a team would have if each half were a separate match. They show when a team earns its results, not real points."
               + (" Goals and xG here are per-90 adjusted." if adjust else ""))

# ---------------------------------------------------------------- match view
with tabs[3]:
    mm = D.m.sort_values("date")
    labels = {r.match_id: f"{r.date:%d %b} | {r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}" for r in mm.itertuples()}
    mid = st.selectbox("Match", list(labels), format_func=lambda i: labels[i], key="hv_match")
    r = D.m[D.m["match_id"] == mid].iloc[0]
    X = H[H["match_id"] == mid]
    if X.empty:
        st.info("No half data for this match.")
    else:
        stats = [k for k in ext.HALF_STATS]
        rows = []
        for k in stats:
            lab, kind = ext.HALF_STATS[k]
            row = {"Statistic": lab}
            for side, team in (("home", r["home_team"]), ("away", r["away_team"])):
                for per, ph in (("1ST", "1st half"), ("2ND", "2nd half")):
                    v = X[(X["side"] == side) & (X["period"] == per)][k]
                    row[f"{team} ({ph})"] = float(v.iloc[0]) if len(v) else np.nan
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, height=720)
        st.caption(f"Half-time score {int(r['home_ht'])}-{int(r['away_ht'])}, full time {int(r['home_score'])}-{int(r['away_score'])}."
                   + (" Counting stats are per-90 adjusted." if adjust else ""))
