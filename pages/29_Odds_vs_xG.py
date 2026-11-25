import numpy as np
import pandas as pd
import streamlit as st

from unl import ext
from unl.bootstrap import setup
from unl.charts import scatter_labeled
from unl.charts2 import grouped_bars
from unl.kpi import kpi_row
from unl.pitch import show
from unl.sample import add_flag, sample_note

D = setup("Odds vs xG | Nations League 2026-27")
st.title("Bookmaker odds vs xG")
st.caption("Three views of every match: what the bookmakers expected before kick-off, what actually happened, and what the chances created deserved. "
           "Odds are the last pre-match prices for home win, draw and away win, converted to probabilities with the bookmaker margin removed.")

if D.odds is None:
    st.info("This page needs unl_2026_27_extra.xlsx next to Home.py (sheet 'odds_1x2'). "
            + (f"Loading it failed with: {D.ext_error}" if getattr(D, "ext_error", None) else "The file was not found."))
    st.stop()

O = D.odds.copy()
T = D.odds_team.copy()
n = len(O)
fav_won = (~O["upset"] & (O["favourite"] != "D")).sum()
fav_n = (O["favourite"] != "D").sum()
over = T["Pts - market"].idxmax()
under = T["Pts - market"].idxmin()
kpi_row([("Matches with odds", n, f"average bookmaker margin {O['margin'].mean():.1f}%", "blue"),
         ("Favourites that won", f"{fav_won / fav_n * 100:.0f}%", f"{fav_won} of {fav_n} matches with a clear favourite", "green"),
         ("Upsets", int(O["upset"].sum()), "the favourite did not win", "red"),
         ("Beat the market most", over, f"{T.loc[over, 'Pts - market']:+.1f} pts vs market expectation", "gold"),
         ("Fell short most", under, f"{T.loc[under, 'Pts - market']:+.1f} pts vs market expectation", "teal")], min_width=200)

tabs = st.tabs(["Teams", "Upsets", "Match by match", "Market vs xG model"])

# ---------------------------------------------------------------- teams
with tabs[0]:
    league = st.radio("League", ["All"] + sorted(T["League"].unique()), horizontal=True, key="od_league")
    S = T if league == "All" else T[T["League"] == league]
    st.markdown("**Points, market expectation and xG expectation.** Market xPts add up each match's win and draw probabilities from the odds "
                "(3 x win + draw). xG xPts do the same from the chances each team created and allowed.")
    team = st.selectbox("Highlight a team", [""] + list(S.index), key="od_team")
    c1, c2 = st.columns(2)
    with c1:
        show(scatter_labeled(S["Market xPts"], S["Pts"], S.index, "Points the market expected", "Actual points", diag=True, highlight=team or None, label_n=8,
                             title="Results vs market expectation"))
    with c2:
        show(scatter_labeled(S["Market xPts"], S["xG xPts"], S.index, "Points the market expected", "Points the chances deserved", diag=True,
                             highlight=team or None, label_n=8, title="Chances vs market expectation"))
    st.caption("Left: above the line = more points than the odds suggested. Right: above the line = created better chances than the market expected, "
               "whatever the score said. A team high on the right and low on the left was unlucky; the reverse was lucky.")
    out = S.reset_index().rename(columns={"team": "Team"})
    out["Avg win prob"] = out["Avg win prob"] * 100
    out = out.rename(columns={"Avg win prob": "Avg win prob %"}).drop(columns=["Group"])
    out = add_flag(out, "Matches", 5).sort_values("Pts - market", ascending=False)
    st.dataframe(out.round(2), hide_index=True, height=420)
    sample_note(out, what="teams")

# ---------------------------------------------------------------- upsets
with tabs[1]:
    U = O[O["upset"]].copy()
    U["Match"] = U["home_team"] + " " + U["home_score"].astype(int).astype(str) + "-" + U["away_score"].astype(int).astype(str) + " " + U["away_team"]
    U["Result probability % (market)"] = U["pm_actual"] * 100
    U["Result probability % (xG)"] = U["px_actual"] * 100
    U["Favourite probability %"] = U["fav_prob"] * 100
    U["Verdict"] = np.where(U["px_actual"] >= 0.40, "Chances backed the result",
                            np.where(U["px_actual"] < 0.20, "Against the chances", "Open game"))
    lg = st.radio("League", ["All"] + sorted(U["league"].unique()), horizontal=True, key="od_up_league")
    if lg != "All":
        U = U[U["league"] == lg]
    st.markdown(f"**{len(U)} upsets.** Sorted by how unlikely the actual result looked before the match.")
    cols = ["date", "Match", "fav_team", "Favourite probability %", "Result probability % (market)", "Result probability % (xG)", "Verdict"]
    out = U.sort_values("pm_actual")[cols].rename(columns={"date": "Date", "fav_team": "Favourite"})
    out["Date"] = out["Date"].dt.strftime("%d %b")
    st.dataframe(out.round(1), hide_index=True, height=480)
    st.caption("Result probability is the chance of the result that actually happened (a draw counts as a result). Verdict uses the xG model: 40% or more = the chances supported the "
               "result, under 20% = the result went against the chances. The xG model is computed after the match, so it describes what was deserved, not what was expected.")

# ---------------------------------------------------------------- match by match
with tabs[2]:
    f = st.columns([2, 2, 2])
    lg = f[0].radio("League", ["All"] + sorted(O["league"].unique()), horizontal=True, key="od_m_league")
    only_up = f[1].checkbox("Only upsets", False, key="od_m_up")
    M = O if lg == "All" else O[O["league"] == lg]
    if only_up:
        M = M[M["upset"]]
    out = pd.DataFrame({"Date": M["date"].dt.strftime("%d %b"), "Home": M["home_team"], "Score": M["home_score"].astype(int).astype(str) + "-" + M["away_score"].astype(int).astype(str),
                        "Away": M["away_team"], "Odds 1": M["close_home"], "Odds X": M["close_draw"], "Odds 2": M["close_away"],
                        "Market H %": M["pm_home"] * 100, "Market D %": M["pm_draw"] * 100, "Market A %": M["pm_away"] * 100,
                        "xG H %": M["px_home"] * 100, "xG D %": M["px_draw"] * 100, "xG A %": M["px_away"] * 100, "Result": M["result"]})
    st.dataframe(out.round(1), hide_index=True, height=520)
    st.caption("Odds are decimal. Market % remove the bookmaker margin; xG % come from the shot-by-shot expected-goals model used on the True table page. "
               "Result: H = home win, D = draw, A = away win.")

# ---------------------------------------------------------------- market vs xG model
with tabs[3]:
    st.subheader("How good were the probabilities?")
    b_open, b_close, b_xg = O["pmo_brier"].mean(), O["pm_brier"].mean(), O["px_brier"].mean()
    kpi_row([("Opening odds score", f"{b_open:.3f}", "Brier score, lower is better", "slate"),
             ("Closing odds score", f"{b_close:.3f}", "after the line moved", "blue"),
             ("xG model score", f"{b_xg:.3f}", "known only after the match", "teal"),
             ("Home win: market", f"{O['pm_home'].mean() * 100:.0f}%", f"actual {(O['result'] == 'H').mean() * 100:.0f}%", "gold"),
             ("Draw: market", f"{O['pm_draw'].mean() * 100:.0f}%", f"actual {(O['result'] == 'D').mean() * 100:.0f}%", "red")], min_width=190)
    st.caption("Brier score is the average squared gap between the probabilities and what happened (0 = perfect). The xG model scores better because it sees the match's own chances; "
               "the bookmakers had to forecast before kick-off, so the useful comparison is opening against closing odds. With 156 matches, small gaps are noise.")
    cal = ext.calibration(O)
    c1, c2 = st.columns(2)
    for col, nm in ((c1, "Market"), (c2, "xG model")):
        cc = cal[cal["model"] == nm]
        cats = [f"{b}\n(n={int(k)})" for b, k in zip(cc["bin"], cc["n"])]
        with col:
            show(grouped_bars(cats, [("Predicted", cc["predicted"].values * 100, "#1D5FAF"), ("Happened", cc["actual"].values * 100, "#D61F3D")],
                              ylabel="Percent", title=f"{nm}: predicted vs actual frequency", fmt="{:.0f}", figsize=(6.2, 3.4)))
    st.caption("Each bar pair groups every home, draw and away probability in a range and compares the average prediction with how often that outcome happened. "
               "Bars of equal height mean well calibrated. Few matches per bin, so read the pattern, not single bars.")
