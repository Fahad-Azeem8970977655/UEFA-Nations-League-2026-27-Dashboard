import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.sample import add_flag, sample_note, MIN_MINUTES
from unl.charts import bar_chart
from unl.charts2 import forest, scatter_trend
from unl.kpi import kpi_row
from unl.pitch import show
from unl.ratings import over_under_rated
from unl.style import AWAY, HOME

D = setup("Ratings | Nations League 2026-27")
st.title("Ratings")
st.caption("Team strength adjusted for the opposition inside each league, and what actually drives the SofaScore player rating.")

TR, RM = D.rate, D.rmodel
tab1, tab2 = st.tabs(["Team ratings", "What drives player ratings"])

with tab1:
    league = st.radio("League", sorted(TR.table["league"].unique()), horizontal=True, key="rt_league")
    t = TR.table[TR.table["league"] == league].sort_values("Rating", ascending=False)
    lam, cv, base = TR.lam[league]
    top, adj = t.iloc[0], t.loc[t["Adjustment"].abs().idxmax()]
    kpi_row([("Top-rated team", top["team"], f"{top['Rating']:+.2f} xGD per match vs an average side", "blue"),
             ("Spread top to bottom", f"{t['Rating'].max() - t['Rating'].min():.2f}", "xG per match", "red"),
             ("Largest schedule effect", adj["team"], f"{adj['Adjustment']:+.2f} between raw and adjusted xGD", "green"),
             ("Elo leader", t.sort_values("Elo").iloc[-1]["team"], f"{t['Elo'].max():.0f}", "gold"),
             ("Model error vs a flat guess", f"{cv / base * 100:.0f}%", "leave-one-match-out (lower is better)", "teal")], min_width=200)
    left, right = st.columns([1, 1])
    with left:
        show(forest(list(t["team"]), t["Rating"].values, (t["Rating"] - 1.96 * t["Rating SE"]).values, (t["Rating"] + 1.96 * t["Rating SE"]).values,
                    title=f"{league}: opponent-adjusted rating, 95% range", xlabel="xG difference per match vs average team"))
    with right:
        hl = st.selectbox("Highlight a team", [""] + list(t["team"]), key="rt_hl")
        show(scatter_trend(t["Attack"], -t["Defence"], t["team"], "Attack (xG scored above average)", "Defence (xG conceded below average)",
                           highlight=hl or None, trend=False, label_n=20, title="Attack against defence, opponent-adjusted"))
    out = t[["Rank in league", "team", "group_key", "Rating", "Rating SE", "Attack", "Defence", "Elo", "Raw xGD / match", "Schedule strength", "Adjustment"]].copy()
    out.columns = ["Rank", "Team", "Group", "Rating", "SE", "Attack", "Defence", "Elo", "Raw xGD / match", "Schedule strength", "Adjustment"]
    st.dataframe(out.round(2), hide_index=True, height=440)
    st.caption("Rating: ridge (Bradley-Terry / Massey style) fit of the xG margin of every match in the league, with the penalty chosen by "
               "leave-one-match-out validation. It is shrunk, so top and bottom teams sit closer to zero than their raw xGD. "
               "Attack and Defence split the same model into xG scored and conceded above average. "
               "Schedule strength is the average rating of the opponents faced. Elo runs on the same xG margins. Defence is shown so that higher is better.")

with tab2:
    c = st.columns(3)
    pos = c[0].selectbox("Position", ["G", "D", "M", "F"], index=1, key="rd_pos", format_func=lambda k: {"G": "Goalkeepers", "D": "Defenders", "M": "Midfielders", "F": "Forwards"}[k])
    mm = c[1].number_input("Minimum matches for the over / under list", 2, 6, 3, key="rd_matches")
    co = RM.coefs[RM.coefs["position"] == pos].copy()
    co = co.reindex(co["coef_per_sd"].abs().sort_values(ascending=False).index)
    ou = over_under_rated(RM, mm)
    ou = ou[ou["position"] == pos]
    kpi_row([("Variance explained", f"{RM.r2[pos] * 100:.0f}%", "of match-to-match rating", "blue"),
             ("Biggest driver", co.iloc[0]["driver"], f"{co.iloc[0]['coef_per_sd']:+.2f} rating per SD", "red"),
             ("Player-matches", int((RM.matches['position'] == pos).sum()), "30+ minutes played", "green"),
             ("Most underrated", ou.sort_values("Residual").iloc[-1]["player"] if len(ou) else "n/a",
              f"{ou['Residual'].max():+.2f} above stats" if len(ou) else None, "gold")], min_width=200)
    left, right = st.columns([1, 1])
    with left:
        top = co.head(14)
        show(bar_chart(list(top["driver"]), top["coef_per_sd"].values, f"Rating change per 1 SD of the stat (ridge, {pos})", fmt="{:+.2f}",
                       color=HOME))
    with right:
        out = co[["driver", "coef_per_unit", "coef_per_sd"]].head(14)
        out.columns = ["Driver", "Rating per unit", "Rating per SD"]
        st.dataframe(out.round(3), hide_index=True)
        st.caption("Positive = raises the rating. Units are counts in a match, so one goal is worth about the 'per unit' figure for the position.")
    st.subheader("Rated above or below what the stats suggest")
    if ou.empty:
        st.info("Not enough players. Lower the minimum matches.")
    else:
        l2, r2 = st.columns(2)
        with l2:
            d = ou.sort_values("Residual", ascending=False).head(10)
            show(bar_chart([f"{r.player} ({r.team})" for r in d.itertuples()], d["Residual"].values, "Rated above their stats", fmt="{:+.2f}", color=HOME))
        with r2:
            d = ou.sort_values("Residual").head(10)
            show(bar_chart([f"{r.player} ({r.team})" for r in d.itertuples()], d["Residual"].values, "Rated below their stats", fmt="{:+.2f}", color=AWAY))
        tbl = ou.sort_values("Residual", ascending=False)[["player", "team", "Matches", "Avg rating", "Stat-implied rating", "Residual", "z"]]
        tbl.columns = ["Player", "Team", "Matches", "Avg rating", "Stat-implied", "Rating minus stats", "z"]
        tbl = add_flag(tbl, "Matches", 4)
        st.dataframe(tbl.round(2), hide_index=True, height=360)
        sample_note(tbl, what="players")
        st.caption("Stat-implied rating is the model's rating from the box-score stats alone, used as our own performance index. "
                   "A positive gap means SofaScore rates the player higher than the counted stats explain: things the stats miss, such as off-ball work, "
                   "or an algorithm that rewards what is not in this data. z compares the gap with its sampling noise; above 2 is unlikely to be chance.")
