import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.helpers import num
from unl.kpi import kpi_row

D = setup("Data quality | Nations League 2026-27")
st.title("Data quality")
st.caption("What is in the data and what is missing, per table and per match, so you know which charts to trust.")

mids = D.m["match_id"].unique()
tables = {"Shots": D.shots, "Passes": D.passes, "Carries": D.carries, "Defensive actions": D.defensive, "Touch map points": D.heat,
          "Average positions": D.avg, "Momentum": D.mom, "Incidents": D.inc, "Player stats": D.ps, "Lineups": D.lineups, "Team stats": D.tm}
cov = pd.DataFrame(index=mids)
for name, t in tables.items():
    cov[name] = pd.Series(t["match_id"].value_counts()).reindex(mids).fillna(0).astype(int)
ok = cov > 0
full = ok.all(axis=1)

shots = D.shots
on_t = shots[shots["shot_type"].isin(["goal", "save", "post"])]
checks = [
    ("Shots without xG", shots["xg"].isna().mean() * 100, len(shots)),
    ("On-target shots without xGOT", on_t["xgot"].isna().mean() * 100, len(on_t)),
    ("On-target shots without goal-mouth zone", on_t["goal_mouth_location"].isna().mean() * 100, len(on_t)),
    ("Passes without end coordinates", D.passes["end_x"].isna().mean() * 100, len(D.passes)),
    ("Player rows without rating", num(D.ps, "rating").isna().mean() * 100, len(D.ps)),
    ("Player rows with minutes but no position", (num(D.ps, "minutesplayed").fillna(0).gt(0) & D.ps["position"].isna()).mean() * 100, len(D.ps)),
    ("Team rows without distance covered", num(D.tm, "distance_covered").isna().mean() * 100, len(D.tm)),
    ("Team rows without possession", num(D.tm, "ball_possession").isna().mean() * 100, len(D.tm)),
    ("Player-match rows without average position", 100 - D.avg[["match_id", "player_id"]].drop_duplicates().shape[0] / max(D.ps.loc[num(D.ps, "minutesplayed").fillna(0).gt(0), ["match_id", "player_id"]].drop_duplicates().shape[0], 1) * 100, len(D.ps)),
]
Q = pd.DataFrame(checks, columns=["Check", "Missing %", "Rows checked"])
worst = Q.sort_values("Missing %").iloc[-1]
kpi_row([("Matches with every table", f"{full.sum()} of {len(full)}", f"{full.mean() * 100:.0f}%", "green"), ("Tables checked", len(tables), None, "blue"),
         ("Largest gap", f"{worst['Missing %']:.0f}%", worst["Check"], "red"), ("Shots with xG", f"{100 - Q.iloc[0]['Missing %']:.0f}%", None, "gold"),
         ("Passes with coordinates", f"{100 - Q.iloc[3]['Missing %']:.1f}%", None, "teal")], min_width=180)

tab1, tab2, tab3 = st.tabs(["Missing values", "Coverage by match", "Table sizes"])
with tab1:
    st.dataframe(Q.round(1), hide_index=True)
    st.caption("Under 1% missing is harmless. Anything above 10% affects the charts built on it: set that against the page you are reading.")
with tab2:
    only = st.checkbox("Show only incomplete matches", True, key="dq_only")
    lab = D.m.set_index("match_id").apply(lambda r: f"{r['date']:%d %b} {r['home_team']} v {r['away_team']}", axis=1)
    v = ok.replace({True: "yes", False: "MISSING"})
    v.insert(0, "Match", lab.reindex(v.index))
    if only:
        v = v[~full.reindex(v.index)]
    if v.empty:
        st.success("Every match has data in every table.")
    else:
        st.dataframe(v.reset_index(drop=True), hide_index=True)
    miss = (~ok).sum().sort_values(ascending=False)
    st.caption("Matches missing, by table: " + ", ".join(f"{k} {int(n)}" for k, n in miss.items() if n) if miss.any() else "No table is missing for any match.")
with tab3:
    t = pd.DataFrame({"Table": list(tables), "Rows": [len(x) for x in tables.values()], "Matches covered": [int(ok[n].sum()) for n in tables]})
    st.dataframe(t, hide_index=True)
    st.caption("Coordinates come from SofaScore tracking and are approximate, especially at the edges of the pitch. Distance data exists only where the provider supplied it.")
