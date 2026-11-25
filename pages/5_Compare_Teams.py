import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import radar_chart
from unl.metrics import RADAR_LABELS, TEAM_METRICS, team_percentiles
from unl.pitch import show
from unl.style import AWAY, HOME

D = setup("Compare teams | Nations League 2026-27")



st.title("Compare teams")
c1, c2 = st.columns(2)
a = c1.selectbox("Team A", D.teams, index=0, key="cmp_a")
b = c2.selectbox("Team B", D.teams, index=min(1, len(D.teams) - 1), key="cmp_b")
summ = D.team_summary
pct = team_percentiles(summ, RADAR_LABELS)
_ra, _rb = D.rate.table.set_index("team").loc[a], D.rate.table.set_index("team").loc[b]
kpi_row([("xG per match", f"{summ.loc[a, 'xG']:.2f} v {summ.loc[b, 'xG']:.2f}"), ("xG against", f"{summ.loc[a, 'xG against']:.2f} v {summ.loc[b, 'xG against']:.2f}"),
         ("Possession", f"{summ.loc[a, 'Possession %']:.0f}% v {summ.loc[b, 'Possession %']:.0f}%"),
         ("Points", f"{int(summ.loc[a, 'Points'])} v {int(summ.loc[b, 'Points'])}"), ("Adjusted rating", f"{_ra['Rating']:+.2f} v {_rb['Rating']:+.2f}", "different leagues are not comparable" if _ra["league"] != _rb["league"] else None)], min_width=170)
left, right = st.columns([1, 1])
with left:
    show(radar_chart([(a, pct.loc[a], HOME), (b, pct.loc[b], AWAY)]))
with right:
    rows = []
    for label, _, better, dec in TEAM_METRICS:
        va, vb = summ.loc[a, label], summ.loc[b, label]
        if pd.isna(va) or pd.isna(vb):
            continue
        rows.append({"Metric (per match)": label, a: round(float(va), dec), b: round(float(vb), dec),
                     "Difference": round(float(va - vb), dec)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
h2h = D.m[((D.m["home_team"] == a) & (D.m["away_team"] == b)) | ((D.m["home_team"] == b) & (D.m["away_team"] == a))]
st.subheader("Head to head this season")
if h2h.empty:
    st.caption("These teams have not met in the group stage.")
else:
    st.dataframe(pd.DataFrame({"Date": h2h["date"].dt.strftime("%d %b"), "Home": h2h["home_team"],
                               "Score": h2h["home_score"].astype(int).astype(str) + "-" + h2h["away_score"].astype(int).astype(str),
                               "Away": h2h["away_team"]}), hide_index=True)
