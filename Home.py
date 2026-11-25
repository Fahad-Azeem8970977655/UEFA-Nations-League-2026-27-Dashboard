"""
UEFA Nations League 2026-27 | league phase dashboard (Streamlit, multipage)

Run:  streamlit run Home.py
Pages live in ./pages and appear in the sidebar automatically.
Shared code lives in ./unl (data loading, metrics, pitch drawing, charts, analytics).

Data (put next to this file, or set UNL_DATA_DIR):
    unl_2026_27_all*.xlsx   summary tables
    unl_2026_27_xy*.xlsx    coordinate tables
    unl_2026_27_extra*.xlsx halves, referees, odds and player profiles (optional; the pages that need it say so when it is missing)
If the Excel files are missing, CSVs in ./unl_data/ with the same table names are used.
The first run reads the Excel files (the large one can take a few minutes) and caches them in ./.cache.
"""
import base64

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import ASSETS, setup
from unl.kpi import kpi_row

D = setup("Nations League 2026-27")

logo = base64.b64encode((ASSETS / "unl_logo_tight.png").read_bytes()).decode()
_n = D.m["match_id"].nunique()
_goals = int(D.m["home_score"].sum() + D.m["away_score"].sum())
_tm = D.tm
_cs = (_tm["ga"] == 0).mean() * 100
_red = int(((D.inc["type"] == "card") & D.inc["class"].isin(["red", "yellowRed"])).sum())
_gl = D.shots[D.shots["shot_type"] == "goal"]
_late = (pd.to_numeric(_gl["minute"], errors="coerce") > 75).mean() * 100
tiles = [  # icon, value, label, style
    ("📅", _n, "Matches played", ""), ("👕", len(D.teams), "Teams in 4 leagues", "light"), ("⚽", _goals, "Goals scored", ""),
    ("📊", f"{_goals / _n:.2f}", "Goals per match", "gold"),
    ("🏠", f"{(D.m['home_score'] > D.m['away_score']).mean() * 100:.0f}%", "Home wins", "light"),
    ("🎯", f"{_tm['expected_goals'].mean():.2f}", "xG per team-match", ""), ("🧤", f"{_cs:.0f}%", "Clean sheets", "green"),
    ("📨", f"{_tm['pass_acc'].mean():.0f}%", "Pass accuracy", "light"),
    ("🥅", f"{_tm['total_shots'].mean():.1f}", "Shots per team-match", ""), ("🟥", _red, "Red cards", "light"),
    ("⏳", f"{_late:.0f}%", "Goals after minute 75", "gold"),
]
rows = [tiles[:4], tiles[4:7], tiles[7:]]
hexes = "".join(
    '<div class="hex-row" style="margin-left:%dpx">%s</div>' % ((43 if i == 1 else 0), "".join(
        f'<div class="hex {sty}"><div class="hex-ico">{ic}</div><div class="hex-val">{v}</div><div class="hex-lab">{lab}</div></div>'
        for ic, v, lab, sty in r)) for i, r in enumerate(rows))
st.markdown(
    f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-logo"><img src="data:image/png;base64,{logo}" alt="UEFA Nations League logo"></div>
    <div class="hero-title">UEFA Nations<br>League</div>
    <div class="hero-kicker">LEAGUE PHASE ANALYTICS DASHBOARD</div>
    <div class="hero-season">Season <b>2026&ndash;27</b></div>
    <div class="hero-pill">Key Findings Dashboard</div>
    <div class="hero-sub">{_n} matches &middot; {len(D.teams)} teams &middot; Leagues A, B, C and D &middot; data from SofaScore</div>
  </div>
  <div class="hex-wrap">{hexes}</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Hexagons summarise the whole league phase. Use the sidebar to open a page; every page starts with its own key numbers.")

st.subheader("League highlights")
top = D.players.sort_values(["goals", "expectedgoals"], ascending=False).iloc[0]
best_att = D.team_summary["xG"].idxmax()
best_def = D.team_summary["xG against"].idxmin()
rt = D.rate.table.sort_values("Rating", ascending=False).iloc[0]
kpi_row([("Top scorer", f"{int(top['goals'])} goals", f"{top['player']} ({top['team']})", "gold"),
         ("Best attack by xG", best_att, f"{D.team_summary.loc[best_att, 'xG']:.2f} xG per match", "teal"),
         ("Best defence by xG", best_def, f"{D.team_summary.loc[best_def, 'xG against']:.2f} xG against per match", "blue"),
         ("Top-rated team", rt["team"], f"{rt['league']}, {rt['Rating']:+.2f} xGD vs average", "green")], min_width=220)

st.subheader("Where to look")
groups = [
    ("Results and teams", [("pages/1_Groups.py", "Groups", "standings, form and results"),
                           ("pages/2_Teams.py", "Teams", "profiles, match logs and maps"),
                           ("pages/3_Matches.py", "Matches", "momentum, shots, lineups, xG race"),
                           ("pages/5_Compare_Teams.py", "Compare teams", "side by side and head to head"),
                           ("pages/19_Ratings.py", "Ratings", "opponent-adjusted team ratings and rating drivers"),
                           ("pages/21_Match_Predictor.py", "Match predictor", "result odds and group luck simulation"),
                          ("pages/31_Bayes_Predictor.py", "Bayesian predictor", "strengths with ranges and a fair model test"),
                          ("pages/30_What_If.py", "What-if simulator", "change a match or a result and see the effect"),
                           ("pages/26_Team_Report.py", "Team report", "strengths, weaknesses, key players")]),
    ("Players", [("pages/4_Players.py", "Players", "leaderboards, profiles, percentiles"),
                 ("pages/13_Player_Roles.py", "Player roles", "roles detected from touch maps"),
                 ("pages/20_Scouting.py", "Scouting", "shortlist builder with export"),
                 ("pages/32_Similar_Players.py", "Similar players", "find players with a similar profile"),
                 ("pages/33_Market_Value.py", "Market value model", "over and undervalued players, by machine learning"),
                 ("pages/18_Finishing.py", "Finishing and goalkeepers", "skill versus luck, shot-stopping, set pieces"),
                 ("pages/25_Player_Form.py", "Player form", "match-by-match rating and form")]),
    ("Tactics and shape", [("pages/14_Defensive_Shape.py", "Defensive shape", "line height, compactness and xGA"),
                           ("pages/8_Pressing.py", "Pressing", "PPDA and regain heights"),
                           ("pages/9_Passing.py", "Passing", "pass maps and flow"),
                           ("pages/15_Pass_Model.py", "Pass model", "expected pass completion and risk"),
                           ("pages/16_Progression.py", "Progression", "channels, passes versus carries"),
                           ("pages/12_Team_Styles.py", "Team styles", "clusters of how teams play")]),
    ("Chances and flow", [("pages/7_Shot_Map.py", "Shot map", "where shots and goals came from"),
                          ("pages/10_True_Table.py", "True table", "expected points and luck"),
                          ("pages/11_Expected_Threat.py", "Expected threat", "xT model, team and player threat"),
                          ("pages/17_Match_Flow.py", "Match flow", "game state, substitutions, time of match"),
                          ("pages/22_Formations.py", "Formations", "shapes used and how they perform"),
                          ("pages/23_Discipline.py", "Discipline", "cards, penalties, VAR and referees"),
                          ("pages/28_Halves.py", "Halves", "first half versus second half"),
                          ("pages/29_Odds_vs_xG.py", "Odds vs xG", "bookmaker expectation versus results and chances"),
                          ("pages/27_Data_Quality.py", "Data quality", "what is missing and where")]),
]
cols = st.columns(len(groups))
for col, (name, links) in zip(cols, groups):
    with col:
        st.markdown(f'<div class="group-label">{name}</div>', unsafe_allow_html=True)
        for path, label, desc in links:
            st.page_link(path, label=label, help=desc)
            st.caption(desc)