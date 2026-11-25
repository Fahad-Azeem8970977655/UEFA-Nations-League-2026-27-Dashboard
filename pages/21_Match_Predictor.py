import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from unl import predict as PR
from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.charts2 import stacked_share
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, GREY, HOME, INK

D = setup("Match predictor | Nations League 2026-27")
st.title("Match predictor")
st.caption("Win, draw and loss chances and likely scorelines for two teams in the same league, from opponent-adjusted attack and defence strength. "
           "A second tab replays every group many times to show how much of the table was luck.")


@st.cache_data(show_spinner=False)
def strengths():
    return PR.fit_strength(D)


@st.cache_data(show_spinner="Simulating groups...")
def sims():
    return PR.simulate_groups(D, 4000)


S = strengths()
bt = PR.backtest(D, S)
kpi_row([("Matches used to fit", bt["n"], "played league-phase matches"), ("Hit rate, in sample", f"{bt['hit'] * 100:.0f}%", "most likely result was right"),
         ("Probability given to real result", f"{bt['mean_prob'] * 100:.0f}%", f"flat guess {bt['base_prob'] * 100:.0f}%"),
         ("Log loss", f"{bt['logloss']:.2f}", f"flat guess {bt['base_logloss']:.2f} (lower is better)"), ("Leagues", len(S), "separate models")], min_width=180)

tab1, tab2, tab3 = st.tabs(["Predict a match", "How lucky was each table?", "How it works"])

with tab1:
    c = st.columns(4)
    league = c[0].radio("League", list(S), key="pr_league")
    teams = S[league]["teams"]
    home = c[1].selectbox("Home team", teams, index=0, key="pr_home")
    away = c[2].selectbox("Away team", [t for t in teams if t != home], index=0, key="pr_away")
    neutral = c[3].checkbox("Neutral venue", False, key="pr_neutral")
    lh, la = PR.expected_goals(S, league, home, away, neutral)
    M = PR.scoreline_probs(lh, la)
    ph, pdw, pa = PR.outcome_probs(M)
    top = PR.top_scorelines(M, 5)
    kpi_row([(f"{home} win", f"{ph * 100:.0f}%", None, "teal"), ("Draw", f"{pdw * 100:.0f}%", None, "gold"), (f"{away} win", f"{pa * 100:.0f}%", None, "magenta"),
             ("Expected xG", f"{lh:.2f} - {la:.2f}", f"{home} - {away}", "blue"), ("Most likely score", f"{top[0][0]}-{top[0][1]}", f"{top[0][2] * 100:.0f}% chance", "red")], min_width=170)
    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        k = 6
        im = ax.imshow(M[:k, :k] * 100, cmap="GnBu", origin="upper")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xlabel(f"{away} goals", fontsize=9)
        ax.set_ylabel(f"{home} goals", fontsize=9)
        ax.set_title("Scoreline chances (%)", fontsize=10, loc="left", color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Chance (%)", fontsize=8)
        fig.tight_layout()
        show(fig)
    with right:
        show(bar_chart([f"{home} win", "Draw", f"{away} win"], [ph * 100, pdw * 100, pa * 100], "Result chances", fmt="{:.0f}%", xlabel="Chance (%)"))
        st.dataframe(pd.DataFrame([{"Score": f"{i}-{j}", "Chance %": round(p * 100, 1)} for i, j, p in top]), hide_index=True)
    s = S[league]
    t = pd.DataFrame({"Attack": [s["att"][home], s["att"][away]], "Defence (higher = leakier)": [s["dfn"][home], s["dfn"][away]]}, index=[home, away])
    st.dataframe(t.round(2))
    st.caption("Attack is xG created above an average team of the league; the defence column is the xG a team concedes less than average when high, "
               "so higher is better for both. Home edge in this league: "
               f"{s['home']:+.2f} xG.")

with tab2:
    T = sims()
    league = st.radio("League", sorted(T["Group"].str.extract(r"(League \w)")[0].unique()), horizontal=True, key="sim_league")
    groups = sorted(g for g in T["Group"].unique() if g.startswith(league))
    gk = st.selectbox("Group", groups, key="sim_group")
    g = T[T["Group"] == gk].sort_values("Actual position")
    pcols = [c for c in g.columns if c.startswith("P") and c[1:].isdigit()]
    unlucky = g.assign(diff=g["Actual points"] - g["Expected points"]).sort_values("diff")
    kpi_row([("Teams", len(g)), ("Most fortunate", unlucky.index[-1], f"{unlucky['diff'].iloc[-1]:+.1f} points vs simulated"),
             ("Least fortunate", unlucky.index[0], f"{unlucky['diff'].iloc[0]:+.1f} points vs simulated"),
             ("Winner's chance", f"{g.loc[g['Actual position'].idxmin(), 'P1']:.0f}%", f"{g['Actual position'].idxmin()}, to finish first again")], min_width=190)
    show(stacked_share(list(g.index), {f"Finish {c[1:]}": g[c].values for c in pcols}, ["#1D5FAF", "#0E7C86", "#D99A00", "#B5326E", "#8A97A6"][:len(pcols)],
                       title="How often each team finishes in each position (% of 4,000 replays)"))
    out = g[["Actual position", "Actual points", "Expected points"] + pcols].copy()
    out.columns = ["Actual position", "Actual points", "Expected points"] + [f"Finish {c[1:]} %" for c in pcols]
    st.dataframe(out.round(1))
    st.caption("Each match is replayed by sampling a scoreline from the xG of its shots. Table order uses points, then goal difference, then goals scored. "
               "A team whose actual position has a low chance on its row was probably helped or hurt by luck.")

with tab3:
    st.markdown("""
**Predictor.** A ridge model per league fits the xG each side created in every played match: league base level, a home edge, the team's attack, and the opponent's defence.
Expected xG for each side drives two independent Poisson distributions, which give the scoreline grid, and from it win, draw and loss.

**Limits**
- About six matches per team, so every strength is shrunk and uncertain. Treat 45% against 35% as a lean, not a call.
- It cannot see injuries, lineups, rest or motivation.
- xG, not goals, is what is fitted, so it predicts chance creation and then converts it at an average rate.
- The check in the KPI strip is in sample (the model saw these matches), so it flatters the real accuracy.
- Leagues are separate pools; teams from different leagues are not compared.
""")
