import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from unl import bayes as B
from unl import predict as PR
from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.charts2 import grouped_bars
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import GREY, INK

D = setup("Bayesian predictor | Nations League 2026-27")
st.title("Bayesian match predictor")
st.caption("A second match model next to the Match predictor: team strengths with credible ranges, Dixon-Coles low-score correction, and a fair out-of-sample test "
           "against the current ridge model. Every number on this page is checked by leaving one matchday out and predicting it.")


@st.cache_data(show_spinner="Cross-validating the models...")
def cv():
    return B.cross_validate(D)


@st.cache_data(show_spinner=False)
def final_fit(tau, w):
    return B.fit_all(B.match_table(D), tau, w)


@st.cache_data(show_spinner=False)
def ridge():
    return PR.fit_strength(D)


@st.cache_data(show_spinner="Sampling the posterior...")
def post_draws(league, home, away, neutral, tau, w):
    return B.predict_draws(final_fit(tau, w), league, home, away, neutral, n=400)


@st.cache_data(show_spinner=False)
def strengths(league, tau, w):
    return B.strength_table(final_fit(tau, w), league, n=400)


summ, per, best = cv()
fit = final_fit(best["tau"], best["w"])
S0 = ridge()
get = lambda name, col: float(summ.loc[summ["Model"] == name, col].iloc[0])
gap, glo, ghi = B.paired_gap(per, "Ridge on xG (current)", best["name"])
kpi_row([("Flat base rate", f"{get('Flat base rate', 'Log loss'):.3f}", "log loss, lower is better", "slate"),
         ("Ridge model (current)", f"{get('Ridge on xG (current)', 'Log loss'):.3f}", f"hit rate {get('Ridge on xG (current)', 'Hit rate %'):.0f}%", "teal"),
         ("Bayesian + Dixon-Coles", f"{get(best['name'], 'Log loss'):.3f}", f"hit rate {get(best['name'], 'Hit rate %'):.0f}%", "blue"),
         ("Gain over ridge", f"{gap:+.3f}", f"90% range {glo:+.3f} to {ghi:+.3f}", "green"),
         ("Settings chosen", f"prior {best['tau']:g}, xG weight {best['w']:g}", f"rho {fit['rho']:+.2f}", "gold")], min_width=190)

tab1, tab2, tab3, tab4 = st.tabs(["Predict a match", "Team strengths", "Model check", "How it works"])

# ---------------------------------------------------------------- predict
with tab1:
    c = st.columns(4)
    league = c[0].radio("League", list(fit["leagues"]), key="by_league")
    teams = fit["leagues"][league]["teams"]
    home = c[1].selectbox("Home team", teams, index=0, key="by_home")
    away = c[2].selectbox("Away team", [t for t in teams if t != home], index=0, key="by_away")
    neutral = c[3].checkbox("Neutral venue", False, key="by_neutral")
    o = st.columns(2)
    dc = o[0].checkbox("Dixon-Coles low-score correction", True, key="by_dc", help="Adjusts the chances of 0-0, 1-0, 0-1 and 1-1 for the correlation between the two scores.")
    width = o[1].radio("Credible range", ["80%", "50%"], horizontal=True, key="by_width")
    lo_q, hi_q = (10, 90) if width == "80%" else (25, 75)

    lh, la, M = B.predict(fit, league, home, away, neutral, dc=dc)
    p = B.probs(M)
    dr = post_draws(league, home, away, neutral, best["tau"], best["w"])
    lo, hi = np.percentile(dr, [lo_q, hi_q], axis=0)
    lr, lr2 = PR.expected_goals(S0, league, home, away, neutral)
    pr = np.array(PR.outcome_probs(PR.scoreline_probs(lr, lr2)))
    top = PR.top_scorelines(M, 1)[0]
    kpi_row([(f"{home} win", f"{p[0] * 100:.0f}%", f"{width} range {lo[0] * 100:.0f}-{hi[0] * 100:.0f}%", "teal"),
             ("Draw", f"{p[1] * 100:.0f}%", f"{width} range {lo[1] * 100:.0f}-{hi[1] * 100:.0f}%", "gold"),
             (f"{away} win", f"{p[2] * 100:.0f}%", f"{width} range {lo[2] * 100:.0f}-{hi[2] * 100:.0f}%", "magenta"),
             ("Expected goals", f"{lh:.2f} - {la:.2f}", f"{home} - {away}", "blue"),
             ("Most likely score", f"{top[0]}-{top[1]}", f"{top[2] * 100:.0f}% chance", "red")], min_width=170)
    left, right = st.columns(2)
    with left:
        show(grouped_bars([f"{home} win", "Draw", f"{away} win"], [("Ridge (current)", pr * 100, "#8A97A6"), ("Bayesian", p * 100, "#1D5FAF")],
                          ylabel="Chance (%)", title="Result chances: two models", fmt="{:.0f}", figsize=(6.2, 3.4)))
    with right:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        k = 6
        im = ax.imshow(M[:k, :k] * 100, cmap="GnBu", origin="upper")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xlabel(f"{away} goals", fontsize=9)
        ax.set_ylabel(f"{home} goals", fontsize=9)
        ax.set_title("Scoreline chances (%)", fontsize=10, loc="left", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Chance (%)", fontsize=8)
        fig.tight_layout()
        show(fig)
    st.caption("The range is the spread of the match probability across 400 draws from the posterior of the team strengths. It is wide because each team has only about six matches; "
               "it describes uncertainty about the teams, not the luck of the match itself.")

# ---------------------------------------------------------------- strengths
with tab2:
    lg = st.radio("League", list(fit["leagues"]), horizontal=True, key="by_league2")
    T = strengths(lg, best["tau"], best["w"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(3.2, 0.34 * len(T) + 1.2)), sharey=True)
    y = np.arange(len(T))[::-1]
    for ax, col, ttl in ((axes[0], "Attack", "Attack (higher = scores more)"), (axes[1], "Defence", "Defence (higher = concedes less)")):
        ax.hlines(y, T[f"{col} low"], T[f"{col} high"], color="#8A97A6", lw=2.2)
        ax.plot(T[col], y, "o", color="#1D5FAF", ms=5)
        ax.axvline(0, color=GREY, lw=0.8, ls="--")
        ax.set_title(ttl, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("Log scale, relative to the league average", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(list(T["Team"]), fontsize=9)
    fig.tight_layout()
    show(fig)
    st.caption("Dots are the most likely strength, bars the 80% credible range. A team whose bar crosses the dashed line cannot be told apart from an average team of its league. "
               "+0.30 on this scale means about 35% more goals than average.")
    st.dataframe(T.round(2), hide_index=True, height=360)

# ---------------------------------------------------------------- model check
with tab3:
    st.subheader("Leave-one-matchday-out test")
    main = ["Flat base rate", "Ridge on xG (current)", best["name"] + " (no DC)", best["name"]]
    nm = {"Flat base rate": "Flat base rate", "Ridge on xG (current)": "Ridge (current)", best["name"] + " (no DC)": "Bayesian, independent goals", best["name"]: "Bayesian + Dixon-Coles"}
    t = summ.set_index("Model").loc[main].reset_index()
    t["Model"] = t["Model"].map(nm)
    show(bar_chart(list(t["Model"]), t["Log loss"].values, "Log loss on unseen matchdays (lower is better)", fmt="{:.3f}", xlabel="Log loss"))
    st.dataframe(t[["Model", "Log loss", "Brier", "RPS", "Hit rate %", "Mean draw prob %", "Actual draw %", "n"]].round(3), hide_index=True)
    dcg = B.paired_gap(per, best["name"] + " (no DC)", best["name"])
    st.markdown(f"- **Against the ridge model:** Bayesian + Dixon-Coles is {gap:+.3f} log loss better on the same {int(t['n'].iloc[0])} matches (90% range {glo:+.3f} to {ghi:+.3f}). "
                + ("The range stays above zero, so the gain is unlikely to be noise." if glo > 0 else "The range includes zero, so the gain could be noise."))
    st.markdown(f"- **Dixon-Coles effect:** {dcg[0]:+.3f} log loss (90% range {dcg[1]:+.3f} to {dcg[2]:+.3f}). "
                "The correction moves draw chances towards what is seen in the data but does not change the score much on 156 matches.")
    st.caption("Each matchday is predicted by a model fitted only on the other five matchdays. Log loss punishes confident wrong calls, Brier is the squared error of the three probabilities, "
               "RPS respects the order win-draw-loss. Mean draw probability against actual draw rate shows whether a model over- or under-predicts draws.")
    st.subheader("All settings tried")
    g = summ[summ["Model"].str.startswith("Bayes")].copy()
    g["DC"] = np.where(g["Model"].str.contains("no DC"), "no", "yes")
    g["Setting"] = g["Model"].str.replace(" (no DC)", "", regex=False)
    st.dataframe(g.sort_values("Log loss")[["Setting", "DC", "Log loss", "Brier", "RPS", "Hit rate %", "Mean draw prob %"]].round(3), hide_index=True, height=340)
    st.caption("Prior = precision of the zero-centred prior on each attack and defence value (higher = stronger pull to the league average). XG weight 1 fits xG only, 0 fits goals only. "
               "The chosen setting is the strongest prior within 0.01 log loss of the best one.")

# ---------------------------------------------------------------- how
with tab4:
    st.markdown(f"""
**Model.** Per league, expected goals are `exp(base + home edge + attack of the scorer + leakiness of the opponent)`. Attack and leakiness get zero-centred normal priors, so teams with
little evidence stay close to their league average (a hierarchical, shrinkage model). The fitted target is `xG weight x xG + (1 - xG weight) x goals`, with the Poisson likelihood.

**Posterior.** The posterior is approximated by a normal distribution around its most likely value (Laplace approximation), with its spread scaled by the dispersion of the residuals.
Sampling it gives credible ranges for each team strength and, by pushing every draw through the scoreline model, for every match probability.

**Scorelines.** Independent Poisson goals slightly misjudge the low scores. The Dixon-Coles correction adjusts 0-0, 1-0, 0-1 and 1-1 with one correlation parameter, fitted on all played matches
(rho = {fit['rho']:+.2f}; negative means 0-0 and 1-1 are a little more likely than independence says).

**Choosing the settings.** Prior strength (7 values) and the xG weight (3 values) are compared by leave-one-matchday-out log loss, against two references: a flat base rate and the ridge model
of the Match predictor. The reported scores are on matches the model did not see.

**Limits**
- Six matches per team: strengths stay uncertain, and wide ranges are the honest result.
- Leagues are separate pools; teams from different leagues are never compared.
- No injuries, lineups, rest or motivation. The model sees only chance creation and conceding.
- Choosing the settings on the same 156 matches that report the score makes the score slightly optimistic; the gap to the references is the safer number.
""")
