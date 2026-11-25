import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from unl import value as V
from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, GREY, HOME, INK

D = setup("Market value model | Nations League 2026-27")
st.title("Market value model")
st.caption("A machine-learning model estimates what a player's market value should be from his age, minutes, output, role and club level, then compares it with the real value. "
           "Every estimate comes from a model that never saw that player.")

if "market_value_m" not in D.players.columns or D.players["market_value_m"].notna().sum() < 100:
    st.info("This page needs unl_2026_27_extra.xlsx next to Home.py (sheet 'profiles')."
            + (f" Loading it failed with: {D.ext_error}" if getattr(D, "ext_error", None) else ""))
    st.stop()


@st.cache_data(show_spinner="Training and cross-validating the models...")
def model():
    return V.fit_value_model(D)


M = model()
cv = M["cv"].set_index("Model")
g = cv.loc["Gradient boosting"]
kpi_row([("Players modelled", M["n"], "90+ minutes with a market value", "blue"), ("R² on unseen players", f"{g['R2 (log value)']:.2f}", f"median-by-league baseline {cv.loc['Median by league and role', 'R2 (log value)']:.2f}", "green"),
         ("Typical error", f"{g['Typical error %']:.0f}%", "median gap between model and real value", "gold"), ("Rank correlation", f"{g['Rank correlation']:.2f}", "ordering of players by value", "teal"),
         ("Inputs", M["n_features"], "features per player", "slate")], min_width=190)

tab1, tab2, tab3, tab4 = st.tabs(["Over and undervalued", "What drives value", "Model check", "How it works"])

with tab1:
    P = M["players"]
    f = st.columns(4)
    direction = f[0].radio("Show", ["Worth more than the market says", "Worth less than the market says"], key="mv_dir")
    minm = f[1].number_input("Minimum minutes", 90, 900, 270, step=45, key="mv_min")
    ages = f[2].slider("Age", int(np.floor(P["age"].min())), int(np.ceil(P["age"].max())), (int(np.floor(P["age"].min())), int(np.ceil(P["age"].max()))), key="mv_age")
    roles = f[3].multiselect("Roles", sorted(P["role"].dropna().unique()), default=[], key="mv_roles", placeholder="All roles")
    F = P[(P["minutesplayed"] >= minm) & P["age"].between(ages[0], ages[1] + 0.99)]
    if roles:
        F = F[F["role"].isin(roles)]
    F = F.sort_values("log_gap", ascending=(direction.startswith("Worth more")))
    top = F.head(20)
    out = pd.DataFrame({"Player": top["player"], "Team": top["team"], "Club": top["club"], "Role": top["role"], "Age": top["age"], "Minutes": top["minutesplayed"].astype(int),
                        "Market value (€m)": top["market_value_m"], "Model value (€m)": top["model_value_m"], "Model / market": top["model_value_m"] / top["market_value_m"]})
    st.dataframe(out.round(2), hide_index=True, height=520)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(F["market_value_m"], F["model_value_m"], s=18, color="#C9D2DC", zorder=2)
    ax.scatter(top["market_value_m"], top["model_value_m"], s=40, color=AWAY if direction.startswith("Worth more") else HOME, zorder=3, edgecolor="white", lw=0.6)
    for r in top.head(8).itertuples():
        ax.annotate(r.player, (r.market_value_m, r.model_value_m), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    lo, hi = min(F["market_value_m"].min(), F["model_value_m"].min()), max(F["market_value_m"].max(), F["model_value_m"].max())
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Market value (€m, log scale)", fontsize=9)
    ax.set_ylabel("Model value (€m, log scale)", fontsize=9)
    ax.set_title("Market value against model value", fontsize=10, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6EBF0", lw=0.6, zorder=0)
    fig.tight_layout()
    show(fig)
    st.caption("Above the dashed line the model values the player higher than the market does. A large gap is a question to investigate, not proof of a bargain: market value also reflects "
               "injuries, potential, the club's transfer position and fame, none of which are in this data.")

with tab2:
    imp = M["importance"].head(10)
    show(bar_chart(list(imp.index), imp.values, "What the model relies on (loss of R² when the input is shuffled)", fmt="{:.2f}", xlabel="Permutation importance"))
    pdp = M["pdp"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(pdp["Age"], pdp["Value"], color=HOME, lw=2.4, marker="o", ms=4)
    ax.set_xlabel("Age", fontsize=9)
    ax.set_ylabel("Average model value (€m)", fontsize=9)
    ax.set_title("What age does to value, everything else unchanged", fontsize=10, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6EBF0", lw=0.6)
    fig.tight_layout()
    show(fig)
    st.caption("Club level and the national team's league carry most of the weight, which is how football valuation works: the same performance is worth more at a bigger club. "
               "Age, contract length and rating come next. The age curve shows value peaking in the mid-twenties and falling after about 30.")

with tab3:
    st.subheader("5-fold cross-validation")
    show(bar_chart(list(cv.index), cv["R2 (log value)"].values, "R² on players the model did not see (higher is better)", fmt="{:.2f}", xlabel="R² of log value"))
    st.dataframe(cv.reset_index().round(3), hide_index=True)
    st.caption("Values are modelled on a log scale, because a player worth 100 times more is not 100 times easier to tell apart. Typical error is the median gap as a percentage. "
               "Gradient boosting beats the ridge regression by capturing interactions (for example, age matters differently at different club levels).")

with tab4:
    st.markdown(f"""
**Target.** The log of the SofaScore market value in millions of euros, for the {M['n']} players with at least 90 minutes and a listed value.

**Inputs ({M['n_features']}).** Age, height, minutes, SofaScore rating, contract years left, the league level (A to D) of the national team, preferred foot, role, 16 per-90 output numbers
(xT, goals, xG, xA, key passes, tackles and others) and a club-level feature: the average log value of the player's club-mates in this dataset. The club feature is computed only from training
players and never includes the player himself.

**Models.** A baseline (median value of the same league level and role), a ridge regression and a gradient-boosting model, compared with 5-fold cross-validation. Each player's prediction comes from
the fold in which he was held out, so the over- and undervalued lists are not flattered by the model having seen the answer.

**Limits**
- Only players of the Nations League league phase, with about three to six matches each.
- Club level is only partly observed: many clubs have just one player in the data.
- Market value is an estimate itself and moves with injuries, potential and transfer news.
- The model finds associations. A gap between model and market is a lead for a scout, not a valuation.
""")
