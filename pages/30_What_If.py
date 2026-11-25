import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from unl import predict as PR
from unl import whatif as W
from unl.bootstrap import setup
from unl.charts2 import grouped_bars, stacked_share
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import INK

D = setup("What-if simulator | Nations League 2026-27")
st.title("What-if simulator")
st.caption("Change a match or a result and see what the model says would have happened. The match tab starts from the Match predictor's team strengths and applies changes "
           "(more shots, a red card) whose size is measured on this dataset. The group tab changes a played score and re-ranks the group.")


@st.cache_data(show_spinner=False)
def strengths():
    return PR.fit_strength(D)


@st.cache_data(show_spinner=False)
def levers():
    return W.estimate_levers(D), W.estimate_red_card(D)


@st.cache_data(show_spinner="Refitting the model on resampled matches...")
def boot(league, home, away, neutral, sc_items, n):
    lev, red = levers()
    return W.bootstrap_scenario(D, league, home, away, neutral, dict(sc_items), lev, red, n=n)


@st.cache_data(show_spinner="Replaying the group...")
def group_sim(gk, over_items):
    base = W.simulate_group(D, gk)
    wi = W.simulate_group(D, gk, dict(over_items))
    return base, wi


S = strengths()
LEV, RED = levers()
kpi_row([("xG of a shot inside the box", f"{LEV['b_in']:.2f}", f"average, {LEV['share_in'] * 100:.0f}% of shots", "blue"),
         ("xG of a shot outside the box", f"{LEV['b_out']:.2f}", "average", "teal"),
         ("After a red card: sent-off side", f"{RED['carded']:.2f}x", f"xG rate, 80% range {RED['carded_range'][0]:.2f}-{RED['carded_range'][1]:.2f}", "red"),
         ("After a red card: opponent", f"{RED['opponent']:.2f}x", f"xG rate, 80% range {RED['opponent_range'][0]:.2f}-{RED['opponent_range'][1]:.2f}", "green"),
         ("Red-card matches used", RED["n"], f"{RED['minutes']:.0f} minutes after the card", "gold")], min_width=200)

tab1, tab2, tab3 = st.tabs(["Match what-if", "Group what-if", "How it works"])

# ---------------------------------------------------------------- match what-if
with tab1:
    c = st.columns(4)
    league = c[0].radio("League", list(S), key="wi_league")
    teams = S[league]["teams"]
    home = c[1].selectbox("Home team", teams, index=0, key="wi_home")
    away = c[2].selectbox("Away team", [t for t in teams if t != home], index=0, key="wi_away")
    neutral = c[3].checkbox("Neutral venue", False, key="wi_neutral")

    st.markdown("**Change the match.** Extra shots are added at the average xG of that zone; negative numbers take shots away (for example a better defence).")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"**{home}**")
        h_in = st.slider("Extra shots inside the box", -6, 6, 0, key="wi_hin")
        h_out = st.slider("Extra shots outside the box", -6, 6, 0, key="wi_hout")
    with a2:
        st.markdown(f"**{away}**")
        a_in = st.slider("Extra shots inside the box ", -6, 6, 0, key="wi_ain")
        a_out = st.slider("Extra shots outside the box ", -6, 6, 0, key="wi_aout")
    with a3:
        st.markdown("**Red card**")
        red_side = st.radio("Who is sent off", ["None", "Home", "Away"], horizontal=True, key="wi_red")
        red_min = st.slider("Minute of the red card", 1, 85, 30, key="wi_redmin", disabled=(red_side == "None"))
    sc = {"h_in": h_in, "h_out": h_out, "a_in": a_in, "a_out": a_out, "red_side": red_side, "red_min": red_min}

    lh, la = PR.expected_goals(S, league, home, away, neutral)
    lh2, la2 = W.apply_scenario(lh, la, sc, LEV, RED)
    M0, p0 = W.match_probs(lh, la)
    M1, p1 = W.match_probs(lh2, la2)
    changed = any([h_in, h_out, a_in, a_out]) or red_side != "None"

    nb = st.select_slider("Model refits for the uncertainty range", [50, 100, 150, 300], value=150, key="wi_nb") if hasattr(st, "select_slider") else 150
    base_b, scn_b = boot(league, home, away, neutral, tuple(sorted(sc.items())), nb)
    lo0, hi0 = np.percentile(base_b, [10, 90], axis=0)
    lo1, hi1 = np.percentile(scn_b, [10, 90], axis=0)
    d = scn_b - base_b
    dlo, dmed, dhi = np.percentile(d, [10, 50, 90], axis=0)

    top0, top1 = PR.top_scorelines(M0, 1)[0], PR.top_scorelines(M1, 1)[0]
    kpi_row([(f"{home} win", f"{p1[0] * 100:.0f}%", f"{(p1[0] - p0[0]) * 100:+.0f} pts vs baseline", "teal"),
             ("Draw", f"{p1[1] * 100:.0f}%", f"{(p1[1] - p0[1]) * 100:+.0f} pts vs baseline", "gold"),
             (f"{away} win", f"{p1[2] * 100:.0f}%", f"{(p1[2] - p0[2]) * 100:+.0f} pts vs baseline", "magenta"),
             ("Expected xG", f"{lh2:.2f} - {la2:.2f}", f"baseline {lh:.2f} - {la:.2f}", "blue"),
             ("Most likely score", f"{top1[0]}-{top1[1]}", f"baseline {top0[0]}-{top0[1]}", "red")], min_width=170)
    if not changed:
        st.info("No change applied yet. Move a slider or pick a red card to see the effect.")

    left, right = st.columns(2)
    with left:
        show(grouped_bars([f"{home} win", "Draw", f"{away} win"], [("Baseline", p0 * 100, "#8A97A6"), ("What-if", p1 * 100, "#1D5FAF")],
                          ylabel="Chance (%)", title="Result chances", fmt="{:.0f}", figsize=(6.2, 3.4)))
    with right:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        k = 6
        im = ax.imshow(M1[:k, :k] * 100, cmap="GnBu", origin="upper")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{M1[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xlabel(f"{away} goals", fontsize=9)
        ax.set_ylabel(f"{home} goals", fontsize=9)
        ax.set_title("Scoreline chances in the what-if (%)", fontsize=10, loc="left", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Chance (%)", fontsize=8)
        fig.tight_layout()
        show(fig)

    tbl = pd.DataFrame({
        "Outcome": [f"{home} win", "Draw", f"{away} win"],
        "Baseline %": p0 * 100, "Baseline 80% range": [f"{a:.0f}-{b:.0f}" for a, b in zip(lo0 * 100, hi0 * 100)],
        "What-if %": p1 * 100, "What-if 80% range": [f"{a:.0f}-{b:.0f}" for a, b in zip(lo1 * 100, hi1 * 100)],
        "Change (pts)": (p1 - p0) * 100, "Change 80% range": [f"{a:+.0f} to {b:+.0f}" for a, b in zip(dlo * 100, dhi * 100)]})
    st.dataframe(tbl.round(1), hide_index=True)
    st.caption(f"Ranges come from {len(base_b)} refits of the team-strength model on resampled matches of {league}"
               + (", and for the red card also from resampling the red-card matches" if red_side != "None" else "")
               + ". With about six matches per team the baseline itself is uncertain; the change is usually better pinned down than the level.")

# ---------------------------------------------------------------- group what-if
with tab2:
    st.markdown("Change the score of up to three played matches and see how the group would have finished. The right-hand tables replay every match from its shot-level xG and "
                "compare the finishing chances with and without your change.")
    g1, g2 = st.columns(2)
    league_g = g1.radio("League", sorted(D.m["league"].unique()), horizontal=True, key="wg_league")
    groups = sorted(D.m[D.m["league"] == league_g]["group_key"].unique())
    gk = g2.selectbox("Group", groups, key="wg_group")
    gm = D.m[D.m["group_key"] == gk].sort_values("date")
    labels = {int(r.match_id): f"{r.date:%d %b} | {r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}" for r in gm.itertuples()}
    over = {}
    for i in range(3):
        cc = st.columns([3, 1, 1])
        mid = cc[0].selectbox(f"Match to change {i + 1}", [0] + list(labels), format_func=lambda x: "(no change)" if x == 0 else labels[x], key=f"wg_m{i}")
        if mid:
            row = gm[gm["match_id"] == mid].iloc[0]
            hg = int(cc[1].number_input(f"{row['home_team']} goals", 0, 12, int(row["home_score"]), key=f"wg_h{i}_{mid}"))
            ag = int(cc[2].number_input(f"{row['away_team']} goals", 0, 12, int(row["away_score"]), key=f"wg_a{i}_{mid}"))
            if (hg, ag) != (int(row["home_score"]), int(row["away_score"])):
                over[int(mid)] = (hg, ag)

    t0 = W.group_table(D, gk)
    t1 = W.group_table(D, gk, over)
    cmp_ = t0[["Team", "Pos", "Pts", "GD"]].merge(t1[["Team", "Pos", "Pts", "GD", "GF", "GA", "W", "D", "L"]], on="Team", suffixes=(" before", " after"))
    cmp_["Moved"] = cmp_["Pos before"] - cmp_["Pos after"]
    cmp_ = cmp_.sort_values("Pos after")
    moved = cmp_[cmp_["Moved"] != 0]
    if not over:
        st.info("No match changed yet. Pick a match above and give it a different score.")
    else:
        kpi_row([("Matches changed", len(over)), ("Teams that move", len(moved), "in the final table"),
                 ("Winner", t1["Team"].iloc[0], "was " + t0["Team"].iloc[0] if t1["Team"].iloc[0] != t0["Team"].iloc[0] else "unchanged", "green"),
                 ("Last place", t1["Team"].iloc[-1], "was " + t0["Team"].iloc[-1] if t1["Team"].iloc[-1] != t0["Team"].iloc[-1] else "unchanged", "red")], min_width=190)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Real table**")
        st.dataframe(t0[["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]], hide_index=True)
    with c2:
        st.markdown("**Table with your changes**")
        st.dataframe(cmp_[["Pos after", "Team", "W", "D", "L", "GF", "GA", "GD after", "Pts after", "Moved"]].rename(
            columns={"Pos after": "Pos", "GD after": "GD", "Pts after": "Pts"}), hide_index=True)
    st.caption("Order: points, then goal difference, then goals scored (the same rule as the Match predictor). The real UEFA rule checks head-to-head first, so a tie on points may order differently.")

    base, wi = group_sim(gk, tuple(sorted(over.items())))
    pcols = [c_ for c_ in base.columns if c_.startswith("P") and c_[1:].isdigit()]
    ch = pd.DataFrame({"Team": base.index, "Win group % (xG replay)": base["P1"].values, "Win group % (with change)": wi["P1"].reindex(base.index).values,
                       "Expected pts (xG replay)": base["Expected points"].values, "Expected pts (with change)": wi["Expected points"].reindex(base.index).values})
    ch["Change in win chance (pts)"] = ch["Win group % (with change)"] - ch["Win group % (xG replay)"]
    ch = ch.sort_values("Win group % (with change)", ascending=False)
    if over:
        order = list(wi.sort_values("P1", ascending=False).index)
        show(stacked_share(order, {f"Finish {c_[1:]}": wi.loc[order, c_].values for c_ in pcols},
                           ["#1D5FAF", "#0E7C86", "#D99A00", "#B5326E", "#8A97A6"][:len(pcols)], title="Finishing chances with your change (% of 4,000 replays)"))
    st.dataframe(ch.round(1), hide_index=True)
    st.caption("xG replay: every match not changed is replayed 4,000 times from the scoreline chances of its own shots, and the changed matches are fixed at your scores. "
               "Both columns use the same random draws, so the difference comes only from your change.")

# ---------------------------------------------------------------- how it works
with tab3:
    st.markdown(f"""
**Baseline.** The ridge model of the Match predictor: per league, the xG each side creates is explained by a base level, a home edge, the team's attack and the opponent's defence.
Expected xG for each side drives two Poisson distributions, which give scorelines and win, draw and loss chances.

**Shots.** Each extra shot adds the average xG of a shot in its zone: {LEV['b_in']:.2f} inside the box and {LEV['b_out']:.2f} outside, measured on {LEV['n']:,} shots in this league phase.
The shot table has no zone flag, so the nearest {LEV['share_in'] * 100:.0f}% of shots (the share the team statistics call inside the box) count as inside. A real extra shot is usually a little worse than average, so
treat large changes as an upper bound.

**Red card.** From {RED['n']} matches with a red card up to minute 80, xG after the card was compared with the side's own season-average rate for the same minutes.
The sent-off side creates {RED['carded']:.2f} times its usual xG and the opponent {RED['opponent']:.2f} times. The 80% ranges ({RED['carded_range'][0]:.2f}-{RED['carded_range'][1]:.2f} and
{RED['opponent_range'][0]:.2f}-{RED['opponent_range'][1]:.2f}) are wide because the sample is small. These are averages: a team that is already losing may push forward regardless, which inflates the opponent's number.
The multipliers apply from the minute of the card to the end of the match; the first part of the match keeps the baseline xG.

**Uncertainty.** The matches of the chosen league are resampled with replacement, the strengths are refitted each time, and the scenario is applied to each refit (and, for the red card, to a resample of the red-card matches).
The 10th and 90th percentiles form the 80% range.

**Group what-if.** The table is recomputed from the played scores with your changes. The finishing chances replay every match from its shot-level xG scoreline grid with and without the change.

**Limits**
- About six matches per team: strengths are shrunk and uncertain. This is a what-if on a simple model, not a forecast.
- The levers describe averages across matches, not causes. Pressing more or defending deeper would also change other things the model cannot see.
- Shots are changed independently of game state, injuries, lineups and motivation.
- The group tab changes the score of a match as played; it does not change what happened inside the other matches.
""")
