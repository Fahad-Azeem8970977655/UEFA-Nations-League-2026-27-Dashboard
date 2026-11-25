import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts2 import avg_positions, forest, line_chart, scatter_trend
from unl.kpi import fmt, kpi_row
from unl.pitch import new_figure, show
from unl.shape import line_effect, team_shape_table
from unl.style import AWAY, HOME

D = setup("Defensive shape | Nations League 2026-27")
st.title("Defensive shape")
st.caption("Where a team's outfield players stand on average: how high the back line is, how stretched the team is between "
           "back and front, how wide, and how much space it covers. Then: does a higher line concede better or worse chances?")

S = D.shape
T = team_shape_table(S)
eff_w = line_effect(S, "xga", within=True)
eff_all = line_effect(S, "xga")
hi, lo = T["line_height"].idxmax(), T["line_height"].idxmin()
kpi_row([("League average back line", f"{S['line_height'].mean():.1f}", "0 = own goal, 100 = opponent goal", "blue"),
         ("Highest line", hi, f"{T.loc[hi, 'line_height']:.1f}", "red"),
         ("Deepest line", lo, f"{T.loc[lo, 'line_height']:.1f}", "green"),
         ("xGA per +10 line height", f"{eff_w['effect']:+.2f}", f"within teams, 95% range {eff_w['lo']:+.2f} to {eff_w['hi']:+.2f}", "gold"),
         ("Correlation line vs xGA", f"{eff_all['r']:+.2f}", f"{eff_all['n']} team-matches", "teal")], min_width=190)

tab1, tab2, tab3 = st.tabs(["Team shape", "League table", "Does a high line pay off?"])

with tab1:
    team = st.selectbox("Team", D.teams, key="shape_team")
    s = S[S["team"] == team].sort_values("date")
    a = D.avg[(D.avg["team"] == team) & D.avg["match_id"].isin(s["match_id"])]
    gk_ids = set(D.ps.loc[D.ps["position"] == "G", "player_id"])
    a = a[~a["player_id"].isin(gk_ids)]
    g = a.groupby("player_id").apply(lambda t: pd.Series({
        "average_x": np.average(t["average_x"], weights=t["points_count"]),
        "average_y": np.average(t["average_y"], weights=t["points_count"]),
        "points": t["points_count"].sum(), "name": t["player_name"].iloc[0]}), include_groups=False)
    g = g.sort_values("points", ascending=False).head(10)
    g["name_short"] = g["name"].str.split().str[-1]
    r = T.loc[team]
    kpi_row([("Back line height", f"{r['line_height']:.1f}", f"rank {int(T['line_height'].rank(ascending=False)[team])} of {len(T)}", "blue"),
             ("Team length", f"{r['length']:.1f}", "front three minus back four (lower = tighter)", "red"),
             ("Team width", f"{r['width']:.1f}", "spread of average y", "green"),
             ("Area covered", f"{r['area']:.1f}%", "of the pitch, outfield hull", "gold"),
             ("xGA per match", f"{r['xGA']:.2f}", f"rank {int(T['xGA'].rank()[team])} (1 = fewest)", "teal"),
             ("Line height spread", f"{r['line_sd']:.1f}", "match to match (std)", "magenta")], min_width=170)
    left, right = st.columns([1.2, 1])
    with left:
        fig, axes = new_figure(1, w=7.0)
        avg_positions(axes[0], g, line_height=r["line_height"], color=HOME)
        axes[0].set_title(f"{team}: average outfield positions, attacking right", fontsize=10)
        show(fig)
    with right:
        lab = [f"{o[:3]} {d:%d %b}" for o, d in zip(s["opponent"], s["date"])]
        fig = line_chart(lab, [("Back line height", s["line_height"].values, HOME)], ylabel="Back line height", zero=False, figsize=(5.4, 2.6))
        ax2 = fig.axes[0].twinx()
        ax2.plot(lab, s["xga"].values, color=AWAY, lw=2, marker="o", ms=4)
        ax2.set_ylabel("xG against", color=AWAY, fontsize=9)
        ax2.spines["top"].set_visible(False)
        fig.axes[0].tick_params(axis="x", labelrotation=60)
        show(fig)
        st.caption("Teal: back line height each match. Magenta: xG conceded.")
    out = s[["date", "opponent", "line_height", "length", "width", "area", "xga", "opp_total_shots", "result"]].copy()
    out["date"] = out["date"].dt.strftime("%d %b")
    out.columns = ["Date", "Opponent", "Back line", "Length", "Width", "Area %", "xGA", "Shots against", "Result"]
    st.dataframe(out.round(2), hide_index=True)

with tab2:
    t = T.reset_index().rename(columns={"team": "Team", "line_height": "Back line", "length": "Length", "width": "Width", "area": "Area %",
                                         "line_sd": "Line SD"})
    st.dataframe(t[["Team", "Back line", "Length", "Width", "Area %", "Line SD", "xGA", "Shots against", "xGA per shot", "Matches"]]
                 .sort_values("Back line", ascending=False).round(2), hide_index=True, height=520)
    hl = st.selectbox("Highlight a team", [""] + list(T.index), key="shape_hl")
    c1, c2 = st.columns(2)
    with c1:
        show(scatter_trend(T["line_height"], T["xGA"], T.index, "Average back line height", "xG against per match", highlight=hl or None,
                           title="Teams: back line against xGA"))
    with c2:
        show(scatter_trend(T["length"], T["xGA"], T.index, "Team length (front minus back)", "xG against per match", highlight=hl or None,
                           title="Teams: compactness against xGA"))

with tab3:
    st.markdown("Each bar is the change in the outcome for a back line **10 points higher** (about 10 metres), with a 95% bootstrap range "
                "over matches. A range that stays on one side of zero is a fairly reliable direction.")
    outcomes = [("xga", "xG against"), ("opp_total_shots", "Shots against"), ("xga_per_shot", "xGA per shot (chance quality)"),
                ("opp_big_chances", "Big chances against")]
    specs = [("All team-matches", {}), ("Within the same team", {"within": True}), ("Within team, holding possession fixed", {"within": True, "control_possession": True})]
    cols = st.columns(len(outcomes))
    for col, (ycol, lab) in zip(cols, outcomes):
        rows = []
        for name, kw in specs:
            e = line_effect(S, ycol, n_boot=300, **kw)
            if e:
                rows.append((name, e["effect"], e["lo"], e["hi"]))
        with col:
            if rows:
                show(forest([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                            title=lab, xlabel="change per +10 line height"))
    q = S.dropna(subset=["line_height", "xga"]).copy()
    q["Line"] = pd.qcut(q["line_height"], 3, labels=["Deep third", "Middle third", "High third"])
    tr = q.groupby("Line", observed=True).agg(Matches=("xga", "size"), **{"Line height": ("line_height", "mean"), "xGA": ("xga", "mean"),
          "Shots against": ("opp_total_shots", "mean"), "xGA per shot": ("xga_per_shot", "mean"), "Possession %": ("ball_possession", "mean"),
          "Goals against": ("ga", "mean"), "Points per match": ("points", "mean")})
    st.dataframe(tr.round(2))
    st.caption("Match-by-match groups by how high the back line stood. Higher lines go with fewer chances against, but "
               "stronger teams also play higher and keep the ball more, so the second and third rows of bars (within the same team, "
               "possession held fixed) are the fairer test. Average positions blend attacking and defending phases, "
               "so this describes overall shape, not only the block without the ball.")
