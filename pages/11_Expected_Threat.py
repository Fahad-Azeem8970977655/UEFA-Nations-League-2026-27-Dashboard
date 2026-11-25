import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import bar_chart
from unl.pitch import new_figure, show
from unl.style import HOME, INK
from unl.viz3 import draw_xt_origin, draw_xt_surface

D = setup("Expected threat | Nations League 2026-27")

st.title("Expected threat (xT)")
st.caption("Every zone of the pitch is given a value: the chance that possession there ends in a goal. "
           "A completed pass or carry earns the difference in value between where it ended and where it started.")

XT = D.xt
_t = XT.team
_pl = XT.player.join(D.players.set_index("player_id")[["player", "team"]]).dropna(subset=["player"])
_pl = _pl[_pl["minutes"] >= 180].sort_values("xt_per90", ascending=False)
kpi_row([("Moves valued", f"{len(XT.moves):,}", "passes and carries"), ("Peak zone value", f"{XT.grid.max():.2f}", "goals per possession"),
         ("Top team xT / match", _t.index[0], f"{_t['xT per match'].iloc[0]:.2f}"), ("Top player xT / 90", _pl.iloc[0]["player"], f"{_pl.iloc[0]['xt_per90']:.2f} ({_pl.iloc[0]['team']})")], min_width=200)
tabs = st.tabs(["The model", "Teams", "Players"])

with tabs[0]:
    fig, axes = new_figure(1, w=8.0)
    draw_xt_surface(axes[0], XT.grid)
    axes[0].set_title("xT surface (darker = more dangerous), attacking to the right", fontsize=10, color=INK)
    show(fig)
    top = XT.grid.max()
    st.caption(f"Fitted on this dataset: {len(XT.moves):,} passes and carries and {len(D.shots):,} shots on a 16 x 12 grid. "
               f"A possession in the most dangerous zone is worth about {top:.2f} goals, one on the own byline about "
               f"{XT.grid.min():.3f}. Value comes from the share of actions that are shots, the xG of those shots and "
               "where passes and carries tend to go next, solved iteratively. Penalties are excluded. "
               "Failed passes add nothing; every carry is counted as completed.")

with tabs[1]:
    T = XT.team
    team = st.selectbox("Team", D.teams, key="xt_team")
    r = T.loc[team]
    c = st.columns(4)
    c[0].metric("xT per match", f"{r['xT per match']:.2f}", help="Total xT gained by completed passes and carries per match.")
    c[1].metric("Rank", f"{int(T['xT per match'].rank(ascending=False)[team])} of {len(T)}")
    c[2].metric("From passing", f"{r['Passing xT'] / (r['Total xT'] or 1) * 100:.0f}%")
    c[3].metric("From carrying", f"{r['Carrying xT'] / (r['Total xT'] or 1) * 100:.0f}%")
    mv = XT.moves[XT.moves["team"] == team]
    left, right = st.columns([1.4, 1])
    with left:
        fig, axes = new_figure(1, w=7.0)
        draw_xt_origin(axes[0], mv)
        show(fig)
        st.caption("Where the threat starts: xT gained by completed passes and carries, grouped by where they began. Team attacks to the right.")
    with right:
        pl = XT.player.join(D.players.set_index("player_id")[["player", "position", "team"]])
        pl = pl[pl["team"] == team].sort_values("xt_total", ascending=False).head(10)
        out = pd.DataFrame({"Player": pl["player"], "Pos": pl["position"], "Min": pl["minutes"].fillna(0).astype(int),
                            "Pass xT": pl["xt_pass"].round(2), "Carry xT": pl["xt_carry"].round(2),
                            "Total": pl["xt_total"].round(2), "Per 90": pl["xt_per90"].round(2)})
        st.write("Top contributors")
        st.dataframe(out, hide_index=True)
    st.write("League ranking")
    show_n = st.slider("Teams shown", 10, len(T), 20, key="xt_n")
    R = T["xT per match"].head(show_n)
    show(bar_chart(list(R.index), R.values, "xT per match", fmt="{:.2f}"))

with tabs[2]:
    c = st.columns(4)
    pos = c[0].selectbox("Position", ["All", "Defenders", "Midfielders", "Forwards"], key="xt_pos")
    mm = c[1].number_input("Minimum minutes", 0, 1000, 270, step=30, key="xt_min")
    by = c[2].selectbox("Rank by", ["xT per 90", "Total xT", "Passing xT", "Carrying xT"], key="xt_by")
    team_f = c[3].selectbox("Team", ["All teams"] + D.teams, key="xt_pl_team")
    pl = XT.player.join(D.players.set_index("player_id")[["player", "position", "team"]])
    pl = pl[pl["minutes"].fillna(0) >= mm]
    if pos != "All":
        pl = pl[pl["position"] == {"Defenders": "D", "Midfielders": "M", "Forwards": "F"}[pos]]
    if team_f != "All teams":
        pl = pl[pl["team"] == team_f]
    col = {"xT per 90": "xt_per90", "Total xT": "xt_total", "Passing xT": "xt_pass", "Carrying xT": "xt_carry"}[by]
    pl = pl.sort_values(col, ascending=False).head(25)
    if pl.empty:
        st.info("No players match these filters.")
    else:
        show(bar_chart((pl["player"] + " (" + pl["team"] + ")").tolist(), pl[col].values, by, fmt="{:.2f}"))
        out = pd.DataFrame({"Player": pl["player"], "Team": pl["team"], "Pos": pl["position"],
                            "Min": pl["minutes"].astype(int), "Pass xT": pl["xt_pass"].round(2),
                            "Carry xT": pl["xt_carry"].round(2), "Total xT": pl["xt_total"].round(2),
                            "xT per 90": pl["xt_per90"].round(2)})
        st.dataframe(out, hide_index=True)
        st.caption("xT rewards moving the ball into dangerous zones, so it favours creative midfielders and wide forwards; "
                   "it says nothing about finishing or defending. Goalkeepers are left out of the position filters.")
