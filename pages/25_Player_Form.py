import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts import bar_chart
from unl.charts2 import grouped_bars, line_chart
from unl.helpers import num
from unl.kpi import kpi_row
from unl.pitch import show
from unl.sample import add_flag, sample_note
from unl.style import AWAY, GREY, HOME

D = setup("Player form | Nations League 2026-27")
st.title("Player form")
st.caption("A player's rating, xG and xA match by match, set against the average for his position, plus the players whose recent form differs most from earlier matches.")

ps = D.ps.copy()
ps["mins"] = num(ps, "minutesplayed").fillna(0)
ps["rating_n"] = num(ps, "rating")
ps = ps[ps["mins"] > 0].sort_values("date")
for c in ("goals", "goalassist", "expectedgoals", "expectedassists"):
    ps[c] = num(ps, c).fillna(0)
pos_avg = ps.groupby("position")["rating_n"].mean()

tab1, tab2 = st.tabs(["Player", "Who is in form"])
with tab1:
    c = st.columns(2)
    team = c[0].selectbox("Team", D.teams, key="pf_team")
    roster = D.players[D.players["team"] == team].sort_values("minutesplayed", ascending=False)
    names = {int(r.player_id): f"{r.player} ({r.position})" for r in roster.itertuples()}
    pid = c[1].selectbox("Player", list(names), format_func=lambda k: names[k], key="pf_player")
    g = ps[ps["player_id"] == pid].copy()
    pos = g["position"].mode().iloc[0]
    r = g["rating_n"].dropna()
    se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
    kpi_row([("Matches", len(g), f"{int(g['mins'].sum())} minutes"), ("Average rating", f"{r.mean():.2f}" if len(r) else "n/a", f"position average {pos_avg[pos]:.2f}"),
             ("Best match", f"{r.max():.1f}" if len(r) else "n/a", None), ("Goals + assists", int(g["goals"].sum() + g["goalassist"].sum()), f"{g['goals'].sum():.0f} goals, {g['goalassist'].sum():.0f} assists"),
             ("xG + xA", f"{g['expectedgoals'].sum() + g['expectedassists'].sum():.2f}", "all matches")], min_width=170)
    lab = [f"{o[:3]} {d:%d %b}" for o, d in zip(g["opponent"], g["date"])]
    left, right = st.columns(2)
    with left:
        show(line_chart(lab, [("Match rating", g["rating_n"].values, HOME), (f"{pos} average", np.full(len(g), pos_avg[pos]), GREY)],
                        ylabel="SofaScore rating", xlabel="Match", title="Rating by match", zero=False, figsize=(6.0, 3.2)))
    with right:
        show(grouped_bars(lab, [("xG", g["expectedgoals"].values, HOME), ("xA", g["expectedassists"].values, AWAY)], ylabel="Expected per match",
                          xlabel="Match", title="Chance involvement", figsize=(6.0, 3.2), fmt="{:.2f}"))
    out = g[["date", "opponent", "mins", "rating_n", "goals", "goalassist", "expectedgoals", "expectedassists"]].copy()
    out["date"] = out["date"].dt.strftime("%d %b")
    out.columns = ["Date", "Opponent", "Minutes", "Rating", "Goals", "Assists", "xG", "xA"]
    st.dataframe(out.round(2), hide_index=True)
    if len(r) < 4:
        st.warning("Fewer than four rated matches: the line is too short to call a trend.")
    elif pd.notna(se):
        st.caption(f"Average rating {r.mean():.2f} with a standard error of {se:.2f}: the true level is likely within about {1.96 * se:.2f} either side.")

with tab2:
    c = st.columns(3)
    mm = c[0].number_input("Minimum rated matches", 4, 6, 4, key="pf_minm")
    pos_f = c[1].selectbox("Position", ["All", "G", "D", "M", "F"], key="pf_pos")
    k = c[2].slider("Recent matches counted", 2, 3, 2, key="pf_k")
    rows = []
    for pid_, g in ps.dropna(subset=["rating_n"]).groupby("player_id"):
        if len(g) < mm or len(g) <= k:
            continue
        rec, old = g["rating_n"].tail(k), g["rating_n"].iloc[:-k]
        rows.append({"player": g["player"].iloc[-1], "team": g["team"].iloc[-1], "position": g["position"].mode().iloc[0], "Matches": len(g),
                     "Earlier rating": old.mean(), "Recent rating": rec.mean(), "Change": rec.mean() - old.mean(), "Minutes": g["mins"].sum()})
    f = pd.DataFrame(rows)
    if pos_f != "All":
        f = f[f["position"] == pos_f]
    if f.empty:
        st.info("No players match.")
    else:
        f = add_flag(f, "Minutes", 270)
        up, dn = f.sort_values("Change", ascending=False).head(10), f.sort_values("Change").head(10)
        c1, c2 = st.columns(2)
        with c1:
            show(bar_chart([f"{r.player} ({r.team})" for r in up.itertuples()], up["Change"].values, "Rising: recent minus earlier rating", fmt="{:+.2f}", color=HOME, xlabel="Rating change"))
        with c2:
            show(bar_chart([f"{r.player} ({r.team})" for r in dn.itertuples()], dn["Change"].values, "Falling: recent minus earlier rating", fmt="{:+.2f}", color=AWAY, xlabel="Rating change"))
        st.dataframe(f.sort_values("Change", ascending=False).round(2), hide_index=True, height=360)
        sample_note(f, what="players")
        st.caption("Changes over two or three matches are mostly noise; look for players who are strong in both the rating change and the underlying xG and xA.")
