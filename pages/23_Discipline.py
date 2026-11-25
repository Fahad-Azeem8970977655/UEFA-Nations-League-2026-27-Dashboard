import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.charts import bar_chart
from unl import ext
from unl.charts2 import grouped_bars, scatter_trend, stacked_share
from unl.events import BLOCKS, block_of
from unl.kpi import kpi_row
from unl.pitch import show
from unl.sample import add_flag, sample_note

D = setup("Discipline | Nations League 2026-27")
st.title("Discipline, penalties and VAR")
st.caption("Cards with their reasons and minutes, penalties won and conceded, VAR decisions, and how each referee handled their matches.")

inc = D.inc.copy()
cards = inc[inc["type"] == "card"].copy()
cards["minute"] = pd.to_numeric(cards["minute"], errors="coerce").clip(lower=1)
n_m = D.tm.groupby("team").size()
pens = inc[(inc["type"] == "goal") & (inc["class"] == "penalty")].assign(result="Scored")
missed = inc[(inc["type"] == "inGamePenalty")].assign(result="Missed")
pen = pd.concat([pens, missed])[["match_id", "team", "player", "minute", "result"]]
mm = D.m.set_index("match_id")
pen["against"] = [mm.loc[m, "away_team"] if t == mm.loc[m, "home_team"] else mm.loc[m, "home_team"] for m, t in zip(pen["match_id"], pen["team"])]
var = inc[inc["type"] == "varDecision"]
reds = cards["class"].isin(["red", "yellowRed"]).sum()
kpi_row([("Cards", len(cards), f"{len(cards) / D.m['match_id'].nunique():.2f} per match"), ("Yellow cards", int((cards["class"] == "yellow").sum())),
         ("Red cards", int(reds), "straight red or second yellow"), ("Penalties awarded", len(pen), f"{(pen['result'] == 'Scored').sum()} scored"),
         ("VAR decisions", len(var), "logged in the data")], min_width=160)

tab1, tab2, tab3, tab4 = st.tabs(["Cards", "Penalties", "VAR", "Referees"])
with tab1:
    g = cards.groupby("team").agg(Cards=("class", "size"), Yellow=("class", lambda s: (s == "yellow").sum()),
                                  Red=("class", lambda s: s.isin(["red", "yellowRed"]).sum()))
    g["Matches"] = n_m.reindex(g.index)
    g["Cards per match"] = g["Cards"] / g["Matches"]
    g = g.sort_values("Cards per match", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        t = g.head(15)
        show(bar_chart(list(t.index), t["Cards per match"].values, "Cards per match, top 15 teams", fmt="{:.2f}", xlabel="Cards per match"))
    with c2:
        r = cards["reason"].fillna("Other").value_counts().head(8)
        show(bar_chart(list(r.index), r.values, "Reasons for cards", fmt="{:.0f}", xlabel="Number of cards"))
    cards["block"] = block_of(cards["minute"], pd.Series(np.nan, index=cards.index))
    b = cards.groupby("block").size().reindex(range(6), fill_value=0)
    show(grouped_bars(BLOCKS, [("Cards", b.values, "#D99A00")], ylabel="Cards", xlabel="Match minutes", title="When cards are shown", fmt="{:.0f}", figsize=(7, 3)))
    st.dataframe(add_flag(g.reset_index().rename(columns={"team": "Team"}), "Matches", 5).round(2), hide_index=True, height=360)
    pc = cards.groupby(["player", "team"]).size().sort_values(ascending=False).head(15).reset_index(name="Cards")
    st.subheader("Most carded players")
    st.dataframe(pc, hide_index=True)
with tab2:
    st.dataframe(pen.sort_values("minute")[["team", "player", "against", "minute", "result"]].rename(columns={"team": "Team", "player": "Taker", "against": "Against", "minute": "Minute", "result": "Result"}), hide_index=True)
    w = pen.groupby("team").size().rename("Won")
    cn = pen.groupby("against").size().rename("Conceded")
    pt = pd.concat([w, cn], axis=1).fillna(0).astype(int).sort_values("Won", ascending=False)
    st.dataframe(pt)
    st.caption(f"Penalty conversion: {(pen['result'] == 'Scored').mean() * 100:.0f}% ({(pen['result'] == 'Scored').sum()} of {len(pen)}).")
with tab3:
    v = var.groupby("class").size().sort_values(ascending=False)
    show(bar_chart(list(v.index), v.values, "VAR decisions by type", fmt="{:.0f}", xlabel="Number of decisions"))
    st.dataframe(var[["match_id", "minute", "class", "team", "player"]].rename(columns={"class": "Decision"}), hide_index=True)
with tab4:
    if D.refs is None:
        st.info("The referee view needs unl_2026_27_extra.xlsx next to Home.py (sheet 'match_info'). "
                + (f"Loading it failed with: {D.ext_error}" if getattr(D, "ext_error", None) else "The file was not found."))
    else:
        R = D.refs
        RT = ext.referee_table(R)
        hs = RT["Home cards"].sum() / RT["Cards"].sum() * 100
        top = RT[RT["Matches"] >= 2].sort_values("Cards per match", ascending=False)
        kpi_row([("Referees used", len(RT), f"{int(RT['Matches'].max())} matches at most", "blue"),
                 ("Cards per match", f"{R['cards'].mean():.2f}", f"{R['yellow'].mean():.2f} yellow, {R['red'].mean():.2f} red", "gold"),
                 ("Home share of cards", f"{hs:.0f}%", "of all cards went to the home side", "teal"),
                 ("Stoppage time", f"{R['added_time'].mean():.1f} min", f"{R['injury_time_1'].mean():.1f} first half, {R['injury_time_2'].mean():.1f} second half", "slate"),
                 ("Strictest (2+ matches)", top.index[0] if len(top) else "n/a", f"{top['Cards per match'].iloc[0]:.1f} cards per match" if len(top) else None, "red")], min_width=190)
        mx = int(RT["Matches"].max())
        min_m = st.slider("Minimum matches refereed", 1, mx, 2 if mx >= 2 else 1, key="rf_min",
                          help="Most referees took only one or two league-phase matches, so rankings with a low minimum are mostly noise.")
        F = RT[RT["Matches"] >= min_m]
        if F.empty:
            st.info("Nobody refereed that many matches.")
        else:
            c1, c2 = st.columns(2)
            a = F.sort_values("Cards per match", ascending=False).head(12)
            b = F.sort_values("Added time per match", ascending=False).head(12)
            with c1:
                show(bar_chart(list(a.index), a["Cards per match"].values, "Cards per match", fmt="{:.1f}", xlabel="Cards per match"))
            with c2:
                show(bar_chart(list(b.index), b["Added time per match"].values, "Stoppage time added per match", fmt="{:.1f}", xlabel="Minutes"))
            cols = ["Country", "Matches", "Cards", "Yellow", "Red", "Penalties", "Home cards", "Away cards", "Home share of cards %", "Cards per match",
                    "Yellows per match", "Career yellows per game", "Vs career (yellows)", "Fouls per match", "Penalties per match", "Added time per match"]
            out = add_flag(F[cols].reset_index().rename(columns={"referee": "Referee"}), "Matches", 3).sort_values("Cards per match", ascending=False)
            st.dataframe(out.round(2), hide_index=True, height=420)
            sample_note(out, what="referees")
            st.caption("Career figures are the referee's yellow cards per game over a whole career (all competitions, as listed by SofaScore). "
                       "'Vs career' is this tournament's yellows per match minus the career rate: positive = stricter than usual in these matches. Home share of cards is 50% when both sides are treated alike.")
        st.subheader("Do more fouls mean more cards?")
        show(scatter_trend(R["fouls"], R["cards"], R["home_team"] + " v " + R["away_team"], "Fouls in the match (both teams)", "Cards in the match", label_n=6))
        r = np.corrcoef(R["fouls"], R["cards"])[0, 1]
        st.caption(f"Correlation across {len(R)} matches: {r:.2f}. A match above the line has more cards than its foul count would suggest.")
        st.subheader("One referee's matches")
        ref = st.selectbox("Referee", sorted(RT.index), key="rf_pick")
        one = R[R["referee"] == ref].copy()
        one["Match"] = one["home_team"] + " " + one["home_score"].astype(int).astype(str) + "-" + one["away_score"].astype(int).astype(str) + " " + one["away_team"]
        one["Date"] = one["date"].dt.strftime("%d %b")
        st.dataframe(one[["Date", "Match", "cards", "yellow", "red", "home_cards", "away_cards", "penalties", "fouls", "added_time"]].rename(
            columns={"cards": "Cards", "yellow": "Yellow", "red": "Red", "home_cards": "Home cards", "away_cards": "Away cards", "penalties": "Penalties",
                     "fouls": "Fouls", "added_time": "Added time"}), hide_index=True)
