import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import compare_bars, momentum_chart, xg_race
from unl.helpers import num
from unl.pitch import (PASS_FILTERS, apply_pass_filter, draw_defensive, draw_passes, draw_pitch, draw_shots,
                       finish_pitch, new_figure, shot_xy, show)
from unl.style import AWAY, GOLD, HOME, INK

D = setup("Matches | Nations League 2026-27")

def label_of(row):
    return f"{row.home_team} {int(row.home_score)}-{int(row.away_score)} {row.away_team} ({row.date:%d %b})"


def fmt_minute(minute, added):
    if pd.isna(minute):
        return ""
    if pd.notna(added) and float(added) > 0:
        return f"{int(minute)}+{int(added)}'"
    return f"{int(minute)}'"


def event_rows(inc, home_team, away_team):
    rows = []
    for r in inc.sort_values(["minute", "added_time"], na_position="last").to_dict("records"):
        when = fmt_minute(r.get("minute"), r.get("added_time"))
        if r["type"] == "goal":
            kind = {"penalty": "Penalty goal", "ownGoal": "Own goal"}.get(r.get("class"), "Goal")
            who = str(r.get("player"))
            if pd.notna(r.get("assist1")):
                who += f" (assist: {r['assist1']})"
            rows.append((when, r.get("team"), kind, who))
        elif r["type"] == "card":
            kind = {"yellow": "Yellow card", "red": "Red card", "yellowRed": "Second yellow"}.get(r.get("class"), "Card")
            rows.append((when, r.get("team"), kind, r.get("player")))
        elif r["type"] == "substitution":
            rows.append((when, r.get("team"), "Substitution", f"{r.get('player_in')} on, {r.get('player_out')} off"))
    return pd.DataFrame(rows, columns=["Minute", "Team", "Event", "Detail"])


MATCH_STATS = [("Possession %", "ball_possession", 0), ("xG", "expected_goals", 2), ("Shots", "total_shots", 0),
               ("Shots on target", "shots_on_target", 0), ("Big chances", "big_chances", 0), ("Passes", "passes", 0),
               ("Pass accuracy %", "pass_acc", 0), ("Final third entries", "final_third_entries", 0),
               ("Touches in box", "touches_in_penalty_area", 0), ("Tackles", "tackles", 0),
               ("Interceptions", "interceptions", 0), ("Clearances", "clearances", 0),
               ("Recoveries", "recoveries", 0), ("Fouls", "fouls", 0), ("Corners", "corner_kicks", 0)]


st.title("Matches")
c1, c2, c3 = st.columns([1, 1, 2])
league = c1.selectbox("League", sorted(D.m["league"].unique()), key="match_league")
groups = sorted(D.m[D.m["league"] == league]["group_key"].unique())
group = c2.selectbox("Group", groups, key="match_group")
sub = D.m[D.m["group_key"] == group].sort_values("date")
labels = {r.match_id: label_of(r) for r in sub.itertuples()}
mid = c3.selectbox("Match", list(labels), format_func=lambda k: labels[k], key="match_pick")
r = D.m[D.m["match_id"] == mid].iloc[0]
home_t, away_t = r["home_team"], r["away_team"]

st.markdown(
    f"""<div class="scoreline"><div class="side home">{html.escape(home_t)}</div>
<div class="score">{int(r['home_score'])} - {int(r['away_score'])}</div>
<div class="side away">{html.escape(away_t)}</div></div>
<div class="matchmeta">Matchday {int(r['matchday'])}, {html.escape(r['group_key'])}, {r['date']:%d %b %Y}</div>""",
    unsafe_allow_html=True)

_h = D.tm[(D.tm["match_id"] == mid) & (D.tm["side"] == "home")].iloc[0]
_a = D.tm[(D.tm["match_id"] == mid) & (D.tm["side"] == "away")].iloc[0]
_x = D.xpts_match[D.xpts_match["match_id"] == mid].iloc[0]
kpi_row([("xG", f"{_h['expected_goals']:.2f} - {_a['expected_goals']:.2f}", "home - away"),
         ("Shots", f"{int(_h['total_shots'])} - {int(_a['total_shots'])}", f"on target {int(_h['shots_on_target'])} - {int(_a['shots_on_target'])}"),
         ("Possession", f"{_h['ball_possession']:.0f}% - {_a['ball_possession']:.0f}%"),
         ("Expected points", f"{_x['xpts_home']:.1f} - {_x['xpts_away']:.1f}", "from shot-level xG"),
         ("Home win chance", f"{_x['p_home'] * 100:.0f}%", f"draw {_x['p_draw'] * 100:.0f}%, away {_x['p_away'] * 100:.0f}%")], min_width=170)

tabs = st.tabs(["Summary", "Shots", "Passing and defending", "Positions", "Lineups", "xG race"])
tm_m = D.tm[D.tm["match_id"] == mid]
inc = D.inc[D.inc["match_id"] == mid]

with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        h = tm_m[tm_m["side"] == "home"].iloc[0]
        a = tm_m[tm_m["side"] == "away"].iloc[0]
        rows = []
        for label, col, dec in MATCH_STATS:
            hv, av = h[col], a[col]
            if pd.notna(hv) and pd.notna(av):
                rows.append((label, float(hv), float(av), dec))
        if rows:
            show(compare_bars(rows, names=(home_t, away_t)))
        f_h, f_a = D.lineups[(D.lineups["match_id"] == mid) & (D.lineups["side"] == "home")], \
            D.lineups[(D.lineups["match_id"] == mid) & (D.lineups["side"] == "away")]
        if len(f_h) and len(f_a):
            st.caption(f"Formations: {home_t} {f_h.iloc[0]['formation']}, {away_t} {f_a.iloc[0]['formation']}")
    with right:
        ev = event_rows(inc, home_t, away_t)
        st.dataframe(ev, hide_index=True)
    mom = D.mom[D.mom["match_id"] == mid]
    if len(mom):
        st.subheader("Momentum")
        goals = inc[inc["type"] == "goal"]
        show(momentum_chart(mom, goals, home_t, away_t))
        st.caption("Positive (teal) = home pressure, negative (magenta) = away pressure. Stars mark goals.")

with tabs[1]:
    shots = D.shots[D.shots["match_id"] == mid]
    fig, axes = new_figure(1, w=9.0)
    ax = axes[0]
    draw_shots(ax, shots[shots["team"] == home_t], HOME, attack_right=True)
    # second team on the same pitch attacking left
    draw_pitch_overlay = shots[shots["team"] == away_t]
    if len(draw_pitch_overlay):
        px, py = shot_xy(draw_pitch_overlay, attack_right=False)
        xg = pd.to_numeric(draw_pitch_overlay["xg"], errors="coerce").fillna(0.03)
        goal = (draw_pitch_overlay["shot_type"] == "goal").values
        size = 45 + xg.values * 900
        ax.scatter(px[~goal], py[~goal], s=size[~goal], facecolor=AWAY, edgecolor=AWAY, alpha=0.35, lw=1.2, zorder=3)
        ax.scatter(px[goal], py[goal], s=130 + xg.values[goal] * 380, marker="*", facecolor=GOLD, edgecolor=INK, lw=0.8, zorder=4)
    ax.set_title(f"{home_t} (attacking right) and {away_t} (attacking left)", fontsize=10, color=INK)
    show(fig)
    st.caption("Star = goal. Circle size = xG.")
    t = shots.sort_values(["minute"])[["minute", "team", "player", "shot_type", "xg", "xgot", "situation", "body_part"]].copy()
    t["xg"] = t["xg"].round(2)
    t["xgot"] = pd.to_numeric(t["xgot"], errors="coerce").round(2)
    t.columns = ["Minute", "Team", "Player", "Outcome", "xG", "xGOT", "Situation", "Body part"]
    st.dataframe(t, hide_index=True)
    og = D.own_goals[D.own_goals["match_id"] == mid]
    if len(og):
        st.caption("Own goal (not a shot, left out above): " + ", ".join(
            f"{r.player} {int(r.minute)}' (counted for {r.team})" for r in og.itertuples()))

with tabs[2]:
    who = st.radio("Team", [home_t, away_t], horizontal=True, key="match_team_radio")
    players = D.ps[(D.ps["match_id"] == mid) & (D.ps["team"] == who) &
                   (num(D.ps, "minutesplayed").fillna(0) > 0)].sort_values("minutesplayed", ascending=False)
    pmap = {-1: "Whole team"}
    pmap.update({int(p.player_id): p.player for p in players.itertuples()})
    pid = st.selectbox("Player", list(pmap), format_func=lambda k: pmap[k], key="match_player_pick")
    color = HOME if who == home_t else AWAY
    ps_ = D.passes[(D.passes["match_id"] == mid) & (D.passes["team"] == who)]
    df_ = D.defensive[(D.defensive["match_id"] == mid) & (D.defensive["team"] == who)]
    if pid != -1:
        ps_, df_ = ps_[ps_["player_id"] == pid], df_[df_["player_id"] == pid]
    kind = st.selectbox("Pass type", PASS_FILTERS, key="match_pass_kind")
    ps_f = apply_pass_filter(ps_, kind, False)
    fig, axes = new_figure(2)
    draw_passes(axes[0], ps_f, color)
    axes[0].set_title(f"{kind}: {len(ps_f)} of {len(ps_)}", fontsize=10, color=INK)
    draw_defensive(axes[1], df_)
    axes[1].set_title(f"Defensive actions: {len(df_)}", fontsize=10, color=INK)
    show(fig)
    st.caption("The team attacks to the right. Grey arrows are incomplete passes.")

with tabs[3]:
    a = D.avg[D.avg["match_id"] == mid].copy()
    if a.empty:
        st.info("No average-position data for this match.")
    else:
        nums = D.ps[D.ps["match_id"] == mid][["player_id", "shirt_number", "substitute"]]
        a = a.merge(nums, on="player_id", how="left")
        with_subs = st.checkbox("Include substitutes", value=False, key="avg_subs")
        if not with_subs:
            a = a[~a["substitute"].fillna(False).astype(bool)]
        fig, axes = new_figure(1, w=9.0)
        ax = axes[0]
        draw_pitch(ax)
        for side, color in (("home", HOME), ("away", AWAY)):
            s = a[a["side"] == side]
            px = s["average_x"] if side == "home" else 100 - s["average_x"]
            py = s["average_y"] if side == "home" else 100 - s["average_y"]
            sizes = 160 + s["points_count"].astype(float).clip(upper=120) * 4
            ax.scatter(px, py, s=sizes, facecolor=color, edgecolor="white", lw=1.2, alpha=0.9, zorder=3)
            for xi, yi, n in zip(px, py, s["shirt_number"]):
                if pd.notna(n):
                    ax.text(xi, yi, str(int(n)), ha="center", va="center", color="white", fontsize=8,
                            fontweight="bold", zorder=4)
        finish_pitch(ax)
        ax.set_title(f"{home_t} (left to right) and {away_t} (right to left)", fontsize=10, color=INK)
        show(fig)
        st.caption("Average position of each player's touches, numbered by shirt. Circle size = number of touches. "
                   "Substitutes can be added with the checkbox.")

with tabs[4]:
    for side, team in (("home", home_t), ("away", away_t)):
        st.subheader(team)
        p = D.ps[(D.ps["match_id"] == mid) & (D.ps["team"] == team)].copy()
        p["minutesplayed"] = num(p, "minutesplayed")
        p = p.sort_values(["substitute", "shirt_number"])
        cols = ["shirt_number", "player", "position", "minutesplayed", "rating", "goals", "goalassist",
                "expectedgoals", "totalshots", "keypass", "totalpass", "accuratepass", "totaltackle",
                "interceptionwon", "totalclearance"]
        cols = [c for c in cols if c in p.columns]
        t = p[cols].copy()
        for c in ["goals", "goalassist", "totalshots", "keypass", "totalpass", "accuratepass", "totaltackle",
                  "interceptionwon", "totalclearance"]:
            if c in t:
                t[c] = t[c].fillna(0).astype(int)
        if "expectedgoals" in t:
            t["expectedgoals"] = t["expectedgoals"].round(2)
        t.columns = [{"shirt_number": "#", "player": "Player", "position": "Pos", "minutesplayed": "Min",
                      "rating": "Rating", "goals": "Goals", "goalassist": "Assists", "expectedgoals": "xG",
                      "totalshots": "Shots", "keypass": "Key passes", "totalpass": "Passes",
                      "accuratepass": "Accurate", "totaltackle": "Tackles", "interceptionwon": "Interceptions",
                      "totalclearance": "Clearances"}[c] for c in cols]
        st.dataframe(t, hide_index=True)
    st.caption("Starters first, then substitutes. Players who did not come on have no minutes or stats.")

with tabs[5]:
    xm = D.xpts_match[D.xpts_match["match_id"] == mid].iloc[0]
    hs_, as_ = int(r["home_score"]), int(r["away_score"])
    actual = "home win" if hs_ > as_ else "away win" if hs_ < as_ else "draw"
    probs = {"home win": xm["p_home"], "draw": xm["p_draw"], "away win": xm["p_away"]}
    likely = max(probs, key=probs.get)
    c = st.columns(5)
    c[0].metric(f"{home_t} win", f"{xm['p_home'] * 100:.0f}%")
    c[1].metric("Draw", f"{xm['p_draw'] * 100:.0f}%")
    c[2].metric(f"{away_t} win", f"{xm['p_away'] * 100:.0f}%")
    c[3].metric(f"xPts {home_t}", f"{xm['xpts_home']:.2f}")
    c[4].metric(f"xPts {away_t}", f"{xm['xpts_away']:.2f}")
    verdict = "matched the chances" if likely == actual else f"went against the chances (the numbers favoured a {likely})"
    st.write(f"Result: {actual}, which {verdict}. The chance of exactly this result was {probs[actual] * 100:.0f}%.")
    shots_m = D.shots[D.shots["match_id"] == mid]
    if len(shots_m):
        show(xg_race(shots_m, home_t, away_t))
        st.caption("Cumulative xG through the match. Stars mark goals. Steep steps are big chances.")
    grid = D.xpts_grid[mid][:6, :6] * 100
    gt = pd.DataFrame(grid, index=[f"{home_t} {i}" for i in range(6)], columns=[f"{away_t} {j}" for j in range(6)])
    st.write("Scoreline probabilities from shot xG")
    st.dataframe(gt.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None))
    st.caption("Built from the xG of every shot, treating shots as independent. Own goals have no xG and are left out.")
