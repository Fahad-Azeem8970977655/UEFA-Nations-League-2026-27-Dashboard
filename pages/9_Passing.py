import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.pitch import new_figure, show
from unl.style import HOME, INK
from unl.viz2 import PASS_CATS, classify_passes, draw_pass_arrows, draw_pass_flow

D = setup("Passing | Nations League 2026-27")

st.title("Passing")
st.caption("Pass arrows on the pitch, coloured by pass type or by player, plus a zone-by-zone pass flow.")

team = st.selectbox("Team", D.teams, key="pass_team")
mids = D.tm[D.tm["team"] == team].sort_values("date")
labels = {r.match_id: f"{r.opponent} ({'H' if r.side == 'home' else 'A'}, {r.date:%d %b})" for r in mids.itertuples()}
sel = st.multiselect("Matches included", list(labels), default=list(labels), format_func=lambda k: labels[k], key="pass_matches")
if not sel:
    st.info("Select at least one match.")
    st.stop()

P = D.passes[(D.passes["team"] == team) & D.passes["match_id"].isin(sel) & D.passes["event_action_type"].isin(["pass", "cross"])].copy()
P = P.dropna(subset=["x", "y", "end_x", "end_y"])
P["cat"] = classify_passes(P)

kpi_row([("Passes", f"{len(P):,}", f"{len(P) / len(sel):.0f} per match"), ("Accuracy", f"{P['outcome'].mean() * 100:.0f}%"),
         ("Key passes", int(P["keypass"].sum())), ("Assists", int(P["is_assist"].sum())),
         ("Into final third", int(((P["end_x"] >= 66.7) & (P["x"] < 66.7) & P["outcome"]).sum()), "completed"),
         ("Long balls", f"{P['is_long_ball'].mean() * 100:.0f}%", "of passes")], min_width=140)
tab1, tab2 = st.tabs(["Pass map", "Pass flow"])

with tab1:
    c = st.columns([1, 1.4])
    by = c[0].radio("Colour by", ["Pass type", "Player"], horizontal=True, key="pass_by")
    cats = c[1].multiselect("Pass types shown", PASS_CATS, default=["Assist", "Key pass", "Progressive"] if len(sel) > 2 else [x for x in PASS_CATS if x != "Completed"],
                            key="pass_cats")
    f = st.columns(3)
    pl_all = P["player"].value_counts().index.tolist()
    players = f[0].multiselect("Players", pl_all, default=pl_all[:3] if by == "Player" else [], key="pass_players",
                               help="Leave empty for the whole team (colour by type).")
    min_len = f[1].slider("Minimum pass length (pitch units)", 0, 60, 0, key="pass_minlen")
    max_n = f[2].slider("Max arrows drawn", 100, 1000, 450, step=50, key="pass_maxn")
    d = P
    if players:
        d = d[d["player"].isin(players)]
    if min_len:
        d = d[((d["end_x"] - d["x"]) ** 2 + (d["end_y"] - d["y"]) ** 2) ** 0.5 >= min_len]
    if by == "Player" and not players:
        st.info("Pick one or more players to colour by player (up to 8).")
        st.stop()
    shown = d[d["cat"].isin(cats)]
    fig, axes = new_figure(1, w=8.0)
    draw_pass_arrows(axes[0], d, cats, by="player" if by == "Player" else "type", max_n=max_n, players=players[:8] or None)
    axes[0].set_title(f"{team}: {len(shown):,} passes shown", fontsize=10, color=INK)
    show(fig)
    counts = d["cat"].value_counts().reindex(PASS_CATS).fillna(0).astype(int)
    st.caption("  |  ".join(f"{k}: {v}" for k, v in counts.items()))
    st.caption("Team attacks to the right. Assists and key passes are always drawn; other types are sampled down to the arrow limit. "
               "Progressive = completed pass at least 25% closer to goal that ends beyond 40% of the pitch.")
    st.info("The data has no pass timestamps or receiver, so passes cannot be chained into sequences such as A to B to C to shot. "
            "Each arrow is one pass.")

with tab2:
    nx = st.select_slider("Zones along the pitch", [4, 6, 8], value=6, key="flow_nx")
    fp = st.multiselect("Players (empty = whole team)", pl_all, key="flow_players")
    d = P[P["player"].isin(fp)] if fp else P
    fig, axes = new_figure(1, w=8.0)
    draw_pass_flow(axes[0], d, nx=nx, ny=max(3, nx * 2 // 3))
    axes[0].set_title(f"{team}: average pass from each zone", fontsize=10, color=INK)
    show(fig)
    st.caption("Each arrow goes from the average start to the average end of the passes that began in that zone. "
               "Thicker = more passes, teal = high completion, magenta = low. Zones with fewer than 8 passes are left blank.")
