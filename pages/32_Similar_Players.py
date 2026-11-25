import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from unl import simil as SM
from unl.bootstrap import setup
from unl.charts2 import grouped_bars
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import AWAY, GREY, HOME, INK

D = setup("Similar players | Nations League 2026-27")
st.title("Similar players")
st.caption("Pick a player and find the players whose game looks most like his, measured on the per-90 numbers that matter for his role. "
           "Add an age or value limit to turn it into a search for a younger or cheaper alternative.")

P = D.players
has_profile = "age" in P.columns and P["age"].notna().any()

c = st.columns(4)
min_min = c[0].number_input("Peer pool: minimum minutes", 90, 600, 180, step=30, key="sp_min")
scope = c[1].radio("Compare with", ["Same role", "Whole position group"], horizontal=True, key="sp_scope",
                   help="Same role: only players of the same role. Whole position group: for example all defenders, measured on the target's role metrics.")
mode = c[2].radio("Similarity", ["Style", "Style and level"], horizontal=True, key="sp_mode",
                  help="Style: the shape of the profile, whatever its size. Style and level: shape and size together, so a weaker player ranks lower.")
same_team = c[3].checkbox("Include his own national team", False, key="sp_team")

cand = P[P["minutesplayed"] >= min_min].merge(D.roles[["player_id", "role"]].rename(columns={"role": "role_"}), on="player_id", how="left")
cand = cand[cand["role_"].notna()].sort_values("player")
labels = {int(r.player_id): f"{r.player} ({r.team}, {r.role_})" for r in cand.itertuples()}
if not labels:
    st.info("No players reach that minutes threshold.")
    st.stop()
default = list(labels).index(int(cand.sort_values("minutesplayed", ascending=False).iloc[0]["player_id"]))
pid = st.selectbox("Player", list(labels), index=default, format_func=lambda i: labels[i], key="sp_player")

tb, info = SM.similar(D, int(pid), "role" if scope == "Same role" else "group", float(min_min), "style" if mode == "Style" else "level")
if tb is None:
    st.info("This player is below the minutes threshold, so he has no profile to compare.")
    st.stop()
me = tb.iloc[info["pos"]]

f = st.columns(3)
age_hi = f[0].slider("Maximum age", 17, 45, 45, key="sp_age") if has_profile else 45
vmax = float(np.ceil(P["market_value_m"].max())) if has_profile else 0.0
vcap = f[1].number_input("Maximum market value (€m)", 0.0, vmax, vmax, step=1.0, key="sp_val") if has_profile else 0.0
topn = f[2].slider("Players to show", 5, 30, 15, key="sp_n")

r = tb.drop(index=info["pos"])
if not same_team:
    r = r[r["team"] != me["team"]]
if has_profile:
    if age_hi < 45:
        r = r[r["age"] <= age_hi]
    if vcap < vmax:
        r = r[r["market_value_m"] <= vcap]
r = r.sort_values("Similarity", ascending=False).head(topn)

kpi_row([("Role", info["role"], f"{info['pool']} players in the pool", "blue"), ("Metrics used", len(info["metrics"]), f"{info['ncomp']} components keep {info['explained'] * 100:.0f}% of the variation", "teal"),
         ("Closest match", r["player"].iloc[0] if len(r) else "none", f"{r['Similarity'].iloc[0]:.0f}% similar" if len(r) else None, "green"),
         ("His minutes", int(me["minutesplayed"]), "per-90 figures get noisy under about 270", "gold")], min_width=190)
if r.empty:
    st.info("No player passes the filters. Loosen the age or value limit.")
    st.stop()

out = r.assign(Rank=np.arange(1, len(r) + 1))
cols = ["Rank", "player", "team"] + (["club", "age", "market_value_m", "contract_year"] if has_profile else []) + ["minutesplayed", "Similarity"]
out = out[cols].rename(columns={"player": "Player", "team": "Team", "club": "Club", "age": "Age", "market_value_m": "Value (€m)", "contract_year": "Contract until",
                                "minutesplayed": "Minutes"})
if "Contract until" in out:
    out["Contract until"] = out["Contract until"].astype("Int64")
out["Minutes"] = out["Minutes"].astype(int)
st.dataframe(out.round(1), hide_index=True, height=420)
st.caption("Similarity 100 = identical profile. Players with few minutes can look similar by chance, so check the Minutes column.")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    pool = tb
    ax.scatter(pool["PC1"], pool["PC2"], s=22, color="#C9D2DC", zorder=2)
    ax.scatter(r["PC1"], r["PC2"], s=46, color=HOME, zorder=3, edgecolor="white", lw=0.6)
    for rr in r.head(8).itertuples():
        ax.annotate(rr.player, (rr.PC1, rr.PC2), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.scatter([me["PC1"]], [me["PC2"]], s=120, facecolor="none", edgecolor=AWAY, lw=2, zorder=4)
    ax.annotate(me["player"], (me["PC1"], me["PC2"]), xytext=(6, 6), textcoords="offset points", fontsize=9.5, color=AWAY, fontweight="bold")
    ax.set_xlabel("Main playing-style axis 1", fontsize=9)
    ax.set_ylabel("Main playing-style axis 2", fontsize=9)
    ax.set_title(f"Style map of {info['role']}s" if scope == "Same role" else "Style map of the position group", fontsize=10, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6EBF0", lw=0.6, zorder=0)
    fig.tight_layout()
    show(fig)
with right:
    pick = st.selectbox("Compare his profile with", list(r["player"]), key="sp_pick")
    row = r[r["player"] == pick].iloc[0]
    a, b = info["pct"].iloc[info["pos"]].values, info["pct"].loc[r[r["player"] == pick].index[0]].values
    show(grouped_bars(info["metrics"], [(me["player"], a, "#D61F3D"), (pick, b, "#1D5FAF")], ylabel="Percentile in the pool", fmt="{:.0f}", figsize=(6.4, 4.2)))
st.caption("Percentile = share of the pool a player is ahead of on each role metric. Where the bars differ most is where the two players are least alike.")
