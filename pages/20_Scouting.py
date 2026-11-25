import io

import numpy as np
import pandas as pd
import streamlit as st

from unl import roles as R
from unl.bootstrap import setup
from unl.sample import add_flag, sample_note, MIN_MINUTES
from unl.charts import scatter_labeled
from unl.charts2 import scatter_trend
from unl.kpi import kpi_row
from unl.pitch import show
from unl.ratings import stabilised_per90

D = setup("Scouting | Nations League 2026-27")
st.title("Scouting shortlist")
st.caption("Filter by role, age, market value, minutes and per-90 output. Percentiles are measured against players of the same role, so the shortlist compares like with like. "
           "Use the minutes cap to find players who produce a lot in few minutes; use age and value to find young or affordable ones.")

# ---------------------------------------------------------------- metric table
METRICS = {
    "xT per 90": ("xt_total", True), "xA per 90": ("expectedassists", True), "xG per 90": ("expectedgoals", True),
    "Goals per 90": ("goals", True), "Key passes per 90": ("keypass", True), "Big chances created per 90": ("bigchancecreated", True),
    "Progressive carries per 90": ("progressiveballcarriescount", True), "Dribbles won per 90": ("woncontest", True),
    "Crosses per 90": ("totalcross", True), "Tackles per 90": ("totaltackle", True), "Interceptions per 90": ("interceptionwon", True),
    "Recoveries per 90": ("ballrecovery", True), "Clearances per 90": ("totalclearance", True), "Aerial duels won per 90": ("aerialwon", True),
    "Shots per 90": ("totalshots", True),
}


@st.cache_data(show_spinner=False)
def build(prior_minutes, stabilise):
    p = D.players.copy()
    p = p.merge(D.xpass.player[["player_id", "PoE shrunk per 100"]], on="player_id", how="left")
    cols = {}
    for lab, (col, _) in METRICS.items():
        cols[lab] = stabilised_per90(p, col, prior_minutes) if stabilise else p[col] / p["minutesplayed"] * 90
    p = pd.concat([p, pd.DataFrame(cols)], axis=1)
    p["Passing over expected (per 100)"] = p["PoE shrunk per 100"]
    return p


c = st.columns(4)
stab = c[0].checkbox("Stabilise small samples", True, key="sc_stab",
                     help="Shrinks each per-90 rate towards the role average, as if the player had 180 extra minutes of average output.")
pool_min = c[1].number_input("Peer pool: minimum minutes", 90, 600, 180, step=30, key="sc_pool")
preset = c[2].radio("Preset", ["Custom", "Hidden gems", "Young talents"], horizontal=True, key="sc_preset",
                    help="Hidden gems: strong output in few minutes. Young talents: age 23 or younger with strong output.")
gems_like = preset in ("Hidden gems", "Young talents")
P = build(180.0, stab)
has_profile = "age" in P.columns and P["age"].notna().any()
all_metrics = list(METRICS) + ["Passing over expected (per 100)"]

f1 = st.columns(4)
roles_pick = f1[0].multiselect("Roles", R.ROLES, default=[], key="sc_roles", placeholder="All roles")
leagues = f1[1].multiselect("League", sorted(D.m["league"].unique()), default=[], key="sc_league", placeholder="All leagues")
teams_pick = f1[2].multiselect("Teams", D.teams, default=[], key="sc_teams", placeholder="All teams")
if preset == "Hidden gems":
    mins = f1[3].slider("Minutes played", 0, 900, (45, 270), step=15, key="sc_mins_gems")
elif preset == "Young talents":
    mins = f1[3].slider("Minutes played", 0, 900, (90, 900), step=15, key="sc_mins_young")
else:
    mins = f1[3].slider("Minutes played", 0, 900, (180, 900), step=15, key="sc_mins")

# profile filters (age, market value, contract, foot) come from the extra scrape; skipped when that file is absent
age_lo, age_hi = (int(np.floor(P["age"].min())), int(np.ceil(P["age"].max()))) if has_profile else (16, 45)
ages, vcap, vmax, cyear, foots = (age_lo, age_hi), 0.0, 0.0, "Any", []
if has_profile:
    f2 = st.columns(4)
    ages = f2[0].slider("Age", age_lo, age_hi, (age_lo, min(23, age_hi)) if preset == "Young talents" else (age_lo, age_hi), key=f"sc_age_{preset}")
    vmax = float(np.ceil(P["market_value_m"].max()))
    vcap = f2[1].number_input("Maximum market value (€m)", 0.0, vmax, vmax, step=1.0, key="sc_val",
                              help="SofaScore market value estimate in millions of euros. Players without a value are dropped once you lower this.")
    years = sorted(int(y) for y in P["contract_year"].dropna().unique() if 2026 <= y <= 2032)
    cyear = f2[2].selectbox("Contract ends by", ["Any"] + years, key="sc_contract", help="Contract expiry year at club level. Players without a listed contract are dropped once you set a year.")
    foots = f2[3].multiselect("Preferred foot", ["Right", "Left", "Both"], default=[], key="sc_foot", placeholder="Any foot")
else:
    st.caption("Age, market value and contract filters need unl_2026_27_extra.xlsx next to Home.py (sheet 'profiles').")

d = P[(P["minutesplayed"] >= mins[0]) & (P["minutesplayed"] <= mins[1])].copy()
if roles_pick:
    d = d[d["role"].isin(roles_pick)]
if leagues:
    tin = set(D.m[D.m["league"].isin(leagues)][["home_team", "away_team"]].values.ravel())
    d = d[d["team"].isin(tin)]
if teams_pick:
    d = d[d["team"].isin(teams_pick)]
if has_profile:
    if tuple(ages) != (age_lo, age_hi):
        d = d[d["age"].between(ages[0], ages[1] + 0.99)]
    if vcap < vmax:
        d = d[d["market_value_m"] <= vcap]
    if cyear != "Any":
        d = d[d["contract_year"] <= cyear]
    if foots:
        d = d[d["preferred_foot"].isin(foots)]
d = d[d["role"].notna() & (d["role"] != "Goalkeeper")]

# percentile vs role pool
pool = P[(P["minutesplayed"] >= pool_min) & P["role"].notna()]
pct = pd.DataFrame(index=d.index)
for m in all_metrics:
    vals = {r: np.sort(g[m].dropna().values) for r, g in pool.groupby("role")}
    pct[m] = [np.searchsorted(vals[r], v, side="right") / max(len(vals[r]), 1) * 100 if r in vals and len(vals[r]) and pd.notna(v) else np.nan
              for r, v in zip(d["role"], d[m])]

st.subheader("Filters on output")
n_f = 4
default_metrics = ["xT per 90", "xA per 90", "xG per 90", "Progressive carries per 90"]
fc = st.columns(n_f)
mask = pd.Series(True, index=d.index)
used = []
for i in range(n_f):
    with fc[i]:
        m = st.selectbox(f"Metric {i + 1}", ["(none)"] + all_metrics, index=(all_metrics.index(default_metrics[i]) + 1) if (i < 2 and gems_like) or (i == 0 and preset == "Custom") else 0,
                         key=f"sc_m{i}_{preset}")
        if m != "(none)":
            floor = st.slider("Minimum percentile", 0, 99, 80 if gems_like else 70, key=f"sc_p{i}_{preset}")
            used.append((m, floor))
mode = st.radio("A player must pass", ["All filters", "Any filter"], horizontal=True, key="sc_mode") if len(used) > 1 else "All filters"
if used:
    conds = [pct[m] >= f for m, f in used]
    mask = np.logical_and.reduce(conds) if mode == "All filters" else np.logical_or.reduce(conds)
res = d[mask].copy()

kpi_row([("Players in filter range", len(d), "after role, league, team and minutes", "blue"),
         ("On the shortlist", len(res), f"{mode.lower()} met" if used else "no output filters", "red"),
         ("Median minutes", int(res["minutesplayed"].median()) if len(res) else 0, "shortlist", "green"),
         ("Teams represented", res["team"].nunique() if len(res) else 0, None, "gold"),
         ("Peer pool", len(pool), f"players with {pool_min}+ minutes", "teal")], min_width=190)

if res.empty:
    st.info("Nobody passes these filters. Lower a percentile or widen the minutes range.")
    st.stop()

sort_m = st.selectbox("Sort by", [m for m, _ in used] + [m for m in all_metrics if m not in [u for u, _ in used]], key="sc_sort")
res = res.assign(**{f"{m} pct": pct.loc[res.index, m] for m in all_metrics})
res = res.sort_values(sort_m, ascending=False)
show_m = [m for m, _ in used] or ["xT per 90", "xA per 90"]
prof = ["age", "club", "market_value_m", "contract_year"] if has_profile else []
out = res[["player", "team", "role", "position", "minutesplayed", "Matches"] + prof + show_m].copy()
for m in show_m:
    out[f"{m} (pct)"] = res[f"{m} pct"].round(0)
out = out.rename(columns={"player": "Player", "team": "Team", "role": "Role", "position": "Pos", "minutesplayed": "Minutes", "age": "Age", "club": "Club",
                          "market_value_m": "Value (€m)", "contract_year": "Contract until"})
out["Minutes"] = out["Minutes"].astype(int)
if "Contract until" in out:
    out["Contract until"] = out["Contract until"].astype("Int64")
out = add_flag(out, "Minutes", MIN_MINUTES)
st.dataframe(out.round(2), hide_index=True, height=440)
sample_note(out, what="shortlisted players")

if has_profile:
    vv = res.dropna(subset=["market_value_m", sort_m])
    vv = vv[vv["market_value_m"] > 0]
    if len(vv) >= 5:
        st.subheader("Output against market value")
        show(scatter_labeled(vv["market_value_m"], vv[sort_m], vv["player"], "Market value (€m, log scale)", sort_m, label_n=10, xlog=True))
        st.caption("Players towards the top left produce a lot for a low market value. Market values are SofaScore estimates, and young players are often valued on potential rather than output.")


c1, c2, _ = st.columns([1, 1, 3])
csv = out.round(3).to_csv(index=False).encode("utf-8")
c1.download_button("Download CSV", csv, "unl_scouting_shortlist.csv", "text/csv", key="sc_csv")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    out.round(3).to_excel(xw, sheet_name="Shortlist", index=False)
    pd.DataFrame({"Filter": [f"{m} >= {f}th percentile" for m, f in used] or ["none"], "Mode": mode,
                  "Roles": ", ".join(roles_pick) or "all", "Minutes": f"{mins[0]}-{mins[1]}", "Stabilised": stab}).to_excel(xw, sheet_name="Filters", index=False)
c2.download_button("Download Excel", buf.getvalue(), "unl_scouting_shortlist.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="sc_xlsx")
st.caption("Percentiles compare each player with players of the same role who played at least the peer-pool minutes. Stabilised rates pull low-minute "
           "players towards the role average so a one-match wonder does not top the list. 'Hidden gems' looks for strong output in few minutes: check them with video.")
