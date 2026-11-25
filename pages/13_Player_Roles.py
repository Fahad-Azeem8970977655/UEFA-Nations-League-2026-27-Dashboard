import numpy as np
import pandas as pd
import streamlit as st

from unl import roles as R
from unl.advanced import similar_players
from unl.bootstrap import setup
from unl.charts import radar_chart
from unl.charts2 import role_map, scatter_trend
from unl.kpi import kpi_row
from unl.pitch import show
from unl.style import HOME

D = setup("Player roles | Nations League 2026-27")
st.title("Player roles")
st.caption("SofaScore only says defender, midfielder or forward. Here each player's touch map (average depth and width) "
           "splits those into eleven roles, so a winger is no longer compared with a centre-forward.")

ro = D.roles
det = ro[ro["role_detected"]]
changed = det[det["role"] != det["position"].map(R.GENERIC)]
kpi_row([("Players with a detected role", f"{len(det)} of {len(ro)}", "need at least 60 touches", "blue"),
         ("Roles", ro["role"].nunique(), "from 4 coarse labels", "red"),
         ("Wide players", int(det["role"].isin(["Full-back", "Wing-back", "Winger", "Wide midfielder"]).sum()),
          "full-backs, wing-backs, wingers, wide mids", "green"),
         ("Most common role", det["role"].value_counts().idxmax(), f"{det['role'].value_counts().iloc[0]} players", "gold"),
         ("Relabelled players", f"{len(changed) / len(det) * 100:.0f}%", "role differs from the generic label", "teal")], min_width=190)

tab1, tab2, tab3 = st.tabs(["Role map", "Percentiles and similar players", "How roles are decided"])

with tab1:
    c = st.columns([1.2, 2])
    team = c[0].selectbox("Team", ["All teams"] + D.teams, key="role_team")
    pick = c[1].multiselect("Roles shown", R.ROLES, default=R.ROLES, key="role_pick")
    mm = st.slider("Minimum minutes", 0, 540, 90, step=30, key="role_min")
    d = det[(det["minutesplayed"] >= mm) & det["role"].isin(pick)]
    hl = None
    if team != "All teams":
        hl = d[d["team"] == team]
    left, right = st.columns([1.5, 1])
    with left:
        show(role_map(d, R.ROLE_COLORS, hl))
        st.caption("Each dot is a player at his average touch position, team attacking to the right. "
                   "Ringed names are the selected team's players.")
    with right:
        tbl = d.groupby("role").agg(Players=("player_id", "size"), Depth=("depth", "mean"), Width=("width", "mean")).round(1)
        st.dataframe(tbl.reindex([r for r in R.ROLES if r in tbl.index]))
        if hl is not None and len(hl):
            st.write(f"{team}: players by role")
            x = hl.sort_values(["role", "minutesplayed"], ascending=[True, False])[["player", "role", "minutesplayed"]]
            x.columns = ["Player", "Role", "Minutes"]
            x["Minutes"] = x["Minutes"].astype(int)
            st.dataframe(x, hide_index=True, height=300)

with tab2:
    c = st.columns(4)
    role = c[0].selectbox("Role", R.ROLES, index=R.ROLES.index("Winger"), key="rp_role")
    mm = c[1].number_input("Minimum minutes", 90, 1000, 180, step=30, key="rp_min")
    basis = c[2].radio("Compare with", ["Same role", "Whole position group"], key="rp_basis")
    ident, raw, pct, z = R.role_frame(D.players, D.roles, role, mm, "group" if basis != "Same role" else "role")
    if len(ident) < 5:
        st.info("Fewer than five players match. Lower the minimum minutes or compare with the whole position group.")
    else:
        names = {int(r.player_id): f"{r.player} ({r.team})" for r in ident.sort_values("player").itertuples()}
        pid = c[3].selectbox("Player", list(names), format_func=lambda k: names[k], key="rp_player")
        i = ident.index[ident["player_id"] == pid][0]
        prow = D.roles[D.roles["player_id"] == pid].iloc[0]
        kpi_row([("Role", ident.loc[i, "role"], f"SofaScore position {prow['position']}", "blue"),
                 ("Peers", len(ident), "players in the comparison", "red"),
                 ("Minutes", int(ident.loc[i, "minutesplayed"]), None, "green"),
                 ("Average depth", f"{prow['depth']:.0f}", "0 = own goal, 100 = opponent goal", "gold"),
                 ("Average width", f"{prow['width']:.0f}", "0 = central, 50 = touchline", "teal")])
        left, right = st.columns(2)
        with left:
            show(radar_chart([(names[pid], pct.loc[i], HOME)]))
            st.caption(f"Percentile among {len(ident)} players ({basis.lower()}), counts per 90 minutes. 100 = best of the group.")
        with right:
            st.dataframe(pd.DataFrame({"Metric": raw.columns, "Per 90 / value": raw.loc[i].round(2).values,
                                       "Percentile": pct.loc[i].round(0).astype(int).values}), hide_index=True)
        st.subheader("Similar players")
        sim = similar_players(ident, z, pid, 8)
        out = pd.DataFrame({"Player": sim["player"], "Team": sim["team"], "Role": sim["role"],
                            "Minutes": sim["minutesplayed"].astype(int), "Similarity %": sim["Similarity"].round(0).astype(int)})
        st.dataframe(out, hide_index=True)
        st.caption("Cosine similarity of standardised per-90 profiles, role-specific metric set. Style, not quality.")

with tab3:
    st.markdown(f"""
**Inputs.** Every touch a player had in the match heatmaps. *Depth* is the average x (0 own goal, 100 opponent goal).
*Width* is the average distance from the centre line (0 central, 50 touchline). A role needs at least {R.MIN_TOUCHES} touches.

**Rules, applied inside the SofaScore position**
- Defenders: width of {R.WIDE_D} or more is a wide defender, a **wing-back** when depth is {R.WINGBACK_X}+ and a **full-back** otherwise; the rest are **centre-backs**.
- Midfielders: width of {R.WIDE_M}+ is wide, a **winger** from depth {R.WINGER_X} and a **wide midfielder** below. Central players under depth {R.DM_X} are **holding midfielders**, from {R.AM_X} **attacking midfielders**, in between **central midfielders**.
- Forwards: width of {R.WIDE_F}+ is a **winger**. Central forwards under depth {R.SECOND_X} are **second strikers**, the others **centre-forwards**.
- Goalkeepers keep one role. Players with too few touches keep a generic label and are left out of role comparisons.

Thresholds sit in `unl/roles.py`. Roles describe where a player played in this tournament, so a player who switched role between matches gets the label of his average position.
""")
    x = det[det["position"].isin(["D", "M", "F"])]
    show(scatter_trend(x["depth"], x["width"], x["player"], "Average depth", "Average width", trend=False, label_n=0,
                       title="Every player's touch map position (depth against width)"))
