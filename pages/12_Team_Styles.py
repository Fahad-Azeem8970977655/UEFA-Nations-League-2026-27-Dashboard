import pandas as pd
import streamlit as st

from unl.advanced import STYLE_FEATURES, cluster_teams, nearest_teams
from unl.bootstrap import setup
from unl.kpi import kpi_row
from unl.charts import scatter_labeled
from unl.pitch import show

D = setup("Team styles | Nations League 2026-27")

st.title("Team styles")
st.caption("Eight style measures per team (how they pass, how high they press, how much of the ball they keep) are reduced to "
           "two dimensions with PCA, then grouped with k-means. Teams close together play alike, whatever their quality.")

S = D.style
kpi_row([("Teams", len(S.coords)), ("Style measures", len(S.features.columns)), ("Variation on 2 axes", f"{S.explained.sum() * 100:.0f}%"),
         ("Most direct", S.features["Forward per pass (m)"].idxmax(), f"{S.features['Forward per pass (m)'].max():.1f} m per pass"),
         ("Most pressing", S.features["Press intensity"].idxmax())], min_width=170)
k = st.slider("Number of style groups", 3, 6, 5, key="style_k")
groups = cluster_teams(S, k)
team = st.selectbox("Highlight a team", [""] + list(S.coords.index), key="style_team")
ev = S.explained * 100

left, right = st.columns([1.5, 1])
with left:
    show(scatter_labeled(S.coords["PC1"], S.coords["PC2"], S.coords.index, f"Style axis 1 ({ev[0]:.0f}% of variation)",
                         f"Style axis 2 ({ev[1]:.0f}% of variation)", groups=groups.loc[S.coords.index].values,
                         highlight=team or None, label="all"))
with right:
    st.write("What the axes mean")
    ld = S.loadings.round(2)
    st.dataframe(ld, height=330)
    st.caption("Large positive or negative loadings show which measures drive each axis. Press intensity is 1 / PPDA, "
               "regain height is the average pitch position (0-100) of tackles, interceptions and recoveries.")

st.subheader("Style groups")
for name in sorted(groups.unique()):
    members = sorted(groups[groups == name].index)
    st.markdown(f"**{name}** ({len(members)}): " + ", ".join(members))

if team:
    st.subheader(f"{team}: closest styles")
    nn = nearest_teams(S, team, 6)
    cmp = S.features.loc[[team] + list(nn.index)].round(1)
    cmp.insert(0, "Distance", [0.0] + [round(v, 2) for v in nn.values])
    st.dataframe(cmp)
    z = S.z.loc[team].sort_values()
    st.write("Where the team stands out (standard deviations from the all-team average)")
    st.dataframe(z.round(2).rename("z-score").to_frame().T)

with st.expander("All style measures"):
    st.dataframe(pd.concat([groups.rename("Group"), S.features.round(1)], axis=1).sort_values("Group"))
