import html

import numpy as np
import pandas as pd
import streamlit as st

from unl.bootstrap import setup
from unl.kpi import kpi_row

D = setup("Data notes | Nations League 2026-27")



st.title("Data notes")
n_matches = D.m["match_id"].nunique()
st.write(f"{n_matches} league-phase matches, from {D.m['date'].min():%d %b} to {D.m['date'].max():%d %b %Y}. "
         "Source: SofaScore, collected for personal analysis.")
kpi_row([("Matches", n_matches), ("Teams", len(D.teams)), ("Player-match rows", f"{len(D.ps):,}"), ("Shots", f"{len(D.shots):,}"), ("Passes with coordinates", f"{len(D.passes):,}"),
         ("Touch points", f"{len(D.heat):,}")], min_width=150)
cov = D.tm.groupby("match_id")["distance_covered"].apply(lambda s: s.notna().any())
st.markdown(
    f"""
**Coverage**
- Distance covered and sprint data exists for {int(cov.sum())} of {n_matches} matches, so averages that use it cover fewer matches.
- Player xG is blank for players who took no shots; the dashboard treats that as zero.
- Players who did not come on have no stats and are left out of per-player totals.
- SofaScore lists own goals among the shots, with no xG and the defender as the shooter. The dashboard removes them from shot maps, xG and xPts, so goals on the shot pages match player goals (final scores still include own goals).

**Extra data (unl_2026_27_extra.xlsx)**
- Halves: SofaScore period statistics for the first and second half of every match. Counting stats missing for a half (for example no yellow cards) are treated as zero; possession is not filled. Goals by half come from the half-time scores.
- Referees: referee, career card totals and stoppage time per match from SofaScore match details; cards, penalties and fouls per match come from the dashboard's own incident and team-stat tables.
- Odds: the last pre-match home / draw / away prices (decimal), turned into probabilities with the bookmaker margin removed by proportional scaling. Market expected points = 3 x win probability + draw probability.
- Player profiles: age (at the last league-phase match), market value in millions of euros (SofaScore estimate), contract end year, club and preferred foot. About 2% of players have no market value and about 10% no contract date.

**What-if simulator**
- Shot levers are the average xG of a shot inside and outside the box (the nearest shots, in the share the team statistics report as inside the box). The red-card multipliers compare xG after the first red card with the side's own average rate, over the matches with a red card up to minute 80.
- Ranges come from refitting the team-strength model on resampled matches. It is a what-if on a simple model, not a forecast.

**Machine-learning pages**
- Bayesian predictor: per-league attack and leakiness with normal priors (Laplace approximation), Dixon-Coles low-score correction, settings chosen and scored by leave-one-matchday-out cross-validation against a flat base rate and the ridge model.
- Similar players: z-scores of the role's per-90 metrics, PCA, cosine or distance similarity.
- Market value model: gradient boosting on log market value with 5-fold cross-validation; each player is predicted by a model that did not see him.

**Coordinates**
- Pass, defensive, carry, dribble and heatmap coordinates run 0-100 across the pitch. Every team is shown attacking to the right.
- Shot coordinates come from SofaScore in a different frame (goal at x = 0), so shots are rotated 180 degrees before plotting.
- On passes, a blank "outcome" is treated as an incomplete pass.

**Definitions**
- Progressive pass: completed, gets at least 25% closer to the opponent's goal, and ends beyond 40% of the pitch.
- Into final third: starts before 66.7% of the pitch length and ends beyond it.
- Into penalty area: ends inside the box and does not start inside it.
- Team percentiles on the radar rank each team among all teams; defensive axes are inverted so outward is always better.
- Expected points: each match's scoreline distribution comes from the xG of every shot (exact Poisson-binomial), treating shots as independent. Own goals have no xG and are left out.
- Expected threat (xT): zone values fitted on this dataset on a 16 x 12 grid (shot share, mean shot xG, pass and carry transitions, penalties excluded). A completed pass or carry earns the value of its end zone minus its start zone; all carries count as completed.
- Team styles: PCA of eight standardised style measures, then k-means (k chosen on the page). Groups describe style, not quality.
- Player percentiles and similarity use per-90 rates compared only with players of the same position and a minimum-minutes cut-off.
- Player roles: depth and width of each player's touch map (needs 60+ touches), split inside the SofaScore position by fixed thresholds listed on the Player roles page.
- Defensive shape: average positions of outfield players (20+ touches) per team and match; line height is the mean x of the four deepest. Positions blend attack and defence.
- Expected pass completion: gradient-boosted model, out-of-fold by match; passes over expected is shrunk by n / (n + 250).
- Progression channels: thirds of the pitch width at the end of a progressive pass or carry. Left and right follow the plotted frame (low y = the attacking team's right).
- Game state and substitutions: minute-level timeline; stoppage time is added to the minute, so the first-half stoppage overlaps the early second half slightly. Momentum is positive for the home side.
- Finishing and goalkeeping: empirical-Bayes shrinkage of goals minus xG (penalties excluded) and of xGOT minus goals conceded.
- Team ratings: ridge model of match xG margin per league, penalty set by leave-one-match-out validation. Rating drivers: ridge regression per position of the SofaScore rating on box-score stats.
- Match predictor: ridge model of xG per league (base level, home edge, attack, defence) feeding two Poisson distributions; group luck simulation samples each played match from its shot-level scoreline grid, 4,000 replays. The accuracy check is in sample.
- Player form uses SofaScore match ratings; team report ranks teams only inside their own league. Data quality lists missing values and per-match coverage.
- Rank for volume metrics such as tackles or clearances shows playing style, not quality.
"""
)
