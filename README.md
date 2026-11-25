# UEFA Nations League 2026-27 Analytics Dashboard

An interactive football analytics dashboard for the **UEFA Nations League 2026-27 league phase**, built with Streamlit. It turns match, shot, pass and player data into team ratings, tactical views, scouting tools and machine-learning models.

Data source: SofaScore, collected by scraping.

## What is inside

| | |
|---|---|
| Matches | 156 league-phase matches, 54 teams, 4 leagues (A to D) |
| Players | 7,136 player-match rows, about 1,450 players |
| Shots | 4,136 shots with xG, location and situation |
| Passes | 151,740 passes with coordinates |
| Pages | 33 interactive pages |

## Pages

**Results and teams:** Groups, Teams, Matches, Compare teams, Ratings, Match predictor, Bayesian predictor, What-if simulator, Team report

**Players:** Players, Player roles, Scouting, Similar players, Market value model, Finishing and goalkeepers, Player form

**Tactics and shape:** Defensive shape, Pressing, Passing, Pass model, Progression, Team styles

**Chances and flow:** Shot map, True table, Expected threat, Match flow, Formations, Discipline (with a Referees tab), Halves, Odds vs xG, Data quality

**Reference:** Data notes

## Highlights

- **True table and xG:** expected points show how much of each table was luck.
- **Expected threat (xT) and pass model:** value of passes and carries, and expected pass completion.
- **Scouting:** shortlists by role with per-90 percentiles, plus age, market value, contract and foot filters, and a "Young talents" preset.
- **Halves:** every team statistic split into first and second half, with an optional per-90 adjustment for stoppage time.
- **Odds vs xG:** bookmaker expectation against results and the chances created, with upsets and calibration.
- **Referees:** cards, penalties, fouls and stoppage time per referee, compared with career rates.

## Machine learning

| Page | Method | How it is checked |
|---|---|---|
| Bayesian predictor | Hierarchical team strengths with normal priors (Laplace approximation) and Dixon-Coles scorelines | Leave-one-matchday-out cross-validation against a flat base rate and a ridge model |
| What-if simulator | Ridge strength model plus shot and red-card effects measured on the data, with bootstrap ranges | Credible ranges from resampled matches |
| Similar players | Role z-scores, PCA, cosine or distance similarity | Style map and percentile profile comparison |
| Market value model | Gradient boosting on log market value with a club-level feature | 5-fold cross-validation; every player is predicted by a model that did not see him |

With about six matches per team, the models report their uncertainty instead of single numbers. The pages say where a result could be noise.

## Quick start

```bash
git clone https://github.com/Fahad-Azeem8970977655/UEFA-Nations-League-2026-27-dashboard-.git
cd UEFA-Nations-League-2026-27-dashboard-
pip install -r requirements.txt
streamlit run Home.py
```

The first start builds a local `.cache` folder and takes about a minute. Later starts take seconds.

## Data files

Keep these next to `Home.py`:

| File | Content |
|---|---|
| `unl_2026_27_all.xlsx` | Matches, standings, team and player statistics, shots, incidents, lineups |
| `unl_2026_27_xy.xlsx` | Coordinate tables: passes, defensive actions, carries, dribbles, heatmaps, average positions, momentum |
| `unl_2026_27_extra.xlsx` | Halves, referees, odds and player profiles (optional) |

The dashboard still runs without the extra file. The Halves, Odds vs xG, Referees, Scouting profile filters and Market value pages then show a note instead.

## Project structure

```
Home.py                landing page
pages/                 the interactive pages
unl/                   shared code: data loading, metrics, models, charts, styling
assets/                logos
.streamlit/            theme settings
requirements.txt
```

If you copy this over an older version, delete the old `pages/` folder first, because page file names changed.

## Limits

- The league phase gives about six matches per team, so strengths and player rates are uncertain.
- xG, ratings and market values come from SofaScore models and estimates.
- Models describe associations, not causes. Injuries, lineups and motivation are not modelled.
- Leagues A to D never meet in the league phase, so teams are only compared within a league.

## Tech

Python, Streamlit, pandas, NumPy, matplotlib, scikit-learn, SciPy, openpyxl.
