# v9 (full project)
Run:  streamlit run Home.py
Put unl_2026_27_all.xlsx and unl_2026_27_xy.xlsx next to Home.py.
Install: pip install -r requirements.txt
If you copy this over an older version, delete the old pages/ folder first (page file names changed).
The first start builds .cache and takes about a minute; later starts take seconds.
Also put unl_2026_27_extra.xlsx next to Home.py. It adds player ages and values (Scouting), the Halves page, the Odds vs xG page and the Referees tab on Discipline. The dashboard still runs without it, and those parts show a note.
The What-if simulator (page 30) needs no extra file: it builds on the Match predictor model and the shot, card and score tables already in the dashboard.
Machine-learning pages (31 Bayesian predictor, 32 Similar players, 33 Market value model) use scikit-learn and scipy; scipy comes with scikit-learn.
