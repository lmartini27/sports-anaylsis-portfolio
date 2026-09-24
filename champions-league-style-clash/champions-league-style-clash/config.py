"""
Central config for the Clash of Styles project.
Edit this first -- everything downstream reads from here.
"""

# --- Seasons to pull (FBref format: "2023-2024") -----------------------
# Start with 2-3 seasons. More seasons = more style-matchup data, but
# also more scraping time and more risk of FBref rate-limiting you.
SEASONS = ["2022-2023", "2023-2024", "2024-2025"]

# --- Domestic leagues to source style stats from ------------------------
# These are the leagues whose clubs regularly reach the CL group/league
# stage. Names must match soccerdata's FBref league keys.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# --- Competition to pull match results from ------------------------------
CUP = "Champions League"

# --- Style feature columns --------------------------------------------
# Pulled from FBref's "possession", "passing", and "defense" squad
# tables, then engineered in 02_feature_engineering.py. These become the
# inputs to the style clustering in step 03.
#
# NOTE ON PRESS INTENSITY: true PPDA (passes-allowed-per-defensive-action)
# needs opponent pass counts attributed to each match, which the season
# squad tables don't expose directly. `press_height_pct` below is a proxy
# -- the share of a team's tackles+interceptions that happen in the
# middle/attacking third rather than their own defensive third. It tells
# you how high up the pitch a team defends, which is most of what
# "presses a lot" means for style purposes, but it is NOT identical to
# PPDA. Documented as a limitation in REPORT.md -- don't oversell it.
STYLE_FEATURES = [
    "possession_pct",              # average possession share
    "pass_completion_pct",         # control vs. risk
    "progressive_passes_per90",    # how much a team progresses the ball via passing
    "passes_into_final_third_per90",
    "press_height_pct",            # proxy for pressing height (see note above)
    "shots_per90",
    "touches_att_pen_per90",
]

# --- Number of style archetypes to cluster into -------------------------
# Start at 4 (e.g. Press-heavy / Possession-control / Direct-counter /
# Low-block) and revisit with an elbow/silhouette plot in step 03.
N_STYLE_CLUSTERS = 4

RANDOM_STATE = 42

# --- Paths ---------------------------------------------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FIGURES_DIR = "outputs/figures"
