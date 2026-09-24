# Champions League Clash of Styles

Predicts and explains Champions League matchups not just by team
*strength*, but by team *style* — quantifying which tactical styles
(high press, possession-control, direct/counter, low-block defensive)
tend to beat which, when teams from different leagues collide.

This is project #4 in the
[Sports Analytics & Applied Math Portfolio](../README.md), following the
fantasy football and March Madness predictors.

## The idea

The Champions League is the one competition where Europe's different
tactical cultures — the Premier League's pace and directness, La Liga's
possession control, the Bundesliga's pressing, Serie A's structure — are
forced to play each other. Two teams can have near-identical "quality"
ratings and still produce very lopsided results because one team's style
is a bad matchup for the other's.

This project:

1. Pulls advanced team stats (possession, pressing, pass directness,
   shot volume, defensive line) for Champions League clubs and their
   domestic leagues.
2. Clusters teams into **style archetypes** using unsupervised learning
   (k-means on standardized style features).
3. Builds a historical dataset of CL matches labeled with each side's
   style archetype.
4. Computes a **style vs. style win-rate matrix** — the empirical answer
   to "does pressing beat possession?"
5. Trains a predictive model for match outcomes that includes a
   style-matchup interaction term, and checks whether it beats a
   strength-only baseline.
6. Visualizes everything (style radar charts, matchup heatmap, feature
   importances).

See [REPORT.md](REPORT.md) for full methodology, results, and honest
limitations, and [STEP_BY_STEP.md](STEP_BY_STEP.md) for how to run this
in VS Code from a clean clone.

## Project structure

```
champions-league-style-clash/
├── README.md
├── REPORT.md                  <- full write-up: methodology, results, limitations
├── STEP_BY_STEP.md            <- VS Code setup + run order
├── requirements.txt
├── config.py                  <- seasons, leagues, constants — edit this first
├── data/
│   ├── raw/                   <- scraped FBref stats land here (gitignored)
│   └── processed/             <- cleaned feature tables + match dataset
├── outputs/
│   └── figures/                <- saved PNGs (radar charts, heatmap, etc.)
├── src/
│   ├── 01_collect_data.py      <- pulls squad style stats + CL match results
│   ├── 02_feature_engineering.py  <- builds per-team style vectors
│   ├── 03_style_clustering.py     <- k-means -> style archetype labels
│   ├── 04_build_matchup_dataset.py <- match-level dataset w/ style pairing
│   ├── 05_train_model.py          <- baseline vs. style-aware model
│   └── 06_visualize_results.py    <- radar charts + matchup heatmap
└── notebooks/
    └── exploration.ipynb       <- optional scratch space
```

## Quickstart

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/01_collect_data.py
python src/02_feature_engineering.py
python src/03_style_clustering.py
python src/04_build_matchup_dataset.py
python src/05_train_model.py
python src/06_visualize_results.py
```

Full walkthrough with VS Code-specific setup: [STEP_BY_STEP.md](STEP_BY_STEP.md).
