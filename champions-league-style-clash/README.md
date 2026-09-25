# Champions League Clash of Styles

Do domestic playing styles predict what happens when Big-5 league clubs meet in the Champions League? This project clusters 194 team-seasons into style archetypes and tests them against 127 CL matches.

**Headline finding:** results follow a clear hierarchy along a possession/control ladder rather than rock-paper-scissors matchups. Each step up the ladder is worth about +0.82 goals (R² = 0.158, p < 0.0001). Style is heavily entangled with team quality, though, so the report treats this as a strong signal rather than proof of cause.

Full methodology, results, and limitations: [REPORT.md](REPORT.md)

Project #4 in the [Sports Analytics Portfolio](../README.md).

## Pipeline

| Script | What it does |
|---|---|
| `src/00_inspect_downloads.py` | Prints columns of every CSV in `data/raw/` (use after adding new data) |
| `src/01_collect_data.py` | Scrapes FBref team possession via `soccerdata` (produces `team_standard.csv`) |
| `src/02_feature_engineering.py` | Aggregates player stats to team style features |
| `src/03_style_clustering.py` | K-means clustering into 4 style archetypes |
| `src/04_build_matchup_dataset.py` | Matches CL results to both clubs' styles |
| `src/05_train_model.py` | Style matrix, style-gap regression, cross-validated model comparison |
| `src/06_visualize_results.py` | Radar chart, matchup heatmap, style-gap chart |

## Data (place in `data/raw/`)

- `team_standard.csv`, produced by `01_collect_data.py`. FBref blocks the later detailed stat pages, so the script errors after saving this file; that's expected.
- `players_data-2024_2025.csv`, from Kaggle, *Football Players Stats (2024-2025)* by hubertsidorowicz
- `2022-2023 Football Player Stats.csv`, from Kaggle, *2022-2023 Football Player Stats* by vivovinco
- `uefa_champions_league_historical_match_statistics_2020_2026.csv`, from Kaggle

Seasons used are set in `config.py` (`SEASONS`).

## Run it

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src\02_feature_engineering.py
python src\03_style_clustering.py
python src\04_build_matchup_dataset.py
python src\05_train_model.py
python src\06_visualize_results.py
```
