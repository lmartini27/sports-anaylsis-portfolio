"""
01_collect_data.py

Pulls two things from FBref (via the `soccerdata` package):
  1. Domestic-league squad stats (possession, passing, defense, standard)
     for every team in the 5 leagues in config.LEAGUES, for each season
     in config.SEASONS. This is where the style features come from.
  2. Champions League match schedule/results for the same seasons, which
     becomes the labeled matchup dataset later.

Saves raw pulls as CSVs into data/raw/ so steps 02+ never have to hit
FBref again (be polite to their servers -- this only scrapes once).

NOTE ON COLUMN NAMES: FBref periodically renames/reshuffles columns, and
soccerdata's multi-index columns don't always match 1:1 across stat
types. This script deliberately does NOT rename anything -- it just
flattens and saves the raw pulls, and prints the column names it got
back. Take a look at those printed columns (and/or open the saved CSVs)
before moving to 02_feature_engineering.py, which is where the actual
mapping to config.STYLE_FEATURES happens and where you may need to
adjust a few names if FBref's schema has drifted since this was written.
This is normal for any FBref-scraping project, not a bug.
"""

import sys
import time
import pandas as pd
import soccerdata as sd

sys.path.append("..")
import config


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """soccerdata often returns multi-index columns like ('Standard', 'Poss').
    Collapse to the second level, which is what FBref actually labels them."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def pull_squad_stats(fbref: sd.FBref, stat_type: str) -> pd.DataFrame:
    print(f"  -> pulling '{stat_type}' team stats ...")
    df = fbref.read_team_season_stats(stat_type=stat_type)
    df = flatten_columns(df).reset_index()
    return df


def main():
    print(f"Fetching squad stats for {config.LEAGUES} / {config.SEASONS}")
    fbref = sd.FBref(leagues=config.LEAGUES, seasons=config.SEASONS)

    stat_types = ["standard", "possession", "passing", "defense"]
    frames = {}
    for stat_type in stat_types:
        df = pull_squad_stats(fbref, stat_type)
        frames[stat_type] = df
        out_path = f"{config.RAW_DATA_DIR}/team_{stat_type}.csv"
        df.to_csv(out_path, index=False)
        print(f"     saved {out_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
        print(f"     columns: {list(df.columns)[:15]} ...")
        time.sleep(3)  # be polite between requests

    # --- Champions League schedule / results -----------------------------
    print(f"\nFetching {config.CUP} schedule for {config.SEASONS} ...")
    fbref_cup = sd.FBref(leagues=config.CUP, seasons=config.SEASONS)
    schedule = fbref_cup.read_schedule().reset_index()
    schedule = flatten_columns(schedule)
    out_path = f"{config.RAW_DATA_DIR}/cl_schedule.csv"
    schedule.to_csv(out_path, index=False)
    print(f"  saved {out_path}  ({schedule.shape[0]} matches)")

    print("\nDone. Inspect data/raw/*.csv and the printed column lists above")
    print("before moving to 02_feature_engineering.py -- confirm RENAME_MAP")
    print("in this file matches what actually came back.")


if __name__ == "__main__":
    main()
