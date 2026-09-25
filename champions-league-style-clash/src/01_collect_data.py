"""
01_collect_data.py

FBref restructured their site at some point and retired most advanced
TEAM-level squad stat pages. As of this version of `soccerdata`,
`read_team_season_stats` only supports: standard, keeper, shooting,
playing_time, misc. The advanced categories this project needs
(possession, passing, defense) still exist on FBref, but only at the
PLAYER level now -- so this script pulls those per-player and
aggregates them up to team totals itself.
"""

import sys
import os
import time
import pandas as pd
import soccerdata as sd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def pull_team_stats(fbref: sd.FBref, stat_type: str) -> pd.DataFrame:
    print(f"  -> pulling team '{stat_type}' stats ...")
    df = fbref.read_team_season_stats(stat_type=stat_type)
    return flatten_columns(df).reset_index()


def pull_player_stats(fbref: sd.FBref, stat_type: str) -> pd.DataFrame:
    print(f"  -> pulling PLAYER '{stat_type}' stats (aggregated to team after) ...")
    df = fbref.read_player_season_stats(stat_type=stat_type)
    return flatten_columns(df).reset_index()


def main():
    print(f"Fetching stats for {config.LEAGUES} / {config.SEASONS}")
    fbref = sd.FBref(leagues=config.LEAGUES, seasons=config.SEASONS)

    standard = pull_team_stats(fbref, "standard")
    standard.to_csv(f"{config.RAW_DATA_DIR}/team_standard.csv", index=False)
    print(f"     saved team_standard.csv  ({standard.shape[0]} rows)")
    print(f"     columns: {list(standard.columns)}")
    time.sleep(3)

    for stat_type, fname in [
        ("passing", "player_passing.csv"),
        ("possession", "player_possession.csv"),
        ("defense", "player_defense.csv"),
    ]:
        df = pull_player_stats(fbref, stat_type)
        out_path = f"{config.RAW_DATA_DIR}/{fname}"
        df.to_csv(out_path, index=False)
        print(f"     saved {out_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
        print(f"     columns: {list(df.columns)}")
        time.sleep(3)

    print(f"\nFetching {config.CUP} schedule for {config.SEASONS} ...")
    fbref_cup = sd.FBref(leagues=config.CUP, seasons=config.SEASONS)
    schedule = fbref_cup.read_schedule().reset_index()
    schedule = flatten_columns(schedule)
    out_path = f"{config.RAW_DATA_DIR}/cl_schedule.csv"
    schedule.to_csv(out_path, index=False)
    print(f"  saved {out_path}  ({schedule.shape[0]} matches)")

    print("\nDone. Inspect the printed column lists above before moving to 02.")


if __name__ == "__main__":
    main()