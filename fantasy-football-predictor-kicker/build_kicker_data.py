"""
build_kicker_data.py — builds a weekly kicker stats + fantasy points
dataset from play-by-play data, since your installed nfl_data_py doesn't
include kickers in its weekly stats table (confirmed by your last run).

Kicker fantasy points come from field goals (weighted by distance) and
extra points, which only show up in the play-by-play (PBP) data — every
single play of every game, not a weekly summary. This script pulls that,
filters to field goal and extra point plays, and aggregates them into
one row per kicker per week — the same shape as the weekly data used for
every other position, so it can plug into the rest of the pipeline.

This will take noticeably longer to download than the other scripts —
play-by-play data is much larger than weekly summaries.
"""

import pandas as pd
import nfl_data_py as nfl

YEARS = [2021, 2022, 2023, 2024]

# Standard kicker fantasy scoring — EDIT to match your league's actual
# rules if they differ. Leagues vary a lot on this, especially whether
# missed field goals are penalized.
SCORING = {
    "fg_0_39": 3,
    "fg_40_49": 4,
    "fg_50_plus": 5,
    "pat_made": 1,
    "fg_missed": 0,   # set to -1 if your league penalizes misses
}


def fg_bucket(distance):
    if distance < 40:
        return "fg_0_39"
    elif distance < 50:
        return "fg_40_49"
    else:
        return "fg_50_plus"


def main():
    print(f"Pulling play-by-play data for {YEARS} (this may take a minute)...")
    pbp = nfl.import_pbp_data(YEARS, downcast=True)
    print(f"Loaded {len(pbp)} plays.\n")

    needed_cols = [
        "play_type", "field_goal_result", "kick_distance", "extra_point_result",
        "kicker_player_id", "kicker_player_name", "posteam", "defteam", "season", "week",
    ]
    missing = [c for c in needed_cols if c not in pbp.columns]
    if missing:
        print(f"MISSING EXPECTED COLUMNS: {missing}")
        print("Columns containing 'kick', 'field_goal', or 'extra_point':")
        print([c for c in pbp.columns if any(k in c for k in ["kick", "field_goal", "extra_point"])])
        return

    fg_plays = pbp[pbp["play_type"] == "field_goal"].copy()
    pat_plays = pbp[pbp["play_type"] == "extra_point"].copy()
    print(f"Found {len(fg_plays)} field goal plays and {len(pat_plays)} extra point plays.\n")

    if fg_plays.empty and pat_plays.empty:
        print("No field goal or extra point plays found — something's off with the filter.")
        print("Unique play_type values:", pbp["play_type"].unique())
        return

    fg_plays["fg_bucket"] = fg_plays["kick_distance"].apply(fg_bucket)
    fg_plays["fg_made"] = (fg_plays["field_goal_result"] == "made").astype(int)

    fg_agg = (
        fg_plays.groupby(["season", "week", "posteam", "kicker_player_id", "kicker_player_name", "fg_bucket"])
        .agg(attempts=("fg_made", "count"), made=("fg_made", "sum"))
        .reset_index()
    )
    fg_pivot = fg_agg.pivot_table(
        index=["season", "week", "posteam", "kicker_player_id", "kicker_player_name"],
        columns="fg_bucket",
        values=["attempts", "made"],
        fill_value=0,
    )
    fg_pivot.columns = [f"{stat}_{bucket}" for stat, bucket in fg_pivot.columns]
    fg_pivot = fg_pivot.reset_index()

    pat_plays["pat_made"] = (pat_plays["extra_point_result"] == "good").astype(int)
    pat_agg = (
        pat_plays.groupby(["season", "week", "posteam", "kicker_player_id", "kicker_player_name"])
        .agg(pat_attempts=("pat_made", "count"), pat_made=("pat_made", "sum"))
        .reset_index()
    )

    kicker_df = pd.merge(
        fg_pivot, pat_agg,
        on=["season", "week", "posteam", "kicker_player_id", "kicker_player_name"],
        how="outer",
    ).fillna(0)

    for bucket in ["fg_0_39", "fg_40_49", "fg_50_plus"]:
        for stat in ["attempts", "made"]:
            col = f"{stat}_{bucket}"
            if col not in kicker_df.columns:
                kicker_df[col] = 0

    kicker_df["fantasy_points_ppr"] = (
        kicker_df["made_fg_0_39"] * SCORING["fg_0_39"]
        + kicker_df["made_fg_40_49"] * SCORING["fg_40_49"]
        + kicker_df["made_fg_50_plus"] * SCORING["fg_50_plus"]
        + kicker_df["pat_made"] * SCORING["pat_made"]
        + (kicker_df["attempts_fg_0_39"] - kicker_df["made_fg_0_39"]
           + kicker_df["attempts_fg_40_49"] - kicker_df["made_fg_40_49"]
           + kicker_df["attempts_fg_50_plus"] - kicker_df["made_fg_50_plus"]) * SCORING["fg_missed"]
    )

    kicker_df = kicker_df.rename(columns={
        "posteam": "recent_team",
        "kicker_player_id": "player_id",
        "kicker_player_name": "player_display_name",
    })
    kicker_df["position"] = "K"

    opp_lookup = (
        pbp[pbp["play_type"].isin(["field_goal", "extra_point"])]
        .groupby(["season", "week", "posteam"])["defteam"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"posteam": "recent_team", "defteam": "opponent_team"})
    )
    kicker_df = kicker_df.merge(opp_lookup, on=["season", "week", "recent_team"], how="left")

    print(f"Built {len(kicker_df)} kicker-week rows.\n")
    print(kicker_df.head(10))

    kicker_df.to_csv("kicker_weekly_data.csv", index=False)
    print("\nSaved to kicker_weekly_data.csv — this is what the kicker model will train on.")


if __name__ == "__main__":
    main()
