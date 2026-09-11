"""
build_fg_attempts_data.py — builds an INDIVIDUAL field goal attempt-level
dataset (one row per attempt, not aggregated to weekly totals), including
exact kick distance, weather, and dome/outdoor status.

This supports testing the core decomposition hypothesis: that FG ACCURACY
(make probability) is more predictable than a kicker's total weekly
points, since distance and conditions plausibly explain real variance
that gets buried once everything is summed into one noisy weekly number.
"""

import pandas as pd
import nfl_data_py as nfl

YEARS = [2021, 2022, 2023, 2024]


def main():
    print(f"Pulling play-by-play data for {YEARS} (this may take a minute)...")
    pbp = nfl.import_pbp_data(YEARS, downcast=True)
    print(f"Loaded {len(pbp)} plays.\n")

    needed_cols = ["play_type", "field_goal_result", "kick_distance",
                   "kicker_player_id", "kicker_player_name", "posteam", "defteam",
                   "season", "week"]
    missing = [c for c in needed_cols if c not in pbp.columns]
    if missing:
        print(f"MISSING EXPECTED COLUMNS: {missing}")
        return

    weather_cols = [c for c in ["roof", "temp", "wind", "surface"] if c in pbp.columns]
    print(f"Weather/venue columns found: {weather_cols}")
    if not weather_cols:
        print("No weather/venue columns found in this pbp data — proceeding without them.\n")

    fg = pbp[pbp["play_type"] == "field_goal"].copy()
    print(f"Found {len(fg)} field goal attempts.\n")

    keep_cols = needed_cols + weather_cols
    fg = fg[keep_cols].copy()

    fg["made"] = (fg["field_goal_result"] == "made").astype(int)
    fg = fg.rename(columns={
        "posteam": "team", "defteam": "opponent_team",
        "kicker_player_id": "player_id", "kicker_player_name": "player_display_name",
    })

    print("Sample rows:")
    print(fg.head(10))
    print(f"\nOverall make rate: {fg['made'].mean():.3f}")

    fg.to_csv("fg_attempts_data.csv", index=False)
    print(f"\nSaved {len(fg)} individual field goal attempts to fg_attempts_data.csv")


if __name__ == "__main__":
    main()
