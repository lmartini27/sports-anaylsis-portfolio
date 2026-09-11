"""
build_redzone_data.py — computes each team's red zone tendency: how often
their offense reaches the red zone, and what fraction of those trips end
in a touchdown vs. a field goal — plus the same numbers from the
defense's side (how many red zone trips they allow, and what fraction
they let become touchdowns).

This tests the hypothesis that a kicker's fantasy points depend more on
team-level red zone tendencies than on the kicker's own recent box score
stats (which showed ~zero predictive power on their own).

Known simplification: a defensive/special-teams touchdown that occurs
during an offense's drive (e.g. a pick-six) would be misclassified as
that offense's own touchdown. This is rare enough not to meaningfully
skew results, but it's a real limitation worth noting in any write-up.
"""

import pandas as pd
import nfl_data_py as nfl

YEARS = [2021, 2022, 2023, 2024]
RED_ZONE_YARDLINE = 20  # yards from opponent's end zone


def main():
    print(f"Pulling play-by-play data for {YEARS} (this may take a minute)...")
    pbp = nfl.import_pbp_data(YEARS, downcast=True)
    print(f"Loaded {len(pbp)} plays.\n")

    needed_cols = ["game_id", "drive", "posteam", "defteam", "yardline_100",
                   "touchdown", "field_goal_result", "season", "week"]
    missing = [c for c in needed_cols if c not in pbp.columns]
    if missing:
        print(f"MISSING EXPECTED COLUMNS: {missing}")
        print("Columns containing 'drive', 'yardline', or 'touchdown':")
        print([c for c in pbp.columns if any(k in c for k in ["drive", "yardline", "touchdown"])])
        return

    plays = pbp.dropna(subset=["drive", "posteam"]).copy()
    drive_groups = plays.groupby(["game_id", "drive"])

    drive_summary = drive_groups.agg(
        season=("season", "first"),
        week=("week", "first"),
        posteam=("posteam", "first"),
        defteam=("defteam", "first"),
        reached_red_zone=("yardline_100", lambda x: (x <= RED_ZONE_YARDLINE).any()),
        had_touchdown=("touchdown", lambda x: (x == 1).any()),
        had_made_fg=("field_goal_result", lambda x: (x == "made").any()),
    ).reset_index()

    red_zone_drives = drive_summary[drive_summary["reached_red_zone"]].copy()
    print(f"Found {len(red_zone_drives)} drives that reached the red zone "
          f"(out of {len(drive_summary)} total drives).\n")

    if red_zone_drives.empty:
        print("No red zone drives found — something's off with the yardline filter.")
        return

    def outcome(row):
        if row["had_touchdown"]:
            return "touchdown"
        elif row["had_made_fg"]:
            return "field_goal"
        else:
            return "other"

    red_zone_drives["outcome"] = red_zone_drives.apply(outcome, axis=1)

    # Offense side: how does THIS team perform when THEY reach the red zone?
    offense_agg = (
        red_zone_drives.groupby(["season", "week", "posteam"])
        .agg(
            redzone_trips=("outcome", "count"),
            redzone_tds=("outcome", lambda x: (x == "touchdown").sum()),
            redzone_fgs=("outcome", lambda x: (x == "field_goal").sum()),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defense side: how does THIS team's DEFENSE perform when the OPPONENT reaches the red zone?
    defense_agg = (
        red_zone_drives.groupby(["season", "week", "defteam"])
        .agg(
            redzone_trips_allowed=("outcome", "count"),
            redzone_tds_allowed=("outcome", lambda x: (x == "touchdown").sum()),
            redzone_fgs_forced=("outcome", lambda x: (x == "field_goal").sum()),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    redzone_df = pd.merge(offense_agg, defense_agg, on=["season", "week", "team"], how="outer").fillna(0)

    print(f"Built {len(redzone_df)} team-week red zone rows.\n")
    print(redzone_df.head(10))

    redzone_df.to_csv("redzone_team_data.csv", index=False)
    print("\nSaved to redzone_team_data.csv")


if __name__ == "__main__":
    main()
