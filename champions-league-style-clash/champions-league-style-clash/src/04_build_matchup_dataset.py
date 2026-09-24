"""
04_build_matchup_dataset.py

Joins the CL schedule/results with each team's style cluster (from the
*domestic league season immediately preceding* the CL season, since
that's the most recent style read we have on a team going into Europe).

Produces one row per CL match with:
  home_team, away_team, home_style, away_style, home_goals, away_goals,
  result (H/D/A from the home team's perspective)

NAME MATCHING CAVEAT: FBref sometimes spells a club's name slightly
differently between its domestic-league page and its Champions League
page (e.g. "Paris S-G" vs "Paris Saint-Germain"). This script does an
exact-match join and PRINTS any CL teams it couldn't find a style for --
add those to TEAM_NAME_FIXES below and re-run rather than silently
dropping matches.
"""

import sys
import pandas as pd

sys.path.append("..")
import config

# Map CL schedule name -> domestic league name, only needed for teams
# that print as "unmatched" when you first run this.
TEAM_NAME_FIXES = {
    # "Paris S-G": "Paris Saint-Germain",
}


def prev_season(season: str) -> str:
    """'2023-2024' -> '2022-2023' -- used to look up a team's style
    from the domestic season before the CL campaign in question."""
    start, end = season.split("-")
    return f"{int(start) - 1}-{int(end) - 1}"


def main():
    schedule = pd.read_csv(f"{config.RAW_DATA_DIR}/cl_schedule.csv")
    styles = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/team_style_clusters.csv")

    # normalize name spelling using the manual fix map
    schedule["home_team_clean"] = schedule["home_team"].replace(TEAM_NAME_FIXES)
    schedule["away_team_clean"] = schedule["away_team"].replace(TEAM_NAME_FIXES)

    style_lookup = styles.set_index(["team", "season"])["style_name"].to_dict()

    def lookup_style(team, cl_season):
        # try same-season domestic style first, fall back to prior season
        for season in (cl_season, prev_season(cl_season)):
            key = (team, season)
            if key in style_lookup:
                return style_lookup[key]
        return None

    schedule["home_style"] = schedule.apply(
        lambda r: lookup_style(r["home_team_clean"], r["season"]), axis=1
    )
    schedule["away_style"] = schedule.apply(
        lambda r: lookup_style(r["away_team_clean"], r["season"]), axis=1
    )

    unmatched_home = schedule.loc[schedule["home_style"].isna(), "home_team"].unique()
    unmatched_away = schedule.loc[schedule["away_style"].isna(), "away_team"].unique()
    unmatched = sorted(set(unmatched_home) | set(unmatched_away))
    if unmatched:
        print(f"[!] {len(unmatched)} teams had no style match (likely name-spelling "
              f"mismatches, or clubs outside the 5 tracked leagues):")
        for t in unmatched:
            print(f"    - {t}")
        print("Add fixes to TEAM_NAME_FIXES in this file and re-run if these are "
              "spelling issues. Rows without a style on both sides get dropped below.")

    matched = schedule.dropna(subset=["home_style", "away_style"]).copy()

    # result from home team's perspective, assuming schedule has goal columns
    goal_cols = [c for c in matched.columns if "goal" in c.lower() or c.lower() in ("hg", "ag")]
    if "home_goals" not in matched.columns and goal_cols:
        print(f"Note: expected 'home_goals'/'away_goals' columns; found {goal_cols} "
              f"instead. Rename them to home_goals/away_goals above if needed.")

    matched["result"] = matched.apply(
        lambda r: "H" if r.get("home_goals", 0) > r.get("away_goals", 0)
        else ("A" if r.get("home_goals", 0) < r.get("away_goals", 0) else "D"),
        axis=1,
    )
    matched["style_matchup"] = matched["home_style"] + " vs " + matched["away_style"]

    keep_cols = ["season", "home_team", "away_team", "home_goals", "away_goals",
                 "result", "home_style", "away_style", "style_matchup"]
    keep_cols = [c for c in keep_cols if c in matched.columns]
    out = matched[keep_cols]

    out_path = f"{config.PROCESSED_DATA_DIR}/cl_matchup_dataset.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  ({out.shape[0]} matches with both sides' style, "
          f"out of {schedule.shape[0]} total CL matches pulled)")


if __name__ == "__main__":
    main()
