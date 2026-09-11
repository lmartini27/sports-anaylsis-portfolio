"""
build_vegas_data.py — pulls Vegas point spreads and totals per game, and
computes each team's IMPLIED point total (how many points the betting
market expects that specific team to score). This is the single most
useful external predictor in real sports betting models, since the
market price already incorporates weather, injuries, coaching
tendencies, and everything else at once.

A higher implied team total generally means more scoring drives, which
means more field goal AND extra point opportunities for that team's kicker.

Unlike the rolling-average features elsewhere in this project, this is
NOT something to average over past games — a Vegas line is set fresh for
each specific upcoming game, so we use it directly for that exact
(season, week, team), not as a rolling history.
"""

import pandas as pd
import nfl_data_py as nfl

YEARS = [2021, 2022, 2023, 2024]


def main():
    print(f"Pulling schedule/odds data for {YEARS}...")
    sched = nfl.import_schedules(YEARS)
    print(f"Loaded {len(sched)} games.\n")

    needed_cols = ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    missing = [c for c in needed_cols if c not in sched.columns]
    if missing:
        print(f"MISSING EXPECTED COLUMNS: {missing}")
        print("Columns containing 'spread', 'total', 'line', 'odds', or 'moneyline':")
        print([c for c in sched.columns if any(k in c.lower() for k in
                                                 ["spread", "total", "line", "odds", "moneyline"])])
        return

    sched = sched[needed_cols].copy()

    # spread_line convention (nflverse): from the home team's perspective,
    # negative means the home team is favored.
    sched["home_implied_total"] = (sched["total_line"] - sched["spread_line"]) / 2
    sched["away_implied_total"] = (sched["total_line"] + sched["spread_line"]) / 2

    print("Sample rows — sanity check these: the favored team should show the higher implied total.")
    print(sched.head(10))

    home_rows = sched[["season", "week", "home_team", "home_implied_total"]].rename(
        columns={"home_team": "team", "home_implied_total": "implied_team_total"}
    )
    away_rows = sched[["season", "week", "away_team", "away_implied_total"]].rename(
        columns={"away_team": "team", "away_implied_total": "implied_team_total"}
    )
    team_totals = pd.concat([home_rows, away_rows], ignore_index=True)

    team_totals.to_csv("vegas_team_totals.csv", index=False)
    print(f"\nSaved {len(team_totals)} team-week implied totals to vegas_team_totals.csv")


if __name__ == "__main__":
    main()
