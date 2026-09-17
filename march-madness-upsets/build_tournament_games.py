"""
build_tournament_games.py — pulls actual NCAA tournament game results
(seeds + winner) for every season, and labels each game as an upset
(the higher seed NUMBER — i.e. the worse-ranked team — winning) or not.

Seed 1 is the best team, so "upset" = the team with the WORSE (higher
number) seed won.
"""

import os
import pandas as pd
import cbbd
from cbbd.rest import ApiException

SEASONS = list(range(2015, 2026))


def main():
    api_key = os.environ.get("CBBD_API_KEY")
    if not api_key:
        print("ERROR: Set the CBBD_API_KEY environment variable first.")
        return

    configuration = cbbd.Configuration(access_token=api_key)

    rows = []
    with cbbd.ApiClient(configuration) as api_client:
        games_api = cbbd.GamesApi(api_client)

        for season in SEASONS:
            print(f"Fetching NCAA tournament games for {season}...")
            try:
                games = games_api.get_games(season=season, tournament="NCAA")
            except ApiException as e:
                print(f"  API error for season {season}: {e}")
                continue

            print(f"  Found {len(games)} tournament games.")

            for g in games:
                if g.home_seed is None or g.away_seed is None:
                    continue
                if g.home_seed == g.away_seed:
                    continue  # same-seed play-in games — "upset" isn't meaningful here
                if g.home_winner is None:
                    continue

                if g.home_seed < g.away_seed:
                    favorite_team, underdog_team = g.home_team, g.away_team
                    favorite_seed, underdog_seed = g.home_seed, g.away_seed
                    favorite_won = g.home_winner
                else:
                    favorite_team, underdog_team = g.away_team, g.home_team
                    favorite_seed, underdog_seed = g.away_seed, g.home_seed
                    favorite_won = g.away_winner

                rows.append({
                    "season": season,
                    "game_id": g.id,
                    "home_team": g.home_team, "away_team": g.away_team,
                    "home_seed": g.home_seed, "away_seed": g.away_seed,
                    "home_points": g.home_points, "away_points": g.away_points,
                    "favorite_team": favorite_team, "underdog_team": underdog_team,
                    "favorite_seed": favorite_seed, "underdog_seed": underdog_seed,
                    "upset": 0 if favorite_won else 1,
                })

    if not rows:
        print("No tournament games found — check the 'tournament' filter value or season range.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("tournament_games.csv", index=False)
    print(f"\nSaved {len(df)} tournament games to tournament_games.csv")
    print(f"Overall upset rate: {df['upset'].mean():.1%}")

    print("\nUpset rate by seed matchup (favorite seed vs underdog seed), most common matchups:")
    matchup_stats = (
        df.groupby(["favorite_seed", "underdog_seed"])["upset"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
    )
    print(matchup_stats.head(10))


if __name__ == "__main__":
    main()
