"""
build_efficiency_ratings.py — pulls each team's adjusted net efficiency
rating (KenPom-style: offense minus defense, adjusted for opponent
strength) for every team, every season. This applies to ALL teams, not
just tournament qualifiers — fixing a real limitation of the upset
model, which could only use teams good enough to make the bracket.
"""

import os
import time
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
        ratings_api = cbbd.RatingsApi(api_client)

        for season in SEASONS:
            print(f"Fetching efficiency ratings for {season}...")
            try:
                ratings = ratings_api.get_adjusted_efficiency(season=season)
            except ApiException as e:
                print(f"  API error for season {season}: {e}")
                continue

            print(f"  Got {len(ratings)} team ratings.")
            for r in ratings:
                rows.append({
                    "season": r.season,
                    "team": r.team,
                    "offensive_rating": r.offensive_rating,
                    "defensive_rating": r.defensive_rating,
                    "net_rating": r.net_rating,
                })
            time.sleep(1)

    if not rows:
        print("No ratings collected — check API access.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("efficiency_ratings.csv", index=False)
    print(f"\nSaved {len(df)} team-season ratings to efficiency_ratings.csv")
    print(df["net_rating"].describe())


if __name__ == "__main__":
    main()
