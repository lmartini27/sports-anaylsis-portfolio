"""
check_era_shift.py — tests whether the relationship between a team's
prior-season rating and its rating CHANGE has shifted between an
earlier era and the transfer-portal era, which would explain why a
model trained on older data generalizes poorly to recent seasons.
"""

import pandas as pd
from scipy.stats import pearsonr

CONTINUITY_CSV = "roster_continuity.csv"
RATINGS_CSV = "efficiency_ratings.csv"


def main():
    ratings = pd.read_csv(RATINGS_CSV)
    ratings = ratings[ratings["season"] != 2020].copy()
    ratings = ratings[ratings["net_rating"].between(-45, 45)].copy()

    prior_ratings = ratings.copy()
    prior_ratings["season"] = prior_ratings["season"] + 1
    prior_ratings = prior_ratings.rename(columns={"net_rating": "prior_net_rating"})

    df = ratings.merge(
        prior_ratings[["season", "team", "prior_net_rating"]],
        on=["season", "team"], how="inner",
    )
    df["rating_change"] = df["net_rating"] - df["prior_net_rating"]

    early_era = df[df["season"] <= 2020]
    late_era = df[df["season"] >= 2022]

    for label, era_df in [("Early era (through 2020)", early_era),
                           ("Late/portal era (2022+)", late_era)]:
        r, p = pearsonr(era_df["prior_net_rating"], era_df["rating_change"])
        print(f"{label}: n={len(era_df)}, correlation={r:.3f} (p={p:.4f})")
        print(f"  Mean |rating_change|: {era_df['rating_change'].abs().mean():.2f}")
        print(f"  Std of rating_change: {era_df['rating_change'].std():.2f}\n")


if __name__ == "__main__":
    main()
