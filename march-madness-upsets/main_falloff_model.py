"""
main_falloff_model.py — does roster continuity predict how much a
team's quality rises or falls from one season to the next?

Unlike the upset model (which only used tournament teams), this uses
EVERY team's year-over-year net efficiency rating change — a bigger,
less biased sample, and a more direct test: continuity's relationship
to team quality itself, rather than filtered through seed and
single-elimination game variance.

Controls for regression-to-the-mean (bad teams tend to improve, good
teams tend to decline, regardless of continuity) by including the
PRIOR season's rating as a feature — otherwise continuity could look
predictive just by proxying for "was already a good/bad team."
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

CONTINUITY_CSV = "roster_continuity.csv"
RATINGS_CSV = "efficiency_ratings.csv"

PORTAL_ERA_START = 2022  # current season must be in the portal era; using an
                          # earlier season's rating as the "prior" baseline is fine
TEST_SEASONS = [2024, 2025]


def evaluate(train, test, features, label):
    model = LinearRegression()
    model.fit(train[features], train["rating_change"])
    preds = model.predict(test[features])
    mae = mean_absolute_error(test["rating_change"], preds)
    r2 = r2_score(test["rating_change"], preds)
    print(f"--- {label} ---")
    print(f"MAE: {mae:.2f}   R^2: {r2:.3f}")
    for name, coef in zip(features, model.coef_):
        print(f"  {name}: {coef:+.4f}")
    print()
    return r2


def main():
    continuity = pd.read_csv(CONTINUITY_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    print(f"Loaded {len(continuity)} continuity rows, {len(ratings)} rating rows.\n")

    # 2020 was the COVID-shortened season (tournament cancelled, incomplete
    # schedules) and produces wildly unstable ratings — exclude it entirely,
    # both as a "current" season and as a "prior" season for 2021.
    before_2020_filter = len(ratings)
    ratings = ratings[ratings["season"] != 2020].copy()
    print(f"Dropped {before_2020_filter - len(ratings)} rows from the unreliable 2020 season.")

    # General safety net for any other unrealistic values
    before_range_filter = len(ratings)
    ratings = ratings[ratings["net_rating"].between(-45, 45)].copy()
    print(f"Dropped {before_range_filter - len(ratings)} additional rows outside a plausible rating range.\n")

    # Prior season's rating, for each (season, team)
    prior_ratings = ratings.copy()
    prior_ratings["season"] = prior_ratings["season"] + 1
    prior_ratings = prior_ratings.rename(columns={"net_rating": "prior_net_rating"})

    df = ratings.merge(
        prior_ratings[["season", "team", "prior_net_rating"]],
        on=["season", "team"], how="inner",
    )
    df = df.merge(
        continuity[["season", "team", "continuity_pct"]],
        on=["season", "team"], how="inner",
    )

    df["rating_change"] = df["net_rating"] - df["prior_net_rating"]

    before_era_filter = len(df)
    df = df[df["season"] >= PORTAL_ERA_START].copy()
    print(f"Restricting to portal era (season >= {PORTAL_ERA_START}): "
          f"{len(df)} of {before_era_filter} rows kept.\n")

    print(f"{len(df)} team-seasons have both continuity and a prior-season rating to compare.\n")

    print("Mean rating_change by season (checking for a level shift, not just a slope issue):")
    print(df.groupby("season")["rating_change"].agg(["mean", "std", "count"]))
    print()

    train = df[~df["season"].isin(TEST_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    print(f"Train: {len(train)}   Test: {len(test)}\n")

    if len(test) == 0:
        print("Not enough data — check season ranges.")
        return

    r2_baseline = evaluate(train, test, ["prior_net_rating"],
                            "Baseline: prior rating only (regression-to-the-mean)")
    r2_full = evaluate(train, test, ["prior_net_rating", "continuity_pct"],
                        "Full: prior rating + continuity")

    print("--- The key question ---")
    print(f"R^2 improvement from adding continuity: {r2_full - r2_baseline:+.3f}")


if __name__ == "__main__":
    main()
