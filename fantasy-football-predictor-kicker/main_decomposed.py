"""
main_decomposed.py — the full decomposed kicker model.

Instead of regressing directly on the noisy combined weekly point total,
this predicts FG/PAT ATTEMPT VOLUME (separate regressions per distance
bucket + PATs) and ACCURACY (make probability per attempt, from
main_accuracy.py's model) separately, then combines them into one
expected-points number via:

    E[points] = sum over buckets of (E[attempts in bucket] * P(make) * points)
                + E[PAT attempts] * PAT make rate

This tests whether decomposing the problem reveals real predictive power
that gets hidden when regressing on the combined total directly.
"""

import pandas as pd

from src.kicker_pipeline import (
    load_kicker_data, build_kicker_features, load_redzone_data, add_redzone_features,
    load_vegas_data,
)
from src.model import train_and_evaluate
from src.accuracy_model import (
    load_fg_attempts_data, prepare_accuracy_features, train_accuracy_model,
)

CSV_PATH = "kicker_weekly_data.csv"
REDZONE_CSV_PATH = "redzone_team_data.csv"
VEGAS_CSV_PATH = "vegas_team_totals.csv"
FG_ATTEMPTS_CSV_PATH = "fg_attempts_data.csv"

BUCKET_DISTANCES = {
    "attempts_fg_0_39": 30,
    "attempts_fg_40_49": 45,
    "attempts_fg_50_plus": 55,
}
BUCKET_POINTS = {
    "attempts_fg_0_39": 3,
    "attempts_fg_40_49": 4,
    "attempts_fg_50_plus": 5,
}


def train_volume_models(df_model, feature_cols):
    """Trains a separate regression for each attempt-volume target."""
    volume_models = {}
    for target in list(BUCKET_DISTANCES.keys()) + ["pat_attempts"]:
        results, _ = train_and_evaluate(df_model, feature_cols, target_col=target)
        volume_models[target] = results
        print(f"\n--- Volume model: {target} ---")
        for name, r in results.items():
            print(f"{name:20s} MAE = {r['mae']:.2f}   R^2 = {r['r2']:.3f}")
    return volume_models


def compute_expected_points(volume_preds, accuracy_model, is_dome, wind, temp, pat_make_rate):
    """
    Combines predicted attempt VOLUME per bucket with the accuracy
    model's predicted MAKE PROBABILITY at each bucket's representative
    distance, producing one final expected fantasy points number.
    """
    total_points = 0.0
    for bucket, distance in BUCKET_DISTANCES.items():
        X = pd.DataFrame([{
            "kick_distance": distance,
            "is_dome": is_dome,
            "wind_filled": 0 if is_dome else wind,
            "temp_filled": 70 if is_dome else temp,
        }])
        make_prob = accuracy_model.predict_proba(X)[0, 1]
        total_points += max(0, volume_preds[bucket]) * make_prob * BUCKET_POINTS[bucket]

    total_points += max(0, volume_preds["pat_attempts"]) * pat_make_rate * 1
    return total_points


def main():
    print("Loading kicker data...")
    df = load_kicker_data(CSV_PATH)

    print("Loading red zone + Vegas data...")
    redzone_raw = load_redzone_data(REDZONE_CSV_PATH)
    df, redzone_rolling, own_col, opp_rz_col = add_redzone_features(df, redzone_raw, rolling_window=3)
    vegas_df = load_vegas_data(VEGAS_CSV_PATH)

    print("Building shared feature set...")
    df_model, feature_cols = build_kicker_features(
        df, rolling_window=3, extra_feature_cols=[own_col, opp_rz_col], vegas_df=vegas_df
    )
    print(f"{len(df_model)} rows, {len(feature_cols)} features.\n")

    print("Training volume models (one per FG bucket + PAT)...")
    volume_models = train_volume_models(df_model, feature_cols)

    print("\nLoading FG attempt data and training accuracy model...")
    fg_df = load_fg_attempts_data(FG_ATTEMPTS_CSV_PATH)
    fg_model_df, acc_feature_cols = prepare_accuracy_features(fg_df)
    acc_results, _ = train_accuracy_model(fg_model_df, acc_feature_cols)
    accuracy_model = acc_results["model"]
    print(f"Accuracy model ROC-AUC: {acc_results['roc_auc']:.3f}")

    pat_make_rate = df["pat_made"].sum() / df["pat_attempts"].sum()
    print(f"League-average PAT make rate: {pat_make_rate:.3f}\n")

    # --- Backtest: compare the combined model's prediction to reality
    # for several historical rows, using each game's ACTUAL conditions ---
    game_conditions = fg_df.drop_duplicates(subset=["season", "week", "team"])[
        ["season", "week", "team", "roof", "wind", "temp"]
    ]

    print("--- Backtest: decomposed model vs. actual outcomes (5 examples) ---")
    for i in range(min(5, len(df_model))):
        example_row = df_model.iloc[i]
        volume_preds = {
            target: volume_models[target]["linear_regression"]["model"].predict(
                df_model.iloc[[i]][feature_cols]
            )[0]
            for target in list(BUCKET_DISTANCES.keys()) + ["pat_attempts"]
        }

        match = game_conditions[
            (game_conditions["season"] == example_row["season"]) &
            (game_conditions["week"] == example_row["week"]) &
            (game_conditions["team"] == example_row["recent_team"])
        ]
        if not match.empty:
            roof_val = match["roof"].iloc[0]
            is_dome = 1 if roof_val in ["dome", "closed"] else 0
            wind = match["wind"].iloc[0] if pd.notna(match["wind"].iloc[0]) else 0
            temp = match["temp"].iloc[0] if pd.notna(match["temp"].iloc[0]) else 70
        else:
            is_dome, wind, temp = 0, 5, 60

        predicted_points = compute_expected_points(
            volume_preds, accuracy_model, is_dome=is_dome, wind=wind, temp=temp,
            pat_make_rate=pat_make_rate,
        )
        actual_points = example_row["fantasy_points_ppr"]
        print(f"{example_row['player_display_name']:20s} "
              f"season {int(example_row['season'])} week {int(example_row['week']):<3} "
              f"predicted={predicted_points:5.1f}  actual={actual_points:5.1f}")


if __name__ == "__main__":
    main()
