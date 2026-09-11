"""
main_k.py — trains and evaluates the kicker-specific fantasy predictor,
including red zone tendency features and Vegas-implied team totals.

Run build_kicker_data.py, build_redzone_data.py, AND build_vegas_data.py
first to generate the CSVs this script needs, then run this.
"""

from src.kicker_pipeline import (
    load_kicker_data, build_kicker_features, compute_defense_fg_pressure,
    predict_next_week_kicker, load_redzone_data, add_redzone_features,
    load_vegas_data,
)
from src.model import train_and_evaluate, print_feature_importance

CSV_PATH = "kicker_weekly_data.csv"
REDZONE_CSV_PATH = "redzone_team_data.csv"
VEGAS_CSV_PATH = "vegas_team_totals.csv"


def main():
    print("Loading kicker data...")
    df = load_kicker_data(CSV_PATH)
    print(f"Loaded {len(df)} kicker-week rows.\n")

    print("Loading red zone data...")
    redzone_raw = load_redzone_data(REDZONE_CSV_PATH)
    df, redzone_rolling, own_col, opp_rz_col = add_redzone_features(df, redzone_raw, rolling_window=3)
    print(f"Added red zone features: {own_col}, {opp_rz_col}\n")

    print("Loading Vegas data...")
    vegas_df = load_vegas_data(VEGAS_CSV_PATH)
    print(f"Loaded {len(vegas_df)} team-week Vegas lines.\n")

    print("Building features...")
    df_model, feature_cols = build_kicker_features(
        df, rolling_window=3, extra_feature_cols=[own_col, opp_rz_col], vegas_df=vegas_df
    )
    print(f"{len(df_model)} rows ready for modeling, using {len(feature_cols)} features.\n")

    print("Training models...")
    results, splits = train_and_evaluate(df_model, feature_cols)

    print("\n--- Kicker Results (Red Zone + Vegas Features) ---")
    for name, r in results.items():
        print(f"{name:20s} MAE = {r['mae']:.2f} pts   R^2 = {r['r2']:.3f}")

    print_feature_importance(results["random_forest"]["model"], feature_cols)

    # --- Example prediction ---
    defense_df, opp_col = compute_defense_fg_pressure(df, rolling_window=3)
    example_row = df_model.iloc[0]
    example_player = example_row["player_display_name"]
    example_opponent = example_row["opponent_team"]
    example_season = example_row["season"]
    example_week = example_row["week"]
    example_team = example_row["recent_team"]

    # For this backtest demo, look up the ACTUAL historical Vegas line for
    # this specific game. For a real future game, you'd pass in today's
    # actual sportsbook line instead, since it won't exist in this CSV yet.
    vegas_match = vegas_df[
        (vegas_df["season"] == example_season) & (vegas_df["week"] == example_week) &
        (vegas_df["team"] == example_team)
    ]
    example_implied_total = vegas_match["implied_team_total"].iloc[0] if not vegas_match.empty else None

    print("\n--- Example Prediction ---")
    predicted = predict_next_week_kicker(
        player_display_name=example_player,
        upcoming_opponent=example_opponent,
        df=df,
        defense_df=defense_df,
        opp_col=opp_col,
        model=results["linear_regression"]["model"],
        feature_cols=feature_cols,
        redzone_df=redzone_rolling,
        redzone_own_col=own_col,
        redzone_opp_col=opp_rz_col,
        implied_team_total=example_implied_total,
    )
    print(f"Predicted points for {example_player} vs {example_opponent}: {predicted:.1f}")


if __name__ == "__main__":
    main()
