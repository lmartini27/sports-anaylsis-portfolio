"""
Fantasy Football Success Predictor — main pipeline.

Run this file to load data, build features, train models, and see results:
    python main.py
"""

from src.data_loader import load_weekly_data
from src.opponent import add_opponent_strength, compute_team_defense_by_position, add_rolling_defense_strength
from src.features import build_features
from src.model import train_and_evaluate, print_feature_importance
from src.predict import predict_next_week


def main():
    print("Loading data...")
    df = load_weekly_data(years=[2021, 2022, 2023])
    print(f"Loaded {len(df)} player-week rows.\n")

    print("Computing opponent defensive strength...")
    df, opponent_col = add_opponent_strength(df, window=3)
    print(f"Added opponent-strength feature: {opponent_col}\n")

    print("Building features...")
    df_model, feature_cols = build_features(df, rolling_window=3, extra_feature_cols=[opponent_col])
    print(f"{len(df_model)} rows ready for modeling, using {len(feature_cols)} features.\n")

    print("Training models...")
    results, splits = train_and_evaluate(df_model, feature_cols)

    print("\n--- Results ---")
    for name, r in results.items():
        print(f"{name:20s} MAE = {r['mae']:.2f} pts   R^2 = {r['r2']:.3f}")

    print_feature_importance(results["random_forest"]["model"], feature_cols)

    # --- Example: predict a specific player's next game ---
    # Rebuild the defense lookup table (predict.py needs the un-merged
    # version so it can look up ANY team, not just ones already in df)
    def_table = compute_team_defense_by_position(df)
    def_table, _ = add_rolling_defense_strength(def_table, window=3)

    print("\n--- Example Prediction ---")
    # EDIT THESE TWO LINES with a real player name and opponent team code
    # from your own dataset (see the README for how to look these up)
    example_player = df["player_display_name"].iloc[0]
    example_opponent = df["opponent_team"].iloc[0]

    predicted = predict_next_week(
        player_display_name=example_player,
        upcoming_opponent=example_opponent,
        df=df,
        def_table=def_table,
        model=results["linear_regression"]["model"],
        feature_cols=feature_cols,
    )
    print(f"Predicted points for {example_player} vs {example_opponent}: {predicted:.1f}")


if __name__ == "__main__":
    main()
