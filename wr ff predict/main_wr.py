"""
main_wr.py — trains and evaluates a WR-specific fantasy points predictor.

First of a set of per-position models (WR now; RB/TE/QB can follow the
same pattern — copy this file, change POSITION, done).

Once the 2026 season is underway, add 2026 to YEARS below to include
those games in training and in the rolling "recent form" calculations —
no other code changes needed.
"""

from src.data_loader import load_weekly_data
from src.opponent import add_opponent_strength, compute_team_defense_by_position, add_rolling_defense_strength
from src.position_pipeline import build_position_features, predict_next_week_position
from src.model import train_and_evaluate, print_feature_importance

POSITION = "WR"
YEARS = [2021, 2022, 2023, 2024]  # add 2026 once real games exist


def main():
    print(f"Loading data for seasons {YEARS}...")
    df = load_weekly_data(years=YEARS)
    print(f"Loaded {len(df)} player-week rows.\n")

    print("Computing opponent defensive strength...")
    df, opponent_col = add_opponent_strength(df, window=3)
    print(f"Added opponent-strength feature: {opponent_col}\n")

    print(f"Building {POSITION}-specific features...")
    df_model, feature_cols = build_position_features(
        df, POSITION, rolling_window=3, opponent_col=opponent_col
    )
    print(f"{len(df_model)} {POSITION} rows ready for modeling, using {len(feature_cols)} features.\n")

    print("Training models...")
    results, splits = train_and_evaluate(df_model, feature_cols)

    print(f"\n--- {POSITION} Results ---")
    for name, r in results.items():
        print(f"{name:20s} MAE = {r['mae']:.2f} pts   R^2 = {r['r2']:.3f}")

    print_feature_importance(results["random_forest"]["model"], feature_cols)

    # --- Example prediction ---
    def_table = compute_team_defense_by_position(df)
    def_table, _ = add_rolling_defense_strength(def_table, window=3)

    # EDIT with a real WR name + opponent code once you're using real data
    example_player = df_model["player_display_name"].iloc[0]
    example_opponent = df["opponent_team"].iloc[0]

    print("\n--- Example Prediction ---")
    predicted = predict_next_week_position(
        player_display_name=example_player,
        upcoming_opponent=example_opponent,
        df=df,
        def_table=def_table,
        model=results["linear_regression"]["model"],
        feature_cols=feature_cols,
        position=POSITION,
    )
    print(f"Predicted points for {example_player} ({POSITION}) vs {example_opponent}: {predicted:.1f}")


if __name__ == "__main__":
    main()
