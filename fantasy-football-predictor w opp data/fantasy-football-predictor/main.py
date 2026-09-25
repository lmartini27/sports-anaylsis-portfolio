"""
Fantasy Football Success Predictor -- main pipeline.

    python main.py

Predicts each player's PPR fantasy points in their next game, using their
recent form and the next opponent's recent defense against their position.
Evaluated walk-forward by season against a naive "recent average" baseline,
with and without the opponent feature.
"""

from src.data_loader import load_weekly_data
from src.features import add_next_game, build_features
from src.opponent import add_opponent_strength
from src.model import train_and_evaluate, print_results, print_feature_importance

YEARS = [2021, 2022, 2023]
WINDOW = 3


def main():
    print("Loading data...")
    df = load_weekly_data(years=YEARS)
    print(f"Loaded {len(df)} player-week rows.")

    df = add_next_game(df)
    df, opp_col = add_opponent_strength(df, window=WINDOW)
    df_model, all_features = build_features(df, rolling_window=WINDOW, extra_feature_cols=[opp_col])
    base_features = [c for c in all_features if c != opp_col]
    baseline_col = f"fantasy_points_ppr_avg_last{WINDOW}"
    print(f"{len(df_model)} rows ready for modeling ({len(all_features)} features).")

    res_without, _ = train_and_evaluate(df_model, base_features, baseline_col)
    res_with, rf = train_and_evaluate(df_model, all_features, baseline_col)

    print_results(res_without, "WITHOUT opponent feature")
    print_results(res_with, "WITH next-opponent feature")

    gain = res_with["linear_regression"]["r2"] - res_without["linear_regression"]["r2"]
    print(f"\nOpponent feature R^2 change (Linear Regression): {gain:+.3f}")
    print_feature_importance(rf, all_features)


if __name__ == "__main__":
    main()
