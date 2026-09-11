"""
Fantasy Football Success Predictor — main pipeline.

Run this file to load data, build features, train models, and see results:
    python main.py
"""

from src.data_loader import load_weekly_data
from src.features import build_features
from src.model import train_and_evaluate, print_feature_importance


def main():
    print("Loading data...")
    df = load_weekly_data(years=[2021, 2022, 2023])
    print(f"Loaded {len(df)} player-week rows.\n")

    print("Building features...")
    df_model, feature_cols = build_features(df, rolling_window=3)
    print(f"{len(df_model)} rows ready for modeling, using {len(feature_cols)} features.\n")

    print("Training models...")
    results, splits = train_and_evaluate(df_model, feature_cols)

    print("\n--- Results ---")
    for name, r in results.items():
        print(f"{name:20s} MAE = {r['mae']:.2f} pts   R^2 = {r['r2']:.3f}")

    print_feature_importance(results["random_forest"]["model"], feature_cols)


if __name__ == "__main__":
    main()
