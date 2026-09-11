"""
main_accuracy.py — trains and evaluates the field goal accuracy model
(predicts make/miss per individual attempt, not weekly point totals).

Run build_fg_attempts_data.py first to generate fg_attempts_data.csv.
"""

from src.accuracy_model import (
    load_fg_attempts_data, prepare_accuracy_features,
    train_accuracy_model, print_accuracy_results,
)

CSV_PATH = "fg_attempts_data.csv"


def main():
    print("Loading field goal attempt data...")
    df = load_fg_attempts_data(CSV_PATH)
    print(f"Loaded {len(df)} individual attempts.\n")

    print("Preparing features...")
    df_model, feature_cols = prepare_accuracy_features(df)
    print(f"{len(df_model)} attempts ready, using features: {feature_cols}\n")

    print("Training accuracy model...")
    results, splits = train_accuracy_model(df_model, feature_cols)

    print("\n--- Field Goal Accuracy Model Results ---")
    print_accuracy_results(results, feature_cols)


if __name__ == "__main__":
    main()
