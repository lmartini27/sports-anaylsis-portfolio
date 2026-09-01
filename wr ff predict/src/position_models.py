"""
Per-position modeling: instead of one combined model across all
positions, train a SEPARATE model for each position (QB, RB, WR, TE).

See EXTENSION_ROADMAP.md for the full explanation of why this matters.
Fill in the TODOs below — this is meant to be written by you, not copied.
"""

from src.model import train_and_evaluate


def train_position_models(df_model, feature_cols, position_col_prefix="pos_"):
    """
    Trains a separate model (Linear Regression + Random Forest, via
    train_and_evaluate) for each position, using only that position's rows.

    Returns a dict like:
        { "QB": <results dict from train_and_evaluate>, "RB": {...}, ... }
    """
    position_cols = [c for c in feature_cols if c.startswith(position_col_prefix)]
    results_by_position = {}

    for pos_col in position_cols:
        position_name = pos_col.replace(position_col_prefix, "")
        print(f"Training model for position: {position_name}")

        # TODO 1: filter df_model down to only rows where this position's
        # one-hot column equals 1.
        # Hint: df_model[df_model[pos_col] == 1]
        position_df = None

        # TODO 2: decide which feature columns to pass in for this subset.
        # Should the one-hot position columns still be included, now that
        # every row in position_df is the same position? Think about
        # whether a column that's the same value (1) for every row can
        # possibly help a model make different predictions.
        position_feature_cols = None

        # TODO 3: call train_and_evaluate(position_df, position_feature_cols)
        # the same way main.py already does for the combined model.
        results = None

        results_by_position[position_name] = results

    return results_by_position


def compare_to_combined_baseline(results_by_position, combined_results):
    """
    Prints a comparison: for each position's model, show its MAE/R^2
    next to the combined (all-positions) model's overall MAE/R^2.

    TODO: Loop through results_by_position and print a row for each
    position showing:
        position name | position-specific MAE | position-specific R^2
    versus the combined model's MAE/R^2 (from combined_results, which has
    the same shape as one entry from train_and_evaluate — check model.py
    if you need a reminder of that structure).

    Once this prints, look at the numbers and answer for yourself: did
    splitting by position help? Did it help some positions more than
    others? Any hypothesis for why?
    """
    pass
