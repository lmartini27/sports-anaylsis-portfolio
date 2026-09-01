"""
Feature engineering: turns raw weekly stats into rolling-average features
that can be used to predict a player's NEXT week fantasy performance.
"""

import pandas as pd


def build_features(df, rolling_window=3, extra_feature_cols=None):
    """
    For each player-week, compute rolling averages of that player's own
    recent performance, looking only at PAST weeks (never the target week
    itself — this avoids data leakage, one of the most common mistakes in
    sports prediction projects).

    extra_feature_cols: optional list of columns already computed elsewhere
        (e.g. opponent defensive strength from src/opponent.py) to include
        alongside the rolling player-stat features built here.

    Returns (df_model, feature_cols):
        df_model    - DataFrame ready for modeling, with a
                      `target_next_week_points` column (what we predict)
        feature_cols - list of column names to use as model inputs
    """
    extra_feature_cols = extra_feature_cols or []
    df = df.sort_values(["player_id", "season", "week"]).copy()

    stat_cols = [
        "fantasy_points_ppr", "targets", "carries",
        "receiving_yards", "rushing_yards", "passing_yards", "receptions",
    ]

    grouped = df.groupby("player_id")

    for col in stat_cols:
        # shift(1) means "don't peek at the current week" — only prior weeks
        df[f"{col}_avg_last{rolling_window}"] = (
            grouped[col]
            .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
        )

    # The target: next week's fantasy points for this player
    df["target_next_week_points"] = grouped["fantasy_points_ppr"].shift(-1)

    # One-hot encode position (QB/RB/WR/TE) so the model can use it
    df = pd.get_dummies(df, columns=["position"], prefix="pos")

    feature_cols = [c for c in df.columns if "_avg_last" in c or c.startswith("pos_")]
    feature_cols += [c for c in extra_feature_cols if c not in feature_cols]

    # Drop rows missing history (first weeks) or missing a target (last week per player)
    df_model = df.dropna(subset=feature_cols + ["target_next_week_points"])

    return df_model, feature_cols
