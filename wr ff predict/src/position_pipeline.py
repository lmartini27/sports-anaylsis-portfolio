"""
position_pipeline.py — builds features and predictions for a SINGLE
position at a time (e.g. WR), instead of mixing all positions into one
model. Different positions score fantasy points in very different ways
(a WR's points come from receptions/targets/receiving yards; a QB's come
from passing), so a dedicated model per position should outperform one
combined model trying to compromise across all of them.

Unlike features.py, this does NOT one-hot encode position — every row is
already the same position, so that column would be constant and useless.
"""

import pandas as pd

STAT_COLS = [
    "fantasy_points_ppr", "targets", "carries",
    "receiving_yards", "rushing_yards", "passing_yards", "receptions",
]


def build_position_features(df, position, rolling_window=3, opponent_col=None):
    """
    Filters to a single position and builds rolling-average features,
    using the same leakage-safe approach as features.py (only ever
    looking at PAST weeks to predict the next one).

    Returns (df_model, feature_cols).
    """
    pos_df = (
        df[df["position"] == position]
        .sort_values(["player_id", "season", "week"])
        .copy()
    )

    grouped = pos_df.groupby("player_id")
    for col in STAT_COLS:
        pos_df[f"{col}_avg_last{rolling_window}"] = (
            grouped[col]
            .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
        )

    pos_df["target_next_week_points"] = grouped["fantasy_points_ppr"].shift(-1)

    feature_cols = [f"{col}_avg_last{rolling_window}" for col in STAT_COLS]
    if opponent_col and opponent_col in pos_df.columns and opponent_col not in feature_cols:
        feature_cols.append(opponent_col)

    df_model = pos_df.dropna(subset=feature_cols + ["target_next_week_points"])

    return df_model, feature_cols


def predict_next_week_position(player_display_name, upcoming_opponent, df, def_table,
                                model, feature_cols, position, rolling_window=3):
    """
    Same idea as src/predict.py, but for a single-position model — no
    one-hot position column needs to be set since the model already
    only knows about one position.
    """
    player_df = df[
        (df["player_display_name"] == player_display_name) & (df["position"] == position)
    ].sort_values(["season", "week"])

    if player_df.empty:
        raise ValueError(f"No {position} data found for player: {player_display_name!r}")

    recent_games = player_df.tail(rolling_window)
    current_features = {
        f"{col}_avg_last{rolling_window}": recent_games[col].mean()
        for col in STAT_COLS
    }

    opponent_col_name = f"def_points_allowed_avg_last{rolling_window}"
    opp_rows = def_table[
        (def_table["team"] == upcoming_opponent) & (def_table["position"] == position)
    ].sort_values(["season", "week"])

    if opp_rows.empty:
        raise ValueError(f"No defensive data found for team {upcoming_opponent!r} vs {position!r}")

    opponent_strength = opp_rows[opponent_col_name].iloc[-1]

    row = {col: 0 for col in feature_cols}
    row.update({k: v for k, v in current_features.items() if k in row})
    if opponent_col_name in row:
        row[opponent_col_name] = opponent_strength

    X = pd.DataFrame([row])[feature_cols]
    return model.predict(X)[0]
