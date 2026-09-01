"""
predict.py — use a trained model to generate an actual next-week fantasy
point prediction for a specific player against a specific opponent.
"""

import pandas as pd


STAT_COLS = [
    "fantasy_points_ppr", "targets", "carries",
    "receiving_yards", "rushing_yards", "passing_yards", "receptions",
]


def get_player_current_features(df, player_display_name, rolling_window=3):
    """
    Finds a player's most recent `rolling_window` games and averages them —
    the same calculation features.py does for training rows, but anchored
    to "right now" (their most recent games) instead of a specific
    mid-season week.
    """
    player_df = df[df["player_display_name"] == player_display_name].sort_values(["season", "week"])
    if player_df.empty:
        raise ValueError(f"No data found for player: {player_display_name!r}")

    recent_games = player_df.tail(rolling_window)
    current_features = {
        f"{col}_avg_last{rolling_window}": recent_games[col].mean()
        for col in STAT_COLS
    }
    position = player_df["position"].iloc[-1]
    return current_features, position


def get_opponent_current_strength(def_table, upcoming_opponent, position, rolling_window=3):
    """
    Looks up the upcoming opponent's most recent defensive strength
    against this position — their rolling average over their last
    `rolling_window` games, which reflects their strength heading INTO
    their next game.
    """
    col_name = f"def_points_allowed_avg_last{rolling_window}"
    opp_rows = def_table[
        (def_table["team"] == upcoming_opponent) & (def_table["position"] == position)
    ].sort_values(["season", "week"])

    if opp_rows.empty:
        raise ValueError(f"No defensive data found for team {upcoming_opponent!r} vs {position!r}")

    return opp_rows[col_name].iloc[-1], col_name


def predict_next_week(player_display_name, upcoming_opponent, df, def_table,
                       model, feature_cols, rolling_window=3):
    """
    Builds a feature row for a specific player's upcoming matchup and
    returns the model's predicted fantasy points for that game.
    """
    current_features, position = get_player_current_features(df, player_display_name, rolling_window)
    opponent_strength, opponent_col = get_opponent_current_strength(
        def_table, upcoming_opponent, position, rolling_window
    )

    # Start every feature at 0 (this correctly handles one-hot position columns)
    row = {col: 0 for col in feature_cols}
    row.update({k: v for k, v in current_features.items() if k in row})
    if opponent_col in row:
        row[opponent_col] = opponent_strength

    pos_col = f"pos_{position}"
    if pos_col in row:
        row[pos_col] = 1

    X = pd.DataFrame([row])[feature_cols]  # enforce exact training column order
    prediction = model.predict(X)[0]
    return prediction
