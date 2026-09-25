"""
Feature engineering: turns raw weekly stats into features for predicting a
player's fantasy points in their NEXT game.

Timing convention (important for avoiding leakage):
  - Each row is one player-game in week t.
  - Features use games up to and INCLUDING week t -- all of that is known
    before the next game kicks off.
  - The target is the player's points in their next game of the SAME
    season (week t+1, or t+2 if there was a bye). Rows whose next game is
    in a different season, or more than 2 weeks later (e.g. an injury
    absence), have no target and are dropped.
"""

import pandas as pd

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]
STAT_COLS = [
    "fantasy_points_ppr", "targets", "carries",
    "receiving_yards", "rushing_yards", "passing_yards", "receptions",
]
MAX_WEEK_GAP = 2  # allows for one bye week


def add_next_game(df):
    """
    Keeps the four fantasy positions and adds, for each player-game, the
    week, opponent, and fantasy points of that player's NEXT game in the
    same season. `next_opponent` is what the opponent-strength feature
    must describe, since that's the game being predicted.
    """
    df = df[df["position"].isin(FANTASY_POSITIONS)].copy()
    df = df.sort_values(["player_id", "season", "week"])
    g = df.groupby(["player_id", "season"])

    df["next_week"] = g["week"].shift(-1)
    df["next_opponent"] = g["opponent_team"].shift(-1)
    df["target_next_week_points"] = g["fantasy_points_ppr"].shift(-1)

    too_far = (df["next_week"] - df["week"]) > MAX_WEEK_GAP
    df.loc[too_far, ["next_week", "next_opponent", "target_next_week_points"]] = pd.NA
    return df


def build_features(df, rolling_window=3, extra_feature_cols=None):
    """
    Rolling averages of each player's own recent stats, over their last
    `rolling_window` games up to and including the current week, computed
    within a season so last year's form doesn't carry over.

    Returns (df_model, feature_cols).
    """
    extra_feature_cols = extra_feature_cols or []
    df = df.sort_values(["player_id", "season", "week"]).copy()
    g = df.groupby(["player_id", "season"])

    for col in STAT_COLS:
        df[f"{col}_avg_last{rolling_window}"] = g[col].transform(
            lambda s: s.rolling(rolling_window, min_periods=1).mean()
        )

    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=int)
    for pos in FANTASY_POSITIONS:  # guarantee all four columns exist
        if f"pos_{pos}" not in df.columns:
            df[f"pos_{pos}"] = 0

    feature_cols = [f"{c}_avg_last{rolling_window}" for c in STAT_COLS]
    feature_cols += [f"pos_{p}" for p in FANTASY_POSITIONS]
    feature_cols += [c for c in extra_feature_cols if c not in feature_cols]

    df_model = df.dropna(subset=feature_cols + ["target_next_week_points"]).copy()
    df_model["target_next_week_points"] = df_model["target_next_week_points"].astype(float)
    return df_model, feature_cols
