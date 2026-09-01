"""
Opponent strength features: estimates how many fantasy points a defense
has recently allowed to each position, so the model can account for
matchup difficulty (e.g. facing a weak pass defense vs. a shutdown one) —
not just how well the player themselves has been playing lately.
"""

import pandas as pd


def compute_team_defense_by_position(df):
    """
    For each (season, week, defending team, position), sums the fantasy
    points scored BY THAT POSITION AGAINST that team that week. This is,
    from the defense's perspective, literally "points allowed."
    """
    def_df = (
        df.groupby(["season", "week", "opponent_team", "position"])["fantasy_points_ppr"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team", "fantasy_points_ppr": "points_allowed"})
    )
    return def_df


def add_rolling_defense_strength(def_df, window=3):
    """
    Rolling average of points allowed to each position by each team,
    using only PAST weeks (shift(1)) so we never leak future information
    into a prediction.
    """
    def_df = def_df.sort_values(["team", "position", "season", "week"]).copy()
    grouped = def_df.groupby(["team", "position"])
    col_name = f"def_points_allowed_avg_last{window}"
    def_df[col_name] = (
        grouped["points_allowed"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return def_df, col_name


def add_opponent_strength(df, window=3):
    """
    Convenience function: computes each team's rolling defensive strength
    by position, then merges the UPCOMING opponent's strength onto each
    player-week row as a new feature (how many points this position
    typically scores against this week's opponent).

    Returns (df_with_feature, feature_column_name).
    """
    def_table = compute_team_defense_by_position(df)
    def_table, col_name = add_rolling_defense_strength(def_table, window=window)

    merge_cols = ["season", "week", "team", "position"]
    def_small = def_table[merge_cols + [col_name]].rename(columns={"team": "opponent_team"})

    df = df.merge(def_small, on=["season", "week", "opponent_team", "position"], how="left")

    return df, col_name
