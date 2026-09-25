"""
Opponent strength: how many fantasy points the defense a player faces in
their NEXT game has recently allowed to that player's position.

For a game in week w, the defense's strength is its average points
allowed to the position over its previous `window` games that season
(weeks before w only), so nothing from the predicted game leaks in.
"""


def compute_team_defense_by_position(df):
    """Points scored BY each position AGAINST each team, per week = points allowed."""
    return (
        df.groupby(["season", "week", "opponent_team", "position"])["fantasy_points_ppr"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team", "fantasy_points_ppr": "points_allowed"})
    )


def add_rolling_defense_strength(def_df, window=3):
    """
    For each (team, position, season, week): average points allowed over the
    team's previous `window` games that season. shift(1) excludes week w itself.
    """
    def_df = def_df.sort_values(["team", "position", "season", "week"]).copy()
    g = def_df.groupby(["team", "position", "season"])
    col_name = f"next_opp_def_points_allowed_avg_last{window}"
    def_df[col_name] = g["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return def_df, col_name


def add_opponent_strength(df, window=3):
    """
    Attaches the NEXT opponent's defensive strength (as of the next game's
    week) to each row. Requires `next_week` and `next_opponent` columns from
    features.add_next_game().
    """
    def_table = compute_team_defense_by_position(df)
    def_table, col_name = add_rolling_defense_strength(def_table, window=window)

    lookup = def_table[["season", "week", "team", "position", col_name]].rename(
        columns={"week": "next_week", "team": "next_opponent"}
    )
    df = df.copy()
    df["next_week"] = df["next_week"].astype("Float64")
    lookup["next_week"] = lookup["next_week"].astype("Float64")
    df = df.merge(lookup, on=["season", "next_week", "next_opponent", "position"], how="left")
    return df, col_name
