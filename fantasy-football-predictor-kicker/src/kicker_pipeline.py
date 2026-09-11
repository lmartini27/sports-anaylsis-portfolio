"""
kicker_pipeline.py — feature engineering, opponent strength, and
prediction for the kicker-specific model. Built on kicker_weekly_data.csv
(produced by build_kicker_data.py), since kicker stats aren't included in
nfl_data_py's standard weekly data.
"""

import pandas as pd
import numpy as np

STAT_COLS = [
    "attempts_fg_0_39", "attempts_fg_40_49", "attempts_fg_50_plus",
    "made_fg_0_39", "made_fg_40_49", "made_fg_50_plus",
    "pat_attempts", "pat_made", "fantasy_points_ppr",
]


def load_kicker_data(csv_path="kicker_weekly_data.csv"):
    return pd.read_csv(csv_path)


def load_redzone_data(csv_path="redzone_team_data.csv"):
    return pd.read_csv(csv_path)


def load_vegas_data(csv_path="vegas_team_totals.csv"):
    return pd.read_csv(csv_path)


def add_redzone_features(df, redzone_df, rolling_window=3):
    """
    Adds two red-zone-based features:
      - the kicker's own team's rolling red zone trip volume (more trips
        means more overall scoring opportunities, TD or FG)
      - the upcoming opponent's rolling red zone TD rate ALLOWED (lower
        means a defense that's stingy near the goal line, which should
        mean MORE field goal opportunities for the visiting kicker)

    Returns (df_with_features, redzone_rolling_table, own_col, opp_col).
    The rolling table is also returned separately so predict_next_week_kicker
    can look up "current" values for a team not yet in df's next game.
    """
    redzone_df = redzone_df.copy()
    redzone_df["redzone_td_rate_allowed"] = (
        redzone_df["redzone_tds_allowed"] / redzone_df["redzone_trips_allowed"].replace(0, np.nan)
    )
    redzone_df = redzone_df.sort_values(["team", "season", "week"])
    grouped = redzone_df.groupby("team")

    own_col = f"own_redzone_trips_avg_last{rolling_window}"
    redzone_df[own_col] = grouped["redzone_trips"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )

    opp_col = f"opp_redzone_td_rate_allowed_avg_last{rolling_window}"
    redzone_df[opp_col] = grouped["redzone_td_rate_allowed"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )

    merged = df.merge(
        redzone_df[["season", "week", "team", own_col]].rename(columns={"team": "recent_team"}),
        on=["season", "week", "recent_team"], how="left",
    )
    merged = merged.merge(
        redzone_df[["season", "week", "team", opp_col]].rename(columns={"team": "opponent_team"}),
        on=["season", "week", "opponent_team"], how="left",
    )

    return merged, redzone_df, own_col, opp_col


def compute_defense_fg_pressure(df, rolling_window=3):
    """
    For each defense, how many total field goal attempts have they
    recently faced? A defense that's tough near the goal line (forcing
    field goals instead of touchdowns) will show a HIGHER number here —
    the OPPOSITE relationship from skill positions, where a tough
    defense means fewer points allowed.
    """
    df = df.copy()
    df["total_fg_attempts"] = (
        df["attempts_fg_0_39"] + df["attempts_fg_40_49"] + df["attempts_fg_50_plus"]
    )

    defense_df = (
        df.groupby(["season", "week", "opponent_team"])["total_fg_attempts"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team"})
    )
    defense_df = defense_df.sort_values(["team", "season", "week"])
    col_name = f"opp_fg_attempts_faced_avg_last{rolling_window}"
    defense_df[col_name] = (
        defense_df.groupby("team")["total_fg_attempts"]
        .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
    )
    return defense_df, col_name


def build_kicker_features(df, rolling_window=3, extra_feature_cols=None, vegas_df=None):
    """
    Builds rolling-average features: the kicker's own recent performance
    plus the upcoming opponent's recent field-goal-pressure — using the
    same leakage-safe (past-weeks-only) approach as the rest of the project.

    ALIGNMENT FIX: the target is this row's OWN fantasy_points_ppr value
    (not shifted forward an extra week). The "_avg_lastN" features already
    only look at PRIOR weeks (via shift(1) before rolling), so pairing them
    with this row's own score correctly represents "predict this game using
    only games that came before it" — which matches exactly how
    predict_next_week_kicker uses the most recent completed games to
    predict the next (not-yet-played) one. Shifting the target an
    additional week (as earlier versions did) silently skipped each
    player's most recent completed game, which is also the single most
    informative one.

    vegas_df: optional DataFrame from load_vegas_data() with columns
    (season, week, team, implied_team_total). Unlike other features, this
    is NOT rolled/averaged — a Vegas line is set fresh for that specific
    upcoming game, so it's merged in directly for the matching game.

    extra_feature_cols: optional list of columns already merged into df
    elsewhere (e.g. red zone features from add_redzone_features) to
    include alongside the features built here.
    """
    extra_feature_cols = extra_feature_cols or []
    df = df.sort_values(["player_id", "season", "week"]).copy()

    grouped = df.groupby("player_id")
    for col in STAT_COLS:
        df[f"{col}_avg_last{rolling_window}"] = (
            grouped[col]
            .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
        )

    df["target_next_week_points"] = df["fantasy_points_ppr"]

    defense_df, opp_col = compute_defense_fg_pressure(df, rolling_window)
    df = df.merge(
        defense_df[["season", "week", "team", opp_col]].rename(columns={"team": "opponent_team"}),
        on=["season", "week", "opponent_team"],
        how="left",
    )

    feature_cols = [f"{col}_avg_last{rolling_window}" for col in STAT_COLS] + [opp_col]

    if vegas_df is not None:
        df = df.merge(
            vegas_df[["season", "week", "team", "implied_team_total"]].rename(
                columns={"team": "recent_team"}
            ),
            on=["season", "week", "recent_team"],
            how="left",
        )
        feature_cols.append("implied_team_total")

    feature_cols += [c for c in extra_feature_cols if c not in feature_cols]

    df_model = df.dropna(subset=feature_cols + ["target_next_week_points"])

    return df_model, feature_cols


def predict_next_week_kicker(player_display_name, upcoming_opponent, df, defense_df, opp_col,
                              model, feature_cols, rolling_window=3,
                              redzone_df=None, redzone_own_col=None, redzone_opp_col=None,
                              implied_team_total=None):
    """
    Builds a feature row for a specific kicker's upcoming matchup and
    returns the model's predicted fantasy points.

    implied_team_total: the Vegas-implied point total for the kicker's
    OWN team in the upcoming game (from load_vegas_data(), or a number
    you supply yourself — e.g. from a current sportsbook line for a real
    future game not yet in any historical CSV). Unlike other arguments,
    this can't be looked up automatically for a truly future game, since
    it doesn't exist until a sportsbook sets it.
    """
    player_df = df[df["player_display_name"] == player_display_name].sort_values(["season", "week"])
    if player_df.empty:
        raise ValueError(f"No kicker data found for: {player_display_name!r}")

    recent_games = player_df.tail(rolling_window)
    current_features = {
        f"{col}_avg_last{rolling_window}": recent_games[col].mean()
        for col in STAT_COLS
    }

    opp_rows = defense_df[defense_df["team"] == upcoming_opponent].sort_values(["season", "week"])
    if opp_rows.empty:
        raise ValueError(f"No defensive data found for team: {upcoming_opponent!r}")
    opponent_strength = opp_rows[opp_col].iloc[-1]

    row = {col: 0 for col in feature_cols}
    row.update({k: v for k, v in current_features.items() if k in row})
    if opp_col in row:
        row[opp_col] = opponent_strength

    if "implied_team_total" in row and implied_team_total is not None:
        row["implied_team_total"] = implied_team_total

    if redzone_df is not None:
        own_team = player_df["recent_team"].iloc[-1]
        own_rows = redzone_df[redzone_df["team"] == own_team].sort_values(["season", "week"])
        if not own_rows.empty and redzone_own_col in row:
            row[redzone_own_col] = own_rows[redzone_own_col].iloc[-1]

        opp_rz_rows = redzone_df[redzone_df["team"] == upcoming_opponent].sort_values(["season", "week"])
        if not opp_rz_rows.empty and redzone_opp_col in row:
            row[redzone_opp_col] = opp_rz_rows[redzone_opp_col].iloc[-1]

    X = pd.DataFrame([row])[feature_cols]
    return model.predict(X)[0]
