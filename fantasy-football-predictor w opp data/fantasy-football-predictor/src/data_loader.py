"""
Data loading module for the Fantasy Football Success Predictor.

Primary source: nfl_data_py (weekly player stats, pulled from nflfastR).
Install with: pip install nfl_data_py

If nfl_data_py isn't installed (or you have no internet access), this
module falls back to generating a small synthetic dataset so you can
test the rest of the pipeline (features, modeling) before running with
real data.
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "player_id", "player_display_name", "position", "season", "week",
    "recent_team", "opponent_team",
    "fantasy_points_ppr", "targets", "carries", "receiving_yards",
    "rushing_yards", "passing_yards", "receptions", "season_type",
]


def load_weekly_data(years):
    """
    Load weekly player stats for the given list of seasons (e.g. [2022, 2023, 2024]).
    Returns a DataFrame with one row per player per week.
    """
    try:
        import nfl_data_py as nfl
        df = nfl.import_weekly_data(years)
        df = df[[c for c in REQUIRED_COLUMNS if c in df.columns]].copy()
        # Regular season only: playoff weeks only include playoff teams and
        # would make "next game" and opponent averages inconsistent.
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].drop(columns="season_type")
        return df
    except ImportError:
        print("nfl_data_py not installed — falling back to synthetic demo data.")
        print("Run `pip install nfl_data_py` to use real NFL stats.\n")
        return _generate_synthetic_data(years)


def _generate_synthetic_data(years, n_players=60, weeks_per_season=17, seed=42):
    """Generates fake-but-plausible weekly fantasy data for offline testing."""
    rng = np.random.default_rng(seed)
    positions = ["QB", "RB", "WR", "TE"]
    position_base = {"QB": 18, "RB": 12, "WR": 11, "TE": 8}
    position_std = {"QB": 6, "RB": 7, "WR": 7, "TE": 5}
    teams = [f"TEAM_{letter}" for letter in "ABCDEFGH"]

    rows = []
    for season in years:
        for player_id in range(n_players):
            pos = positions[player_id % len(positions)]
            base = position_base[pos] + rng.normal(0, 3)
            team = teams[player_id % len(teams)]
            other_teams = [t for t in teams if t != team]
            for week in range(1, weeks_per_season + 1):
                opponent = other_teams[rng.integers(0, len(other_teams))]
                points = max(0, rng.normal(base, position_std[pos]))
                rows.append({
                    "player_id": player_id,
                    "player_display_name": f"Player_{player_id}",
                    "position": pos,
                    "season": season,
                    "week": week,
                    "recent_team": team,
                    "opponent_team": opponent,
                    "fantasy_points_ppr": round(points, 1),
                    "targets": max(0, int(rng.normal(6, 3))) if pos in ["WR", "TE", "RB"] else 0,
                    "carries": max(0, int(rng.normal(12, 5))) if pos == "RB" else 0,
                    "receiving_yards": max(0, rng.normal(50, 25)) if pos in ["WR", "TE", "RB"] else 0,
                    "rushing_yards": max(0, rng.normal(45, 20)) if pos in ["RB", "QB"] else 0,
                    "passing_yards": max(0, rng.normal(230, 60)) if pos == "QB" else 0,
                    "receptions": max(0, int(rng.normal(4, 2))) if pos in ["WR", "TE", "RB"] else 0,
                })
    return pd.DataFrame(rows)
