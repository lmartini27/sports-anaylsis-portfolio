"""
explore_roster_continuity.py — RUN THIS FIRST, before building the full
roster continuity model.

Tests the core idea on ONE team across two seasons: what fraction of a
team's total minutes were played by athletes who were also on the roster
the PREVIOUS season? This is the standard "returning production" metric
used in real college basketball analytics (KenPom, EvanMiya, etc.).

Requires: pip install cbbd
Requires: CBBD_API_KEY environment variable set to your free API key
          from https://collegebasketballdata.com/key
"""

import os
import cbbd
from cbbd.rest import ApiException

TEAM = "Duke"          # change to any team name the API recognizes
LATER_SEASON = 2025     # the season we're computing returning-production FOR
EARLIER_SEASON = 2024   # the prior season we check for returning players


def main():
    api_key = os.environ.get("CBBD_API_KEY")
    if not api_key:
        print("ERROR: Set the CBBD_API_KEY environment variable first.")
        return

    configuration = cbbd.Configuration(access_token=api_key)

    with cbbd.ApiClient(configuration) as api_client:
        teams_api = cbbd.TeamsApi(api_client)
        stats_api = cbbd.StatsApi(api_client)

        print(f"Fetching {TEAM} roster for {EARLIER_SEASON} and {LATER_SEASON}...")
        try:
            earlier_roster = teams_api.get_team_roster(season=EARLIER_SEASON, team=TEAM)
            later_roster = teams_api.get_team_roster(season=LATER_SEASON, team=TEAM)
        except ApiException as e:
            print(f"API error fetching rosters: {e}")
            return

        if not earlier_roster or not later_roster:
            print("Got an empty roster back — check the team name spelling and season.")
            return

        earlier_players = earlier_roster[0].players if hasattr(earlier_roster[0], "players") else []
        later_players = later_roster[0].players if hasattr(later_roster[0], "players") else []

        print(f"{EARLIER_SEASON} roster size: {len(earlier_players)}")
        print(f"{LATER_SEASON} roster size: {len(later_players)}")
        print(f"\nSample player record: {later_players[0] if later_players else 'none found'}\n")

        earlier_ids = {p.id for p in earlier_players}

        print(f"Fetching {TEAM} player season stats for {LATER_SEASON}...")
        try:
            later_stats = stats_api.get_player_season_stats(season=LATER_SEASON, team=TEAM)
        except ApiException as e:
            print(f"API error fetching player stats: {e}")
            return

        if not later_stats:
            print("Got no player stats back — check team name/season.")
            return

        print(f"\nSample stats record: {later_stats[0]}\n")

        total_minutes = sum(p.minutes or 0 for p in later_stats)
        returning_minutes = sum(
            p.minutes or 0 for p in later_stats if p.athlete_id in earlier_ids
        )

        if total_minutes == 0:
            print("Total minutes came back as zero — check the 'minutes' field name/data.")
            return

        continuity_pct = returning_minutes / total_minutes
        print(f"\n--- {TEAM}: {LATER_SEASON} Returning Production ---")
        print(f"Total team minutes:     {total_minutes:.0f}")
        print(f"Returning-player minutes: {returning_minutes:.0f}")
        print(f"Roster continuity:      {continuity_pct:.1%}")


if __name__ == "__main__":
    main()
