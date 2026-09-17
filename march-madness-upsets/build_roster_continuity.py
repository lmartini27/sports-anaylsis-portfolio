"""
build_roster_continuity.py — computes roster continuity (% of a team's
total minutes played by athletes who were also on the roster the prior
season) for EVERY team across multiple seasons.

Tries the efficient approach first: calling get_team_roster/
get_player_season_stats with NO team filter, hoping this returns data
for all teams in one call per season. If that looks wrong (suspiciously
few teams returned), falls back to looping through every team
individually via get_teams().

Requires: pip install cbbd
Requires: CBBD_API_KEY environment variable set
"""

import os
import time
import pandas as pd
import cbbd
from cbbd.rest import ApiException

SEASONS = list(range(2015, 2026))
MIN_EXPECTED_TEAMS = 50  # sanity threshold — real D-I has 350+ teams


def fetch_all_teams_bulk(teams_api, stats_api, season):
    """Attempt the efficient bulk fetch (no team filter)."""
    rosters = teams_api.get_team_roster(season=season)
    stats = stats_api.get_player_season_stats(season=season)
    return rosters, stats


def fetch_all_teams_looped(teams_api, stats_api, season):
    """Fallback: loop through every team individually."""
    team_list = teams_api.get_teams(season=season)
    rosters, stats = [], []
    for t in team_list:
        try:
            r = teams_api.get_team_roster(season=season, team=t.school)
            s = stats_api.get_player_season_stats(season=season, team=t.school)
            rosters.extend(r)
            stats.extend(s)
        except ApiException:
            continue
        time.sleep(0.2)
    return rosters, stats


def main():
    api_key = os.environ.get("CBBD_API_KEY")
    if not api_key:
        print("ERROR: Set the CBBD_API_KEY environment variable first.")
        return

    configuration = cbbd.Configuration(access_token=api_key)

    all_rosters = {}
    all_stats = {}
    use_bulk = True

    with cbbd.ApiClient(configuration) as api_client:
        teams_api = cbbd.TeamsApi(api_client)
        stats_api = cbbd.StatsApi(api_client)

        for season in SEASONS:
            print(f"Fetching season {season}...")
            try:
                if use_bulk:
                    rosters, stats = fetch_all_teams_bulk(teams_api, stats_api, season)
                    if len(rosters) < MIN_EXPECTED_TEAMS:
                        print(f"  Only got {len(rosters)} teams — bulk call may need a "
                              f"team filter after all. Switching to per-team looping "
                              f"(this will be much slower).")
                        use_bulk = False
                        rosters, stats = fetch_all_teams_looped(teams_api, stats_api, season)
                else:
                    rosters, stats = fetch_all_teams_looped(teams_api, stats_api, season)
            except ApiException as e:
                print(f"  API error for season {season}: {e}")
                continue

            all_rosters[season] = {
                r.team: {p.id for p in (r.players or [])} for r in rosters
            }
            all_stats[season] = stats
            print(f"  Got {len(rosters)} team rosters, {len(stats)} player stat lines.")
            time.sleep(1)

    print("\nComputing continuity...")
    rows = []
    for season in SEASONS:
        prior_season = season - 1
        if season not in all_stats or prior_season not in all_rosters:
            continue

        prior_rosters_by_team = all_rosters[prior_season]

        stats_by_team = {}
        for s in all_stats[season]:
            stats_by_team.setdefault(s.team, []).append(s)

        for team, player_stats in stats_by_team.items():
            total_minutes = sum(p.minutes or 0 for p in player_stats)
            if total_minutes == 0:
                continue

            prior_ids = prior_rosters_by_team.get(team, set())
            returning_minutes = sum(
                p.minutes or 0 for p in player_stats if p.athlete_id in prior_ids
            )
            continuity_pct = returning_minutes / total_minutes

            rows.append({
                "season": season,
                "team": team,
                "total_minutes": total_minutes,
                "returning_minutes": returning_minutes,
                "continuity_pct": continuity_pct,
            })

    if not rows:
        print("No continuity rows computed — check the diagnostic output above for issues.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("roster_continuity.csv", index=False)
    print(f"\nSaved {len(df)} team-season continuity rows to roster_continuity.csv")
    print(df["continuity_pct"].describe())


if __name__ == "__main__":
    main()
