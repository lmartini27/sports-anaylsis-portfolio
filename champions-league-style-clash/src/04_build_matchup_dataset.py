"""
04_build_matchup_dataset.py

Joins Champions League results (uefa_champions_league_historical_match_
statistics_2020_2026.csv -- Portuguese columns: mandante/visitante =
home/away, placar = "X x Y", temporada = "24/25") to each team's style.

Club names differ between the CL file and FBref ("Bayer 04 Leverkusen"
vs "Leverkusen", "AC Milan" vs "Milan"), so names are normalized and
matched automatically within each season. Every unmatched name is saved
to data/processed/unmatched_cl_teams.csv so you can check nothing
important was missed -- clubs outside the Big 5 leagues SHOULD be there.
"""

import os
import re
import sys
import unicodedata
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CL_FILE = "uefa_champions_league_historical_match_statistics_2020_2026.csv"

# Manual overrides: CL name -> FBref name. Add here if a Big-5 club shows up unmatched.
MANUAL_FIXES = {}

STOP = {"fc", "ac", "as", "ss", "ssc", "sc", "cf", "afc", "club", "kv", "bc", "de", "cd", "rc",
        "ud", "sv", "vfb", "vfl", "tsg", "bv", "us", "ogc", "losc", "calcio", "hotspur", "bsc",
        "stade", "olympique", "1", "04", "05", "09", "29", "1899", "1909", "1846", "1900"}
PHRASE_FIXES = [("munchen", "munich"), ("utd", "united"), ("eint", "eintracht"),
                ("paris s g", "paris saint germain"), ("psg", "paris saint germain"),
                ("nottham", "nottingham"), ("brestois", "brest"), ("rennais", "rennes"),
                ("lyonnais", "lyon"), ("m gladbach", "gladbach"), ("monchengladbach", "gladbach"),
                ("mgladbach", "gladbach"), ("internazionale", "inter"), ("inter milan", "inter"),
                ("athletic bilbao", "athletic")]


def normalize(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = f" {re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()} "
    for old, new in PHRASE_FIXES:
        s = s.replace(f" {old} ", f" {new} ")
    return " ".join(t for t in s.split() if t not in STOP)


def match_team(cl_name, index):
    if cl_name in MANUAL_FIXES:
        return MANUAL_FIXES[cl_name] if any(n == MANUAL_FIXES[cl_name] for n, _ in index) else None
    tokens = set(normalize(cl_name).split())
    if not tokens:
        return None
    exact = [n for n, t in index if t == tokens]
    if len(exact) == 1:
        return exact[0]
    subset = [(n, t) for n, t in index if t and t <= tokens and len(t) / len(tokens) >= 0.5]
    if not subset:
        return None
    best = max(len(t) for _, t in subset)
    cands = [n for n, t in subset if len(t) == best]
    return cands[0] if len(cands) == 1 else None


def parse_score(placar):
    m = re.match(r"\s*(\d+)\s*x\s*(\d+)", str(placar))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    raw = pd.read_csv(os.path.join(config.RAW_DATA_DIR, CL_FILE))
    raw = raw[raw["campeonato"] == "UEFA Champions League"].copy()
    raw["season"] = raw["temporada"].apply(lambda t: f"20{t.split('/')[0]}-20{t.split('/')[1]}")
    raw = raw[raw["season"].isin(config.SEASONS)].copy()

    scores = raw["placar"].apply(parse_score)
    raw["home_goals"] = scores.str[0]
    raw["away_goals"] = scores.str[1]
    raw = raw.dropna(subset=["home_goals", "away_goals"])

    styles = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "team_style_clusters.csv"))
    style_lookup = styles.set_index(["team", "season"])["style_name"].to_dict()
    indexes = {s: [(n, set(normalize(n).split())) for n in grp["team"].unique()]
               for s, grp in styles.groupby("season")}

    matched_names, unmatched = {}, set()

    def resolve(cl_name, season):
        key = (cl_name, season)
        if key not in matched_names:
            team = match_team(cl_name, indexes.get(season, []))
            matched_names[key] = team
            if team is None:
                unmatched.add(cl_name)
        return matched_names[key]

    raw["home_team"] = [resolve(n, s) for n, s in zip(raw["mandante"], raw["season"])]
    raw["away_team"] = [resolve(n, s) for n, s in zip(raw["visitante"], raw["season"])]
    raw["home_style"] = [style_lookup.get((t, s)) for t, s in zip(raw["home_team"], raw["season"])]
    raw["away_style"] = [style_lookup.get((t, s)) for t, s in zip(raw["away_team"], raw["season"])]

    print("Name matches made (CL name -> FBref name), only where they differ:")
    for cl, team in sorted({(cl, t) for (cl, _), t in matched_names.items() if t and t != cl}):
        print(f"  {cl} -> {team}")

    pd.Series(sorted(unmatched), name="cl_team").to_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "unmatched_cl_teams.csv"), index=False)
    print(f"\n{len(unmatched)} CL clubs not matched to a Big-5 style (full list saved to "
          f"data/processed/unmatched_cl_teams.csv -- skim it for any Big-5 club).")

    m = raw.dropna(subset=["home_style", "away_style"]).copy()
    m["result"] = ["H" if h > a else ("A" if h < a else "D")
                   for h, a in zip(m["home_goals"], m["away_goals"])]
    m["style_matchup"] = m["home_style"] + " vs " + m["away_style"]

    cols = ["season", "home_team", "away_team", "home_goals", "away_goals",
            "result", "home_style", "away_style", "style_matchup"]
    out_path = os.path.join(config.PROCESSED_DATA_DIR, "cl_matchup_dataset.csv")
    m[cols].to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(m)} matches with both sides' style "
          f"(out of {len(raw)} CL matches in {config.SEASONS})")
    print(m.groupby("season").size().rename("matches per season"))


if __name__ == "__main__":
    main()
