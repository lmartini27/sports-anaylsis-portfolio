"""
02_feature_engineering.py

Builds one row of style features per team-season.

Handles two player-data formats found in data/raw/:
  - FBref-style Kaggle files:  players_data-YYYY_YYYY.csv  (or _light)
      comma-separated, season TOTALS, columns like Cmp / Att / Def 3rd
  - vivovinco-style files:     YYYY-YYYY Football Player Stats.csv
      semicolon-separated, latin-1 encoded, PER-90 values,
      columns like PasTotCmp / PasTotAtt / TklDef3rd

Only seasons listed in config.SEASONS are used. If a season has more
than one file, the full FBref file is preferred over light, then vivovinco.

Team-level per-90 features are per TEAM match (sum of player 90s / 11).
"""

import os
import re
import sys
import difflib
import unicodedata
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TEAM_KEYS = ["league", "season", "team"]
COUNT_COLS = ["passes_cmp", "passes_att", "final_third", "touches_att_pen",
              "tkl_def3rd", "tkl_mid3rd", "tkl_att3rd", "shots"]

LEAGUE_MAP = [
    ("Premier League", "ENG-Premier League"),
    ("La Liga", "ESP-La Liga"),
    ("Serie A", "ITA-Serie A"),
    ("Bundesliga", "GER-Bundesliga"),
    ("Ligue 1", "FRA-Ligue 1"),
]

FORMATS = {
    "fbref_full": {
        "pattern": re.compile(r"^players_data-(\d{4})_(\d{4})\.csv$"), "priority": 0,
        "read": dict(low_memory=False),
        "cols": {"Squad": "team", "Comp": "comp", "90s": "nineties", "Cmp": "passes_cmp",
                 "Att": "passes_att", "1/3": "final_third", "Att Pen": "touches_att_pen",
                 "Def 3rd": "tkl_def3rd", "Mid 3rd": "tkl_mid3rd", "Att 3rd": "tkl_att3rd",
                 "Sh": "shots"},
    },
    "fbref_light": {
        "pattern": re.compile(r"^players_data_light-(\d{4})_(\d{4})\.csv$"), "priority": 1,
        "read": dict(low_memory=False),
        "cols": None,  # same as fbref_full, filled in below
    },
    "vivovinco": {
        "pattern": re.compile(r"^(\d{4})-(\d{4}) Football Player Stats\.csv$"), "priority": 2,
        "read": dict(sep=";", encoding="latin1"),
        "cols": {"Squad": "team", "Comp": "comp", "90s": "nineties", "PasTotCmp": "passes_cmp",
                 "PasTotAtt": "passes_att", "Pas3rd": "final_third", "TouAttPen": "touches_att_pen",
                 "TklDef3rd": "tkl_def3rd", "TklMid3rd": "tkl_mid3rd", "TklAtt3rd": "tkl_att3rd",
                 "Shots": "shots"},
    },
}
FORMATS["fbref_light"]["cols"] = FORMATS["fbref_full"]["cols"]


# --- team-name normalization (same logic as 04) so abbreviations like
# "Paris S-G" / "Eint Frankfurt" match full names like "Paris Saint-Germain"
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


def map_league(comp):
    for keyword, code in LEAGUE_MAP:
        if keyword in str(comp):
            return code
    return None


def to_num(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def discover_files():
    chosen = {}
    for fname in sorted(os.listdir(config.RAW_DATA_DIR)):
        for fmt, spec in FORMATS.items():
            m = spec["pattern"].match(fname)
            if m:
                season = f"{m.group(1)}-{m.group(2)}"
                if season not in chosen or spec["priority"] < FORMATS[chosen[season][1]]["priority"]:
                    chosen[season] = (fname, fmt)
    return chosen


def load_players(fname, fmt, season):
    spec = FORMATS[fmt]
    df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, fname), **spec["read"])
    missing = [c for c in spec["cols"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fname} is missing columns {missing}. For FBref-style files this usually "
            f"means the season's detailed passing/defense tables aren't published yet -- "
            f"remove that season from config.SEASONS."
        )
    out = df[list(spec["cols"])].rename(columns=spec["cols"]).copy()
    out["nineties"] = to_num(out["nineties"])
    for c in COUNT_COLS:
        out[c] = to_num(out[c])

    # vivovinco files are per-90; detect that and convert back to season totals
    regulars = out[out["nineties"] >= 10]
    if len(regulars) and regulars["passes_att"].median() < 150:
        print(f"  {fname}: values look per-90 -> converting to season totals (x player 90s)")
        for c in COUNT_COLS:
            out[c] = out[c] * out["nineties"]

    print(f"  {fname}: max player 90s = {out['nineties'].max():.1f} "
          f"(a full season is ~34-38; much lower means a partial season)")

    out["league"] = out["comp"].apply(map_league)
    out = out[out["league"].notna()].copy()
    out["season"] = season
    return out


def aggregate(players):
    agg = players.groupby(TEAM_KEYS)[["nineties"] + COUNT_COLS].sum().reset_index()
    team_matches = agg["nineties"] / 11
    agg["pass_completion_pct"] = agg["passes_cmp"] / agg["passes_att"] * 100
    total_tkl = agg["tkl_def3rd"] + agg["tkl_mid3rd"] + agg["tkl_att3rd"]
    agg["press_height_pct"] = (agg["tkl_mid3rd"] + agg["tkl_att3rd"]) / total_tkl * 100
    agg["passes_into_final_third_per90"] = agg["final_third"] / team_matches
    agg["touches_att_pen_per90"] = agg["touches_att_pen"] / team_matches
    agg["shots_per90"] = agg["shots"] / team_matches
    return agg


def load_possession():
    df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "team_standard.csv"))
    df = df.rename(columns={"Unnamed: 5": "possession_pct"})

    def label(code):
        s = str(code).strip()
        return f"20{s[:2]}-20{s[2:]}" if s.isdigit() and len(s) == 4 else s

    df["season"] = df["season"].apply(label)
    return df[TEAM_KEYS + ["possession_pct"]]


def main():
    files = discover_files()
    use = {s: f for s, f in files.items() if s in config.SEASONS}
    skipped = sorted(set(files) - set(use))
    print(f"Seasons in config.SEASONS: {config.SEASONS}")
    print(f"Using files: {use}")
    if skipped:
        print(f"Ignoring seasons not in config.SEASONS: {skipped}")
    missing = [s for s in config.SEASONS if s not in use]
    if missing:
        raise SystemExit(f"No player file found for seasons {missing} in data/raw/.")

    frames = [aggregate(load_players(fname, fmt, season)) for season, (fname, fmt) in use.items()]
    teams = pd.concat(frames, ignore_index=True)

    poss = load_possession()
    teams["name_key"] = teams["team"].apply(normalize)
    poss["name_key"] = poss["team"].apply(normalize)
    poss = poss.rename(columns={"team": "orig_team"}).drop_duplicates(["league", "season", "name_key"])
    merged = teams.merge(poss, on=["league", "season", "name_key"], how="left")

    # Safety net: within each league-season, pair any leftover unmatched
    # teams with the closest-spelled unused name from the possession file.
    fixes = []
    for (lg, se), grp in merged[merged["possession_pct"].isna()].groupby(["league", "season"]):
        used = set(merged.loc[(merged["league"] == lg) & (merged["season"] == se), "name_key"])
        pool = poss[(poss["league"] == lg) & (poss["season"] == se) & (~poss["name_key"].isin(used))]
        pool_keys = pool["name_key"].tolist()
        for idx, row in grp.iterrows():
            best = difflib.get_close_matches(row["name_key"], pool_keys, n=1, cutoff=0.0)
            if not best:
                continue
            hit = pool[pool["name_key"] == best[0]].iloc[0]
            merged.loc[idx, "possession_pct"] = hit["possession_pct"]
            fixes.append((se, row["team"], hit["orig_team"]))
            pool_keys.remove(best[0])
    if fixes:
        print("\nClosest-spelling matches used for possession (check these look right):")
        for se, a, b in fixes:
            print(f"  {se}: {a}  <->  {b}")
    no_poss = merged[merged["possession_pct"].isna()]
    if len(no_poss):
        print(f"\n[!] {len(no_poss)} team-seasons had no possession match (name mismatch "
              f"between files) and will be dropped:")
        print(no_poss[["season", "team"]].to_string(index=False))

    out = merged[TEAM_KEYS + config.STYLE_FEATURES].dropna()
    out_path = os.path.join(config.PROCESSED_DATA_DIR, "team_style_features.csv")
    out.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\nSaved {out_path}  ({out.shape[0]} team-seasons)")
    print(out.groupby("season").size().rename("teams per season"))
    print(out[config.STYLE_FEATURES].describe().round(2))


if __name__ == "__main__":
    main()
