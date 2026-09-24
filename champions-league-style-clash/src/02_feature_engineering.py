"""
02_feature_engineering.py

Merges the four raw FBref squad tables (standard, possession, passing,
defense) into one row per team-season, and engineers the style features
listed in config.STYLE_FEATURES.

Because FBref's exact column names can drift between seasons/scraper
versions, this script finds columns by keyword match instead of hard
-coding exact names. If a column can't be found, it prints the full
column list for that table so you can fix the keywords below.
"""

import sys
import pandas as pd

sys.path.append("..")
import config

ID_COLS_CANDIDATES = ["team", "league", "season"]


def find_col(df: pd.DataFrame, keywords, table_name: str) -> str:
    """Return the single column whose name contains ALL keywords
    (case-insensitive). Errors loudly with the full column list if
    zero or multiple matches are found, so you can fix `keywords`."""
    matches = [
        c for c in df.columns
        if all(k.lower() in str(c).lower() for k in keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    print(f"\n[!] Could not uniquely match {keywords} in '{table_name}' table.")
    print(f"    Candidates found: {matches}")
    print(f"    Full column list: {list(df.columns)}")
    raise ValueError(
        f"Fix the keyword list for {keywords} in 02_feature_engineering.py "
        f"based on the columns printed above."
    )


def id_cols(df: pd.DataFrame) -> list:
    return [c for c in ID_COLS_CANDIDATES if c in df.columns]


def load_raw(stat_type: str) -> pd.DataFrame:
    return pd.read_csv(f"{config.RAW_DATA_DIR}/team_{stat_type}.csv")


def main():
    standard = load_raw("standard")
    possession = load_raw("possession")
    passing = load_raw("passing")
    defense = load_raw("defense")

    # --- 90s played, for per-90 normalization -----------------------------
    nineties_col = find_col(standard, ["90s"], "standard")

    # --- possession table ---------------------------------------------------
    poss_col = find_col(possession, ["poss"], "possession")
    touches_pen_col = find_col(possession, ["att pen"], "possession")
    poss_feats = possession[id_cols(possession) + [poss_col, touches_pen_col]].copy()
    poss_feats = poss_feats.rename(columns={
        poss_col: "possession_pct",
        touches_pen_col: "touches_att_pen_raw",
    })

    # --- passing table --------------------------------------------------
    cmp_pct_col = find_col(passing, ["cmp%"], "passing")
    prgp_col = find_col(passing, ["prgp"], "passing")
    final_third_col = find_col(passing, ["1/3"], "passing")
    pass_feats = passing[id_cols(passing) + [cmp_pct_col, prgp_col, final_third_col]].copy()
    pass_feats = pass_feats.rename(columns={
        cmp_pct_col: "pass_completion_pct",
        prgp_col: "progressive_passes_raw",
        final_third_col: "passes_into_final_third_raw",
    })

    # --- defense table: pressing-height proxy -------------------------------
    # share of tackles+interceptions made in the middle/attacking third
    # vs. the team's own defensive third -> higher = defends higher up.
    def_third_col = find_col(defense, ["def 3rd"], "defense")
    mid_third_col = find_col(defense, ["mid 3rd"], "defense")
    att_third_col = find_col(defense, ["att 3rd"], "defense")
    def_feats = defense[id_cols(defense) + [def_third_col, mid_third_col, att_third_col]].copy()
    def_feats = def_feats.rename(columns={
        def_third_col: "tkl_def3rd",
        mid_third_col: "tkl_mid3rd",
        att_third_col: "tkl_att3rd",
    })
    total_tkl = def_feats["tkl_def3rd"] + def_feats["tkl_mid3rd"] + def_feats["tkl_att3rd"]
    def_feats["press_height_pct"] = (
        (def_feats["tkl_mid3rd"] + def_feats["tkl_att3rd"]) / total_tkl.replace(0, pd.NA) * 100
    )

    # --- standard table: shots, 90s --------------------------------------
    shots_col = find_col(standard, ["sh", "90"], "standard")  # Sh/90 or similar
    std_feats = standard[id_cols(standard) + [nineties_col, shots_col]].copy()
    std_feats = std_feats.rename(columns={
        nineties_col: "nineties",
        shots_col: "shots_per90",
    })

    # --- merge everything on team/league/season -----------------------------
    merge_keys = id_cols(standard)
    merged = (
        std_feats
        .merge(poss_feats, on=merge_keys, how="left")
        .merge(pass_feats, on=merge_keys, how="left")
        .merge(def_feats[merge_keys + ["press_height_pct"]], on=merge_keys, how="left")
    )

    # --- normalize the "raw" (season-total) columns to per-90 --------------
    merged["progressive_passes_per90"] = merged["progressive_passes_raw"] / merged["nineties"]
    merged["passes_into_final_third_per90"] = merged["passes_into_final_third_raw"] / merged["nineties"]
    merged["touches_att_pen_per90"] = merged["touches_att_pen_raw"] / merged["nineties"]

    final_cols = merge_keys + config.STYLE_FEATURES
    out = merged[final_cols].dropna()

    out_path = f"{config.PROCESSED_DATA_DIR}/team_style_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({out.shape[0]} team-seasons, {len(config.STYLE_FEATURES)} style features)")
    print(out.describe())


if __name__ == "__main__":
    main()
