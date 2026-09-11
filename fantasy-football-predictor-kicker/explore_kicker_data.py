"""
explore_kicker_data.py — RUN THIS FIRST, before we build the kicker model.

Kickers score fantasy points completely differently than skill players:
field goals by distance and extra points, not yards or receptions. This
means the columns used for QB/RB/WR/TE (targets, carries, receiving_yards,
etc.) are meaningless here — we need to know exactly which kicking-related
columns your data actually has before building real features.

Run this and paste me the full output.
"""

import nfl_data_py as nfl

YEARS = [2023, 2024]

print(f"Pulling weekly data for {YEARS}...")
df = nfl.import_weekly_data(YEARS)

kickers = df[df["position"] == "K"]
print(f"\nFound {len(kickers)} kicker-week rows out of {len(df)} total rows.\n")

if kickers.empty:
    print("No rows with position == 'K' found. Here are the position labels that DO exist:")
    print(df["position"].unique())
else:
    numeric_cols = kickers.select_dtypes(include="number").columns
    summary = kickers[numeric_cols].describe().T
    useful = summary[summary["max"] > 0][["count", "mean", "max"]]

    print("Numeric columns with nonzero data for kickers (these are the real signal):")
    print(useful.sort_values("max", ascending=False))

    print("\nFull column list (for reference):")
    print(list(df.columns))
