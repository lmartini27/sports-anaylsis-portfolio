"""
00_inspect_downloads.py

Prints shape and columns of every CSV in data/raw/. Tries utf-8 first,
falls back to latin-1/cp1252 for files that aren't UTF-8 (common with
scraped data containing accented player/club names).
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def read_csv_robust(path):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not read {path} with utf-8, latin1, or cp1252")


def main():
    raw_dir = config.RAW_DATA_DIR
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]

    if not files:
        print(f"No CSV files found in {raw_dir}/")
        return

    for fname in sorted(files):
        path = os.path.join(raw_dir, fname)
        try:
            full_df, enc = read_csv_robust(path)
            print(f"\n{'='*70}")
            print(f"FILE: {fname}  (read with encoding={enc})")
            print(f"Shape: {full_df.shape[0]} rows x {full_df.shape[1]} columns")
            print(f"Columns:\n{list(full_df.columns)}")
            print(f"\nFirst row sample:")
            print(full_df.head(1).T)
        except Exception as e:
            print(f"\n[!] Could not read {fname}: {e}")


if __name__ == "__main__":
    main()