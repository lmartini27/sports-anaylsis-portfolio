"""
06_visualize_results.py

Saves three figures to outputs/figures/:
  1. style_radar_chart.png   -- average feature profile of each style
  2. matchup_heatmap.png     -- home win % by style pair, annotated with n
  3. style_gap_results.png   -- H/D/A split by style gap (the headline chart)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

P, F = config.PROCESSED_DATA_DIR, config.FIGURES_DIR


def radar_chart():
    prof = pd.read_csv(os.path.join(P, "cluster_profiles.csv"), index_col=0)
    feats = list(prof.columns)
    normed = (prof - prof.min()) / (prof.max() - prof.min())
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, row in normed.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", " ") for f in feats], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Style archetype profiles (scaled 0-1 across styles)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.savefig(os.path.join(F, "style_radar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved outputs/figures/style_radar_chart.png")


def matchup_heatmap():
    win = pd.read_csv(os.path.join(P, "home_win_rate_matrix.csv"), index_col=0)
    n = pd.read_csv(os.path.join(P, "matchup_counts_matrix.csv"), index_col=0)
    labels = win.round(0).astype("Int64").astype(str) + "%\n(n=" + n.fillna(0).astype(int).astype(str) + ")"
    labels = labels.where(n.notna(), "")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(win, annot=labels, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                cbar_kws={"label": "Home win rate (%)"}, ax=ax)
    ax.set_xlabel("Away team style")
    ax.set_ylabel("Home team style")
    ax.set_title("CL home win rate by style matchup (small n = unreliable)")
    fig.savefig(os.path.join(F, "matchup_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved outputs/figures/matchup_heatmap.png")


def gap_chart():
    tbl = pd.read_csv(os.path.join(P, "style_gap_results.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = tbl.index.astype(str)
    ax.bar(x, tbl["home_win_pct"], label="Home win", color="#2e7d32")
    ax.bar(x, tbl["draw_pct"], bottom=tbl["home_win_pct"], label="Draw", color="#9e9e9e")
    ax.bar(x, tbl["away_win_pct"], bottom=tbl["home_win_pct"] + tbl["draw_pct"], label="Away win", color="#c62828")
    for i, n in enumerate(tbl["n"]):
        ax.text(i, 102, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Style gap (home rank - away rank); positive = home team higher on the control ladder")
    ax.set_ylabel("% of matches")
    ax.set_title("Champions League results by style gap")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(os.path.join(F, "style_gap_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved outputs/figures/style_gap_results.png")


def main():
    os.makedirs(F, exist_ok=True)
    radar_chart()
    matchup_heatmap()
    gap_chart()


if __name__ == "__main__":
    main()
