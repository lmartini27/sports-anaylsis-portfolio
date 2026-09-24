"""
06_visualize_results.py

Produces the two visuals that carry this project:
  1. A radar chart per style cluster, showing its average feature profile
     (what actually defines "High Press" vs "Possession Control", etc.)
  2. A heatmap of home win rate by style matchup (home_style x away_style)

Saves both as PNGs in outputs/figures/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
import config


def radar_chart(cluster_profile: pd.DataFrame, out_path: str):
    features = list(cluster_profile.columns)
    n = len(features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # normalize each feature to 0-1 across clusters so the radar is readable
    normed = (cluster_profile - cluster_profile.min()) / (
        cluster_profile.max() - cluster_profile.min()
    )

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for cluster_id, row in normed.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Style {cluster_id}")
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Style archetype feature profiles (normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def matchup_heatmap(win_rate: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        win_rate, annot=True, fmt=".1f", cmap="RdYlGn", center=50,
        cbar_kws={"label": "Home win rate (%)"}, ax=ax,
    )
    ax.set_xlabel("Away team style")
    ax.set_ylabel("Home team style")
    ax.set_title("Champions League home win rate by style matchup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    cluster_profile = pd.read_csv(
        f"{config.PROCESSED_DATA_DIR}/cluster_profiles.csv", index_col=0
    )
    radar_chart(cluster_profile, f"{config.FIGURES_DIR}/style_radar_chart.png")

    win_rate = pd.read_csv(
        f"{config.PROCESSED_DATA_DIR}/home_win_rate_matrix.csv", index_col=0
    )
    matchup_heatmap(win_rate, f"{config.FIGURES_DIR}/matchup_heatmap.png")

    print("\nAll figures saved to outputs/figures/. Drop the two PNGs into "
          "REPORT.md's Results section once you've eyeballed them.")


if __name__ == "__main__":
    main()
