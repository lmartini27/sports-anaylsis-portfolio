"""
03_style_clustering.py

Standardizes the style features and runs k-means. Clusters are named
AUTOMATICALLY by ranking them on average possession (lowest -> highest),
so you don't need to relabel by hand every time the data changes.
Check the printed profiles and example teams to make sure the names fit.
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Names assigned in order of average possession, lowest first (used when k=4)
NAMES_BY_POSSESSION_RANK = [
    "Low-Block / Reactive",
    "Balanced / Mid-Block",
    "Structured Progressive",
    "Possession Control",
]


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "team_style_features.csv"))
    X = StandardScaler().fit_transform(df[config.STYLE_FEATURES])

    print("Elbow check (inertia by k) -- look for where the drop levels off:")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(X)
        print(f"  k={k}: inertia={km.inertia_:.1f}")

    km = KMeans(n_clusters=config.N_STYLE_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
    df["style_cluster"] = km.fit_predict(X)

    order = df.groupby("style_cluster")["possession_pct"].mean().sort_values().index.tolist()
    if config.N_STYLE_CLUSTERS == len(NAMES_BY_POSSESSION_RANK):
        name_map = {c: NAMES_BY_POSSESSION_RANK[i] for i, c in enumerate(order)}
    else:
        name_map = {c: f"Style {i + 1} (possession rank)" for i, c in enumerate(order)}
    df["style_name"] = df["style_cluster"].map(name_map)

    profile = df.groupby("style_name")[config.STYLE_FEATURES].mean().round(1)
    profile = profile.loc[[name_map[c] for c in order]]
    profile["n_teams"] = df["style_name"].value_counts()
    print(f"\nCluster profiles at k={config.N_STYLE_CLUSTERS} (sorted by possession):")
    print(profile)

    print("\nExample teams per style (highest possession first):")
    for name in profile.index:
        ex = (df[df["style_name"] == name].sort_values("possession_pct", ascending=False)
              .head(5).apply(lambda r: f"{r['team']} ({r['season'][2:4]}/{r['season'][7:9]})", axis=1))
        print(f"  {name}: {', '.join(ex)}")

    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "team_style_clusters.csv"), index=False)
    profile.drop(columns="n_teams").to_csv(os.path.join(config.PROCESSED_DATA_DIR, "cluster_profiles.csv"))
    print("\nSaved team_style_clusters.csv and cluster_profiles.csv")


if __name__ == "__main__":
    main()
