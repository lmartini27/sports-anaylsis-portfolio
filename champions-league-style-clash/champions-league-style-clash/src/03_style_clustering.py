"""
03_style_clustering.py

Standardizes the style features and runs k-means to assign each
team-season a style archetype (0..N_STYLE_CLUSTERS-1). Also prints an
elbow-method summary so you can sanity check config.N_STYLE_CLUSTERS
before trusting the labels.

After clustering, it prints the average feature values per cluster so
you can hand-label them with real names (e.g. "High Press",
"Possession Control", "Direct Counter", "Low Block") -- update
CLUSTER_NAME_MAP below once you've eyeballed them.
"""

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append("..")
import config

# Fill this in AFTER you've run this script once and looked at the
# per-cluster feature averages it prints. Cluster numbers aren't
# meaningful on their own -- the printed averages tell you which
# cluster is "presses high" vs "sits deep", etc.
CLUSTER_NAME_MAP = {
    0: "Cluster 0 (name me)",
    1: "Cluster 1 (name me)",
    2: "Cluster 2 (name me)",
    3: "Cluster 3 (name me)",
}


def main():
    df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/team_style_features.csv")

    X = df[config.STYLE_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- elbow check: inertia for k = 2..8 ----------------------------------
    print("Elbow check (inertia by k) -- look for where the drop levels off:")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        print(f"  k={k}: inertia={km.inertia_:.1f}")

    # --- final clustering at config.N_STYLE_CLUSTERS -------------------------
    kmeans = KMeans(
        n_clusters=config.N_STYLE_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=10,
    )
    df["style_cluster"] = kmeans.fit_predict(X_scaled)
    df["style_name"] = df["style_cluster"].map(CLUSTER_NAME_MAP)

    print(f"\nFinal clustering at k={config.N_STYLE_CLUSTERS}:")
    print(df["style_cluster"].value_counts().sort_index())

    print("\nPer-cluster average style features (use this to name the clusters):")
    cluster_profile = df.groupby("style_cluster")[config.STYLE_FEATURES].mean().round(1)
    print(cluster_profile)

    out_path = f"{config.PROCESSED_DATA_DIR}/team_style_clusters.csv"
    df.to_csv(out_path, index=False)
    cluster_profile.to_csv(f"{config.PROCESSED_DATA_DIR}/cluster_profiles.csv")
    print(f"\nSaved {out_path} and cluster_profiles.csv")
    print(
        "\nNext: open cluster_profiles.csv, name each cluster based on its "
        "feature averages, and update CLUSTER_NAME_MAP at the top of this "
        "file. Re-run once named -- 04 onward reads 'style_name'."
    )


if __name__ == "__main__":
    main()
