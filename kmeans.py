# pyright: basic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from config import KMEANS_DIR

FEATURE_1 = "y_std"
FEATURE_2 = "straightness_ratio"


def perform_clustering(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Cannot find '{csv_path}'")
        return

    # Features used for Clustering
    features = df[[FEATURE_1, FEATURE_2]]

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(features_scaled)

    # Analysis
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    print("--- Cluster Centroids (Original Scale) ---")
    print(pd.DataFrame(centroids_original, columns=[FEATURE_1, FEATURE_2]))
    print("\n")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=FEATURE_2,
        y=FEATURE_1,
        hue="cluster",
        data=df,
        palette="viridis",
        s=100,  # Size
        alpha=0.8,  # Opacity
    )

    plt.scatter(
        centroids_original[:, 1],
        centroids_original[:, 0],
        marker="x",
        s=300,
        c="red",
        label="Centroids",
    )

    plt.title("Clustering of Pedestrians and PMD Users", fontsize=16)
    plt.xlabel(f"Feature 2 ({FEATURE_2})", fontsize=12)
    plt.ylabel(f"Feature 1 ({FEATURE_1})", fontsize=12)
    plt.legend(title="Cluster")
    plt.grid(True)

    plt.yscale("log")
    plt.title("Clustering Result (Y-axis in Log Scale)", fontsize=16)

    output_image_path = f"{KMEANS_DIR}/kmeans_result.png"
    plt.savefig(output_image_path)
    print(f"Clustering visualization saved to '{output_image_path}'")
    plt.show()

    output_csv_path = f"{KMEANS_DIR}/kmeans_results.csv"
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Clustering results with labels saved to '{output_csv_path}'")


if __name__ == "__main__":
    perform_clustering(f"{KMEANS_DIR}/kmeans_input.csv")
