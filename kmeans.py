# pyright: basic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from config import ANALYSIS_DIR


def perform_clustering(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Cannot find '{csv_path}'")
        return

    # Features used for Clustering
    features_to_use = ["walking_band_energy", "arm_angle_var"]
    features = df[features_to_use]

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
    print(pd.DataFrame(centroids_original, columns=features_to_use))
    print("\n")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=features_to_use[1],
        y=features_to_use[0],
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
    plt.xlabel(f"Feature 2 ({features_to_use[1]})", fontsize=12)
    plt.ylabel(f"Feature 1 ({features_to_use[0]})", fontsize=12)
    plt.legend(title="Cluster")
    plt.grid(True)

    # plt.yscale("log")
    # plt.title("Clustering Result (Y-axis in Log Scale)", fontsize=16)

    output_image_path = f"{ANALYSIS_DIR}/kmeans_result.png"
    plt.savefig(output_image_path)
    print(f"Clustering visualization saved to '{output_image_path}'")
    plt.show()

    output_csv_path = f"{ANALYSIS_DIR}/kmeans_results.csv"
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Clustering results with labels saved to '{output_csv_path}'")


def perform_3d_clustering(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Cannot find '{csv_path}'")
        return

    features_to_use = ["walking_band_energy", "straightness_ratio", "arm_angle_var"]
    features = df[features_to_use]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(features_scaled)

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    print("--- Cluster Centroids (Original Scale) ---")
    print(pd.DataFrame(centroids_original, columns=features_to_use))
    print("\n")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df[features_to_use[1]],  # X: straightness_ratio
        df[features_to_use[2]],  # Y: arm_angle_var
        df[features_to_use[0]],  # Z: walking_band_energy
        c=df["cluster"],
        cmap="viridis",
        s=60,
        alpha=0.8,
    )

    ax.scatter(
        centroids_original[:, 1],
        centroids_original[:, 2],
        centroids_original[:, 0],
        marker="x",
        s=300,
        c="red",
        label="Centroids",
    )

    ax.set_title("3D Clustering Result", fontsize=16)
    ax.set_xlabel(features_to_use[1], fontsize=12)
    ax.set_ylabel(features_to_use[2], fontsize=12)
    ax.set_zlabel(features_to_use[0], fontsize=12)
    ax.legend(*scatter.legend_elements(), title="Cluster")

    output_image_path_3d = f"{ANALYSIS_DIR}/kmeans_3d_result.png"
    plt.savefig(output_image_path_3d)
    print(f"3D Clustering visualization saved to '{output_image_path_3d}'")
    plt.show()

    print("Generating Pair Plot...")
    pair_plot = sns.pairplot(
        df,
        vars=features_to_use,
        hue="cluster",
        palette="viridis",
        plot_kws={"alpha": 0.7, "s": 80},
    )  # s: 점 크기

    pair_plot.fig.suptitle(
        "Pair Plot of Features by Cluster", y=1.02, fontsize=16
    )  # y: 제목 위치 조절

    output_image_path_pair = f"{ANALYSIS_DIR}/kmeans_pairplot_result.png"
    pair_plot.savefig(output_image_path_pair)
    print(f"Pair Plot visualization saved to '{output_image_path_pair}'")
    plt.show()

    output_csv_path = f"{ANALYSIS_DIR}/kmeans_3d_results.csv"
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Clustering results with labels saved to '{output_csv_path}'")


if __name__ == "__main__":
    perform_clustering(f"{ANALYSIS_DIR}/kmeans_input.csv")
    # perform_3d_clustering(f"{ANALYSIS_DIR}/kmeans_input.csv")
