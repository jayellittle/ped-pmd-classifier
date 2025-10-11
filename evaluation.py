# pyright: basic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from config import VERSION


def load_and_prepare_data(results_path, answers_path):
    results_df = pd.read_csv(results_path)
    answers_df = pd.read_csv(answers_path)

    results_df = results_df.sort_values(["video_name", "track_id"]).reset_index(
        drop=True
    )
    answers_df = answers_df.sort_values(["video_name", "track_id"]).reset_index(
        drop=True
    )

    y_pred = results_df["cluster"].values
    y_true = answers_df["cluster"].values

    return y_pred, y_true, results_df, answers_df


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)

    return accuracy, precision, recall, precision_per_class, recall_per_class


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    plt.figure(figsize=(12, 10))
    plt.rcParams.update({"font.size": 14})
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 72},
        cbar_kws={"shrink": 0.8},
        square=True,
        linewidths=2,
        linecolor="white",
    )
    plt.title("Confusion Matrix", fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=16, fontweight="bold")

    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        f"results/{VERSION}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return cm


def main():
    results_path = f"results/{VERSION}/kmeans_results.csv"
    answers_path = f"results/{VERSION}/kmeans_answers.csv"

    y_pred, y_true, results_df, answers_df = load_and_prepare_data(
        results_path, answers_path
    )

    print("=== K-Means Clustering Evaluation ===\n")

    print(f"Total Samples: {len(y_true)}")
    print(f"Class (True): {np.bincount(y_true)}")
    print(f"Class (Predicted): {np.bincount(y_pred)}\n")

    accuracy, precision, recall, precision_per_class, recall_per_class = (
        calculate_metrics(y_true, y_pred)
    )

    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision - Weighted: {precision:.4f}")
    print(f"Recall - Weighted: {recall:.4f}\n")

    print("=== Evaluation Metrics by Class ===")
    for i in range(len(precision_per_class)):
        print(f"Class {i}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")

    class_names = ["Pedestrian", "PMD"]
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()
