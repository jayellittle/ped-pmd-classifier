# pyright: basic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import ANALYSIS_DIR


def plot_feature_distributions(file_path):
    csv_file = f"{file_path}/charts.csv"

    try:
        df = pd.read_csv(csv_file)

        features = [
            "y_var",
            "straightness_ratio",
            "walking_band_energy",
            "arm_angle_var",
        ]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.boxplot(
                x="group",
                y=feature,
                data=df,
                ax=axes[i],
                palette="pastel",
                showfliers=False,
            )

            axes[i].set_title(f"Distribution of {feature} by Group", fontsize=14)
            axes[i].set_xlabel("Group", fontsize=12)
            axes[i].set_ylabel(feature, fontsize=12)

        fig.suptitle("Feature Distributions by Group", fontsize=18, fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_image_path = f"{file_path}/distributions.png"
        plt.savefig(output_image_path)

        print(f"Graph was saved as file '{output_image_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    plot_feature_distributions(ANALYSIS_DIR)
