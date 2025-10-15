# pyright: basic

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def analyze_feature_importance():
    try:
        df = pd.read_csv("results/v5/kmeans_answers.csv")

    except FileNotFoundError as e:
        print(f"Error: Cannot load file '{e.filename}'.")
        return

    features_to_use = [
        "y_var",
        "straightness_ratio",
        "walking_band_energy",
        "arm_angle_var",
    ]
    X = df[features_to_use]
    y = df["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Data separated: train {len(X_train)}, validation {len(X_test)}")
    print("-" * 30)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest training completed.")
    print("-" * 30)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"** Model Validation Accuracy: {accuracy * 100:.2f}% **")
    print("-" * 30)

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": features_to_use, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("--- Feature Importances ---")
    print(feature_importance_df)
    print("-" * 30)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pedestrian", "PMD"],
        yticklabels=["Pedestrian", "PMD"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    sns.barplot(x="importance", y="feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig("results/v5/random_forest_analysis.png")
    plt.show()


if __name__ == "__main__":
    analyze_feature_importance()
