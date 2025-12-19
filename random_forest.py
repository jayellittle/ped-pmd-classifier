# pyright: basic

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


FEATURES = [
    "y_var",
    "straightness_ratio",
    "walking_band_energy",
    "arm_angle_var",
]


def evaluate_with_cv(features_to_use):
    df = pd.read_csv("results/v6/analysis_answer.csv")

    # 이상치 제거
    df = df[df["arm_angle_var"] > 0.0]

    X = df[features_to_use]
    y = df["class"]

    # K-Fold Cross-Validation (K=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 각 fold에서의 성능
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"5-Fold CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Std Deviation: {scores.std():.4f}")


def plot_learning_curve(features_to_use):
    df = pd.read_csv("results/v6/analysis_answer.csv")
    df = df[df["arm_angle_var"] > 0.0]

    X = df[features_to_use]
    y = df["class"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/v6/learning_curve.png")
    # plt.show()


def test_multiple_splits(features_to_use):
    df = pd.read_csv("results/v6/analysis_answer.csv")
    df = df[df["arm_angle_var"] > 0.0]

    X = df[features_to_use]
    y = df["class"]

    accuracies = []

    # 10번 다른 split으로 테스트
    for random_state in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"Random State {random_state}: {accuracy:.4f}")

    print(f"\nMean: {np.mean(accuracies):.4f}")
    print(f"Std:  {np.std(accuracies):.4f}")
    print(f"Min:  {np.min(accuracies):.4f}")
    print(f"Max:  {np.max(accuracies):.4f}")


def comprehensive_validation(features_to_use):
    """종합적인 과적합 검증"""

    print("=" * 50)
    print("1. K-Fold Cross-Validation")
    print("=" * 50)
    evaluate_with_cv(features_to_use)

    print("\n" + "=" * 50)
    print("2. Multiple Random State Test")
    print("=" * 50)
    test_multiple_splits(features_to_use)

    print("\n" + "=" * 50)
    print("3. Learning Curve")
    print("=" * 50)
    plot_learning_curve(features_to_use)


def analyze_feature_importance(features_to_use):
    try:
        df = pd.read_csv("results/v6/analysis_answer.csv")

    except FileNotFoundError as e:
        print(f"Error: Cannot load file '{e.filename}'.")
        return

    nan_rows = df[df["class"].isnull()]

    if not nan_rows.empty:
        print("--- 'class' 열에 결측치가 있는 행 ---")
        print(nan_rows)
        print("-" * 30)
    else:
        print("--- 'class' 열에 결측치가 없습니다. ---")
        print("-" * 30)

    X = df[features_to_use]
    y = df["class"]

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
    plt.savefig("results/v6/random_forest_analysis.png")
    # plt.show()

    print("--- Feature Correlations ---")
    correlation_matrix = X.corr()
    print(correlation_matrix)
    print("-" * 30)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("results/v6/feature_correlation.png")
    # plt.show()


    comprehensive_validation(FEATURES)


def test_all_combinations():
    import itertools

    df = pd.read_csv("results/v6/analysis_answer.csv")
    
    # 공정한 비교를 위해 모든 실험을 '모든 변수가 정상인 데이터'로 통일
    # 만약 특정 조합에서는 전체 데이터를 쓰고 싶다면 별도 로직이 필요하지만,
    # '변수의 성능'을 비교할 때는 데이터셋을 고정하는 것이 과학적입니다.
    df_filtered = df[df["arm_angle_var"] > 0.0].copy()
    y = df_filtered["class"]
    
    print(f"\n{'='*60}")
    print(f"Feature Combination Analysis (Samples: {len(df_filtered)})")
    print(f"{'='*60}")

    results = []

    # 1개부터 전체 개수까지 모든 조합 생성
    for r in range(1, len(FEATURES) + 1):
        for combo in itertools.combinations(FEATURES, r):
            features = list(combo)
            X = df_filtered[features]
            
            # CV 점수 계산
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            results.append({
                "features": ", ".join(features),
                "n_features": len(features),
                "mean_accuracy": mean_score,
                "std": std_score
            })

    # 결과 출력 (정확도 내림차순 정렬)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mean_accuracy", ascending=False)

    print(results_df.to_string(index=False, float_format="%.4f"))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # analyze_feature_importance(FEATURES)
    # comprehensive_validation(FEATURES)
    test_all_combinations()
