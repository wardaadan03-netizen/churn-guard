"""
ChurnGuard
End-to-End Customer Churn Machine Learning Pipeline
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocess import (
    load_data,
    clean_data,
    create_features,
    build_preprocessor,
    get_feature_names,
    save_artifacts,
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = os.path.join(
    DATA_DIR,
    "Sales - Marketing customer dataset.csv"
)


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

def prepare_data(df):

    df = df.copy()

    if "churn" not in df.columns:
        raise ValueError(
            "The dataset does not contain a 'churn' column."
        )

    y = df["churn"]

    X = df.drop(
        columns=["churn"]
    )

    return X, y


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Active",
            "Churned"
        ],
        yticklabels=[
            "Active",
            "Churned"
        ]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name
        .replace(" ", "_")
        .lower()
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"cm_{filename}.png"
        ),
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names
):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return

    importances = (
        model.feature_importances_
    )

    indices = np.argsort(
        importances
    )[::-1][:20]

    plt.figure(
        figsize=(12, 8)
    )

    plt.bar(
        range(len(indices)),
        importances[indices]
    )

    plt.xticks(
        range(len(indices)),
        [
            feature_names[i]
            for i in indices
        ],
        rotation=45,
        ha="right"
    )

    plt.title(
        "Top 20 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        "✅ Feature importance saved."
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("🚀 CHURNGUARD TRAINING PIPELINE")
    print("=" * 60)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\n📥 Loading dataset...")

    print(
        f"📍 Dataset path:\n{DATA_PATH}"
    )

    df = load_data(
        DATA_PATH
    )

    print(
        f"✅ Raw data shape: {df.shape}"
    )

    # -----------------------------------------------------
    # CLEAN DATA
    # -----------------------------------------------------

    print("\n🧹 Cleaning data...")

    df = clean_data(df)

    print(
        f"✅ After cleaning: {df.shape}"
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    print("\n⚙️ Creating features...")

    df = create_features(df)

    print(
        f"✅ Final dataset shape: {df.shape}"
    )

    # -----------------------------------------------------
    # X / Y
    # -----------------------------------------------------

    X, y = prepare_data(
        df
    )

    print("\n📊 Target distribution:")

    print(
        y.value_counts()
    )

    print(
        f"\n📈 Churn rate: {y.mean():.2%}"
    )

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\n✅ Training samples: {len(X_train)}"
    )

    print(
        f"✅ Testing samples: {len(X_test)}"
    )

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    print("\n🔧 Building preprocessing pipeline...")

    preprocessor = build_preprocessor()

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    feature_names = (
        get_feature_names(
            preprocessor
        )
    )

    print(
        f"✅ Transformed features: "
        f"{X_train_transformed.shape[1]}"
    )

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1
    )

    models = {

        "Logistic Regression":
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss"
            )
    }

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    results = []

    best_model = None
    best_model_name = None
    best_f1 = -1

    print("\n🤖 Training models...")

    for name, model in models.items():

        print(
            f"\n➡️ Training {name}..."
        )

        model.fit(
            X_train_transformed,
            y_train
        )

        y_pred = model.predict(
            X_test_transformed
        )

        y_probability = (
            model.predict_proba(
                X_test_transformed
            )[:, 1]
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        results.append({

            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc

        })

        print(
            f"   Accuracy:  {accuracy:.4f}"
        )

        print(
            f"   Precision: {precision:.4f}"
        )

        print(
            f"   Recall:    {recall:.4f}"
        )

        print(
            f"   F1:        {f1:.4f}"
        )

        print(
            f"   ROC-AUC:   {roc_auc:.4f}"
        )

        plot_confusion_matrix(
            y_test,
            y_pred,
            name
        )

        if f1 > best_f1:

            best_f1 = f1

            best_model = model

            best_model_name = name

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE")
    print("=" * 60)

    print(
        results_df.round(4)
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\n✅ Metrics saved to:\n{results_path}"
    )

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    print("\n🏆 BEST MODEL")

    print(
        f"Model: {best_model_name}"
    )

    print(
        f"F1 Score: {best_f1:.4f}"
    )

    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------

    print(
        "\n🔄 Running 5-fold cross-validation..."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        best_model,
        X_train_transformed,
        y_train,
        cv=cv,
        scoring="f1"
    )

    print(
        f"CV F1: {cv_scores.mean():.4f}"
    )

    print(
        f"CV Std: {cv_scores.std():.4f}"
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    plot_feature_importance(
        best_model,
        feature_names
    )

    # -----------------------------------------------------
    # SAVE ARTIFACTS
    # -----------------------------------------------------

    print(
        "\n💾 Saving model artifacts..."
    )

    save_artifacts(
        preprocessor,
        feature_names,
        best_model
    )

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    y_pred_best = (
        best_model.predict(
            X_test_transformed
        )
    )

    print(
        "\n📋 Classification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred_best,
            target_names=[
                "Active",
                "Churned"
            ],
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # VERIFY FILES
    # -----------------------------------------------------

    print("\n🔎 VERIFYING MODEL FILES...")

    required_files = [

        os.path.join(
            MODEL_DIR,
            "best_model.pkl"
        ),

        os.path.join(
            MODEL_DIR,
            "preprocessor.pkl"
        ),

        os.path.join(
            MODEL_DIR,
            "feature_names.json"
        )
    ]

    all_exist = True

    for file_path in required_files:

        if os.path.exists(file_path):

            size = (
                os.path.getsize(
                    file_path
                ) / 1024
            )

            print(
                f"✅ {file_path} "
                f"({size:.1f} KB)"
            )

        else:

            print(
                f"❌ MISSING: {file_path}"
            )

            all_exist = False

    if all_exist:

        print(
            "\n🎉 TRAINING COMPLETE!"
        )

        print(
            "Your model files are ready."
        )

    else:

        print(
            "\n❌ Some model files are missing."
        )


if __name__ == "__main__":
    main()