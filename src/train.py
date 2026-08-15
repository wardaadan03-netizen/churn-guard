"""
ChurnGuard - Machine Learning Training Pipeline
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
    save_artifacts
)

warnings.filterwarnings("ignore")


# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "Sales - Marketing customer dataset.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    y = df["churn"]

    X = df.drop(
        "churn",
        axis=1
    )

    return X, y


# ============================================================
# CONFUSION MATRIX
# ============================================================

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
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
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

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_{filename}.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance_plot(
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
        figsize=(10, 7)
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
        "Top 20 Feature Importance"
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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                y_probability
            )
    }

    return metrics, y_pred


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "🚀 Starting ChurnGuard Training Pipeline"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n📥 Loading dataset...")

    df = load_data(
        DATA_PATH
    )

    print(
        f"✅ Raw dataset: {df.shape}"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    print("\n🧹 Cleaning data...")

    df = clean_data(df)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print(
        "\n⚙️ Creating features..."
    )

    df = create_features(df)

    print(
        f"✅ Final dataset: {df.shape}"
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print(
        "\n🎯 Target distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        f"\n📊 Churn rate: {y.mean():.2%}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print(
        "\n🔧 Building preprocessing pipeline..."
    )

    preprocessor = (
        build_preprocessor()
    )

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
        "Processed features:",
        X_train_transformed.shape[1]
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    negative = sum(
        y_train == 0
    )

    positive = sum(
        y_train == 1
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
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
                eval_metric="logloss",
                random_state=42
            )
    }

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\n🤖 Training models..."
    )

    results = {}

    best_model = None
    best_model_name = None
    best_f1 = -1

    for name, model in models.items():

        print(
            f"\n➡️ Training {name}..."
        )

        model.fit(
            X_train_transformed,
            y_train
        )

        metrics, y_pred = (
            evaluate_model(
                model,
                X_test_transformed,
                y_test
            )
        )

        results[name] = metrics

        print(
            f"Accuracy : {metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['Recall']:.4f}"
        )

        print(
            f"F1       : {metrics['F1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['ROC-AUC']:.4f}"
        )

        plot_confusion_matrix(
            y_test,
            y_pred,
            name
        )

        if metrics["F1"] > best_f1:

            best_f1 = metrics["F1"]

            best_model = model

            best_model_name = name

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "📊 MODEL COMPARISON"
    )

    print(
        "=" * 65
    )

    results_df = pd.DataFrame(
        results
    ).T

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    )

    print(
        results_df.round(4)
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_metrics.csv"
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    print(
        f"\n🏆 Best Model: {best_model_name}"
    )

    print(
        f"🏆 Best F1 Score: {best_f1:.4f}"
    )

    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

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
        f" ± {cv_scores.std():.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\n📈 Creating feature importance..."
    )

    feature_importance_plot(
        best_model,
        feature_names
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    print(
        "\n💾 Saving model artifacts..."
    )

    save_artifacts(
        preprocessor,
        feature_names,
        best_model,
        output_dir=MODEL_DIR
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "✅ CHURNGUARD TRAINING COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        "\nCreated:"
    )

    print(
        "📁 models/best_model.pkl"
    )

    print(
        "📁 models/preprocessor.pkl"
    )

    print(
        "📁 models/feature_names.json"
    )

    print(
        "📁 outputs/model_metrics.csv"
    )

    print(
        "📁 outputs/confusion_matrix_*.png"
    )

    print(
        "📁 outputs/feature_importance.png"
    )


if __name__ == "__main__":
    main()