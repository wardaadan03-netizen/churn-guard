"""
Data preprocessing pipeline for ChurnGuard.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(filepath):
    """Load the customer dataset."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nDataset not found:\n{filepath}\n\n"
            f"Make sure the CSV is inside the data folder."
        )

    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

def clean_data(df):
    """Clean and prepare raw customer data."""

    df = df.copy()

    # -----------------------------
    # Missing values
    # -----------------------------

    numeric_fill_columns = [
        "satisfaction_score",
        "nps_score",
        "age",
        "total_spent",
        "avg_order_value",
        "marketing_spend_per_user"
    ]

    for col in numeric_fill_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # -----------------------------
    # Age
    # -----------------------------

    if "age" in df.columns:
        df["age"] = df["age"].clip(18, 90)

    # -----------------------------
    # Coupon
    # -----------------------------

    if "coupon_code" in df.columns:
        df["used_coupon"] = df["coupon_code"].notna().astype(int)
        df.drop(columns=["coupon_code"], inplace=True)

    # -----------------------------
    # Boolean columns
    # -----------------------------

    boolean_columns = [
        "is_premium_user",
        "discount_used",
        "refund_requested"
    ]

    for col in boolean_columns:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map({
                        "yes": 1,
                        "true": 1,
                        "1": 1,
                        "no": 0,
                        "false": 0,
                        "0": 0
                    })
                )

            df[col] = df[col].fillna(0).astype(int)

    # -----------------------------
    # Dates
    # -----------------------------

    if "signup_date" in df.columns:

        df["signup_date"] = pd.to_datetime(
            df["signup_date"],
            errors="coerce"
        )

        df["days_since_signup"] = (
            pd.Timestamp.now() - df["signup_date"]
        ).dt.days

        df["days_since_signup"] = (
            df["days_since_signup"]
            .fillna(df["days_since_signup"].median())
            .clip(lower=0)
        )

    else:
        df["days_since_signup"] = 0

    if "last_purchase_date" in df.columns:

        df["last_purchase_date"] = pd.to_datetime(
            df["last_purchase_date"],
            errors="coerce"
        )

        df["days_since_last_purchase"] = (
            pd.Timestamp.now() - df["last_purchase_date"]
        ).dt.days

        max_days = df["days_since_last_purchase"].max()

        df["days_since_last_purchase"] = (
            df["days_since_last_purchase"]
            .fillna(max_days)
            .clip(lower=0)
        )

    else:
        df["days_since_last_purchase"] = 0

    # Remove original dates
    for col in ["signup_date", "last_purchase_date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # -----------------------------
    # Customer ID
    # -----------------------------

    if "customer_id" in df.columns:
        df.drop(columns=["customer_id"], inplace=True)

    return df


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(df):
    """Create additional business features."""

    df = df.copy()

    # Engagement
    df["engagement_score"] = (
        df["total_visits"]
        * df["avg_session_time"]
        * df["pages_per_session"]
    )

    df["engagement_score"] = df["engagement_score"].clip(0, 1000)

    # Revenue per visit
    df["revenue_per_visit"] = (
        df["total_spent"]
        / (df["total_visits"] + 1)
    )

    # Risk score
    df["risk_score"] = (
        df["support_tickets"] * 0.3
        + df["refund_requested"] * 0.7
    )

    # Loyalty
    df["is_loyal"] = (
        (df["nps_score"] >= 8)
        & (df["is_premium_user"] == 1)
    ).astype(int)

    # Email interaction
    df["interaction_rate"] = (
        df["email_click_rate"]
        / (df["email_open_rate"] + 0.01)
    )

    return df


# ---------------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------------

def build_preprocessor():

    numeric_features = [
        "age",
        "total_visits",
        "avg_session_time",
        "pages_per_session",
        "email_open_rate",
        "email_click_rate",
        "total_spent",
        "avg_order_value",
        "discount_used",
        "support_tickets",
        "refund_requested",
        "delivery_delay_days",
        "marketing_spend_per_user",
        "lifetime_value",
        "last_3_month_purchase_freq",
        "days_since_signup",
        "days_since_last_purchase",
        "engagement_score",
        "revenue_per_visit",
        "risk_score",
        "interaction_rate"
    ]

    categorical_features = [
        "gender",
        "country",
        "acquisition_channel",
        "device_type",
        "subscription_type",
        "payment_method",
        "used_coupon",
        "is_premium_user",
        "is_loyal"
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ---------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------

def get_feature_names(preprocessor):

    numeric_features = [
        "age",
        "total_visits",
        "avg_session_time",
        "pages_per_session",
        "email_open_rate",
        "email_click_rate",
        "total_spent",
        "avg_order_value",
        "discount_used",
        "support_tickets",
        "refund_requested",
        "delivery_delay_days",
        "marketing_spend_per_user",
        "lifetime_value",
        "last_3_month_purchase_freq",
        "days_since_signup",
        "days_since_last_purchase",
        "engagement_score",
        "revenue_per_visit",
        "risk_score",
        "interaction_rate"
    ]

    categorical_features = [
        "gender",
        "country",
        "acquisition_channel",
        "device_type",
        "subscription_type",
        "payment_method",
        "used_coupon",
        "is_premium_user",
        "is_loyal"
    ]

    cat_pipeline = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    cat_names = cat_pipeline.get_feature_names_out(
        categorical_features
    )

    return numeric_features + cat_names.tolist()


# ---------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------

def save_artifacts(
    preprocessor,
    feature_names,
    model
):

    os.makedirs(MODEL_DIR, exist_ok=True)

    preprocessor_path = os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    feature_path = os.path.join(
        MODEL_DIR,
        "feature_names.json"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    joblib.dump(
        model,
        model_path
    )

    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print("\n✅ MODEL ARTIFACTS SAVED")
    print(f"   Model:        {model_path}")
    print(f"   Preprocessor: {preprocessor_path}")
    print(f"   Features:     {feature_path}")