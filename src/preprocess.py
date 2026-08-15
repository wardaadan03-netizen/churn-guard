"""
ChurnGuard - Data Preprocessing Pipeline
Customer churn prediction using the Sales & Marketing dataset.
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):
    """Load the Sales & Marketing customer dataset."""

    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """Clean and prepare raw customer data."""

    df = df.copy()

    # --------------------------------------------------------
    # Numeric missing values
    # --------------------------------------------------------

    numeric_median_columns = [
        "satisfaction_score",
        "nps_score",
        "age",
        "total_spent",
        "avg_order_value",
        "marketing_spend_per_user"
    ]

    for column in numeric_median_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

    # --------------------------------------------------------
    # Age limits
    # --------------------------------------------------------

    if "age" in df.columns:

        df["age"] = df["age"].clip(
            lower=18,
            upper=90
        )

    # --------------------------------------------------------
    # Coupon usage
    # --------------------------------------------------------

    if "coupon_code" in df.columns:

        df["used_coupon"] = (
            df["coupon_code"]
            .notna()
            .astype(int)
        )

        df.drop(
            "coupon_code",
            axis=1,
            inplace=True
        )

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    if "signup_date" in df.columns:

        df["signup_date"] = pd.to_datetime(
            df["signup_date"],
            errors="coerce"
        )

    if "last_purchase_date" in df.columns:

        df["last_purchase_date"] = pd.to_datetime(
            df["last_purchase_date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Date-based features
    # --------------------------------------------------------

    reference_date = pd.Timestamp.today().normalize()

    if "signup_date" in df.columns:

        df["days_since_signup"] = (
            reference_date - df["signup_date"]
        ).dt.days

        df["days_since_signup"] = (
            df["days_since_signup"]
            .fillna(
                df["days_since_signup"].median()
            )
            .clip(lower=0)
        )

    if "last_purchase_date" in df.columns:

        df["days_since_last_purchase"] = (
            reference_date - df["last_purchase_date"]
        ).dt.days

        df["days_since_last_purchase"] = (
            df["days_since_last_purchase"]
            .fillna(
                df["days_since_last_purchase"].median()
            )
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # Remove original dates
    # --------------------------------------------------------

    for column in [
        "signup_date",
        "last_purchase_date"
    ]:

        if column in df.columns:

            df.drop(
                column,
                axis=1,
                inplace=True
            )

    # --------------------------------------------------------
    # Remove customer ID
    # --------------------------------------------------------

    if "customer_id" in df.columns:

        df.drop(
            "customer_id",
            axis=1,
            inplace=True
        )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Create additional customer behavior features."""

    df = df.copy()

    # --------------------------------------------------------
    # Engagement score
    # --------------------------------------------------------

    df["engagement_score"] = (
        df["total_visits"]
        * df["avg_session_time"]
        * df["pages_per_session"]
    )

    df["engagement_score"] = (
        df["engagement_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 1000)
    )

    # --------------------------------------------------------
    # Revenue per visit
    # --------------------------------------------------------

    df["revenue_per_visit"] = (
        df["total_spent"]
        / df["total_visits"].replace(0, np.nan)
    )

    df["revenue_per_visit"] = (
        df["revenue_per_visit"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Customer risk score
    # --------------------------------------------------------

    df["risk_score"] = (
        df["support_tickets"] * 0.3
        + df["refund_requested"] * 0.7
    )

    # --------------------------------------------------------
    # Loyalty indicator
    # --------------------------------------------------------

    df["is_loyal"] = (
        (df["nps_score"] >= 8)
        & (df["is_premium_user"] == 1)
    ).astype(int)

    # --------------------------------------------------------
    # Email interaction rate
    # --------------------------------------------------------

    df["interaction_rate"] = (
        df["email_click_rate"]
        / (df["email_open_rate"] + 0.01)
    )

    df["interaction_rate"] = (
        df["interaction_rate"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():
    """Build numerical and categorical preprocessing pipeline."""

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
        "satisfaction_score",
        "nps_score",
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
        "city",
        "acquisition_channel",
        "device_type",
        "subscription_type",
        "payment_method",
        "used_coupon"
    ]

    numeric_pipeline = Pipeline(
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

    categorical_pipeline = Pipeline(
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
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor):
    """Return feature names after preprocessing."""

    feature_names = []

    # Numerical features
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
        "satisfaction_score",
        "nps_score",
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

    feature_names.extend(numeric_features)

    # Categorical features
    categorical_features = [
        "gender",
        "country",
        "city",
        "acquisition_channel",
        "device_type",
        "subscription_type",
        "payment_method",
        "used_coupon"
    ]

    encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    encoded_names = encoder.get_feature_names_out(
        categorical_features
    )

    feature_names.extend(
        encoded_names.tolist()
    )

    return feature_names


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    preprocessor,
    feature_names,
    model,
    output_dir="models"
):
    """Save trained model and preprocessing artifacts."""

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    joblib.dump(
        model,
        os.path.join(
            output_dir,
            "best_model.pkl"
        )
    )

    joblib.dump(
        preprocessor,
        os.path.join(
            output_dir,
            "preprocessor.pkl"
        )
    )

    with open(
        os.path.join(
            output_dir,
            "feature_names.json"
        ),
        "w"
    ) as file:

        json.dump(
            feature_names,
            file,
            indent=2
        )

    print(
        f"✅ Artifacts saved to {output_dir}/"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    filepath = (
        "data/"
        "Sales - Marketing customer dataset.csv"
    )

    df = load_data(filepath)

    print(
        f"Loaded dataset: {df.shape}"
    )

    df = clean_data(df)

    df = create_features(df)

    print(
        f"Processed dataset: {df.shape}"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )