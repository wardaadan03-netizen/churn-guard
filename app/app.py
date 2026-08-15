"""
ChurnGuard
Streamlit Customer Churn Prediction Application
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🔮",
    layout="wide"
)


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }

    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }

    .risk-low {
        background-color: #00c853;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():

    model_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    preprocessor_path = os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )

    feature_path = os.path.join(
        MODEL_DIR,
        "feature_names.json"
    )

    # Check files
    missing = []

    if not os.path.exists(model_path):
        missing.append(model_path)

    if not os.path.exists(preprocessor_path):
        missing.append(preprocessor_path)

    if not os.path.exists(feature_path):
        missing.append(feature_path)

    if missing:

        st.error(
            "❌ Model artifacts are missing."
        )

        st.code(
            "\n".join(missing)
        )

        st.info(
            "Run this from the Churn Guard project root:"
        )

        st.code(
            "python src/train.py"
        )

        return None, None, None

    try:

        model = joblib.load(
            model_path
        )

        preprocessor = joblib.load(
            preprocessor_path
        )

        with open(
            feature_path,
            "r"
        ) as f:

            feature_names = json.load(
                f
            )

        return (
            model,
            preprocessor,
            feature_names
        )

    except Exception as e:

        st.error(
            f"❌ Error loading artifacts: {e}"
        )

        return None, None, None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-header">🔮 ChurnGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'AI-Powered Customer Churn Intelligence System'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model, preprocessor, feature_names = (
    load_artifacts()
)


if model is None:
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🎯 About ChurnGuard")

    st.write(
        """
        ChurnGuard predicts whether a customer
        is likely to churn using machine learning.
        """
    )

    st.divider()

    st.subheader("🤖 Models")

    st.write(
        """
        • Logistic Regression  
        • Random Forest  
        • XGBoost
        """
    )

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write(
        """
        Python  
        Scikit-learn  
        XGBoost  
        Pandas  
        Streamlit
        """
    )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2 = st.tabs(
    [
        "🔮 Churn Prediction",
        "📊 Model Insights"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header(
        "Customer Churn Prediction"
    )

    st.write(
        "Enter customer information to estimate churn risk."
    )

    with st.form(
        "prediction_form"
    ):

        col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # DEMOGRAPHICS
        # -------------------------------------------------

        with col1:

            st.subheader(
                "👤 Customer"
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Other"
                ]
            )

            age = st.slider(
                "Age",
                18,
                90,
                35
            )

            country = st.selectbox(
                "Country",
                [
                    "Germany",
                    "India",
                    "UK",
                    "France",
                    "USA",
                    "Canada",
                    "Australia"
                ]
            )

            acquisition_channel = st.selectbox(
                "Acquisition Channel",
                [
                    "Organic",
                    "Google Ads",
                    "Facebook Ads",
                    "Referral",
                    "Email Campaign",
                    "Other"
                ]
            )

            device_type = st.selectbox(
                "Device",
                [
                    "Mobile",
                    "Desktop",
                    "Tablet"
                ]
            )

            subscription_type = st.selectbox(
                "Subscription",
                [
                    "Monthly",
                    "Annual"
                ]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Credit Card",
                    "Debit Card",
                    "UPI",
                    "Net Banking",
                    "Wallet"
                ]
            )

            is_premium_user = st.selectbox(
                "Premium User",
                [
                    "Yes",
                    "No"
                ]
            )

        # -------------------------------------------------
        # ENGAGEMENT
        # -------------------------------------------------

        with col2:

            st.subheader(
                "📱 Engagement"
            )

            total_visits = st.number_input(
                "Total Visits",
                min_value=0,
                value=50
            )

            avg_session_time = st.number_input(
                "Average Session Time",
                min_value=0.0,
                value=5.0,
                step=0.5
            )

            pages_per_session = st.number_input(
                "Pages per Session",
                min_value=0.0,
                value=5.0,
                step=0.5
            )

            email_open_rate = st.slider(
                "Email Open Rate",
                0.0,
                1.0,
                0.40
            )

            email_click_rate = st.slider(
                "Email Click Rate",
                0.0,
                1.0,
                0.10
            )

            purchase_frequency = st.number_input(
                "Purchases - Last 3 Months",
                min_value=0,
                value=3
            )

            support_tickets = st.number_input(
                "Support Tickets",
                min_value=0,
                value=0
            )

        # -------------------------------------------------
        # FINANCIAL
        # -------------------------------------------------

        with col3:

            st.subheader(
                "💰 Financial"
            )

            total_spent = st.number_input(
                "Total Spent",
                min_value=0.0,
                value=500.0,
                step=10.0
            )

            avg_order_value = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=50.0,
                step=5.0
            )

            discount_used = st.selectbox(
                "Discount Used",
                [
                    "Yes",
                    "No"
                ]
            )

            refund_requested = st.selectbox(
                "Refund Requested",
                [
                    "Yes",
                    "No"
                ]
            )

            delivery_delay_days = st.number_input(
                "Delivery Delay Days",
                min_value=0,
                value=0
            )

            marketing_spend = st.number_input(
                "Marketing Spend per User",
                min_value=0.0,
                value=50.0,
                step=5.0
            )

            lifetime_value = st.number_input(
                "Lifetime Value",
                min_value=0.0,
                value=1000.0,
                step=50.0
            )

        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if submitted:

        try:

            premium = (
                1
                if is_premium_user == "Yes"
                else 0
            )

            discount = (
                1
                if discount_used == "Yes"
                else 0
            )

            refund = (
                1
                if refund_requested == "Yes"
                else 0
            )

            # Create raw input
            input_data = {

                "gender": gender,
                "age": age,
                "country": country,
                "acquisition_channel":
                    acquisition_channel,
                "device_type": device_type,
                "subscription_type":
                    subscription_type,
                "is_premium_user":
                    premium,
                "total_visits":
                    total_visits,
                "avg_session_time":
                    avg_session_time,
                "pages_per_session":
                    pages_per_session,
                "email_open_rate":
                    email_open_rate,
                "email_click_rate":
                    email_click_rate,
                "total_spent":
                    total_spent,
                "avg_order_value":
                    avg_order_value,
                "discount_used":
                    discount,
                "support_tickets":
                    support_tickets,
                "refund_requested":
                    refund,
                "delivery_delay_days":
                    delivery_delay_days,
                "payment_method":
                    payment_method,
                "marketing_spend_per_user":
                    marketing_spend,
                "lifetime_value":
                    lifetime_value,
                "last_3_month_purchase_freq":
                    purchase_frequency,
                "used_coupon":
                    0,

                # Date-derived features
                "days_since_signup":
                    365,
                "days_since_last_purchase":
                    30
            }

            input_df = pd.DataFrame(
                [input_data]
            )

            # -------------------------------------------------
            # Feature engineering
            # -------------------------------------------------

            input_df["engagement_score"] = (
                input_df["total_visits"]
                * input_df["avg_session_time"]
                * input_df["pages_per_session"]
            ).clip(0, 1000)

            input_df["revenue_per_visit"] = (
                input_df["total_spent"]
                / (
                    input_df["total_visits"]
                    + 1
                )
            )

            input_df["risk_score"] = (
                input_df["support_tickets"]
                * 0.3
                +
                input_df["refund_requested"]
                * 0.7
            )

            input_df["is_loyal"] = (
                (
                    input_df["is_premium_user"]
                    == 1
                )
                &
                (
                    input_df["nps_score"]
                    if "nps_score" in input_df
                    else True
                )
            ).astype(int)

            # Since NPS isn't collected in the UI,
            # use premium status as a simple loyalty proxy.
            input_df["is_loyal"] = (
                input_df["is_premium_user"]
                == 1
            ).astype(int)

            input_df["interaction_rate"] = (
                input_df["email_click_rate"]
                /
                (
                    input_df["email_open_rate"]
                    + 0.01
                )
            )

            # -------------------------------------------------
            # Transform
            # -------------------------------------------------

            transformed = (
                preprocessor.transform(
                    input_df
                )
            )

            # -------------------------------------------------
            # Predict
            # -------------------------------------------------

            probability = (
                model.predict_proba(
                    transformed
                )[0, 1]
            )

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.divider()

            st.header(
                "📊 Prediction Result"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )

            with c2:

                if probability >= 0.50:

                    st.markdown(
                        '<div class="risk-high">'
                        '🔴 HIGH RISK'
                        '</div>',
                        unsafe_allow_html=True
                    )

                elif probability >= 0.25:

                    st.markdown(
                        '<div class="risk-medium">'
                        '🟠 MEDIUM RISK'
                        '</div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        '<div class="risk-low">'
                        '🟢 LOW RISK'
                        '</div>',
                        unsafe_allow_html=True
                    )

            with c3:

                if prediction == 1:

                    st.error(
                        "⚠️ Customer likely to CHURN"
                    )

                else:

                    st.success(
                        "✅ Customer likely to STAY"
                    )

            # -------------------------------------------------
            # BUSINESS RECOMMENDATION
            # -------------------------------------------------

            st.subheader(
                "💡 Recommended Action"
            )

            if probability >= 0.50:

                st.warning(
                    """
                    **Immediate retention action recommended.**

                    Consider offering a personalized discount,
                    contacting the customer, improving support,
                    or providing a subscription incentive.
                    """
                )

            elif probability >= 0.25:

                st.info(
                    """
                    **Monitor this customer.**

                    Increase engagement through targeted
                    emails, offers, and personalized content.
                    """
                )

            else:

                st.success(
                    """
                    **Customer appears stable.**

                    Continue normal engagement and loyalty
                    activities.
                    """
                )

        except Exception as e:

            st.error(
                f"❌ Prediction error: {e}"
            )

            st.exception(e)


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header(
        "📊 Model Insights"
    )

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    )

    importance_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )

    if os.path.exists(
        metrics_path
    ):

        metrics_df = pd.read_csv(
            metrics_path
        )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        st.subheader(
            "🏆 Best Model"
        )

        best_row = metrics_df.loc[
            metrics_df["f1"].idxmax()
        ]

        st.metric(
            "Best Model",
            best_row["model"]
        )

        st.metric(
            "F1 Score",
            f"{best_row['f1']:.3f}"
        )

        st.metric(
            "ROC-AUC",
            f"{best_row['roc_auc']:.3f}"
        )

    else:

        st.info(
            "Model metrics will appear after training."
        )

    if os.path.exists(
        importance_path
    ):

        st.subheader(
            "🌟 Feature Importance"
        )

        st.image(
            importance_path,
            use_container_width=True
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777;">
        ChurnGuard | Machine Learning Fundamentals Capstone
        <br>
        Built with Python, Scikit-learn, XGBoost and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)