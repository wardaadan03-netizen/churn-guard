"""
ChurnGuard - Streamlit Customer Churn Prediction App
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🔮",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }

    .risk-high {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
    }

    .risk-medium {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        background-color: #ffa500;
        color: white;
    }

    .risk-low {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        background-color: #00c853;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

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

    if not os.path.exists(model_path):

        return None, None, None, (
            "best_model.pkl is missing. "
            "Run `python src/train.py` first."
        )

    if not os.path.exists(
        preprocessor_path
    ):

        return None, None, None, (
            "preprocessor.pkl is missing. "
            "Run `python src/train.py` first."
        )

    if not os.path.exists(
        feature_path
    ):

        return None, None, None, (
            "feature_names.json is missing. "
            "Run `python src/train.py` first."
        )

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
        ) as file:

            feature_names = json.load(
                file
            )

        return (
            model,
            preprocessor,
            feature_names,
            None
        )

    except Exception as error:

        return (
            None,
            None,
            None,
            str(error)
        )


# ============================================================
# HEADER
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

model, preprocessor, feature_names, error = (
    load_artifacts()
)

if error:

    st.error(
        f"⚠️ Error loading artifacts: {error}"
    )

    st.info(
        "Run this command from the Churn Guard "
        "project root:\n\n"
        "`python src/train.py`"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🎯 About ChurnGuard"
    )

    st.write(
        "ChurnGuard predicts whether a customer "
        "is likely to churn using machine learning."
    )

    st.divider()

    st.subheader(
        "🤖 Models"
    )

    st.write(
        "• Logistic Regression\n"
        "• Random Forest\n"
        "• XGBoost"
    )

    st.divider()

    st.subheader(
        "📊 Dataset"
    )

    st.write(
        "Sales & Marketing Customer Dataset"
    )

    st.divider()

    st.caption(
        "Built with Python, Scikit-learn, "
        "XGBoost and Streamlit."
    )


# ============================================================
# TABS
# ============================================================

tab_prediction, tab_insights = st.tabs(
    [
        "🧑 Customer Prediction",
        "📊 Model Insights"
    ]
)


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

with tab_prediction:

    st.header(
        "Predict Customer Churn"
    )

    st.write(
        "Enter customer information below "
        "to estimate their churn probability."
    )

    with st.form(
        "prediction_form"
    ):

        # ----------------------------------------------------
        # Demographics
        # ----------------------------------------------------

        st.subheader(
            "👤 Customer Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Other"
                ]
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=90,
                value=35
            )

            country = st.text_input(
                "Country",
                value="USA"
            )

            city = st.text_input(
                "City",
                value="New York"
            )

        with col2:

            acquisition_channel = (
                st.selectbox(
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
            )

            device_type = st.selectbox(
                "Device Type",
                [
                    "Mobile",
                    "Desktop",
                    "Tablet"
                ]
            )

            subscription_type = st.selectbox(
                "Subscription Type",
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

        with col3:

            is_premium_user = st.selectbox(
                "Premium User",
                [
                    "Yes",
                    "No"
                ]
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

            used_coupon = st.selectbox(
                "Coupon Used",
                [
                    "Yes",
                    "No"
                ]
            )

        # ----------------------------------------------------
        # Engagement
        # ----------------------------------------------------

        st.subheader(
            "📱 Engagement"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

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

        with col2:

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
                0.40,
                0.01
            )

        with col3:

            email_click_rate = st.slider(
                "Email Click Rate",
                0.0,
                1.0,
                0.10,
                0.01
            )

            purchase_frequency = (
                st.number_input(
                    "Purchases - Last 3 Months",
                    min_value=0,
                    value=3
                )
            )

        # ----------------------------------------------------
        # Financial
        # ----------------------------------------------------

        st.subheader(
            "💰 Financial & Support"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

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

            lifetime_value = st.number_input(
                "Lifetime Value",
                min_value=0.0,
                value=1000.0,
                step=50.0
            )

        with col2:

            support_tickets = st.number_input(
                "Support Tickets",
                min_value=0,
                value=1
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

        with col3:

            satisfaction_score = st.number_input(
                "Satisfaction Score",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

            nps_score = st.number_input(
                "NPS Score",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

            signup_days = st.number_input(
                "Days Since Signup",
                min_value=0,
                value=365
            )

            purchase_days = st.number_input(
                "Days Since Last Purchase",
                min_value=0,
                value=30
            )

        # ----------------------------------------------------
        # Submit
        # ----------------------------------------------------

        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================

    if submitted:

        try:

            # -----------------------------------------------
            # Convert categorical yes/no values
            # -----------------------------------------------

            premium_value = (
                1
                if is_premium_user == "Yes"
                else 0
            )

            discount_value = (
                1
                if discount_used == "Yes"
                else 0
            )

            refund_value = (
                1
                if refund_requested == "Yes"
                else 0
            )

            coupon_value = (
                1
                if used_coupon == "Yes"
                else 0
            )

            # -----------------------------------------------
            # Derived features
            # -----------------------------------------------

            engagement_score = (
                total_visits
                * avg_session_time
                * pages_per_session
            )

            engagement_score = min(
                engagement_score,
                1000
            )

            revenue_per_visit = (
                total_spent
                / total_visits
                if total_visits > 0
                else 0
            )

            risk_score = (
                support_tickets * 0.3
                + refund_value * 0.7
            )

            is_loyal = (
                1
                if (
                    nps_score >= 8
                    and premium_value == 1
                )
                else 0
            )

            interaction_rate = (
                email_click_rate
                / (email_open_rate + 0.01)
            )

            # -----------------------------------------------
            # Input dataframe
            # -----------------------------------------------

            input_data = {

                "gender":
                    gender,

                "age":
                    age,

                "country":
                    country,

                "city":
                    city,

                "acquisition_channel":
                    acquisition_channel,

                "device_type":
                    device_type,

                "subscription_type":
                    subscription_type,

                "is_premium_user":
                    premium_value,

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
                    discount_value,

                "support_tickets":
                    support_tickets,

                "refund_requested":
                    refund_value,

                "delivery_delay_days":
                    delivery_delay_days,

                "payment_method":
                    payment_method,

                "satisfaction_score":
                    satisfaction_score,

                "nps_score":
                    nps_score,

                "marketing_spend_per_user":
                    marketing_spend,

                "lifetime_value":
                    lifetime_value,

                "last_3_month_purchase_freq":
                    purchase_frequency,

                "used_coupon":
                    coupon_value,

                "days_since_signup":
                    signup_days,

                "days_since_last_purchase":
                    purchase_days,

                "engagement_score":
                    engagement_score,

                "revenue_per_visit":
                    revenue_per_visit,

                "risk_score":
                    risk_score,

                "interaction_rate":
                    interaction_rate,

                "is_loyal":
                    is_loyal
            }

            input_df = pd.DataFrame(
                [input_data]
            )

            # -----------------------------------------------
            # Transform
            # -----------------------------------------------

            input_transformed = (
                preprocessor.transform(
                    input_df
                )
            )

            # -----------------------------------------------
            # Prediction
            # -----------------------------------------------

            probability = (
                model.predict_proba(
                    input_transformed
                )[0][1]
            )

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            # -----------------------------------------------
            # Results
            # -----------------------------------------------

            st.divider()

            st.subheader(
                "📊 Prediction Results"
            )

            result1, result2, result3 = (
                st.columns(3)
            )

            with result1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )

            with result2:

                if probability >= 0.50:

                    risk = "HIGH"
                    css_class = "risk-high"

                elif probability >= 0.25:

                    risk = "MEDIUM"
                    css_class = "risk-medium"

                else:

                    risk = "LOW"
                    css_class = "risk-low"

                st.markdown(
                    f"""
                    <div class="{css_class}">
                    Risk: {risk}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result3:

                if prediction == 1:

                    st.error(
                        "⚠️ Customer likely to CHURN"
                    )

                else:

                    st.success(
                        "✅ Customer likely to STAY"
                    )

            # -----------------------------------------------
            # Risk drivers
            # -----------------------------------------------

            st.subheader(
                "🎯 Potential Risk Drivers"
            )

            drivers = []

            if purchase_days > 60:

                drivers.append(
                    "Low recent purchase activity"
                )

            if total_visits < 20:

                drivers.append(
                    "Low website engagement"
                )

            if purchase_frequency < 2:

                drivers.append(
                    "Low purchase frequency"
                )

            if support_tickets > 2:

                drivers.append(
                    "Multiple support tickets"
                )

            if refund_value == 1:

                drivers.append(
                    "Refund requested"
                )

            if email_open_rate < 0.20:

                drivers.append(
                    "Low email engagement"
                )

            if premium_value == 0:

                drivers.append(
                    "Customer is not a premium user"
                )

            if not drivers:

                st.success(
                    "No major manually identified "
                    "risk factors."
                )

            else:

                for driver in drivers:

                    st.warning(
                        f"📌 {driver}"
                    )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# MODEL INSIGHTS
# ============================================================

with tab_insights:

    st.header(
        "🤖 Model Insights"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    )

    if os.path.exists(
        metrics_path
    ):

        st.subheader(
            "📈 Model Performance"
        )

        metrics_df = pd.read_csv(
            metrics_path,
            index_col=0
        )

        st.dataframe(
            metrics_df.round(4),
            use_container_width=True
        )

    else:

        st.info(
            "Run the training pipeline to "
            "generate model metrics."
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )

    if os.path.exists(
        importance_path
    ):

        st.subheader(
            "📊 Feature Importance"
        )

        st.image(
            importance_path,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance will appear "
            "after model training."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    **ChurnGuard v1.0**

    Customer Churn Intelligence System

    Built with Python • Scikit-learn • XGBoost • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)