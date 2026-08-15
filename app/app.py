"""
Streamlit application for customer churn prediction using the Sales & Marketing dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="ChurnGuard - Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #FF4B4B;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        font-size: 1.4rem;
    }
    .risk-medium {
        background-color: #FFA500;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        font-size: 1.4rem;
    }
    .risk-low {
        background-color: #00C853;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    """Load model, preprocessor, and selector."""
    try:
        model = joblib.load('../models/best_model.pkl')
        preprocessor = joblib.load('../models/preprocessor.pkl')
        
        with open('../models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load selector if it exists
        selector = None
        if os.path.exists('../models/selector.pkl'):
            selector = joblib.load('../models/selector.pkl')
            print(f"✅ Selector loaded with {selector.n_features_in_} features")
        
        return model, preprocessor, feature_names, selector
    except Exception as e:
        st.error(f"⚠️ Error loading artifacts: {e}")
        return None, None, None, None

# Header
st.markdown('<h1 class="main-header">🔮 ChurnGuard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Churn Intelligence System for Sales & Marketing</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 About")
    st.markdown("This system predicts customer churn using a trained machine learning model.")
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Customers", "15,000")
    st.metric("Churn Rate", "~15%", "Industry standard")
    st.markdown("---")
    st.markdown("### 📈 Business Impact")
    st.metric("Retention Value", "$2.4M", "Annual potential saved")

# Load model
model, preprocessor, feature_names, selector = load_artifacts()

if model is None:
    st.stop()

# Main app
tab1, tab2 = st.tabs(["🧑 Single Customer Prediction", "📊 Model Insights"])

# Tab 1: Single Customer Prediction
with tab1:
    st.header("Predict Churn for a Single Customer")
    st.markdown("Enter customer details below to get a churn probability and risk assessment.")
    
    col1, col2, col3 = st.columns(3)
    
    with st.form("prediction_form"):
        with col1:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.slider("Age", 18, 90, 35, 1)
            country = st.selectbox("Country", ["Germany", "India", "UK", "France", "USA", "Canada", "Australia"])
            subscription_type = st.selectbox("Subscription Type", ["Monthly", "Annual"])
            is_premium_user = st.selectbox("Premium User", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"])
        
        with col2:
            st.subheader("📱 Engagement")
            total_visits = st.number_input("Total Website Visits", min_value=0, max_value=1000, value=50)
            avg_session_time = st.number_input("Avg Session Time (minutes)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
            pages_per_session = st.number_input("Pages per Session", min_value=0, max_value=50, value=5)
            email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.4, 0.01)
            email_click_rate = st.slider("Email Click Rate", 0.0, 0.5, 0.1, 0.01)
            last_3_month_purchase_freq = st.number_input("Purchase Frequency (last 3 months)", min_value=0, max_value=50, value=3)
        
        with col3:
            st.subheader("💰 Financial")
            total_spent = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
            avg_order_value = st.number_input("Avg Order Value ($)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
            discount_used = st.selectbox("Used Discount", ["Yes", "No"])
            support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=0)
            refund_requested = st.selectbox("Requested Refund", ["Yes", "No"])
            delivery_delay_days = st.number_input("Delivery Delay (days)", min_value=0, max_value=30, value=0)
            marketing_spend_per_user = st.number_input("Marketing Spend per User ($)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
            lifetime_value = st.number_input("Customer Lifetime Value ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
            
            acquisition_channel = st.selectbox("Acquisition Channel", 
                ["Organic", "Google Ads", "Facebook Ads", "Referral", "Email Campaign", "Other"])
            device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        try:
            # Prepare input data
            input_data = {
                'gender': gender,
                'age': age,
                'country': country,
                'acquisition_channel': acquisition_channel,
                'device_type': device_type,
                'subscription_type': subscription_type,
                'is_premium_user': 1 if is_premium_user == "Yes" else 0,
                'total_visits': total_visits,
                'avg_session_time': avg_session_time,
                'pages_per_session': pages_per_session,
                'email_open_rate': email_open_rate,
                'email_click_rate': email_click_rate,
                'total_spent': total_spent,
                'avg_order_value': avg_order_value,
                'discount_used': 1 if discount_used == "Yes" else 0,
                'support_tickets': support_tickets,
                'refund_requested': 1 if refund_requested == "Yes" else 0,
                'delivery_delay_days': delivery_delay_days,
                'payment_method': payment_method,
                'marketing_spend_per_user': marketing_spend_per_user,
                'lifetime_value': lifetime_value,
                'last_3_month_purchase_freq': last_3_month_purchase_freq,
                'used_coupon': 0
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Add derived features
            engagement_score = total_visits * avg_session_time * pages_per_session
            revenue_per_visit = total_spent / (total_visits + 1)
            risk_score = support_tickets * 0.3 + (1 if refund_requested == "Yes" else 0) * 0.7
            is_loyal = 1 if (is_premium_user == "Yes") else 0
            
            input_df['engagement_score'] = engagement_score
            input_df['revenue_per_visit'] = revenue_per_visit
            input_df['risk_score'] = risk_score
            input_df['is_loyal'] = is_loyal
            input_df['days_since_signup'] = 365
            input_df['days_since_last_purchase'] = 30
            
            # Preprocess
            input_transformed = preprocessor.transform(input_df)
            
            # Apply feature selection if selector exists
            if selector is not None:
                input_transformed = selector.transform(input_transformed)
            
            # Make prediction
            prob = model.predict_proba(input_transformed)[0, 1]
            prediction = 1 if prob >= 0.5 else 0
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric("Churn Probability", f"{prob:.1%}")
            
            with col_result2:
                risk_level = "High" if prob > 0.5 else "Medium" if prob > 0.25 else "Low"
                risk_class = "risk-high" if prob > 0.5 else "risk-medium" if prob > 0.25 else "risk-low"
                st.markdown(f'<div class="{risk_class}">Risk: {risk_level}</div>', unsafe_allow_html=True)
            
            with col_result3:
                if prediction == 1:
                    st.error("⚠️ Customer predicted to CHURN")
                else:
                    st.success("✅ Customer predicted to STAY")
            
            # Key drivers
            st.markdown("### 🎯 Key Risk Drivers")
            
            drivers = []
            if subscription_type == "Monthly":
                drivers.append("📌 Monthly subscription (higher churn risk)")
            if total_visits < 20:
                drivers.append("📌 Low website visits (under 20 visits)")
            if support_tickets > 2:
                drivers.append(f"📌 Multiple support tickets ({support_tickets} tickets)")
            if last_3_month_purchase_freq < 2:
                drivers.append("📌 Low purchase frequency")
            if is_premium_user == "No":
                drivers.append("📌 Non-premium user")
            if email_open_rate < 0.2:
                drivers.append("📌 Low email engagement")
            
            if drivers:
                for driver in drivers:
                    st.warning(driver)
            else:
                st.info("✅ No major risk factors detected. Customer appears to be low risk.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please make sure the model was trained correctly with `python src/train.py`")

# Tab 2: Model Insights
with tab2:
    st.header("🤖 Model Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Metrics")
        try:
            metrics_df = pd.read_csv('../outputs/model_metrics.csv', index_col=0)
            st.dataframe(metrics_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        except:
            st.info("Run training first to see metrics.")
    
    with col2:
        st.subheader("📊 Feature Importance")
        try:
            img_path = '../outputs/feature_importance.png'
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info("Run training first to generate feature importance plot.")
        except:
            st.info("Feature importance plot not available.")
    
    st.subheader("🔍 SHAP Explanation")
    try:
        img_path = '../outputs/shap_summary.png'
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Run training first to generate SHAP summary.")
    except:
        st.info("SHAP plot not available.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>ChurnGuard v1.0 | Built with Python, Streamlit, and XGBoost</p>
</div>
""", unsafe_allow_html=True)