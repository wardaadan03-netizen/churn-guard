"""
Data preprocessing pipeline for customer churn prediction.
This handles the Sales & Marketing dataset with its intentional messiness.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import re

def load_data(filepath):
    """Load the Sales & Marketing customer dataset."""
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def clean_data(df):
    """Clean the dataset - handle missing values, outliers, and type conversions."""
    df = df.copy()
    
    # Handle missing values in key columns
    # satisfaction_score: fill with median
    df['satisfaction_score'].fillna(df['satisfaction_score'].median(), inplace=True)
    
    # nps_score: fill with median
    df['nps_score'].fillna(df['nps_score'].median(), inplace=True)
    
    # age: cap outliers at 18-90 range and fill missing with median
    df['age'] = df['age'].clip(18, 90)
    df['age'].fillna(df['age'].median(), inplace=True)
    
    # total_spent: fill missing with median
    df['total_spent'].fillna(df['total_spent'].median(), inplace=True)
    
    # avg_order_value: fill missing with median
    df['avg_order_value'].fillna(df['avg_order_value'].median(), inplace=True)
    
    # marketing_spend_per_user: fill missing with median
    df['marketing_spend_per_user'].fillna(df['marketing_spend_per_user'].median(), inplace=True)
    
    # Handle coupon_code - convert to binary (used/not used)
    df['used_coupon'] = df['coupon_code'].notna().astype(int)
    df.drop('coupon_code', axis=1, inplace=True)
    
    # Convert signup_date and last_purchase_date to datetime
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    
    # Feature: days since signup
    df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
    df['days_since_signup'].fillna(df['days_since_signup'].median(), inplace=True)
    
    # Feature: days since last purchase (churn indicator)
    df['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days
    df['days_since_last_purchase'].fillna(df['days_since_last_purchase'].max(), inplace=True)
    
    # Drop original date columns (we've engineered features from them)
    df.drop(['signup_date', 'last_purchase_date'], axis=1, inplace=True)
    
    # Drop customer_id - not useful for modeling
    df.drop('customer_id', axis=1, inplace=True)
    
    return df

def create_features(df):
    """Create additional features for better prediction."""
    df = df.copy()
    
    # Engagement score (combination of visits and session time)
    df['engagement_score'] = (df['total_visits'] * df['avg_session_time'] * df['pages_per_session']).clip(0, 1000)
    
    # Revenue per visit
    df['revenue_per_visit'] = df['total_spent'] / (df['total_visits'] + 1)
    
    # Customer risk score (based on support tickets and refunds)
    df['risk_score'] = (df['support_tickets'] * 0.3 + df['refund_requested'] * 0.7)
    
    # Loyalty indicator (high NPS + premium user)
    df['is_loyal'] = ((df['nps_score'] >= 8) & (df['is_premium_user'] == 1)).astype(int)
    
    # Interaction rate (clicks relative to opens)
    df['interaction_rate'] = df['email_click_rate'] / (df['email_open_rate'] + 0.01)
    
    return df

def build_preprocessor():
    """Build the preprocessing pipeline."""
    # Define column groups
    numeric_features = [
        'age', 'total_visits', 'avg_session_time', 'pages_per_session',
        'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value',
        'discount_used', 'support_tickets', 'refund_requested', 'delivery_delay_days',
        'marketing_spend_per_user', 'lifetime_value', 'last_3_month_purchase_freq',
        'days_since_signup', 'days_since_last_purchase', 'engagement_score',
        'revenue_per_visit', 'risk_score'
    ]
    
    categorical_features = [
        'gender', 'country', 'acquisition_channel', 'device_type',
        'subscription_type', 'is_premium_user', 'payment_method',
        'used_coupon', 'is_loyal'
    ]
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def get_feature_names(preprocessor, df):
    """Get feature names after preprocessing."""
    # Get numeric feature names
    numeric_features = [
        'age', 'total_visits', 'avg_session_time', 'pages_per_session',
        'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value',
        'discount_used', 'support_tickets', 'refund_requested', 'delivery_delay_days',
        'marketing_spend_per_user', 'lifetime_value', 'last_3_month_purchase_freq',
        'days_since_signup', 'days_since_last_purchase', 'engagement_score',
        'revenue_per_visit', 'risk_score'
    ]
    
    # Get categorical feature names
    cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_features = [
        'gender', 'country', 'acquisition_channel', 'device_type',
        'subscription_type', 'is_premium_user', 'payment_method',
        'used_coupon', 'is_loyal'
    ]
    cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
    
    return numeric_features + cat_feature_names.tolist()

def save_artifacts(preprocessor, feature_names, model, output_dir='models/'):
    """Save preprocessing artifacts and model."""
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(preprocessor, f'{output_dir}/preprocessor.pkl')
    with open(f'{output_dir}/feature_names.json', 'w') as f:
        import json
        json.dump(feature_names, f)
    joblib.dump(model, f'{output_dir}/best_model.pkl')
    
    print(f"✅ Artifacts saved to {output_dir}/")

if __name__ == "__main__":
    # Test the pipeline
    df = load_data('../data/sales_marketing_customer.csv')
    df = clean_data(df)
    df = create_features(df)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")