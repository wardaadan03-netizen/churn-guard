"""
Prediction utilities for customer churn classification.
Supports single predictions and batch predictions from CSV.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Union, Dict, List, Tuple

class ChurnPredictor:
    """
    Customer churn prediction class.
    Handles loading artifacts and making predictions.
    """
    
    def __init__(self, model_dir: str = '../models/'):
        """
        Initialize the predictor with trained artifacts.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load the trained model, preprocessor, and feature names."""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'best_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Load feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.json')
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            
            print(f"✅ Artifacts loaded successfully from {self.model_dir}")
            print(f"   Model: {self.model.__class__.__name__}")
            print(f"   Features: {len(self.feature_names)}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading artifacts: {e}")
            print("   Please run 'python train.py' first to train the model.")
            return False
    
    def preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data using the fitted preprocessor.
        
        Args:
            df: Input DataFrame with raw customer data
            
        Returns:
            Transformed features as numpy array
        """
        # Calculate derived features
        df_copy = df.copy()
        
        # Add engagement_score if not present
        if 'engagement_score' not in df_copy.columns:
            if all(col in df_copy.columns for col in ['total_visits', 'avg_session_time', 'pages_per_session']):
                df_copy['engagement_score'] = (
                    df_copy['total_visits'] * 
                    df_copy['avg_session_time'] * 
                    df_copy['pages_per_session']
                ).clip(0, 1000)
            else:
                df_copy['engagement_score'] = 0
        
        # Add revenue_per_visit if not present
        if 'revenue_per_visit' not in df_copy.columns:
            if 'total_spent' in df_copy.columns and 'total_visits' in df_copy.columns:
                df_copy['revenue_per_visit'] = df_copy['total_spent'] / (df_copy['total_visits'] + 1)
            else:
                df_copy['revenue_per_visit'] = 0
        
        # Add risk_score if not present
        if 'risk_score' not in df_copy.columns:
            risk_components = []
            if 'support_tickets' in df_copy.columns:
                risk_components.append(df_copy['support_tickets'] * 0.3)
            if 'refund_requested' in df_copy.columns:
                risk_components.append(df_copy['refund_requested'] * 0.7)
            df_copy['risk_score'] = sum(risk_components) if risk_components else 0
        
        # Add is_loyal if not present
        if 'is_loyal' not in df_copy.columns:
            df_copy['is_loyal'] = 0
        
        # Add placeholder columns if they don't exist
        if 'days_since_signup' not in df_copy.columns:
            df_copy['days_since_signup'] = 365
        if 'days_since_last_purchase' not in df_copy.columns:
            df_copy['days_since_last_purchase'] = 30
        if 'used_coupon' not in df_copy.columns:
            df_copy['used_coupon'] = 0
        
        # Transform using preprocessor
        transformed = self.preprocessor.transform(df_copy)
        return transformed
    
    def predict(self, customer_data: Union[pd.DataFrame, Dict]) -> Tuple[int, float, Dict]:
        """
        Predict churn for a single customer or batch.
        
        Args:
            customer_data: DataFrame or dictionary with customer features
            
        Returns:
            Tuple of (prediction (0/1), probability, risk factors dict)
        """
        # Convert dict to DataFrame if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Ensure all required columns exist
        required_cols = ['gender', 'age', 'country', 'acquisition_channel', 'device_type',
                        'subscription_type', 'is_premium_user', 'total_visits', 
                        'avg_session_time', 'pages_per_session', 'email_open_rate',
                        'email_click_rate', 'total_spent', 'avg_order_value',
                        'discount_used', 'support_tickets', 'refund_requested',
                        'delivery_delay_days', 'payment_method', 'marketing_spend_per_user',
                        'lifetime_value', 'last_3_month_purchase_freq']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['gender', 'country', 'acquisition_channel', 'device_type', 
                          'subscription_type', 'payment_method']:
                    df[col] = 'Unknown'
                elif col in ['is_premium_user', 'discount_used', 'refund_requested']:
                    df[col] = 0
                else:
                    df[col] = 0
        
        # Preprocess
        transformed = self.preprocess_input(df)
        
        # Get predictions
        probabilities = self.model.predict_proba(transformed)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Generate risk factors for each customer
        risk_factors = []
        for i in range(len(df)):
            prob = probabilities[i]
            pred = predictions[i]
            
            factors = {
                'prediction': int(pred),
                'probability': float(prob),
                'risk_level': 'High' if prob > 0.5 else 'Medium' if prob > 0.25 else 'Low',
                'key_drivers': self._get_risk_factors(df.iloc[i], prob)
            }
            risk_factors.append(factors)
        
        # Return single or batch
        if len(df) == 1:
            return predictions[0], probabilities[0], risk_factors[0]
        else:
            return predictions, probabilities, risk_factors
    
    def _get_risk_factors(self, customer: pd.Series, probability: float) -> Dict:
        """Identify key risk factors for a customer."""
        factors = {}
        
        # Contract type risk
        if customer.get('subscription_type') == 'Monthly':
            factors['Subscription'] = 'Monthly contract (higher risk)'
        elif customer.get('subscription_type') == 'Annual':
            factors['Subscription'] = 'Annual contract (lower risk)'
        
        # Engagement risk
        if customer.get('total_visits', 0) < 10:
            factors['Engagement'] = 'Low website visits'
        if customer.get('avg_session_time', 0) < 2:
            factors['Engagement'] = 'Short session duration'
        
        # Support risk
        if customer.get('support_tickets', 0) > 2:
            factors['Support'] = f"{customer.get('support_tickets', 0)} support tickets"
        
        # Purchase risk
        if customer.get('last_3_month_purchase_freq', 0) < 1:
            factors['Purchase'] = 'No recent purchases'
        
        # Email engagement
        if customer.get('email_open_rate', 0) < 0.2:
            factors['Email'] = 'Low email engagement'
        
        return factors
    
    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Batch predict from a CSV file.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with predictions appended
        """
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📥 Loaded {len(df)} customers from {csv_path}")
        
        # Make predictions
        predictions, probabilities, risk_factors = self.predict(df)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['churn_prediction'] = predictions
        result_df['churn_probability'] = probabilities
        result_df['risk_level'] = [f['risk_level'] for f in risk_factors]
        
        # Save if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
        
        # Summary statistics
        churn_count = predictions.sum()
        print(f"\n📊 Prediction Summary:")
        print(f"   Total customers: {len(predictions)}")
        print(f"   Predicted churn: {churn_count} ({churn_count/len(predictions):.1%})")
        print(f"   Average probability: {probabilities.mean():.2%}")
        
        return result_df

# Standalone functions for quick use
def quick_predict(customer_dict: Dict) -> Tuple[int, float, Dict]:
    """
    Quick function to predict churn for a single customer.
    
    Args:
        customer_dict: Dictionary of customer features
        
    Returns:
        Tuple of (prediction, probability, risk_factors)
    """
    predictor = ChurnPredictor()
    return predictor.predict(customer_dict)

def batch_predict(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Quick function for batch prediction from CSV.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to save results (optional)
        
    Returns:
        DataFrame with predictions
    """
    predictor = ChurnPredictor()
    return predictor.predict_from_csv(csv_path, output_path)

if __name__ == "__main__":
    # Example usage
    print("🔮 ChurnGuard Prediction Utility")
    print("=" * 40)
    
    # Example 1: Single customer prediction
    print("\n📋 Example 1: Single Customer Prediction")
    sample_customer = {
        'gender': 'Male',
        'age': 35,
        'country': 'USA',
        'acquisition_channel': 'Google Ads',
        'device_type': 'Mobile',
        'subscription_type': 'Monthly',
        'is_premium_user': 0,
        'total_visits': 25,
        'avg_session_time': 4.5,
        'pages_per_session': 3,
        'email_open_rate': 0.3,
        'email_click_rate': 0.08,
        'total_spent': 350.0,
        'avg_order_value': 50.0,
        'discount_used': 0,
        'support_tickets': 2,
        'refund_requested': 0,
        'delivery_delay_days': 1,
        'payment_method': 'Credit Card',
        'marketing_spend_per_user': 45.0,
        'lifetime_value': 1200.0,
        'last_3_month_purchase_freq': 2
    }
    
    predictor = ChurnPredictor()
    pred, prob, factors = predictor.predict(sample_customer)
    
    print(f"   Prediction: {'CHURN' if pred == 1 else 'STAY'}")
    print(f"   Probability: {prob:.1%}")
    print(f"   Risk Level: {factors['risk_level']}")
    print("   Key Drivers:")
    for key, value in factors['key_drivers'].items():
        print(f"      - {key}: {value}")
    
    # Example 2: Batch prediction
    print("\n📋 Example 2: Batch Prediction from CSV")
    # Uncomment below to run batch prediction
    # df = batch_predict('../data/test_customers.csv', '../outputs/batch_predictions.csv')