"""
Model training pipeline for customer churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, classification_report)
import matplotlib
# Fix: Use 'Agg' backend to prevent GUI errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
import os

# Import SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

from preprocess import load_data, clean_data, create_features, build_preprocessor, save_artifacts

def prepare_data(df):
    """Prepare features and target for modeling."""
    df = df.copy()
    
    # Target variable
    y = df['churn']
    
    # Features - drop churn
    X = df.drop('churn', axis=1)
    
    return X, y

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics, y_pred, y_proba

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir='../outputs/'):
    """Plot confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cm_{model_name.replace(" ", "_")}.png', dpi=150)
    plt.close()

def shap_explain(model, X_sample, preprocessor, feature_names, output_dir='../outputs/'):
    """Generate SHAP explanations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform data
    X_transformed = preprocessor.transform(X_sample)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed[:100])
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_transformed[:100], 
                      feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot of top features
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed[:100], 
                      feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP plots saved to {output_dir}/")
    return shap_values

def feature_importance_plot(model, feature_names, output_dir='../outputs/'):
    """Plot feature importance for tree-based models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(min(20, len(indices))), 
                importances[indices[:20]], 
                align='center')
        plt.xticks(range(min(20, len(indices))), 
                   [feature_names[i] for i in indices[:20]], 
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
        plt.close()
        print(f"✅ Feature importance plot saved to {output_dir}/")

def get_feature_names(preprocessor, X_train):
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
    
    # Get categorical feature names from one-hot encoder
    cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_features = [
        'gender', 'country', 'acquisition_channel', 'device_type',
        'subscription_type', 'is_premium_user', 'payment_method',
        'used_coupon', 'is_loyal'
    ]
    cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
    
    return numeric_features + cat_feature_names.tolist()

def main():
    """Main training pipeline."""
    print("🚀 Starting ChurnGuard Training Pipeline")
    print("📊 Dataset: Sales & Marketing Customer Dataset")
    
    # Create output directory
    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # 1. Load and clean data
    print("📥 Loading data...")
    
    # Try multiple possible filenames
    possible_files = [
        'data/sales_marketing_customer.csv',
        'data/Sales - Marketing customer dataset.csv',
        'data/Sales_Marketing_Customer_Dataset.csv',
        'data/customer_dataset.csv'
    ]
    
    df = None
    for file_path in possible_files:
        try:
            df = load_data(file_path)
            print(f"✅ Loaded from: {file_path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("❌ Error: Dataset not found!")
        print("Please ensure your dataset is in the 'data/' folder with one of these names:")
        for f in possible_files:
            print(f"  - {f}")
        print("\n💡 Download from: https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset")
        return
    
    df = clean_data(df)
    df = create_features(df)
    print(f"✅ Data shape: {df.shape}")
    
    # 2. Prepare X and y
    X, y = prepare_data(df)
    print(f"✅ Target distribution:\n{y.value_counts()}")
    print(f"   Churn rate: {y.mean():.2%}")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 4. Build and fit preprocessor
    print("🔧 Building preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor, X_train)
    print(f"✅ Preprocessing complete: {X_train_transformed.shape[1]} features")
    
    # Apply SMOTE for class imbalance
    print("⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)
    print(f"✅ Balanced training set: {X_train_balanced.shape[0]} samples")
    print(f"   Class 0: {sum(y_train_balanced==0)}")
    print(f"   Class 1: {sum(y_train_balanced==1)}")
    
    # Feature selection to reduce noise
    print("🔍 Selecting top features...")
    selector = SelectKBest(f_classif, k=min(40, X_train_balanced.shape[1]))
    X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
    X_test_selected = selector.transform(X_test_transformed)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"✅ Selected {len(selected_features)} features")
    
    # ✅ SAVE THE SELECTOR
    joblib.dump(selector, '../models/selector.pkl')
    print(f"✅ Selector saved to ../models/selector.pkl")
    
    # Calculate scale_pos_weight AFTER SMOTE
    scale_pos_weight = len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1])
    print(f"✅ Scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 5. Train models with improved hyperparameters
    print("🤖 Training models...")
    
    models = {
        'Logistic_Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42, 
            C=1.0,
            solver='liblinear'
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample', 
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_selected, y_train_balanced)
        
        metrics, y_pred, y_proba = evaluate_model(model, X_test_selected, y_test, name)
        results[name] = metrics
        print(f"    F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, output_dir='../outputs/')
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
    
    # 6. Results summary
    print("\n📊 Model Performance Summary:")
    print("-" * 80)
    metrics_df = pd.DataFrame(results).T
    print(metrics_df.round(4))
    metrics_df.to_csv('../outputs/model_metrics.csv')
    
    print(f"\n🏆 Best Model: {best_model.__class__.__name__} (F1: {best_f1:.4f})")
    
    # 7. Cross-validation on best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_selected, y_train_balanced, cv=cv, scoring='f1')
    print(f"✅ Cross-validation F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 8. SHAP analysis
    print("🔍 Generating SHAP explanations...")
    shap_explain(best_model, X_test, preprocessor, selected_features, output_dir='../outputs/')
    
    # 9. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance_plot(best_model, selected_features, output_dir='../outputs/')
    
    # 10. Save artifacts
    save_artifacts(preprocessor, selected_features, best_model, output_dir='../models/')
    
    # 11. Classification report
    y_pred_best = best_model.predict(X_test_selected)
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Active', 'Churned']))
    
    print("\n✅ Training pipeline complete!")
    print("📁 Outputs saved to ../outputs/")
    print("📁 Models saved to ../models/")

if __name__ == "__main__":
    main()