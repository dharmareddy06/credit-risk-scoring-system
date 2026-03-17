import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from data_preprocessing import load_data, preprocess_data, encode_categoricals, split_and_scale
from feature_engineering import engineer_features

def train_and_evaluate():
    # 1. Load and prepare data
    print("Loading data...")
    df = load_data('data/credit_risk_data.csv')
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Encoding categoricals...")
    df, le = encode_categoricals(df)
    
    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    
    # 2. Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_auc = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"{name} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        results[name] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'model': model
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with AUC: {best_auc:.4f}")
    
    # 3. Save the best model and artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': list(feature_names),
        'model_name': best_model_name,
        'background_sample': X_train[:100]  # Save first 100 rows as background
    }
    
    with open('models/credit_model_artifacts.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
        
    print("Model artifacts saved to models/credit_model_artifacts.pkl")
    
    # Generate a simple report
    with open('models/evaluation_report.txt', 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"ROC-AUC: {res['auc']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall: {res['recall']:.4f}\n\n")
        f.write(f"Chosen Model: {best_model_name}\n")

if __name__ == "__main__":
    train_and_evaluate()
