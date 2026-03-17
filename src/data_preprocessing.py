import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values
    df['income'] = df['income'].fillna(df['income'].median())
    df['employment_years'] = df['employment_years'].fillna(df['employment_years'].median())
    
    # Drop applicant_id as it's not a feature
    if 'applicant_id' in df.columns:
        df = df.drop('applicant_id', axis=1)
        
    return df

def encode_categoricals(df):
    le = LabelEncoder()
    if 'loan_intent' in df.columns:
        df['loan_intent'] = le.fit_transform(df['loan_intent'])
    return df, le

def split_and_scale(df, target_col='loan_status'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

if __name__ == "__main__":
    df = load_data('data/credit_risk_data.csv')
    df = preprocess_data(df)
    df, le = encode_categoricals(df)
    X_train, X_test, y_train, y_test, scaler, columns = split_and_scale(df)
    print("Preprocessing complete.")
    print(f"Features: {list(columns)}")
