import pandas as pd

def engineer_features(df):
    # Loan-to-Income Ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    
    # Age-Employment Ratio (Stability)
    df['employment_stability'] = df['employment_years'] / df['age']
    
    # Combined Risk Factor (Heuristic)
    df['risk_factor'] = (df['debt_to_income'] * 0.5) + (df['previous_defaults'] * 0.5)
    
    return df

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    df = load_data('data/credit_risk_data.csv')
    df = preprocess_data(df)
    df = engineer_features(df)
    print("Feature engineering complete.")
    print(df.head())
