import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1000, output_path='data/credit_risk_data.csv'):
    np.random.seed(42)
    
    data = {
        'applicant_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 75, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'loan_intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], n_samples),
        'loan_term': np.random.choice([3, 6, 12, 24, 36, 48, 60], n_samples),
        'previous_defaults': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'debt_to_income': np.random.uniform(0.05, 0.6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target 'loan_status' (1 for default, 0 for paid) based on some logic
    # Higher risk if low credit score, high debt_to_income, previous defaults
    risk_score = (
        (850 - df['credit_score']) / 550 * 0.4 +
        df['debt_to_income'] * 0.3 +
        df['previous_defaults'] * 0.3 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['loan_status'] = (risk_score > 0.5).astype(int)
    
    # Add some noise/missing values to make it realistic
    df.loc[np.random.choice(df.index, 20), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 10), 'employment_years'] = np.nan
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
