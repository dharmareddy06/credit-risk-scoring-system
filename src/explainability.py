import shap
import pandas as pd
import pickle

def get_shap_explainer(model, X_train_sample):
    # For Logistic Regression, we might use LinearExplainer
    # For tree-based models, use TreeExplainer
    # KernelExplainer is a fallback for all
    if hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_train_sample)
    else:
        try:
            explainer = shap.TreeExplainer(model)
        except:
            explainer = shap.KernelExplainer(model.predict, X_train_sample)
    return explainer

def explain_prediction(model, scaler, feature_names, input_data, background_sample):
    # input_data should be a dataframe with raw features
    # 1. Scale input
    input_scaled = scaler.transform(input_data)
    input_df_scaled = pd.DataFrame(input_scaled, columns=feature_names)
    
    # background_sample is already scaled if we saved it from X_train_scaled, 
    # but in my current train_model.py, X_train is the scaled array.
    
    # 2. Get explanation
    explainer = get_shap_explainer(model, background_sample)
    shap_values = explainer.shap_values(input_df_scaled)
    
    # For LinearExplainer with LogisticRegression:
    # shap_values could be (1, features) array
    if len(shap_values.shape) == 2:
        shap_vals = shap_values[0]
    elif len(shap_values.shape) == 3:
        # For multi-class (rare here but just in case)
        shap_vals = shap_values[0][1]
    else:
        shap_vals = shap_values
        
    explanation = dict(zip(feature_names, shap_vals))
    # Convert np.float64 to float for JSON serialization
    explanation = {k: float(v) for k, v in explanation.items()}
    return explanation

if __name__ == "__main__":
    with open('models/credit_model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # Test sample
    test_data = pd.DataFrame([{
        'age': 30, 'income': 50000, 'employment_years': 5, 'credit_score': 700,
        'loan_amount': 10000, 'loan_intent': 1, 'loan_term': 36, 'previous_defaults': 0,
        'debt_to_income': 0.2, 'loan_to_income_ratio': 0.2, 'employment_stability': 0.16,
        'risk_factor': 0.1
    }])
    
    # Add dummy columns for any missing features if needed
    for col in artifacts['feature_names']:
        if col not in test_data.columns:
            test_data[col] = 0
            
    # Ensure column order
    test_data = test_data[artifacts['feature_names']]
    
    expl = explain_prediction(artifacts['model'], artifacts['scaler'], artifacts['feature_names'], test_data)
    print("Explanation:", expl)
