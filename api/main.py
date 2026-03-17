from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import sys

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import preprocess_data, encode_categoricals
from feature_engineering import engineer_features
from explainability import explain_prediction

app = FastAPI(title="Credit Risk AI API", description="Explainable Credit Risk Scoring System")

# Load model artifacts
try:
    with open('models/credit_model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
except FileNotFoundError:
    artifacts = None

class LoanApplication(BaseModel):
    age: int
    income: float
    employment_years: float
    credit_score: int
    loan_amount: float
    loan_intent: str
    loan_term: int
    previous_defaults: int
    debt_to_income: float

@app.get("/")
def read_root():
    return {"message": "Credit Risk AI API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": artifacts is not None}

@app.post("/predict")
def predict_risk(application: LoanApplication):
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Convert input to DataFrame
    input_dict = application.dict()
    df = pd.DataFrame([input_dict])
    
    # 2. Preprocess & Feature Engineering
    # We need to map loan_intent string to integer using the label encoder
    try:
        df['loan_intent'] = artifacts['label_encoder'].transform(df['loan_intent'])
    except Exception as e:
        # Default to a safe fallback or raise error
        df['loan_intent'] = 0 
        
    df = engineer_features(df)
    
    # 3. Ensure feature order matches trainer
    feature_names = artifacts['feature_names']
    X = df[feature_names]
    
    # 4. Predict
    X_scaled = artifacts['scaler'].transform(X)
    probability = artifacts['model'].predict_proba(X_scaled)[:, 1][0]
    prediction = int(artifacts['model'].predict(X_scaled)[0])
    
    decision = "Approved" if prediction == 0 else "Rejected"
    risk_score = round(probability * 100, 2)
    
    # 5. Explain
    explanation = explain_prediction(
        artifacts['model'], 
        artifacts['scaler'], 
        feature_names, 
        X, 
        artifacts['background_sample']
    )
    
    return {
        "risk_probability": probability,
        "risk_score": risk_score,
        "prediction": prediction,
        "loan_decision": decision,
        "explanation": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
