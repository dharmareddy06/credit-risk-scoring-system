# AI Explainable Credit Risk Scoring System

This project is an end-to-end Machine Learning system that predicts credit risk while providing transparent AI explanations using SHAP.

## Project Structure

- `data/`: Contains synthetic credit risk data.
- `src/`: Core logic for preprocessing, training, and explainability.
- `api/`: FastAPI backend for real-time predictions.
- `dashboard/`: Streamlit dashboard for interactive risk assessment.
- `models/`: Saved model artifacts and evaluation reports.

## Features

- **Risk Prediction**: Probability of default and categorical risk assessment.
- **Explainable AI**: SHAP force plots and feature importance visualization.
- **Interactive Dashboard**: Easy-to-use interface for data entry and analysis.
- **Scaleable API**: FASTAPI endpoint for integration with other systems.

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd credit-risk-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. **Train the Model**:
   ```bash
   python src/train_model.py
   ```

2. **Start the API**:
   ```bash
   python api/main.py
   ```

3. **Start the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## API Usage

### Predict Risk

- **URL**: `POST /predict`
- **Payload**:
  ```json
  {
    "age": 30,
    "income": 50000,
    "employment_years": 5,
    "credit_score": 720,
    "loan_amount": 15000,
    "loan_intent": "PERSONAL",
    "loan_term": 36,
    "previous_defaults": 0,
    "debt_to_income": 0.2
  }
  ```

## Business Impact

- **Transparency**: Explains why a loan was rejected, helping with regulatory compliance.
- **Accuracy**: Uses advanced ML models (XGBoost/Random Forest) to improve decision accuracy.
- **Efficiency**: Automates initial risk assessment for loan officers.
