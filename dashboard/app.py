import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Credit Risk AI dashboard",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: var(--background-color);
    }
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    .stMetric:hover {
        transform: translateY(-5px);
    }
    /* Interactive Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 15px rgba(79, 70, 229, 0.4);
        opacity: 0.95;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0) scale(0.98);
    }

    /* Interactive Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 10px 0;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--secondary-background-color);
        border-radius: 12px 12px 0 0;
        padding: 10px 20px;
        transition: all 0.2s ease;
        border: 1px solid rgba(128, 128, 128, 0.1);
        color: var(--text-color);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(128, 128, 128, 0.15);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--background-color) !important;
        border-bottom: 4px solid var(--primary-color) !important;
        font-weight: bold;
        color: var(--primary-color) !important;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.05);
    }
    
    /* Tables and other elements */
    .stTable {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 10px;
    }

    /* Fade-in effect for sections */
    .stVerticalBlock {
        animation: fadeIn 0.6s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("AI-Powered Credit Risk Intelligence")
st.markdown("---")

# Load data for benchmarking
@st.cache_data
def load_dataset():
    data_path = "data/credit_risk_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df_bench = load_dataset()

# Helper for API calls
def get_prediction(data):
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

# Sidebar for Input
with st.sidebar:
    st.header("👤 Applicant Data")
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
    employment_years = st.number_input("Years of Employment", 0, 50, 5)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    loan_amount = st.number_input("Loan Amount Requested ($)", 0, 1000000, 15000)
    loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_term = st.selectbox("Loan Term (Months)", [3, 6, 12, 24, 36, 48, 60])
    previous_defaults = st.radio("Previous Defaults?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2)
    
    input_data = {
        'age': age,
        'income': income,
        'employment_years': employment_years,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_intent': loan_intent,
        'loan_term': loan_term,
        'previous_defaults': previous_defaults,
        'debt_to_income': debt_to_income
    }

# Main Layout
tab1, tab2, tab3 = st.tabs(["Single Analysis", "What-If Simulator", "Population Benchmarks"])

with tab1:
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Application Summary")
        summary_df = pd.DataFrame([input_data]).T.rename(columns={0: 'Value'})
        st.table(summary_df)
        
        analyze_btn = st.button("Run Full Analysis", type="primary")
        
    if analyze_btn:
        with st.spinner("AI is evaluating risk factors..."):
            result = get_prediction(input_data)
            
            if result:
                with col_b:
                    st.subheader("AI Prediction")
                    
                    risk_score = result['risk_score']
                    decision = result['loan_decision']
                    
                    # Risk Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Score Probability (%)", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#2c3e50"},
                            'steps' : [
                                {'range': [0, 30], 'color': "#27ae60"},
                                {'range': [30, 70], 'color': "#f1c40f"},
                                {'range': [70, 100], 'color': "#e74c3c"}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score}
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if decision == "Approved":
                        st.success(f"### Decision: {decision} ✅")
                    else:
                        st.error(f"### Decision: {decision} ❌")
                
                st.markdown("---")
                st.subheader("Explainable AI Insights (SHAP)")
                
                expl = result['explanation']
                expl_df = pd.DataFrame(list(expl.items()), columns=['Feature', 'Impact'])
                expl_df['Color'] = expl_df['Impact'].apply(lambda x: 'Increase Risk' if x > 0 else 'Decrease Risk')
                
                fig_shap = px.bar(
                    expl_df.sort_values(by='Impact'), 
                    x='Impact', 
                    y='Feature', 
                    color='Color',
                    orientation='h',
                    color_discrete_map={'Increase Risk': '#e74c3c', 'Decrease Risk': '#3498db'},
                    title="How features influenced this decision"
                )
                fig_shap.update_layout(xaxis_title="SHAP Value (Impact)", yaxis_title="")
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.error("API Error. Please check if the backend server is running on port 8000.")

with tab2:
    st.subheader("Interactive 'What-If' Analysis")
    st.write("Modify key parameters in real-time to see how the risk score changes.")
    
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        w_income = st.slider("Annual Income ($)", 10000, 200000, int(income), step=5000, key="w_inc")
        w_credit = st.slider("Credit Score", 300, 850, int(credit_score), key="w_cred")
        w_debt = st.slider("Debt-to-Income", 0.0, 1.0, float(debt_to_income), step=0.05, key="w_debt")
        w_loan = st.number_input("Loan Amount ($)", 0, 1000000, int(loan_amount), step=1000, key="w_loan")

    # Update input data with what-if values
    what_if_data = input_data.copy()
    what_if_data.update({
        'income': w_income,
        'credit_score': w_credit,
        'debt_to_income': w_debt,
        'loan_amount': w_loan
    })
    
    with col_w2:
        res_wi = get_prediction(what_if_data)
        if res_wi:
            wi_score = res_wi['risk_score']
            
            # Simple Delta Display
            delta = wi_score - risk_score if 'risk_score' in globals() else 0
            
            st.metric("Simulated Risk Score", f"{wi_score}%", delta=f"{delta:.2f}%", delta_color="inverse")
            
            # Mini Gauge
            fig_mini = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = wi_score,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3498db"}},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
            fig_mini.update_layout(height=250)
            st.plotly_chart(fig_mini, use_container_width=True)
            
            if res_wi['loan_decision'] == "Approved":
                st.info("💡 With these changes, the loan would be **Approved**.")
            else:
                st.warning("⚠️ With these changes, the loan would be **Rejected**.")

with tab3:
    st.subheader("Population Benchmarking")
    if df_bench is not None:
        st.write("Compare the current applicant against our historical dataset.")
        
        bench_col = st.selectbox("Select Feature to Benchmark", ['income', 'credit_score', 'age', 'debt_to_income'])
        
        fig_hist = px.histogram(
            df_bench, 
            x=bench_col, 
            color="loan_status",
            marginal="box",
            title=f"Distribution of {bench_col} (Color: Loan Status)",
            color_discrete_map={0: "#27ae60", 1: "#e74c3c"}
        )
        
        # Add vertical line for current applicant
        current_val = input_data[bench_col.replace('income', 'income').replace('credit_score', 'credit_score')]
        fig_hist.add_vline(x=current_val, line_dash="dash", line_color="black", annotation_text="Current Applicant")
        
        st.plotly_chart(fig_hist, use_container_width=True)
        st.write(f"The current applicant's **{bench_col}** is **{current_val}**.")
    else:
        st.info("Dataset not found for benchmarking. Ensure `data/credit_risk_data.csv` is correctly placed.")

st.markdown("---")
with st.expander("About the AI Model"):
    st.write("""
    This system uses a **Gradient Boosting** approach for risk estimation, integrated with **SHAP** for interpretability. 
    It analyzes 9 features to provide a probability of default.
    """)
