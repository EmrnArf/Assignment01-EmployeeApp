import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
import os
import json

# Setting up the page
st.set_page_config(page_title="Employee Churn Predictor", page_icon="🏢", layout="centered")

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.title("Employee Churn Predictor")
st.markdown("<p style='text-align: center; color: #cbd5e1;'>Will they stay or will they go? Predict employee retention using our advanced AI model.</p>", unsafe_allow_html=True)

# Helper function to predict manually using extracted weights (so we don't need joblib if we just use the weights)
def predict_churn(features):
    # This matches the model we trained (Model C: hidden_layer_sizes=(64, 32), activation='relu')
    # Let's load weights.js that we created earlier, which contains our trained model's weights.
    # We will read it as JSON
    try:
        # Check if model_weights.json exists in artifacts
        json_path = r"c:\Users\meanj\.gemini\antigravity\brain\fc3d1193-ae90-4829-96b4-fcfd3c82e297\artifacts\model_weights.json"
        with open(json_path, 'r') as f:
            model_data = json.load(f)
    except:
        st.error("Model weights not found. Please ensure 'model_weights.json' is available.")
        return 0.5

    def relu(x):
        return np.maximum(0, x)
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    w1 = np.array(model_data["weights"][0])
    b1 = np.array(model_data["biases"][0])
    w2 = np.array(model_data["weights"][1])
    b2 = np.array(model_data["biases"][1])
    w3 = np.array(model_data["weights"][2])
    b3 = np.array(model_data["biases"][2])
    
    # Layer 1
    h1 = relu(np.dot(features, w1) + b1)
    # Layer 2
    h2 = relu(np.dot(h1, w2) + b2)
    # Output Layer
    out = sigmoid(np.dot(h2, w3) + b3)
    
    return out[0]

with st.form("prediction_form"):
    st.subheader("Employee Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction = st.number_input("Satisfaction Level (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        evaluation = st.number_input("Last Evaluation (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        projects = st.number_input("Number of Projects", min_value=1, max_value=20, value=3)
        hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=150)
        time = st.number_input("Time Spend Company (Years)", min_value=1, max_value=20, value=3)
        
    with col2:
        accident = st.selectbox("Work Accident", options=["No", "Yes"])
        promotion = st.selectbox("Promotion Last 5 Years", options=["No", "Yes"])
        
        dept_options = {"IT": 0, "RandD": 1, "Accounting": 2, "HR": 3, "Management": 4, 
                        "Marketing": 5, "Product Mgt": 6, "Sales": 7, "Technical": 8, "Support": 9}
        department = st.selectbox("Department", options=list(dept_options.keys()))
        
        salary_options = {"High": 0, "Low": 1, "Medium": 2}
        salary = st.selectbox("Salary", options=list(salary_options.keys()))

    submit_button = st.form_submit_button(label="Predict Churn")

if submit_button:
    # Convert inputs to numerical format
    features = np.array([
        satisfaction,
        evaluation,
        projects,
        hours,
        time,
        1 if accident == "Yes" else 0,
        1 if promotion == "Yes" else 0,
        dept_options[department],
        salary_options[salary]
    ])
    
    with st.spinner('Analyzing patterns...'):
        prob = predict_churn(features)
        
    st.markdown("---")
    st.subheader("Result")
    
    prob_percentage = prob * 100
    
    if prob > 0.5:
        st.error(f"High Risk of Leaving! The model predicts this employee is likely to churn. (Probability: {prob_percentage:.1f}%)")
        st.progress(int(prob_percentage))
    else:
        st.success(f"Likely to Stay! The model predicts this employee is happy and will stay. (Probability: {prob_percentage:.1f}%)")
        st.progress(int(prob_percentage))
        
