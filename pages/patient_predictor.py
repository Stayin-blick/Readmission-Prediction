import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.stats import boxcox

# Load trained model
model = joblib.load("output/readmission_predictor.pkl")

# Load dataset to check expected feature names
final_df = pd.read_csv("inputs/readmission_dataset/final_df.csv")
expected_features = final_df.drop(columns=["readmitted"]).columns  # Drop target column

# Define age mapping (same as training)
age_mapping = {
    "0-20": 15, "20-40": 30, "40-60": 50, "60-80": 70, "80+": 90
}

# Categorical feature options (from training data)
categorical_options = {
    "medical_specialty": ["Cardiology", "Neurology", "Pediatrics", "Surgery", "Other"],
    "diag_1": ["Circulatory", "Respiratory", "Diabetes", "Injury", "Other"],
    "diag_2": ["Circulatory", "Respiratory", "Diabetes", "Injury", "Other"],
    "diag_3": ["Circulatory", "Respiratory", "Diabetes", "Injury", "Other"],
    "glucose_test": ["Normal", "High"],
    "A1Ctest": ["None", "Normal", "High"],
    "change": ["No", "Yes"],
    "diabetes_med": ["No", "Yes"]
}

# Transformation Functions
def apply_boxcox(series, lambda_value):
    return (np.power(series, lambda_value) - 1) / lambda_value if lambda_value != 0 else np.log(series)

def preprocess_input(data):
    """Apply transformations to match model training."""
    
    # One-Hot Encode categorical features
    data = pd.get_dummies(data, columns=categorical_options.keys(), drop_first=True)
    
    # Apply Box-Cox Transformations
    lambda_values = {
        "time_in_hospital": 0.4,  
        "n_outpatient": 0.2,  
        "n_inpatient": 0.3,  
        "n_emergency": 0.1  
    }
    for col, lam in lambda_values.items():
        if col in data.columns:
            data[f"boxcox_{col}"] = apply_boxcox(data[col] + 1, lam)  # Avoid zero values

    # Apply Square Root Transformation
    if "n_procedures" in data.columns:
        data["sqrt_n_procedures"] = np.sqrt(data["n_procedures"])
        data.drop(columns=["n_procedures"], inplace=True)

    # Apply Log Transformation
    if "n_medications" in data.columns:
        data["log_n_medications"] = np.log1p(data["n_medications"])
        data.drop(columns=["n_medications"], inplace=True)

    # Drop original numerical features that were transformed
    data.drop(columns=lambda_values.keys(), inplace=True, errors="ignore")

    # Ensure columns match training
    data = data.reindex(columns=expected_features, fill_value=0)

    return data

# Function to make predictions
def make_prediction(data):
    processed_data = preprocess_input(data)  # Apply transformations

    # 🔍 Debugging: Print processed data shape
    st.write("🛠 Model Input Shape:", processed_data.shape)

    # Ensure feature order matches training
    processed_data = processed_data[expected_features]  

    # Get probability predictions if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_data)[:, 1]  # Probability of "Readmitted"
        prediction = (proba > 0.4).astype(int)  # Adjust threshold from 0.5 to 0.4
    else:
        prediction = model.predict(processed_data)

    return prediction


st.title("⚕️ Patient Readmission Predictor")
st.markdown("Enter patient details manually or upload a CSV file for batch predictions.")

# Sidebar for input method selection
input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

# Manual Input Form
if input_method == "Manual Input":
    st.subheader("📋 Enter Patient Details")

    # User inputs
    age_group = st.selectbox("Age Group", ["0-20", "20-40", "40-60", "60-80", "80+"])
    time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 3)
    n_inpatient = st.slider("Number of Inpatient Visits", 0, 20, 1)
    n_emergency = st.slider("Number of Emergency Visits", 0, 10, 0)
    n_outpatient = st.slider("Number of Outpatient Visits", 0, 30, 5)
    n_procedures = st.slider("Number of Procedures", 0, 10, 1)
    n_medications = st.slider("Number of Medications", 0, 50, 10)

    # Dropdowns for categorical features
    medical_specialty = st.selectbox("Medical Specialty", categorical_options["medical_specialty"])
    diag_1 = st.selectbox("Primary Diagnosis", categorical_options["diag_1"])
    diag_2 = st.selectbox("Secondary Diagnosis", categorical_options["diag_2"])
    diag_3 = st.selectbox("Tertiary Diagnosis", categorical_options["diag_3"])
    glucose_test = st.selectbox("Glucose Test Result", categorical_options["glucose_test"])
    A1Ctest = st.selectbox("A1C Test Result", categorical_options["A1Ctest"])
    change = st.radio("Medication Change", categorical_options["change"])
    diabetes_med = st.radio("Diabetes Medication", categorical_options["diabetes_med"])

    # Convert inputs to match model format
    input_data = pd.DataFrame([[
        age_mapping[age_group], time_in_hospital, n_inpatient, n_emergency, n_outpatient,
        n_procedures, n_medications,  # Numerical
        medical_specialty, diag_1, diag_2, diag_3, glucose_test, A1Ctest, change, diabetes_med  # Categorical
    ]], columns=["age", "time_in_hospital", "n_inpatient", "n_emergency", "n_outpatient",
                 "n_procedures", "n_medications",
                 "medical_specialty", "diag_1", "diag_2", "diag_3", 
                 "glucose_test", "A1Ctest", "change", "diabetes_med"])

    # Predict button
    if st.button("🔍 Predict Readmission"):
        result = make_prediction(input_data)
        st.subheader(f"Prediction: **{'Readmitted' if result[0] == 1 else 'Not Readmitted'}**")

# CSV Upload for Batch Prediction
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)

        # Apply the same age mapping to the CSV
        if "age" in batch_data.columns:
            batch_data["age"] = batch_data["age"].map(age_mapping)
        
        # Apply transformations before prediction
        processed_batch_data = preprocess_input(batch_data)
        
        # Make predictions
        predictions = make_prediction(processed_batch_data)
        batch_data["Readmission Prediction"] = ["Readmitted" if p == 1 else "Not Readmitted" for p in predictions]
        
        st.write(batch_data)
        st.download_button("⬇️ Download Predictions", batch_data.to_csv(index=False), file_name="predictions.csv")
