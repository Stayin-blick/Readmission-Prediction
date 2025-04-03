import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("inputs/readmission_dataset/cleaned_hospital_readmissions.csv")

df = load_data()

st.title("📌 Project Summary")

# Business Overview
st.markdown("""
### 🔍 Overview  
This project predicts **patient readmission within 30 days** using medical data.  
The goal is to help hospitals reduce avoidable readmissions and improve patient care.  
""")

# Dataset Overview
st.header("📊 Dataset Overview")
st.write(f"**Total Records:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

# Display sample data
if st.checkbox("Show sample dataset"):
    st.dataframe(df.head())

# Feature Descriptions
st.subheader("📝 Key Features")
st.markdown("""
- **n_inpatient:** Number of past inpatient visits  
- **n_outpatient:** Number of outpatient visits  
- **n_emergency:** Number of emergency visits  
- **age:** Patient's age group  
- **diabetes_med:** Whether the patient is on diabetes medication  
- **readmitted:** Target variable (Yes/No)  
""")

# Missing Values
st.subheader("🛑 Missing Values Check")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    st.success("No missing values found! ✅")
else:
    st.warning("Missing values detected! Consider handling them.")
    st.write(missing_values[missing_values > 0])
