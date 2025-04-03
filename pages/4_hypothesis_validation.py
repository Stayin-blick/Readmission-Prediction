import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("inputs/readmission_dataset/cleaned_hospital_readmissions.csv")

df = load_data()

st.title("📌 Hypothesis & Validation")

# Hypothesis Explanation
st.markdown("""
### 🔍 Project Hypothesis  
- **Hypothesis:** Patients with multiple prior hospital visits and higher medication counts are more likely to be readmitted.  
- **Validation Approach:** Using statistical analysis and visualization to test these assumptions.
""")

# Box Plot - Inpatient Visits & Readmission
st.subheader("📊 Distribution of Inpatient Visits by Readmission Status")
fig, ax = plt.subplots()
sns.boxplot(x=df["readmitted"], y=df["n_inpatient"], ax=ax)
st.pyplot(fig)

# Count Plot - Readmission by Age Group
st.subheader("📈 Readmission Rate by Age Group")
fig, ax = plt.subplots()
sns.countplot(x=df["age"], hue=df["readmitted"], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
