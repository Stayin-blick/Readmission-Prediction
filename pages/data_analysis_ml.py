import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("inputs/readmission_dataset/cleaned_hospital_readmissions.csv")

df = load_data()

st.title("📊 Data Analytics & ML")

# Exploratory Data Analysis
st.header("🔍 Exploratory Data Analysis")

# Dropdown for Feature Selection
selected_feature = st.selectbox("Select a numerical feature to visualize:", df.select_dtypes(include=['int64', 'float64']).columns)

# Histogram Plot
st.subheader(f"📈 Distribution of {selected_feature}")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Feature Relationship Plot
st.subheader("🔗 Relationship Between Two Features")
x_feature = st.selectbox("Select X-axis feature:", df.select_dtypes(include=['int64', 'float64']).columns, index=1)
y_feature = st.selectbox("Select Y-axis feature:", df.select_dtypes(include=['int64', 'float64']).columns, index=2)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.6)
st.pyplot(fig)