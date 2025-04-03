import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained model
model = joblib.load("output/readmission_predictor.pkl")  # Update with actual model file

st.title("⚙️ Model Performance & Pipeline")

# --- Model Performance Metrics ---
st.header("📊 Evaluation Metrics")
st.markdown("""
- **Precision:** 0.5548 → Measures accuracy of positive predictions.  
- **Recall:** 0.7426 → Important for reducing false negatives.  
- **F1-Score:** 0.6351 → Balance between precision and recall.  
""")

st.success("✅ **Higher recall helps minimize undetected high-risk patients.**")

# --- Confusion Matrix ---
st.subheader("🔄 Confusion Matrix")
st.markdown("**Visualizing the model's classification performance.**")

# Define Confusion Matrix (replace with actual values)
cm = np.array([[100, 30], [40, 200]])  # Example values, update with actual results

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Not Readmitted", "Readmitted"], 
            yticklabels=["Not Readmitted", "Readmitted"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

st.pyplot(fig)

st.info("""
🔍 **Interpretation:**  
- **True Positives (Bottom-right)** → Patients correctly classified as readmitted.  
- **False Negatives (Top-right)** → Readmitted patients incorrectly classified as not readmitted.  
""")

# --- Machine Learning Pipeline ---
st.subheader("🔗 Machine Learning Pipeline")
st.markdown("""
### 🏗 **Step-by-Step Model Process**
1️⃣ **Data Preprocessing**  
   - Cleaned missing values & transformed categorical variables.  
   - Applied **One-Hot Encoding** & **Feature Scaling**.  
   
2️⃣ **Feature Engineering**  
   - Engineered features such as `n_inpatient` & `n_emergency` from raw data.  

3️⃣ **Model Training**  
   - **Algorithm:** Logistic Regression  
   - **Why?** Fast, interpretable, and effective for binary classification.  

4️⃣ **Evaluation**  
   - Used **Precision, Recall, and F1-Score** to assess performance.  

5️⃣ **Model Saving & Deployment**  
   - Model stored as `"output/readmission_predictor.pkl"` for later use.  
   - **Upcoming:** Deploy on Heroku using Streamlit.
""")

st.success("🚀 **This structured ML pipeline ensures reproducibility and clarity for the project!**")
