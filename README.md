# **Hospital Readmission Prediction**

A machine learning model to predict whether a patient will be readmitted to the hospital, helping healthcare providers manage resources efficiently.

---

## **1. Dataset Content**

This project utilizes a **hospital readmissions dataset** from Kaggle, covering **10 years of patient records**. It includes numerical and categorical features such as:

- **Patient history:** Number of inpatient, outpatient, and emergency visits  
- **Medical details:** Diagnoses, lab procedures, medications, and treatment history  
- **Outcome variable:** Whether the patient was readmitted (**Yes/No**)  

The dataset is **manageable in size**, allowing for efficient training and deployment of a **Supervised Binary Classification** model.

---

## **2. Business Requirements**

This project aims to help **healthcare providers** optimize **hospital resources** by predicting **which patients are likely to be readmitted**.  

### **Objectives**

1. **Understanding Readmissions:**  
   - Identify key factors contributing to **hospital readmissions**.
   - Use **data visualization** to explore trends in patient history and medical details.

2. **Predicting Readmissions:**  
   - Develop a **machine learning model** to predict **whether a patient will be readmitted**.
   - Ensure the model is **interpretable** so medical staff can understand key risk factors.

---

## **3. Hypothesis and Validation**

| **Hypothesis** | **Validation Approach** |
|---------------|--------------------|
| **Patients with a higher number of past inpatient visits are more likely to be readmitted.** | **Correlation analysis** between inpatient visit count and readmission rates. |
| **Certain diagnoses (e.g., chronic illnesses) increase the likelihood of readmission.** | **Feature importance evaluation** using ML models. |
| **Frequent emergency visits indicate a higher readmission risk.** | **Statistical tests** on emergency visit frequency vs. readmission outcomes. |

---

## **4. Machine Learning Business Case**

### **Problem Type:**  

- **Supervised Learning → Binary Classification**  
- **Target Variable:** `readmitted (Yes(1)/No(0))`

### **Model Success Metrics:**

- **Precision, Recall, and F1-score** to evaluate model performance.
- Focus on **recall** to minimize false negatives (patients who should have been flagged as high risk).

### **Final Model Performance:**

| Metric  | Score |
|---------|-------|
| **Precision** | `0.5548` |
| **Recall** | `0.7426` |
| **F1 Score** | `0.6351` |

---

## **5. Feature Importance & Model Saving**

The most predictive features include:

- **BoxCox transformed variables:** `n_emergency`, `n_outpatient`, `n_inpatient`
- **Medical history features:** `diabetes_med`, `diagnosis categories`  

### **Model Saving & Deployment**

- The final trained model is saved as:
 'output/readmission_predictor.pkl')

## **6. Streamlit Dashboard Overview**

The **Streamlit dashboard** provides an interactive way to explore the dataset, analyze key insights, and predict patient readmissions. It includes the following sections:

1) **Project Summary** – Overview of the hospital readmission problem, dataset, and business objectives.
2) **Data Analysis & Visualization** – Exploratory Data Analysis (EDA) with visualizations of key features and correlations.
3) **Hypothesis & Validation** – Testing business hypotheses and validating key insights with statistical analysis.
4) **Model Training & Evaluation** – Machine learning model performance metrics, feature importance, and confusion matrix.
5) **Patient Readmission Predictor** – User inputs patient data (manual or CSV upload) to receive readmission predictions.
6) **Technical Details** – Explanation of data preprocessing, feature engineering, model selection, and deployment.

### **Live Demo:**

- **[Readmission Predictor (Heroku)](https://readmission-predictor-live-876b8e020239.herokuapp.com/)**

The dashboard is designed to be **user-friendly for healthcare professionals**, allowing them to gain insights and make **data-driven decisions efficiently**.
