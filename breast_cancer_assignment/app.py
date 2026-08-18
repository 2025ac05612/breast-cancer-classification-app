import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Set page layout and title
st.set_page_config(page_title="2025ac05612 - Assignment 2", layout="wide")
st.title("Breast Cancer Classification Model Evaluator")
st.write("Upload test data and select a model to view performance metrics and evaluation plots.")

# Get the absolute directory path of the current app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to saved models dynamically
model_files = {
    'Logistic Regression': os.path.join(BASE_DIR, 'model', 'logistic_regression.pkl'),
    'Decision Tree': os.path.join(BASE_DIR, 'model', 'decision_tree.pkl'),
    'kNN': os.path.join(BASE_DIR, 'model', 'knn.pkl'),
    'Naive Bayes': os.path.join(BASE_DIR, 'model', 'naive_bayes.pkl'),
    'Random Forest (Ensemble)': os.path.join(BASE_DIR, 'model', 'random_forest_ensemble.pkl')
}
scaler_file = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Step a: Dataset upload option (CSV)
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

# Step b: Model selection dropdown
st.sidebar.header("2. Choose Model")
selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_files.keys()))

# Check if model files and scaler exist
files_exist = all(os.path.exists(path) for path in model_files.values()) and os.path.exists(scaler_file)

if not files_exist:
    st.error("Error: Model files or scaler.pkl not found in the 'model/' directory. Please ensure you ran your training script successfully.")
else:
    if uploaded_file is not None:
        # Load the uploaded test data
        test_df = pd.read_csv(uploaded_file)
        
        # Verify if the target column is present
        if 'target' not in test_df.columns:
            st.error("The uploaded CSV must contain a 'target' column for evaluation.")
        else:
            st.success("Test dataset loaded successfully!")
            
            # Split features and targets
            X_test = test_df.drop(columns=['target'])
            y_test = test_df['target']
            
            # Load the scaler and transform test data
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            try:
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                st.error(f"Error scaling data. Please ensure test columns match training features. Detail: {e}")
                st.stop()
                
            # Load the selected model
            with open(model_files[selected_model_name], 'rb') as f:
                model = pickle.load(f)
                
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Predict probabilities safely (some classifiers may require special handling, but all our chosen 5 support predict_proba)
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            except AttributeError:
                y_proba = y_pred  # Fallback if probability calculation is not supported
                
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Step c: Display of evaluation metrics
            st.subheader(f"Metrics for {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")
            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            # Step d: Confusion matrix & classification report display
            st.subheader("Visualization & Detailed Report")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Malignant (0)', 'Benign (1)'], 
                            yticklabels=['Malignant (0)', 'Benign (1)'], ax=ax)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
                
            with col_right:
                st.write("**Classification Report**")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4))
                
    else:
        st.info("Please upload your 'test_data.csv' file using the sidebar to begin evaluation.")
