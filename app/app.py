import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("models/logistic_regression_model.pkl")
preprocessor = joblib.load("data/processed/preprocessed_data.pkl")

# Load cleaned dataset
df_sample = pd.read_csv("data/cleaned/cleaned_telco_churn.csv")

df_sample = df_sample.drop(columns=["Churn"])

# Separate columns by type
categorical_cols = df_sample.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
numerical_cols = df_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.title("Customer Churn Prediction")

st.markdown("Enter customer information below:")

# Create input form
user_data = {}
with st.form("user_input_form"):
    for col in categorical_cols:
        options = df_sample[col].dropna().unique().tolist()
        user_data[col] = st.selectbox(col, options)
    
    for col in numerical_cols:
        min_val = int(df_sample[col].min())
        max_val = int(df_sample[col].max())
        mean_val = float(df_sample[col].mean())
        user_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=int(mean_val))
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([user_data])
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Likely to Churn! (Probability: {probability:.2%})")
        else:
            st.success(f"Likely to Stay (Probability of churn: {probability:.2%})")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")
