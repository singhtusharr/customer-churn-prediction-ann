"""Streamlit front-end: collects a customer's profile through input widgets,
runs it through the saved preprocessor + ANN, and shows the churn
probability plus a couple of visual-analytics charts. This is the
"interactive front-end ... for real-time churn probability scoring and
visual analytics" resume bullet.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from src.config import DATA_PATH, DROP_COLUMNS, MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN
from src.preprocessing import load_preprocessor

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("ANN-based real-time churn probability scoring")


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLUMNS)
    return df


try:
    model, preprocessor = load_artifacts()
except Exception:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

reference_df = load_reference_data()

st.subheader("Customer profile")
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", sorted(reference_df["Geography"].unique()))
    gender = st.selectbox("Gender", sorted(reference_df["Gender"].unique()))
    age = st.slider("Age", 18, 95, 38)
    tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes"
    is_active_member = st.selectbox("Active Member?", ["Yes", "No"]) == "Yes"
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

input_df = pd.DataFrame(
    [
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": int(has_cr_card),
            "IsActiveMember": int(is_active_member),
            "EstimatedSalary": estimated_salary,
        }
    ]
)

if st.button("Predict churn probability", type="primary"):
    # .transform() only — the preprocessor was already fit on training data,
    # never refit here. This is the data-leakage boundary in action.
    processed = preprocessor.transform(input_df)
    probability = float(model.predict(processed, verbose=0)[0][0])

    st.subheader("Result")
    st.metric("Churn probability", f"{probability*100:.1f}%")
    st.progress(probability)

    if probability >= 0.5:
        st.warning("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

    st.subheader("Visual analytics")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        churn_rate_by_geo = reference_df.groupby("Geography")[TARGET_COLUMN].mean()
        fig, ax = plt.subplots()
        churn_rate_by_geo.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.axhline(probability, color="red", linestyle="--", label="This customer")
        ax.set_ylabel("Churn rate")
        ax.set_title("Historical churn rate by geography")
        ax.legend()
        st.pyplot(fig)

    with chart_col2:
        fig2, ax2 = plt.subplots()
        reference_df[reference_df[TARGET_COLUMN] == 0]["Age"].hist(ax=ax2, bins=20, alpha=0.5, label="Stayed")
        reference_df[reference_df[TARGET_COLUMN] == 1]["Age"].hist(ax=ax2, bins=20, alpha=0.5, label="Churned")
        ax2.axvline(age, color="red", linestyle="--", label="This customer")
        ax2.set_xlabel("Age")
        ax2.set_title("Age distribution: stayed vs. churned")
        ax2.legend()
        st.pyplot(fig2)
