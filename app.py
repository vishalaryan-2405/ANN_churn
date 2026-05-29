import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details below to predict churn.")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model.h5")

# ---------------- LOAD ENCODERS ----------------
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- USER INPUTS ----------------

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=1000,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=75000.0
)

# ---------------- PREDICTION BUTTON ----------------

if st.button("Predict Churn"):

    # input dictionary
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    # dataframe
    input_df = pd.DataFrame([input_data])

    # encode gender
    input_df['Gender'] = label_encoder_gen.transform(
        input_df['Gender']
    )

    # encode geography
    geo_encoded = one_hot_encoder_geo.transform(
        input_df[['Geography']]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # merge encoded geography
    input_df = pd.concat(
        [input_df.drop('Geography', axis=1), geo_encoded_df],
        axis=1
    )

    # scale input
    input_scaled = scaler.transform(input_df)

    # prediction
    prediction = model.predict(input_scaled)

    prediction_prob = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(f"Churn Probability: **{prediction_prob:.2f}**")

    if prediction_prob > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")