# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 22:55:44 2025

@author: karti
"""


import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load("charge_prediction_model.pkl")

# Streamlit app title
st.title(" Medical Insurance Charge Prediction App")

st.write("### Predict medical charges based on personal and lifestyle details.")

# Collect user inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

# Convert inputs to numerical / encoded form
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

# Region encoding (one-hot style, same as model training)
region_northeast = 0
region_northwest = 0
region_southeast = 0
region_southwest = 0

if region == "northeast":
    region_northeast = 1
elif region == "northwest":
    region_northwest = 1
elif region == "southeast":
    region_southeast = 1
elif region == "southwest":
    region_southwest = 1

# Create input array in correct order
# (Make sure order matches your model training order!)
input_data = np.array([[age, sex, bmi, children, smoker,
                        region_northeast, region_northwest, region_southeast, region_southwest]])

# Predict charges
if st.button("Predict Charge "):
    prediction = model.predict(input_data)
    st.success(f" Estimated Medical Insurance Charge: **$ {-prediction[0]:.2f}**")

