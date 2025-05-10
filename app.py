
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import shap
import matplotlib.pyplot as plt

# Page configuration and title
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("Enter patient data below to predict the likelihood of having diabetes and explain the model's decision using SHAP.")

# Load and train model
@st.cache_data
def load_and_train_model():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model, scaler, X.columns.tolist()

model, scaler, feature_names = load_and_train_model()

# Collect user input
st.subheader("Patient Input")
pregnancies = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Make prediction
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is not likely to have diabetes.")

# Explain prediction with SHAP
if st.button("Explain Prediction"):
    input_scaled = scaler.transform(input_data)
    explainer = shap.Explainer(model, masker=scaler.transform)
    shap_values = explainer(input_data)

    st.subheader("Feature Contribution (SHAP values)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
