
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import shap

# Page config and style
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #008080;
    }
    .stButton button {
        background-color: #008080;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and valid image
st.title("Diabetes Risk Assessment")
st.image("https://i.postimg.cc/8CmCFGr6/diabetes-illustration.jpg", use_column_width=True)
st.markdown("Use the form below to assess the risk of diabetes based on medical data.")

# Load and train model
@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model, scaler, X.columns

model, scaler, features = load_model()

# Sidebar input form
st.sidebar.header("Patient Data Input")

def user_input():
    data = {
        "Pregnancies": st.sidebar.number_input("Number of Pregnancies", 0),
        "Glucose": st.sidebar.number_input("Glucose Level", 0),
        "BloodPressure": st.sidebar.number_input("Blood Pressure", 0),
        "SkinThickness": st.sidebar.number_input("Skin Thickness", 0),
        "Insulin": st.sidebar.number_input("Insulin Level", 0),
        "BMI": st.sidebar.number_input("BMI", 0.0),
        "DiabetesPedigreeFunction": st.sidebar.number_input("Diabetes Pedigree Function", 0.0),
        "Age": st.sidebar.number_input("Age", 0)
    }
    return pd.DataFrame([data])

user_df = user_input()

# Prediction
if st.button("Predict Risk"):
    scaled_input = scaler.transform(user_df)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("This patient is likely to have diabetes.")
    else:
        st.success("This patient is unlikely to have diabetes.")

    st.subheader("Risk Level")
    st.progress(int(prediction_proba * 100))
    st.write(f"Estimated Risk: **{prediction_proba*100:.2f}%**")

    # SHAP Explanation
    st.subheader("Explanation with SHAP")
    explainer = shap.Explainer(model, masker=scaler.transform)
    shap_values = explainer(user_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
