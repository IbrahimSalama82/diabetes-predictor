
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Language selection
lang = st.sidebar.selectbox("Select Language / اختر اللغة", ["English", "العربية"])

# Translations
texts = {
    "English": {
        "title": "Diabetes Risk Assessment",
        "gender": "Gender",
        "age": "Age",
        "bmi": "BMI",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "smoking": "Smoking History",
        "glucose": "HbA1c Level",
        "blood_glucose": "Blood Glucose Level",
        "predict": "Predict",
        "result_0": "The person is NOT diabetic.",
        "result_1": "The person IS diabetic.",
        "male": "Male",
        "female": "Female",
        "yes": "Yes",
        "no": "No",
        "never": "Never",
        "former": "Former",
        "current": "Current"
    },
    "العربية": {
        "title": "تقييم خطر الإصابة بالسكري",
        "gender": "النوع",
        "age": "العمر",
        "bmi": "مؤشر كتلة الجسم",
        "hypertension": "ضغط الدم المرتفع",
        "heart_disease": "أمراض القلب",
        "smoking": "تاريخ التدخين",
        "glucose": "مستوى HbA1c",
        "blood_glucose": "مستوى السكر في الدم",
        "predict": "تنبؤ",
        "result_0": "الشخص غير مصاب بالسكري.",
        "result_1": "الشخص مصاب بالسكري.",
        "male": "ذكر",
        "female": "أنثى",
        "yes": "نعم",
        "no": "لا",
        "never": "لم يدخن أبدًا",
        "former": "مدخن سابق",
        "current": "مدخن حالي"
    }
}

t = texts[lang]

# App title
st.title(t["title"])

# User inputs
gender = st.radio(t["gender"], [t["male"], t["female"]])
age = st.slider(t["age"], 1, 120, 30)
bmi = st.number_input(t["bmi"], 10.0, 50.0, 24.0)
hypertension = st.radio(t["hypertension"], [t["yes"], t["no"]])
heart_disease = st.radio(t["heart_disease"], [t["yes"], t["no"]])
smoking = st.selectbox(t["smoking"], [t["never"], t["former"], t["current"]])
glucose = st.number_input(t["glucose"], 3.0, 10.0, 5.0)
blood_glucose = st.number_input(t["blood_glucose"], 50.0, 300.0, 100.0)

# Encode inputs
gender_val = 1 if gender == t["male"] else 0
hypertension_val = 1 if hypertension == t["yes"] else 0
heart_val = 1 if heart_disease == t["yes"] else 0
smoking_vals = {
    t["never"]: [0, 0, 0, 0, 0],
    t["former"]: [0, 0, 0, 1, 0],
    t["current"]: [1, 0, 0, 0, 0],
}
smoking_encoded = smoking_vals.get(smoking, [0, 0, 0, 0, 0])

# Final feature vector
input_data = [gender_val, age, hypertension_val, heart_val, bmi, glucose, blood_glucose] + smoking_encoded
input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Prediction
if st.button(t["predict"]):
    prediction = model.predict(scaled_input)
    st.success(t["result_1"] if prediction[0] == 1 else t["result_0"])
