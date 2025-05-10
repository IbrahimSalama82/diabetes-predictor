
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load trained pipeline model
model = joblib.load("gradient_boosting.pkl")

# Multilingual support
language = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"], key="lang")
def t(en, ar):
    return ar if language == "Arabic" else en

# App title
st.title(t("Patient Triage Prediction", "تنبؤ بفرز المرضى"))

# Instruction box
st.markdown("""
<div style='padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: #2c3e50;'>ادخل بيانات المريض للحصول على تنبؤ بمستوى الفرز الطبي.</h4>
</div>
""", unsafe_allow_html=True)

# Sidebar form
st.sidebar.header(t("Enter Patient Information", "أدخل معلومات المريض"))

def user_input_features():
    gender = st.sidebar.selectbox(t("Gender", "النوع"), ["Male", "Female"], key="gender")
    chest_pain_type = st.sidebar.selectbox(t("Chest Pain Type", "نوع ألم الصدر"), ["typical", "atypical", "non-anginal", "asymptomatic"], key="chest")
    exercise_angina = st.sidebar.selectbox(t("Exercise Angina", "الذبحة الناتجة عن التمرين"), ["Yes", "No"], key="angina")
    hypertension = st.sidebar.selectbox(t("Hypertension", "ارتفاع ضغط الدم"), ["Yes", "No"], key="htn")
    heart_disease = st.sidebar.selectbox(t("Heart Disease", "أمراض القلب"), ["Yes", "No"], key="hd")
    smoking_status = st.sidebar.selectbox(t("Smoking Status", "حالة التدخين"), ["never smoked", "formerly smoked", "smokes", "Unknown"], key="smoke")

    blood_pressure = st.sidebar.number_input(t("Blood Pressure", "ضغط الدم"), min_value=80, max_value=200, value=120)
    cholesterol = st.sidebar.number_input(t("Cholesterol", "الكوليسترول"), min_value=100, max_value=400, value=180)
    max_heart_rate = st.sidebar.number_input(t("Max Heart Rate", "أقصى معدل ضربات القلب"), min_value=60, max_value=220, value=160)
    plasma_glucose = st.sidebar.number_input(t("Plasma Glucose", "جلوكوز البلازما"), min_value=50, max_value=300, value=90)
    skin_thickness = st.sidebar.number_input(t("Skin Thickness", "سماكة الجلد"), min_value=7, max_value=100, value=25)
    insulin = st.sidebar.number_input(t("Insulin", "الأنسولين"), min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input(t("BMI", "مؤشر كتلة الجسم"), min_value=10.0, max_value=60.0, value=24.5)
    diabetes_pedigree = st.sidebar.number_input(t("Diabetes Pedigree", "النمط الوراثي للسكري"), min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input(t("Age", "العمر"), min_value=1, max_value=120, value=50)

    return pd.DataFrame([{
        "gender": gender,
        "chest pain type": chest_pain_type,
        "blood pressure": blood_pressure,
        "cholesterol": cholesterol,
        "max heart rate": max_heart_rate,
        "exercise angina": exercise_angina,
        "plasma glucose": plasma_glucose,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_status": smoking_status,
        "age": age
    }])

# Get user input
original_input_df = user_input_features()
input_df = original_input_df.copy()

# Show input
st.subheader(t("Patient Input Data", "بيانات المريض المدخلة"))
st.write(original_input_df)

# Label encoding
label_encoders = {
    "gender": LabelEncoder(),
    "chest pain type": LabelEncoder(),
    "exercise angina": LabelEncoder(),
    "hypertension": LabelEncoder(),
    "heart_disease": LabelEncoder(),
    "smoking_status": LabelEncoder()
}

# Predict button
with st.form("predict_form"):
    submitted = st.form_submit_button(t("Predict Risk", "تنبؤ بالخطر"))

if submitted:
    with st.spinner(t("Predicting...", "جارٍ التنبؤ...")):
        try:
            for col, le in label_encoders.items():
                input_df[col] = le.fit_transform(input_df[col])

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            st.markdown(f"### ✅ {t('Prediction Result', 'نتيجة التنبؤ')}")
            st.success(f"{t('Predicted Triage Level', 'مستوى الفرز المتوقع')}: **{prediction}**")

            # Confidence chart
            st.markdown(t("### 🔍 Confidence Levels", "### 🔍 مستويات الثقة"))
            prob_df = pd.DataFrame({
                "Triage Level": model.classes_,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            fig = px.bar(prob_df, x="Triage Level", y="Probability",
                         title=t("Model Confidence per Triage Level", "نسبة الثقة لكل مستوى فرز"),
                         text_auto='.2f', range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"{t('Prediction failed:', 'فشل التنبؤ:')} {e}")

# Footer
st.markdown("<div style='position: fixed; bottom: 10px; right: 10px; font-size: 10px; color: gray;'>FOR ACADEMIC PURPOSES ONLY</div>", unsafe_allow_html=True)
