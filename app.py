
import streamlit as st
import numpy as np
import joblib
from PIL import Image

# تحميل الموديل و StandardScaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# تحميل صورة
image = Image.open("diabetes_image.jpg")
st.image(image, caption='Diabetes Awareness', use_column_width=True)

st.title("🔍 توقع مرض السكري")

# واجهة إدخال البيانات
st.markdown("**أدخل القيم التالية للتنبؤ:**")

preg = st.number_input("عدد مرات الحمل", min_value=0)
glucose = st.number_input("مستوى الجلوكوز", min_value=0)
bp = st.number_input("ضغط الدم", min_value=0)
skin = st.number_input("سمك الجلد", min_value=0)
insulin = st.number_input("الأنسولين", min_value=0)
bmi = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=0.0, format="%.2f")
dpf = st.number_input("عامل الوراثة (DPF)", min_value=0.0, format="%.3f")
age = st.number_input("العمر", min_value=0)

# زر التنبؤ
if st.button("توقّع"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ الشخص غير مصاب بالسكري.")
    else:
        st.error("⚠️ الشخص مصاب بالسكري.")
