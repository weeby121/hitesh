import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor")
st.title("❤ Heart Disease Prediction")

st.header("Enter Patient Details")

# Input features
age = st.slider("Age", 1, 100, 50)
sex = st.selectbox("Sex", ["male", "female"])
cp_type = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
rest_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["yes", "no"])
rest_ecg = st.selectbox("Resting ECG Result", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
ex_angina = st.selectbox("Exercise Induced Angina", ["yes", "no"])
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("ST Segment Slope", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Manual encoding
sex_encoded = 1 if sex == "male" else 0
cp_dict = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
cp_encoded = cp_dict[cp_type]
fbs_encoded = 1 if fbs == "yes" else 0
ecg_dict = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
ecg_encoded = ecg_dict[rest_ecg]
ex_angina_encoded = 1 if ex_angina == "yes" else 0
slope_dict = {"upsloping": 0, "flat": 1, "downsloping": 2}
slope_encoded = slope_dict[slope]
thal_dict = {"normal": 1, "fixed defect": 2, "reversible defect": 3}
thal_encoded = thal_dict[thal]

# Feature engineering
age_group = pd.cut([age], bins=[0, 40, 55, 70, 100], labels=['young', 'middle-aged', 'senior', 'elderly'])[0]
high_bp = int(rest_bp > 130)
high_chol = int(chol > 240)
oldpeak_level = pd.cut([oldpeak], bins=[-1, 1, 2, 10], labels=['low', 'moderate', 'high'])[0]
hr_deficit = (220 - age) - max_hr

# Construct input
input_data = pd.DataFrame([{
    'age': age,
    'sex ': sex_encoded,
    'chest pain type': cp_encoded,
    'resting blood pressure': rest_bp,
    'serum cholestoral': chol,
    'fasting blood sugar': fbs_encoded,
    'resting electrocardiographic results': ecg_encoded,
    'max heart rate': max_hr,
    'exercise induced angina': ex_angina_encoded,
    'oldpeak': oldpeak,
    'ST segment': slope_encoded,
    'major vessels': ca,
    'thal': thal_encoded,
    'age_group': age_group,
    'high_blood_pressure': high_bp,
    'high_cholestoral': high_chol,
    'oldpeak_level': oldpeak_level,
    'hr_deficit': hr_deficit
}])

st.subheader("Model Input Preview")
st.write(input_data)

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ No Heart Disease Detected.")
        else:
            st.error("⚠ High Risk of Heart Disease.")
    except ValueError as e:
        st.error(f"Prediction Error: {e}")