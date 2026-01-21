import streamlit as st
import pandas as pd
import joblib

# Path to your trained pipeline
MODEL_PATH = "D:/MLDP Project/Notebooks/Models/trained_HD_model.pkl"

# Load pipeline
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Title
st.title("❤️ Heart Disease Severity Predictor")

st.write("Enter patient details below. The model will preprocess inputs and predict severity.")

# --- Raw feature inputs ---
age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
resting_bp_mmHg = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
serum_cholesterol_mgdl = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
thalch = st.number_input("Max Heart Rate (thalch)", min_value=60, max_value=220, value=150)
st_depression_exercise = st.number_input("ST Depression (exercise)", min_value=0.0, max_value=10.0, value=1.0)
num_major_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)

sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
fasting_blood_sugar_gt120 = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
resting_ecg_result = st.selectbox("Resting ECG Result", ["normal", "st-t abnormality", "left ventricular hypertrophy"])
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
st_slope_peak_exercise = st.selectbox("ST Slope Peak Exercise", ["upsloping", "flat", "downsloping"])
thal_defect_type = st.selectbox("Thal Defect Type", ["normal", "fixed defect", "reversable defect"])

# --- Prediction ---
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age_years": age_years,
        "resting_bp_mmHg": resting_bp_mmHg,
        "serum_cholesterol_mgdl": serum_cholesterol_mgdl,
        "thalch": thalch,
        "st_depression_exercise": st_depression_exercise,
        "num_major_vessels": num_major_vessels,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "fasting_blood_sugar_gt120": fasting_blood_sugar_gt120,
        "resting_ecg_result": resting_ecg_result,
        "exercise_induced_angina": exercise_induced_angina,
        "st_slope_peak_exercise": st_slope_peak_exercise,
        "thal_defect_type": thal_defect_type
    }])

    try:
        prediction = model.predict(input_data)[0]
        prediction_int = int(round(prediction))

        severity_levels = {
            0: "No heart disease",
            1: "Low risk",
            2: "Moderate risk",
            3: "High risk",
            4: "Very high risk"
        }
        severity = severity_levels.get(prediction_int, f"Unknown severity level: {prediction_int}")

        st.success(f"Predicted severity: {severity}")
        st.info(f"Raw model output: {prediction:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
