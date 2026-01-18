import joblib
import streamlit as st
import pandas as pd

# Configurable model path
MODEL_PATH = r"D:\MLDP Project\Notebooks\Models\trained_HD_model.pkl"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and description
st.title("❤️ Heart Disease Severity Predictor")
st.write("Enter patient details below to predict heart disease presence/severity. "
         "Note: This is for educational purposes only; consult a doctor for medical advice.")

# Collect user inputs (raw values, no manual mapping)
age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
sex_male_female = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
resting_bp_mmHg = st.number_input("Resting BP (mmHg)", min_value=80, max_value=200, value=120)
serum_cholesterol_mgdl = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_blood_sugar_gt120 = st.selectbox("Fasting Blood Sugar > 120", ["True", "False"])
resting_ecg_result = st.selectbox("Resting ECG Result", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.number_input("Max Heart Rate (thalch)", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["True", "False"])
st_depression_exercise = st.number_input("ST Depression (exercise)", min_value=0.0, max_value=10.0, value=1.0)
st_slope_peak_exercise = st.selectbox("ST Slope (peak exercise)", ["upsloping", "flat", "downsloping"])
num_major_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
thal_defect_type = st.selectbox("Thal Defect Type", ["normal", "fixed defect", "reversible defect"])

# Prediction button
if st.button("Predict"):
    # Build DataFrame with raw values
    input_data = pd.DataFrame([{
        'age_years': age_years,
        'sex_male_female': sex_male_female,
        'chest_pain_type': chest_pain_type,
        'resting_bp_mmHg': resting_bp_mmHg,
        'serum_cholesterol_mgdl': serum_cholesterol_mgdl,
        'fasting_blood_sugar_gt120': fasting_blood_sugar_gt120,
        'resting_ecg_result': resting_ecg_result,
        'thalch': thalch,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression_exercise': st_depression_exercise,
        'st_slope_peak_exercise': st_slope_peak_exercise,
        'num_major_vessels': num_major_vessels,
        'thal_defect_type': thal_defect_type,
        'num_major_vessels_outlier': 0  # engineered column required by model
    }])

    # Debug check (optional)
    st.write("Debug input types:", input_data.dtypes)

    # Prediction
    try:
        prediction = model.predict(input_data)[0]

        # Round float output to nearest severity level
        prediction_int = int(round(prediction))

        severity_levels = {
            0: "No heart disease",
            1: "Low risk",
            2: "Moderate risk",
            3: "High risk",
            4: "Very high risk"
        }

        severity = severity_levels.get(prediction_int, f"Unknown severity level: {prediction_int}")

        # Show both raw score and mapped severity
        st.success(f"Predicted severity of heart disease: {severity}")
        st.info(f"Raw model output: {prediction:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}. Check if inputs match the model's expected format.")
