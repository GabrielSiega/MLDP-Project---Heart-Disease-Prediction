import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model
# -------------------------
model_path = r"D:\MLDP Project\Notebooks\Models\heart_disease_rf_v1.pkl"
model = joblib.load(model_path)   # load the actual model object

# -------------------------
# Define ALL categorical levels
# -------------------------
CHEST_PAIN_CATEGORIES = [
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic"
]

THAL_DEFECT_CATEGORIES = [
    "normal",
    "fixed defect",
    "reversable defect"
]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Heart Disease Severity Predictor")

st.title("❤️ Heart Disease Severity Prediction")
st.write("Enter patient information below:")

# Numeric inputs
thalch = st.number_input("Max Heart Rate Achieved (thalch)", 60, 220, 120)
num_major_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
st_depression_exercise = st.number_input("ST Depression (Exercise)", 0.0, 10.0, 2.5)
age_years = st.number_input("Age (Years)", 1, 120, 68)
serum_cholesterol_mgdl = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
resting_bp_mmHg = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 180)

# Place categorical choices side by side
col1, col2 = st.columns(2)

with col1:
    chest_pain_type = st.selectbox("Chest Pain Type", CHEST_PAIN_CATEGORIES)

with col2:
    thal_defect_type = st.selectbox("Thal Defect Type", THAL_DEFECT_CATEGORIES)

# -------------------------
# Prediction logic
# -------------------------
if st.button("Predict Severity"):
    new_patient = pd.DataFrame([{
        "thalch": thalch,
        "num_major_vessels": num_major_vessels,
        "st_depression_exercise": st_depression_exercise,
        "age_years": age_years,
        "serum_cholesterol_mgdl": serum_cholesterol_mgdl,
        "resting_bp_mmHg": resting_bp_mmHg,
        "chest_pain_type": chest_pain_type,
        "thal_defect_type": thal_defect_type
    }])

    # 🔒 Force categorical consistency
    new_patient["chest_pain_type"] = pd.Categorical(
        new_patient["chest_pain_type"],
        categories=CHEST_PAIN_CATEGORIES
    )
    new_patient["thal_defect_type"] = pd.Categorical(
        new_patient["thal_defect_type"],
        categories=THAL_DEFECT_CATEGORIES
    )

    # One-hot encode WITHOUT dropping columns
    new_patient_encoded = pd.get_dummies(new_patient)

    # Align with training feature space using model.feature_names_in_
    new_patient_encoded = new_patient_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    prediction = model.predict(new_patient_encoded)[0]
    st.success(f"Predicted Heart Disease Severity: **{prediction}**")
