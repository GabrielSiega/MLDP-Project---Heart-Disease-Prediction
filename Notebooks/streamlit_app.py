import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model
# -------------------------
model = joblib.load(
    "D:/MLDP Project/Notebooks/Models/trained_HD_model.pkl"
)

# -------------------------
# Feature columns (FROM TRAINING)
# ⚠️ Must match X_train.columns EXACTLY
# -------------------------
FEATURE_COLUMNS = [
    "thalch",
    "num_major_vessels",
    "st_depression_exercise",
    "age_years",
    "serum_cholesterol_mgdl",
    "resting_bp_mmHg",
    "chest_pain_type_typical angina",
    "chest_pain_type_atypical angina",
    "chest_pain_type_non-anginal",
    "chest_pain_type_asymptomatic",
    "thal_defect_type_normal",
    "thal_defect_type_fixed defect",
    "thal_defect_type_reversable defect"
]

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

thalch = st.number_input("Max Heart Rate Achieved (thalch)", 60, 220, 120)
num_major_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
st_depression_exercise = st.number_input("ST Depression (Exercise)", 0.0, 10.0, 2.5)
age_years = st.number_input("Age (Years)", 1, 120, 68)
serum_cholesterol_mgdl = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
resting_bp_mmHg = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 180)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    CHEST_PAIN_CATEGORIES
)

thal_defect_type = st.selectbox(
    "Thal Defect Type",
    THAL_DEFECT_CATEGORIES
)

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

    # 🔒 Force categorical consistency (KEY FIX)
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

    # Align with training feature space
    new_patient_encoded = new_patient_encoded.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # Predict
    prediction = model.predict(new_patient_encoded)[0]

    st.success(f"Predicted Heart Disease Severity: **{prediction}**")
