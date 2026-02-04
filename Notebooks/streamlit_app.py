import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = r"D:\MLDP Project\Notebooks\Models\heart_disease_rf_v1.pkl"

# Attempt to load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")

# -------------------------
# Feature columns (FROM CLEANED TRAINING)
# ⚠️ Updated to include ONLY the numeric columns from your list
# -------------------------
FEATURE_COLUMNS = [
    "age_years",
    "resting_bp_mmHg",
    "serum_cholesterol_mgdl",
    "thalch",
    "st_depression_exercise",
    "num_major_vessels"
]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Heart Disease Severity Predictor")

st.title("❤️ Heart Disease Severity Prediction")
st.write("Enter patient clinical information below:")

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", 1, 120, 50)
    resting_bp_mmHg = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)
    serum_cholesterol_mgdl = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    thalch = st.number_input("Max Heart Rate Achieved (thalch)", 60, 220, 150)
    st_depression_exercise = st.number_input("ST Depression (Exercise)", 0.0, 10.0, 1.0, step=0.1)
    num_major_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

# -------------------------
# Prediction logic
# -------------------------
if st.button("Predict Severity"):
    # Create DataFrame with only the selected 6 features
    new_patient = pd.DataFrame([{
        "age_years": age_years,
        "resting_bp_mmHg": resting_bp_mmHg,
        "serum_cholesterol_mgdl": serum_cholesterol_mgdl,
        "thalch": thalch,
        "st_depression_exercise": st_depression_exercise,
        "num_major_vessels": num_major_vessels
    }])

    # Align with training feature space (ensure column order is correct)
    new_patient_final = new_patient.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # Predict
    try:
        prediction = model.predict(new_patient_final)[0]
        
        # UI Styling based on result
        st.divider()
        if prediction == "Normal":
            st.success(f"Predicted Heart Disease Severity: **{prediction}**")
        else:
            st.warning(f"Predicted Heart Disease Severity: **{prediction}**")
            
    except NameError:
        st.error("Model is not loaded. Please check the file path.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.info("Note: This app is now configured to use only numeric clinical markers for prediction.")