
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load model and encoder
model_path = os.path.join("..", "models", "voting_clf_final.pkl")
encoder_path = os.path.join("..", "encoders", "le_company_size_grouped.pkl")
model = joblib.load(model_path)
le_company_size_grouped = joblib.load(encoder_path)

# Title
st.title("Employee Retention Prediction App")
st.markdown("Fill in the details below to predict whether an employee will leave the company.")

# Grouped dropdown for City Development Index
city_dev_group = st.selectbox(
    "City Development Index Level",
    ["Low (0.40–0.59)", "Medium (0.60–0.74)", "High (0.75–1.00)"]
)

# Convert group to numeric value
if "Low" in city_dev_group:
    city_development_index = 0.55
elif "Medium" in city_dev_group:
    city_development_index = 0.68
else:
    city_development_index = 0.85

# Input fields
training_hours = st.slider("Training Hours", min_value=1, max_value=300, value=50)
gender = st.selectbox("Gender", ["Female", "Male", "Other", "Unknown"])
relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
education_level = st.selectbox("Education Level", ["Graduate", "High School", "Masters", "Phd", "Primary School"])
company_type = st.selectbox("Company Type", ["Early Stage Startup", "Funded Startup", "NGO", "Other", "Public Sector", "Pvt Ltd", "Unknown"])
company_size_grouped = st.selectbox("Grouped Company Size", ['Small', 'Medium', 'Large', 'Very Large', 'Other'])

# Mappings
gender_map = {"Female": 0, "Male": 1, "Other": 2, "Unknown": 3}
experience_map = {"Has relevent experience": 0, "No relevent experience": 1}
education_map = {"Graduate": 0, "High School": 1, "Masters": 2, "Phd": 3, "Primary School": 4}
company_type_map = {
    "Early Stage Startup": 0,
    "Funded Startup": 1,
    "NGO": 2,
    "Other": 3,
    "Public Sector": 4,
    "Pvt Ltd": 5,
    "Unknown": 6
}

# Prediction button
if st.button("Predict"):
    # Construct input dictionary
    input_dict = {
        "city_development_index": city_development_index,
        "gender": gender_map[gender],
        "relevent_experience": experience_map[relevent_experience],
        "enrolled_university": 2,  # default = no_enrollment
        "education_level": education_map[education_level],
        "major_discipline": 5,     # default = STEM
        "experience": 5,           # default or user input if needed
        "company_size": 2,         # default or user input
        "company_type": company_type_map[company_type],
        "last_new_job": 1,         # default
        "training_hours": training_hours,
        "training_hours_log": np.log1p(training_hours),
        "devindex_training_interaction": city_development_index * training_hours,
        "company_size_grouped": [company_size_grouped]  # encoded below
    }

    # Encode grouped company size
    input_dict["company_size_grouped"] = le_company_size_grouped.transform(input_dict["company_size_grouped"])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    st.markdown("### Prediction Result")
    st.write("Will the employee leave?")
    st.success("✅ Yes" if prediction == 1 else "❌ No")
    st.write(f"**Probability of Leaving:** {probability:.2f}")
