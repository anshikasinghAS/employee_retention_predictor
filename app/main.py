import joblib 
import numpy as np
import pandas as pd

# Load model
model = joblib.load('../models/voting_clf_final.pkl')

# Load encoders (update paths as needed)
le_company_size_grouped = joblib.load('../encoders/le_company_size_grouped.pkl')

# Load feature columns to ensure correct input structure
feature_columns = joblib.load('../encoders/feature_columns.pkl')

# Prediction function
def preprocess_and_predict(input_data):
    """
    Preprocess the input data and make predictions using the loaded model.
    """
    df = pd.DataFrame([input_data])

    # Encode the categorical columns (company_size_grouped)
    if 'company_size_grouped' in df.columns:
        df['company_size_grouped'] = le_company_size_grouped.transform(df['company_size_grouped'])

    # Reorder columns to match model's expected input structure
    df = df.reindex(columns=feature_columns)

    # Predict
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return prediction, prob


