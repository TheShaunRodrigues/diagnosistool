import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- Constants (from utils.py or ensemble.py) ---
WEIGHTS = {
    "depressed_mood": [0.15, ['q1', 'q2']], "loss_of_interest": [0.10, ['q3', 'q4']],
    "fatigability": [0.10, ['q5', 'q6']], "concentration_attention": [0.10, ['q11', 'q12']],
    "self_esteem": [0.10, ['q7', 'q8']], "guilt_unworthiness": [0.10, ['q9', 'q10']],
    "pessimistic_views": [0.05, ['q46', 'q47']], "self_harm_suicide": [0.15, ['q16']],
    "disturbed_sleep": [0.05, ['q13']], "diminished_appetite": [0.05, ['q14']]
}
ICD10_COLUMNS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                 'q11', 'q12', 'q13', 'q14', 'q16', 'q46', 'q47']


# --- Model Loading and Artifacts ---
def load_model_and_artifacts(model_path='model.joblib', imputer_path='imputer.joblib', scaler_path='scaler.joblib', target_encoder_path='target_encoder.joblib'):
    """Loads the trained model, imputer, scaler, and target encoder from disk."""
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    target_encoder = joblib.load(target_encoder_path)
    return model, imputer, scaler, target_encoder

# --- Feature Preparation ---
def prepare_features_and_target(agg_df, icd10_columns=ICD10_COLUMNS):
    """Selects features and target variable from the aggregated DataFrame."""
    features_columns = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
                       'age', 'sex', 'happiness.score', 'hour_of_day', 'period.name', 'phq.day', 'start_timestamp',
                       'phq9_total', 'weighted_symptom_score'] + icd10_columns

    features = agg_df[features_columns]
    target = agg_df['phq9_severity'] # target is still severity string here

    return features, target, features_columns


# --- Model Prediction ---
def predict_severity(data_point, model, imputer, scaler, target_encoder, feature_columns):
    """Predicts mental health severity for a single data point."""
    # 1. Create DataFrame from the data point
    input_df = pd.DataFrame([data_point], columns=feature_columns) # Ensure columns are in correct order

    # 2. Impute missing values (using *fitted* imputer)
    input_imputed = imputer.transform(input_df)

    # 3. Scale features (using *fitted* scaler)
    input_scaled = scaler.transform(input_imputed)

    # 4. Make prediction
    prediction_encoded = model.predict(input_scaled)

    # 5. Decode prediction (numerical label back to string severity)
    predicted_severity = target_encoder.inverse_transform(prediction_encoded)[0] # inverse_transform returns array, take first element

    return predicted_severity