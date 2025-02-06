import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib # For saving and loading models


################################################################################
#                            DATA LOADING MODULE                               #
################################################################################

def load_data(csv_path):
    """Loads the dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the dataset.
    """
    df = pd.read_csv(csv_path)
    return df


################################################################################
#                         DATA PREPROCESSING MODULE                            #
################################################################################

def preprocess_data(df):
    """Preprocesses the input DataFrame:
        - Fills missing values
        - Encodes categorical features ('sex', 'period.name', 'time')
        - Converts 'time' and 'start.time' to datetime and extracts features

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """

    df.fillna({
        'phq1': 0, 'phq2': 0, 'phq3': 0, 'phq4': 0, 'phq5': 0,
        'phq6': 0, 'phq7': 0, 'phq8': 0, 'phq9': 0,
        'happiness.score': 0, 'start.time': 0, 'time': 'unknown', 'period.name': 'unknown', 'phq.day': 0,
        'phq9_total': 0, 'phq9_severity': 'unknown'
    }, inplace=True)

    label_encoder = LabelEncoder()
    df['sex'] = label_encoder.fit_transform(df['sex'])
    df['period.name'] = label_encoder.fit_transform(df['period.name'])
    df['time'] = label_encoder.fit_transform(df['time'])

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour_of_day'] = df['time'].dt.hour
    df['start.time'] = pd.to_datetime(df['start.time'], errors='coerce')
    df['start_timestamp'] = df['start.time'].astype(int) / 10**9

    return df

def aggregate_by_user(df):
    """Aggregates the DataFrame by 'user_id', calculating mean for numerical columns
       and 'first' for 'phq9_severity'.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    agg_df = df.groupby('user_id').agg({
        'phq1': 'mean', 'phq2': 'mean', 'phq3': 'mean', 'phq4': 'mean', 'phq5': 'mean',
        'phq6': 'mean', 'phq7': 'mean', 'phq8': 'mean', 'phq9': 'mean',
        'age': 'mean', 'sex': 'mean', 'happiness.score': 'mean', 'hour_of_day': 'mean',
        'period.name': 'mean', 'phq.day': 'mean', 'start_timestamp': 'mean',
        'phq9_total': 'mean', 'phq9_severity': 'first'
    }).reset_index()
    return agg_df



################################################################################
#                         FEATURE ENGINEERING MODULE                           #
################################################################################

# Define weights here - consider moving to utils.py if used elsewhere
WEIGHTS = {
    "depressed_mood": [0.15, ['q1', 'q2']], "loss_of_interest": [0.10, ['q3', 'q4']],
    "fatigability": [0.10, ['q5', 'q6']], "concentration_attention": [0.10, ['q11', 'q12']],
    "self_esteem": [0.10, ['q7', 'q8']], "guilt_unworthiness": [0.10, ['q9', 'q10']],
    "pessimistic_views": [0.05, ['q46', 'q47']], "self_harm_suicide": [0.15, ['q16']],
    "disturbed_sleep": [0.05, ['q13']], "diminished_appetite": [0.05, ['q14']]
}
ICD10_COLUMNS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                 'q11', 'q12', 'q13', 'q14', 'q16', 'q46', 'q47']


def add_icd10_columns(agg_df, original_df):
    """Adds ICD-10 related columns (q1-q14, q16, q46, q47) to the aggregated DataFrame
    by calculating the mean for each user.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame (output of aggregate_by_user).
        original_df (pd.DataFrame): Original, non-aggregated DataFrame.

    Returns:
        pd.DataFrame: DataFrame with ICD-10 columns added.
    """
    agg_df[ICD10_COLUMNS] = original_df.groupby('user_id')[ICD10_COLUMNS].mean()
    return agg_df


def calculate_weighted_symptom_score(agg_df, weights=WEIGHTS):
    """Calculates a weighted symptom score based on predefined weights and questions.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame with relevant 'q' columns.
        weights (dict): Dictionary of symptom weights and corresponding question lists.

    Returns:
        pd.DataFrame: DataFrame with 'weighted_symptom_score' column added.
    """
    weighted_symptom_scores = []
    for _, (weight, questions) in weights.items():
        symptom_score = agg_df[questions].mean(axis=1)
        weighted_symptom_scores.append(symptom_score * weight)

    agg_df['weighted_symptom_score'] = sum(weighted_symptom_scores)
    return agg_df

def prepare_features_and_target(agg_df, icd10_columns=ICD10_COLUMNS):
    """Selects features and target variable from the aggregated DataFrame.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame with all engineered features.
        icd10_columns (list): List of ICD-10 related column names.

    Returns:
        tuple: (features DataFrame, target Series, feature column names)
    """
    features_columns = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
                       'age', 'sex', 'happiness.score', 'hour_of_day', 'period.name', 'phq.day', 'start_timestamp',
                       'phq9_total', 'weighted_symptom_score'] + icd10_columns

    features = agg_df[features_columns]
    target = agg_df['phq9_severity'] # target is still severity string here

    return features, target, features_columns



################################################################################
#                            MODEL TRAINING MODULE                             #
################################################################################

def train_model(X_train, y_train):
    """Trains an ensemble model (RandomForest, XGBoost, Logistic Regression)
       and returns the trained ensemble classifier.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training target.

    Returns:
        VotingClassifier: Trained ensemble classifier.
    """
    # Impute missing values (important to do *before* scaling, and on *training data only* then transform test)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    # Scale features (important to fit on *training data only* and then transform test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)


    # Define models
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss' # eval_metric is needed now
    )
    lr_clf = LogisticRegression(max_iter=1000)

    # Voting Classifier (Hard Voting)
    ensemble_clf = VotingClassifier(estimators=[
        ('rf', rf_clf),
        ('xgb', xgb_clf),
        ('lr', lr_clf)
    ], voting='hard')

    # Train the model
    ensemble_clf.fit(X_train_scaled, y_train) # Train on scaled, imputed data

    return ensemble_clf, imputer, scaler # return imputer and scaler too!

def prepare_training_data(features, target):
    """Prepares data for training: encodes target, splits data into train/test,
       and returns the split data and target encoder.

    Args:
        features (pd.DataFrame): Feature DataFrame.
        target (pd.Series): Target Series (still string labels).

    Returns:
        tuple: (X_train, X_test, y_train, y_test, target_encoder)
    """
    # Encode the target variable (severity strings to numerical)
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target) # encode *before* split!

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, target_encoder


def save_model_and_artifacts(model, imputer, scaler, target_encoder, model_path='model.joblib', imputer_path='imputer.joblib', scaler_path='scaler.joblib', target_encoder_path='target_encoder.joblib'):
    """Saves the trained model, imputer, scaler, and target encoder to disk.

    Args:
        model (VotingClassifier): Trained ensemble classifier.
        imputer (SimpleImputer): Fitted SimpleImputer object.
        scaler (StandardScaler): Fitted StandardScaler object.
        target_encoder (LabelEncoder): Fitted LabelEncoder object for target variable.
        model_path (str): Path to save the model file.
        imputer_path (str): Path to save the imputer file.
        scaler_path (str): Path to save the scaler file.
        target_encoder_path (str): Path to save the target encoder file.
    """
    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(target_encoder, target_encoder_path)
    print(f"Model, imputer, scaler and target encoder saved to {model_path}, {imputer_path}, {scaler_path}, {target_encoder_path}")



################################################################################
#                           MODEL EVALUATION MODULE                            #
################################################################################

def evaluate_model(model, X_test, y_test, scaler_path='scaler.joblib', imputer_path='imputer.joblib'):
    """Evaluates the trained model on the test set and prints accuracy and confusion matrix.

    Args:
        model (VotingClassifier): Trained ensemble classifier.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test target.
        scaler_path (str): Path to the saved scaler.
        imputer_path (str): Path to the saved imputer.
    Returns:
        tuple: (accuracy, confusion_matrix)
    """

    # Load scaler and imputer
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)


    # Apply same preprocessing as during training: imputation and scaling (use *transform* not *fit_transform* on test data!)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)


    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    return accuracy, conf_matrix



################################################################################
#                           MODEL PREDICTION MODULE                            #
################################################################################

def load_model_and_artifacts(model_path='model.joblib', imputer_path='imputer.joblib', scaler_path='scaler.joblib', target_encoder_path='target_encoder.joblib'):
    """Loads the trained model, imputer, scaler, and target encoder from disk.

    Args:
        model_path (str): Path to the model file.
        imputer_path (str): Path to the imputer file.
        scaler_path (str): Path to the scaler file.
        target_encoder_path (str): Path to the target encoder file.

    Returns:
        tuple: (trained model, imputer, scaler, target_encoder)
    """
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    target_encoder = joblib.load(target_encoder_path)
    return model, imputer, scaler, target_encoder


def predict_severity(data_point, model, imputer, scaler, target_encoder, feature_columns):
    """Predicts mental health severity for a single data point (e.g., from user input).

    Args:
        data_point (dict): Dictionary representing a single data point, keys are feature names.
        model (VotingClassifier): Trained ensemble classifier.
        imputer (SimpleImputer): Fitted SimpleImputer object.
        scaler (StandardScaler): Fitted StandardScaler object.
        target_encoder (LabelEncoder): Fitted LabelEncoder object for target variable.
        feature_columns (list): List of feature column names the model expects (order matters!).

    Returns:
        str: Predicted severity label (e.g., "Minimal", "Mild", etc.).
    """
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


################################################################################
#                                 UTILS                                      #
################################################################################

# utils.py (content within the same file now)
WEIGHTS = {
    "depressed_mood": [0.15, ['q1', 'q2']], "loss_of_interest": [0.10, ['q3', 'q4']],
    "fatigability": [0.10, ['q5', 'q6']], "concentration_attention": [0.10, ['q11', 'q12']],
    "self_esteem": [0.10, ['q7', 'q8']], "guilt_unworthiness": [0.10, ['q9', 'q10']],
    "pessimistic_views": [0.05, ['q46', 'q47']], "self_harm_suicide": [0.15, ['q16']],
    "disturbed_sleep": [0.05, ['q13']], "diminished_appetite": [0.05, ['q14']]
}
ICD10_COLUMNS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                 'q11', 'q12', 'q13', 'q14', 'q16', 'q46', 'q47']


################################################################################
#         MAIN FUNCTION (Now for Prediction from External Input)              #
################################################################################

# Load model and artifacts *outside* the function for efficiency (load once on app start)
MODEL, IMPUTER, SCALER, TARGET_ENCODER = load_model_and_artifacts()
_, _, FEATURE_COLUMNS = prepare_features_and_target(pd.DataFrame({'phq9_severity': ['Minimal']})) # dummy DF to get columns


def get_diagnosis_from_input(user_input_json):
    """
    Takes user input data (e.g., from a JSONified HTML form), preprocesses it,
    predicts mental health severity, and returns the prediction.

    Args:
        user_input_json (dict): JSON-like dictionary containing user input features.
                                 Must have keys corresponding to FEATURE_COLUMNS.

    Returns:
        str: Predicted severity label (e.g., "Minimal", "Mild", etc.).
    """
    try:
        # 1. Convert JSON input to DataFrame (assuming input is already a dict in Python after JSON parsing)
        input_data_point_df = pd.DataFrame([user_input_json])

        # 2. Feature Engineering -  *Adapt Preprocessing if needed for single input*
        #    For now, we assume input JSON directly contains the features expected by the model.
        #    If HTML form is collecting raw PHQ/Q questions, you'd need to apply
        #    preprocess_data, aggregate_by_user, add_icd10_columns, calculate_weighted_symptom_score *here*
        #    to generate the features from raw input.

        # 3. Prediction
        predicted_severity = predict_severity(input_data_point_df.iloc[0].to_dict(), MODEL, IMPUTER, SCALER, TARGET_ENCODER, FEATURE_COLUMNS) # pass dict, use pre-loaded artifacts

        return predicted_severity

    except Exception as e:
        print(f"Error during prediction: {e}") # Log error for debugging in real app
        return "Error: Could not generate diagnosis." # Return error message to display to user


if __name__ == '__main__':
    # **Example Usage for Testing `get_diagnosis_from_input`**

    # First, train and save the model artifacts (you only need to do this once)
    if False: # Set to True to re-train and save model (usually you train once and then just load)
        main_training_pipeline() # Call the training pipeline to train and save model
        print("Model training and artifacts saved. Set the 'if False' above to False to use the saved model for prediction.")


    # Example JSON-like input from HTML form (replace with actual form data)
    test_input_json = {
        'phq1': 1, 'phq2': 0, 'phq3': 1, 'phq4': 2, 'phq5': 0, 'phq6': 1, 'phq7': 0, 'phq8': 2, 'phq9': 1,
        'age': 42, 'sex': 0, 'happiness.score': 2, 'hour_of_day': 14, 'period.name': 1, 'phq.day': 10, 'start_timestamp': 1678900000.0,
        'phq9_total': 8, 'weighted_symptom_score': 0.5,
        'q1': 1, 'q2': 1, 'q3': 0, 'q4': 0, 'q5': 1, 'q6': 1, 'q7': 0, 'q8': 0, 'q9': 1, 'q10': 1,
        'q11': 0, 'q12': 0, 'q13': 1, 'q14': 1, 'q16': 0, 'q46': 0, 'q47': 0
    }

    # Get diagnosis
    predicted_diagnosis = get_diagnosis_from_input(test_input_json)
    print(f"\n--- Prediction Test ---")
    print(f"Input JSON: {test_input_json}")
    print(f"Predicted Diagnosis: {predicted_diagnosis}")


################################################################################
#                             MAIN FUNCTION (Example Pipeline)                 #
################################################################################

def main():
    # 1. Load Data
    df = load_data('updated_dataset.csv') # Make sure 'updated_dataset.csv' is in same directory or adjust path

    # 2. Preprocess Data
    processed_df = preprocess_data(df.copy()) # use .copy() to avoid modifying original
    agg_df = aggregate_by_user(processed_df.copy())

    # 3. Feature Engineering
    agg_df_with_icd = add_icd10_columns(agg_df.copy(), processed_df.copy())
    weighted_df = calculate_weighted_symptom_score(agg_df_with_icd.copy())
    features, target, _ = prepare_features_and_target(weighted_df.copy())

    # 4. Prepare Training Data
    X_train, X_test, y_train, y_test, target_encoder = prepare_training_data(features.copy(), target.copy())

    # 5. Train Model
    model, imputer, scaler = train_model(X_train.copy(), y_train.copy())

    # 6. Save Model and artifacts
    save_model_and_artifacts(model, imputer, scaler, target_encoder)

    # 7. Evaluate Model
    evaluate_model(model, X_test.copy(), y_test.copy())


if __name__ == '__main__':
    main()
    print("\nEnd-to-end pipeline run (see model.joblib, imputer.joblib, scaler.joblib, target_encoder.joblib for saved artifacts).")