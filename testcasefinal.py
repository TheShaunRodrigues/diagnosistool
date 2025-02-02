import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from cryptography.fernet import Fernet

# Load the updated dataset
file_path = "updated_dataset.csv"  # Ensure this file has a 'phq9_severity' column
df = pd.read_csv(file_path)

# Define feature columns (PHQ-9 responses)
phq_columns = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']

# Prepare input features (X) and target labels (y)
X = df[phq_columns]
y = df['phq9_severity']

# Encode severity labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'random_forest_depression_model.pkl')
rf_model = joblib.load('random_forest_depression_model.pkl')

# Load Hugging Face Emotion Analysis Model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def analyze_emotion(text):
    emotions = emotion_analyzer(text)
    return max(emotions[0], key=lambda x: x['score'])['label']

# Encryption setup
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(data):
    return cipher_suite.decrypt(data).decode()

# SQLite3 Database for Storing Assessments
def save_assessment(user_id, responses, score, sentiment):
    conn = sqlite3.connect('phq9_assessments.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments
                      (user_id TEXT, responses TEXT, score INTEGER, sentiment REAL)''')
    cursor.execute("INSERT INTO assessments VALUES (?, ?, ?, ?)",
                   (user_id, str(responses), score, sentiment))
    conn.commit()
    conn.close()

# Function to predict depression severity
def predict_depression_severity(responses):
    return label_encoder.inverse_transform([rf_model.predict([responses])[0]])[0]

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", class_report)
    return accuracy

evaluate_model(rf_model, X_test, y_test, label_encoder)

# Example Test Cases
responses = [2, 1, 3, 1, 0, 2, 1, 0, 2]
mood_text = "I feel really down and stressed."
predicted_severity = predict_depression_severity(responses)
predicted_emotion = analyze_emotion(mood_text)
print(f"Predicted Depression Severity: {predicted_severity}")
print(f"Predicted Emotion: {predicted_emotion}")
save_assessment("user123", responses, predicted_severity, predicted_emotion)