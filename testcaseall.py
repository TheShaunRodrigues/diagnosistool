# Install required libraries
"""pip install pandas numpy scikit-learn flask cryptography transformers joblib sqlite3
"""
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import pipeline
import joblib
import sqlite3
from cryptography.fernet import Fernet

# Load the dataset
data = pd.read_csv('Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv')
print(data.head())  # Preview dataset

# Feature and target columns
X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]
y = data.iloc[:, 9]

# Handle missing values by dropping rows with NaN in the target
data = data.dropna(subset=[data.columns[9]])
X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]
y = data.iloc[:, 9]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Depression Severity Prediction Model


# Model 2: Logistic Regression
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Save the model for later use
joblib.dump(model, 'depression_severity_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Emotion Analysis Model

# Model : Default Emotion Model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Function to analyze emotion
def analyze_emotion(text):
    emotions = emotion_analyzer(text)
    emotion = max(emotions[0], key=lambda x: x['score'])['label']
    return emotion

#  Test Predictions with Sample Data

# Sample PHQ-9 responses
sample_responses = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # Replace with actual responses
predicted_severity = model.predict([sample_responses])[0]
print(f"Predicted Depression Severity: {predicted_severity}")

# Sample text for emotion analysis
sample_text = "I am gonna lose myself."
predicted_emotion = analyze_emotion(sample_text)
print(f"Predicted Emotion: {predicted_emotion}")

#  Save Results to Database

# Encrypt/Decrypt utility functions
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(data):
    return cipher_suite.decrypt(data).decode()

# Save assessment to SQLite database
def save_assessment(user_id, responses, score, sentiment):
    conn = sqlite3.connect('phq9_assessments.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments(user_id TEXT, responses TEXT, score INTEGER, sentiment TEXT)''')
    cursor.execute("INSERT INTO assessments VALUES (?, ?, ?, ?)",(user_id, str(responses), score, sentiment))
    conn.commit()
    conn.close()

# Save a sample result
user_id = "user123"
save_assessment(user_id, sample_responses, predicted_severity, predicted_emotion)
print("Assessment saved successfully.")

import sqlite3

conn = sqlite3.connect('phq9_assessments.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM assessments")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()