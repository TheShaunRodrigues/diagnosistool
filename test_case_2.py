# Install required libraries
"""!pip install pandas numpy scikit-learn flask cryptography transformers sqlite3"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Load the dataset
data = pd.read_csv('Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv')

# Print the first few rows of the dataset to identify the column index
print(data.head())

# Feature columns (PHQ-9 questions) and target (severity)
X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]

# Assuming 'depression_severity' is the 10th column (index 9) - adjust if needed
y = data.iloc[:, 9]

# Handle missing values in 'y'
# Option 1: Remove rows with missing values in 'y'
data = data.dropna(subset=[data.columns[9]])  # Drop rows with NaN in the 10th column
X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]
y = data.iloc[:, 9]

# Option 2: Impute missing values with the mean (or median, etc.)
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='mean')  # Or strategy='median'
# y = imputer.fit_transform(y.values.reshape(-1, 1))  # Reshape for imputation
# y = y.ravel()  # Flatten back to 1D array


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Load pre-trained emotion analysis model from Hugging Face
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Function to predict depression severity
def predict_depression_severity(responses):
    return rf_model.predict([responses])[0]

# Function to analyze emotion using Hugging Face model
def analyze_emotion(text):
    emotions = emotion_analyzer(text)
    emotion = max(emotions[0], key=lambda x: x['score'])['label']
    return emotion

# Simulate PHQ-9 responses
responses = [2, 1, 3, 1, 0, 2, 1, 0, 2]  # Replace with actual responses
predicted_severity = predict_depression_severity(responses)
print(f"Predicted Depression Severity: {predicted_severity}")

# Simulate mood text for emotion analysis
mood_text = "I feel really down and stressed."
predicted_emotion = analyze_emotion(mood_text)
print(f"Predicted Emotion: {predicted_emotion}")

# Save the trained RandomForest model
import joblib
joblib.dump(rf_model, 'random_forest_depression_model.pkl')

# Load the model later
rf_model = joblib.load('random_forest_depression_model.pkl')

data['phq9_total'] = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']].sum(axis=1)

y = data.iloc[:, 9]


X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]

model = RandomForestClassifier()
model.fit(X_train, y_train)

emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def analyze_emotion(text):
    emotions = emotion_analyzer(text)
    emotion = max(emotions[0], key=lambda x: x['score'])['label']
    return emotion

def predict_depression_severity(responses):
    return model.predict([responses])[0]

from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(data):
    return cipher_suite.decrypt(data).decode()

def save_assessment(user_id, responses, score, sentiment):
    conn = sqlite3.connect('phq9_assessments.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments
                      (user_id TEXT, responses TEXT, score INTEGER, sentiment REAL)''')
    cursor.execute("INSERT INTO assessments VALUES (?, ?, ?, ?)",
                   (user_id, str(responses), score, sentiment))
    conn.commit()
    conn.close()

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

"""from IPython.core.display import display, HTML
from google.colab import files

# Step 1: Upload Files
print("/content/index.html")
print("/content/style.css")
uploaded = files.upload()

# Step 2: Read Files
html_code = ""
css_code = ""

# Read HTML file
if 'index.html' in uploaded:
    with open('index.html', 'r') as file:
        html_code = file.read()

# Read CSS file
if 'style.css' in uploaded:
    with open('style.css', 'r') as file:
        css_code = "<style>" + file.read() + "</style>"

# Step 3: Render HTML and CSS
if html_code:
    display(HTML(css_code + html_code))
else:
    print("HTML file not uploaded. Please try again.")"""