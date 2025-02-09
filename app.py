from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
from ml_functions import load_model_and_artifacts, prepare_features_and_target, predict_severity # Import ML functions

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # VERY IMPORTANT: CHANGE THIS!
# Load model and artifacts when the app starts
MODEL, IMPUTER, SCALER, TARGET_ENCODER = load_model_and_artifacts()

def predict_depression_severity(phq9_responses, icd10_responses):
    """Predict depression severity using the trained ML model, ensuring all necessary features are included."""

    # Expected feature columns used during model training
    feature_columns = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
                       'age', 'sex', 'happiness.score', 'hour_of_day', 'period.name', 'phq.day', 'start_timestamp',
                       'phq9_total', 'weighted_symptom_score', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8',
                       'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q16', 'q46', 'q47']

    # Calculate missing feature values
    age = 25  # Default age if not collected
    sex = 0   # Default sex (0: Male, 1: Female, if applicable)
    happiness_score = 5  # Neutral score (adjust if needed)
    hour_of_day = 12  # Assume midday response time
    period_name = 1  # Assume "morning" period
    phq_day = 0  # Default value for missing data
    start_timestamp = 1678900000.0  # Example timestamp
    phq9_total = sum(phq9_responses)  # Total PHQ-9 score
    weighted_symptom_score = sum(phq9_responses) / 9  # Example normalization

    # Prepare final feature dictionary
    user_input = {
        'phq1': phq9_responses[0], 'phq2': phq9_responses[1], 'phq3': phq9_responses[2],
        'phq4': phq9_responses[3], 'phq5': phq9_responses[4], 'phq6': phq9_responses[5],
        'phq7': phq9_responses[6], 'phq8': phq9_responses[7], 'phq9': phq9_responses[8],
        'age': age, 'sex': sex, 'happiness.score': happiness_score, 'hour_of_day': hour_of_day,
        'period.name': period_name, 'phq.day': phq_day, 'start_timestamp': start_timestamp,
        'phq9_total': phq9_total, 'weighted_symptom_score': weighted_symptom_score
    }

    # Add ICD-10 responses
    icd10_keys = ["q1", "q2", "q3", "q4", "q5", "q6", "q11", "q12", "q7", "q8", 
                  "q9", "q10", "q46", "q47", "q16", "q13", "q14"]
    
    for key, response in zip(icd10_keys, icd10_responses):
        user_input[key] = response

    # Ensure features are in correct order
    input_df = pd.DataFrame([user_input], columns=feature_columns)

    # Predict using the trained model
    predicted_severity = predict_severity(input_df.iloc[0].to_dict(), MODEL, IMPUTER, SCALER, TARGET_ENCODER, feature_columns)

    return predicted_severity



def save_assessment(user_id, phq9_responses, severity, emotion):
    # Your database or file saving logic here
    print(f"Saving assessment for user {user_id}: Responses={phq9_responses}, Severity={severity}, Emotion={emotion}")

def analyze_emotion(text):
    return "N/A"  # Replace with your emotion analysis logic

def generate_summary_and_feedback(phq9_responses, icd10_responses):
    """Generates a summary and feedback based on both PHQ-9 and ICD-10 responses."""

    total_phq9_score = sum(phq9_responses)
    total_icd10_score = sum(icd10_responses)

    # ICD-10 weighted contribution (adjust weight as necessary)
    combined_score = total_phq9_score + (total_icd10_score * 0.5)  # Giving ICD-10 half the weight

    summary = (
        f"Your total PHQ-9 score is {total_phq9_score}, and your total ICD-10 score is {total_icd10_score}. "
        f"Considering both assessments, your combined depression severity score is {combined_score:.1f}."
    )

    # Determine feedback based on combined score
    if combined_score < 5:
        feedback = "Your score suggests minimal or no depression."
    elif combined_score < 10:
        feedback = "Your score suggests mild depression. Consider stress management techniques and social support."
    elif combined_score < 15:
        feedback = "Your score suggests moderate depression. Seeking a mental health professional is recommended."
    elif combined_score < 20:
        feedback = "Your score suggests severe depression. Please consider seeking professional help immediately."
    else:
        feedback = "Your score suggests very severe depression. **Immediate professional intervention is advised.**"

    return summary, feedback


def generate_responses(phq9_responses, icd10_responses):
    """Generates explanations for PHQ-9 and ICD-10 responses."""

    # PHQ-9 Questions & Explanations
    phq9_questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself—or that you are a failure or have let yourself or your family down?",
        "Trouble concentrating on things, such as reading or watching TV?",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?",
        "Thoughts that you would be better off dead or of hurting yourself in some way?"
    ]

    phq9_explanations = [
        [
            "You still find joy in most activities, which is a good sign.",
            "You sometimes struggle to enjoy things, but this may be temporary. Try engaging in hobbies that previously made you happy.",
            "A noticeable loss of interest may indicate emotional distress. Consider journaling your feelings and engaging in social activities.",
            "A complete lack of pleasure can be a sign of depression. Seeking professional guidance may help you regain motivation."
        ],
        [
            "Your mood seems stable, and you may not be experiencing significant sadness.",
            "You occasionally feel down, which is normal. Try mindfulness or relaxation techniques to manage it.",
            "Frequent sadness and hopelessness could be concerning. Talking to a trusted friend or counselor might help.",
            "Persistent feelings of sadness or hopelessness can be serious. Consider speaking to a mental health professional."
        ],
        [
            "Your sleep patterns seem normal, which is great for mental health.",
            "You occasionally experience sleep issues. Try maintaining a consistent sleep routine.",
            "Frequent sleep disturbances may be affecting your well-being. Avoid caffeine before bedtime and reduce screen time.",
            "Severe sleep issues can worsen emotional distress. Consult a professional if this persists."
        ],
        [
            "You generally feel energetic, which is a positive sign.",
            "Mild fatigue is common but should be monitored. Ensure you're eating well and staying hydrated.",
            "Frequent fatigue may indicate emotional exhaustion. A healthy sleep schedule and regular exercise might help.",
            "Constant exhaustion can be a sign of depression. Seek medical advice if this continues."
        ],
        [
            "Your eating habits appear stable.",
            "Occasional appetite changes are normal, but track if they become frequent.",
            "Moderate appetite fluctuations may be linked to emotional distress. Keeping a food journal might help.",
            "Significant appetite changes can indicate underlying issues. Consult a healthcare provider if needed."
        ],
        [
            "You maintain a balanced self-view, which is positive.",
            "Occasionally feeling down about yourself is normal, but avoid self-criticism.",
            "Frequent self-doubt could indicate low self-esteem. Consider self-compassion exercises.",
            "Severe feelings of worthlessness may require therapy. Seeking support is important."
        ],
        [
            "You seem to have good focus and concentration.",
            "Mild concentration issues can be caused by stress. Try deep breathing exercises.",
            "Moderate difficulty concentrating could impact daily tasks. Structured routines may help.",
            "Severe focus issues might require professional assessment, especially if they persist."
        ],
        [
            "Your movement and energy levels appear normal.",
            "Occasional restlessness or slowness can be due to stress or fatigue.",
            "Frequent changes in movement or energy levels may be concerning. Monitoring patterns can help.",
            "Severe agitation or slowing down significantly could indicate depression. Seek medical advice."
        ],
        [
            "You have no thoughts of self-harm, which is reassuring.",
            "Occasional distressing thoughts can happen but should be addressed early.",
            "Frequent thoughts of self-harm are serious. Please talk to a trusted friend or therapist.",
            "If you are having suicidal thoughts, **please seek immediate help from a professional or a crisis hotline. You are not alone.**"
        ]
    ]

    # ICD-10 Questions & Explanations
    icd10_questions = [
        "Are you feeling depressed?",
        "Are you feeling hopeless?",
        "Do you feel like you are not interested in anything right now?",
        "Do you have less pleasure in doing things you usually enjoy?",
        "Do you currently have considerably less energy?",
        "Are your everyday tasks making you very tired currently?",
        "Is it hard for you to make decisions currently?",
        "Is it hard for you to concentrate currently?",
        "Is your self-confidence clearly lower than usual?",
        "Are you feeling up to your tasks?",
        "Are you blaming yourself currently?",
        "Do you think you are worth less than others right now?",
        "Are you thinking that you will be doing well in the future?",
        "Are you looking hopefully into the future?",
        "Are you thinking about death more often than usual?",
        "Did you sleep badly last night?",
        "Do you have less or no appetite today?"
    ]

    icd10_explanations = [
        [
            "You don’t seem to experience depression frequently.",
            "You occasionally feel depressed, which is normal.",
            "Frequent depressive feelings should be monitored.",
            "Persistent depression is a serious concern. Seeking professional help is advised."
        ],
        [
            "You maintain a hopeful outlook, which is good.",
            "You occasionally feel hopeless, but this may be situational.",
            "Frequent hopelessness can be concerning. Talk to someone you trust.",
            "Strong feelings of hopelessness should not be ignored. Seek professional support."
        ],
        [
            "You maintain interest in activities.",
            "Mild loss of interest can happen due to stress or fatigue.",
            "A moderate loss of interest might indicate emotional distress.",
            "Severe disinterest could be a symptom of depression. Consider professional help."
        ],
        [
            "You still find pleasure in activities.",
            "Occasionally not enjoying things is normal.",
            "A persistent loss of enjoyment might be a concern.",
            "Completely losing interest in once-loved activities can indicate depression."
        ],
        [
            "Your energy levels are stable.",
            "Mild energy loss can happen due to stress.",
            "Moderate fatigue may suggest emotional exhaustion.",
            "Constant exhaustion should be addressed with professional help."
        ],
        [
            "You handle daily tasks well.",
            "Occasional tiredness is normal.",
            "Frequent tiredness might indicate deeper issues.",
            "Severe exhaustion requires further evaluation."
        ],
        [
            "You make decisions confidently.",
            "Occasional indecisiveness is normal.",
            "Frequent trouble making decisions may indicate distress.",
            "Severe difficulty in decision-making is concerning."
        ],
        [
            "You focus well.",
            "Occasional concentration issues happen to everyone.",
            "Persistent focus problems should be addressed.",
            "Severe difficulty concentrating might need professional intervention."
        ],
        [
            "You have a stable self-esteem.",
            "Occasional self-doubt is normal.",
            "Frequent low self-esteem should be monitored.",
            "Severe confidence issues may require therapy."
        ],
        [
            "You feel capable of handling tasks.",
            "Occasional self-doubt is fine.",
            "Frequent difficulty completing tasks may be a concern.",
            "Severe lack of motivation needs professional evaluation."
        ]
    ]

    responses = []
    
    # Process PHQ-9 Responses
    for i, question in enumerate(phq9_questions):
        score = phq9_responses[i] if i < len(phq9_responses) else 0
        explanation = phq9_explanations[i][score]
        responses.append({"question": question, "answer": str(score), "explanation": explanation})

    # Process ICD-10 Responses
    for i, question in enumerate(icd10_questions):
        score = icd10_responses[i] if i < len(icd10_responses) else 0
        explanation = icd10_explanations[i][score]
        responses.append({"question": question, "answer": str(score), "explanation": explanation})

    return responses




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/question')
def question_page():
    return render_template('question.html')

@app.route('/icd10')
def icd10_page():
    return render_template('icd10.html')

@app.route('/emotion-analysis', methods=['GET', 'POST'])
def emotion_analysis_page():
    severity = request.args.get('severity', 'N/A')  # Get severity from the URL
    if request.method == 'POST':
        text = request.form.get('text', '').strip()

        if text:
            predicted_emotion = analyze_emotion(text)
        else:
            predicted_emotion = "N/A"

        return redirect(url_for('results_page', severity=severity, emotion=predicted_emotion))  # Pass severity

    return render_template('emotion.html', severity=severity)  # Pass severity


@app.route('/results')
def results_page():
    severity = request.args.get('severity', 'N/A')  # Get severity from URL
    emotion = request.args.get('emotion', 'N/A')  # Get emotion from URL
    summary = session.get('summary', '')
    feedback = session.get('feedback', '')
    responses = session.get('responses', [])  # Get responses from session

    session.pop('summary', None)  # Clear from session
    session.pop('feedback', None)
    session.pop('responses', None)

    return render_template('results.html', severity=severity, emotion=emotion, summary=summary, feedback=feedback, responses=responses)  # Pass all data to template

@app.route('/process_icd10_assessment', methods=['POST'])
def process_icd10_assessment():
    try:
        data = request.get_json()
        icd10_responses = list(data.values())

        # Retrieve stored PHQ-9 responses
        phq9_responses = session.get('phq9_responses')

        if not phq9_responses:
            return jsonify({'error': 'Missing PHQ-9 responses from session'}), 400

        # Call the updated prediction function
        severity = predict_depression_severity(phq9_responses, icd10_responses)

        return jsonify({'redirect': url_for('results_page', severity=severity)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/process_assessment', methods=['POST'])
def process_assessment():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        phq9_responses = data.get('phq9_responses')

        if not user_id or not phq9_responses:
            return jsonify({'error': 'Missing user_id or PHQ-9 responses'}), 400

        # Store PHQ-9 responses in session for later use
        session['phq9_responses'] = phq9_responses

        return jsonify({'redirect': url_for('icd10_page')})  # Redirect to ICD-10 page

    except Exception as e:
        return jsonify({'error': str(e)}), 500

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    try:
        # Load test dataset (Ensure 'updated_dataset.csv' exists in the same folder)
        df = pd.read_csv('updated_dataset.csv')

        # Preprocess the dataset
        processed_df = preprocess_data(df)
        agg_df = aggregate_by_user(processed_df)
        weighted_df = calculate_weighted_symptom_score(agg_df)
        features, target, feature_columns = prepare_features_and_target(weighted_df)

        # Split into train and test set
        X_train, X_test, y_train, y_test, target_encoder = prepare_training_data(features, target)

        # Load the trained model and preprocessors
        model, imputer, scaler, _ = load_model_and_artifacts()

        # Apply the same preprocessing as training
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Convert numerical predictions back to labels
        y_pred_labels = target_encoder.inverse_transform(y_pred)
        y_test_labels = target_encoder.inverse_transform(y_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
        report = classification_report(y_test_labels, y_pred_labels)

        return jsonify({
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)