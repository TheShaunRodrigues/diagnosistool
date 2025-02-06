from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # VERY IMPORTANT: CHANGE THIS!

# Placeholder functions (replace with your actual implementations)
def predict_depression_severity(phq9_responses):
    total_score = sum(phq9_responses)
    if total_score < 5:
        return "None"
    elif total_score < 10:
        return "Mild"
    elif total_score < 15:
        return "Moderate"
    elif total_score < 20:
        return "Severe"
    else:
        return "Very Severe"

def save_assessment(user_id, phq9_responses, severity, emotion):
    # Your database or file saving logic here
    print(f"Saving assessment for user {user_id}: Responses={phq9_responses}, Severity={severity}, Emotion={emotion}")

def analyze_emotion(text):
    return "N/A"  # Replace with your emotion analysis logic

def generate_summary_and_feedback(phq9_responses):
    total_score = sum(phq9_responses)
    summary = f"Your total PHQ-9 score is {total_score}.  Responses: {phq9_responses}"
    if total_score < 5:
        feedback = "Your score suggests minimal or no depression."
    elif total_score < 10:
        feedback = "Your score suggests mild depression. Consider incorporating stress-reducing techniques and seeking support from friends or family."
    elif total_score < 15:
        feedback = "Your score suggests moderate depression.  It's recommended to consult with a mental health professional."
    elif total_score < 20:
        feedback = "Your score suggests severe depression. Please seek professional help immediately."
    else:
        feedback = "Your score suggests very severe depression.  Please seek professional help immediately."
    return summary, feedback

def generate_responses(phq9_responses):
    questions = [
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

    explanations = [
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

    responses = []
    for i in range(len(questions)):
        score = phq9_responses[i] if i < len(phq9_responses) else 0  # Ensure valid index
        explanation = explanations[i][score]  # Select explanation based on score
        response = {
            "question": questions[i],
            "answer": str(score),
            "explanation": explanation
        }
        responses.append(response)

    return responses




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/question')
def question_page():
    return render_template('question.html')

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

@app.route('/process_assessment', methods=['POST'])
def process_assessment():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        phq9_responses = data.get('phq9_responses')

        if not user_id or not phq9_responses:
            return jsonify({'error': 'Missing user_id or PHQ-9 responses'}), 400

        severity = predict_depression_severity(phq9_responses)
        summary, feedback = generate_summary_and_feedback(phq9_responses)
        responses = generate_responses(phq9_responses)

        session['summary'] = summary
        session['feedback'] = feedback
        session['responses'] = responses  # Store responses in session

        save_assessment(user_id, phq9_responses, severity, "N/A")

        return jsonify({'redirect': url_for('emotion_analysis_page', severity=severity)}) # Pass severity

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)