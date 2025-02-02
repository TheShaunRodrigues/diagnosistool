from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
from testcasefinal import predict_depression_severity, analyze_emotion, save_assessment

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('models/random_forest_depression_model.pkl')

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/question')
def question_page():
    """Render the PHQ-9 assessment page."""
    return render_template('question.html')

@app.route('/emotion-analysis', methods=['GET', 'POST'])
def emotion_analysis_page():
    """Render the emotion analysis page or redirect to results if skipped."""
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        severity = request.form.get('severity', 'N/A')

        if text:
            predicted_emotion = analyze_emotion(text)
        else:
            predicted_emotion = "N/A"

        return redirect(url_for('results_page', severity=severity, emotion=predicted_emotion))
    
    # If accessed directly, show the form
    severity = request.args.get('severity', 'N/A')
    return render_template('emotion.html', severity=severity)

@app.route('/results')
def results_page():
    """Render the results page with severity and optional emotion analysis."""
    severity = request.args.get('severity', 'N/A')
    emotion = request.args.get('emotion', 'N/A')
    return render_template('results.html', severity=severity, emotion=emotion)

@app.route('/process_assessment', methods=['POST'])
def process_assessment():
    """
    Process PHQ-9 responses, predict severity, and redirect to emotion input or results.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        phq9_responses = data.get('phq9_responses')

        if not user_id or not phq9_responses:
            return jsonify({'error': 'Missing user_id or PHQ-9 responses'}), 400

        severity = predict_depression_severity(phq9_responses)
        save_assessment(user_id, phq9_responses, severity, "N/A")

        return jsonify({'redirect': url_for('emotion_analysis_page', severity=severity)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
