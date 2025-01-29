from flask import Flask, request, jsonify, render_template
import joblib
from test_case_2 import analyze_emotion 

app = Flask(__name__)

# Load the saved model
model = joblib.load('models/random_forest_depression_model.pkl')

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/question')
def question_page():
    """Render the PHQ-9 assessment page."""
    return render_template('question.html')

@app.route('/predict', methods=['POST'])
def predict_severity():
    """
    Endpoint to predict depression severity based on PHQ-9 responses.
    """
    try:
        data = request.get_json()
        phq9_responses = data.get('phq9_responses')

        if not phq9_responses:
            return jsonify({'error': 'Missing PHQ-9 responses'}), 400

        predicted_severity = model.predict([phq9_responses])[0]

        return jsonify({'predicted_severity': predicted_severity})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
