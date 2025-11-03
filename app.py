from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Load modelfrom flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# -------------------------------
# Load model and define symptom list
# -------------------------------
model = joblib.load("random_forest_model.pkl")

cleaned_symptoms_list = [
    'itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 
    'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 
    'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 
    'spotting  urination', 'fatigue', 'weight gain', 'anxiety', 
    'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 
    'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 
    'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 
    'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 
    'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 
    'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 
    'yellowing of eyes', 'acute liver failure', 'fluid overload', 
    'swelling of stomach', 'swelled lymph nodes', 'malaise', 
    'blurred and distorted vision', 'phlegm', 'throat irritation', 
    'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 
    'chest pain', 'weakness in limbs', 'fast heart rate', 
    'pain during bowel movements', 'pain in anal region', 'bloody stool', 
    'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 
    'obesity', 'swollen legs', 'swollen blood vessels', 
    'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 
    'swollen extremeties', 'excessive hunger', 'extra marital contacts', 
    'drying and tingling lips', 'slurred speech', 'knee pain', 
    'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 
    'movement stiffness', 'spinning movements', 'loss of balance', 
    'unsteadiness', 'weakness of one body side', 'loss of smell', 
    'bladder discomfort', 'foul smell of urine', 
    'continuous feel of urine', 'passage of gases', 'internal itching', 
    'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 
    'altered sensorium', 'red spots over body', 'belly pain', 
    'abnormal menstruation', 'dischromic  patches', 'watering from eyes', 
    'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 
    'rusty sputum', 'lack of concentration', 'visual disturbances', 
    'receiving blood transfusion', 'receiving unsterile injections', 'coma', 
    'stomach bleeding', 'distention of abdomen', 
    'history of alcohol consumption', 'fluid overload', 'blood in sputum', 
    'prominent veins on calf', 'palpitations', 'painful walking', 
    'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 
    'silver like dusting', 'small dents in nails', 'inflammatory nails', 
    'blister', 'red sore around nose', 'yellow crust ooze'
]

@app.route('/')
def home():
    return "🩺 HealthSnap Symptom Prediction API is running!"

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'symptoms' not in data:
            return jsonify({'error': 'Missing "symptoms" key in JSON input'}), 400

        symptoms = data['symptoms']

        # Create binary input vector (132 symptoms)
        input_vector = [0] * len(cleaned_symptoms_list)
        for symptom in symptoms:
            s = symptom.strip().lower()
            if s in cleaned_symptoms_list:
                idx = cleaned_symptoms_list.index(s)
                input_vector[idx] = 1

        features = np.array(input_vector).reshape(1, -1)

        # Predict disease probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = model.classes_

            # Sort top 3 by probability
            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = [
                {"disease": classes[i], "confidence": float(round(probs[i] * 100, 2))}
                for i in top_indices
            ]

            return jsonify({
                "input_symptoms": symptoms,
                "top_predictions": top_predictions
            })
        else:
            # If model doesn't support probabilities
            prediction = model.predict(features)[0]
            return jsonify({
                "input_symptoms": symptoms,
                "prediction": str(prediction)
            })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 400


if __name__ == '__main__':
    app.run(debug=True)

model = joblib.load("random_forest_model.pkl")

# Master symptom list (length = 132)
cleaned_symptoms_list = [
    'itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 
    'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 
    'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 
    'spotting  urination', 'fatigue', 'weight gain', 'anxiety', 
    'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 
    'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 
    'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 
    'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 
    'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 
    'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 
    'yellowing of eyes', 'acute liver failure', 'fluid overload', 
    'swelling of stomach', 'swelled lymph nodes', 'malaise', 
    'blurred and distorted vision', 'phlegm', 'throat irritation', 
    'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 
    'chest pain', 'weakness in limbs', 'fast heart rate', 
    'pain during bowel movements', 'pain in anal region', 'bloody stool', 
    'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 
    'obesity', 'swollen legs', 'swollen blood vessels', 
    'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 
    'swollen extremeties', 'excessive hunger', 'extra marital contacts', 
    'drying and tingling lips', 'slurred speech', 'knee pain', 
    'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 
    'movement stiffness', 'spinning movements', 'loss of balance', 
    'unsteadiness', 'weakness of one body side', 'loss of smell', 
    'bladder discomfort', 'foul smell of urine', 
    'continuous feel of urine', 'passage of gases', 'internal itching', 
    'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 
    'altered sensorium', 'red spots over body', 'belly pain', 
    'abnormal menstruation', 'dischromic  patches', 'watering from eyes', 
    'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 
    'rusty sputum', 'lack of concentration', 'visual disturbances', 
    'receiving blood transfusion', 'receiving unsterile injections', 'coma', 
    'stomach bleeding', 'distention of abdomen', 
    'history of alcohol consumption', 'fluid overload', 'blood in sputum', 
    'prominent veins on calf', 'palpitations', 'painful walking', 
    'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 
    'silver like dusting', 'small dents in nails', 'inflammatory nails', 
    'blister', 'red sore around nose', 'yellow crust ooze'
]

@app.route('/')
def home():
    return "HealthSnap Symptom Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input format example: {"symptoms": ["headache", "fatigue", "nausea"]}
        data = request.get_json()

        if 'symptoms' not in data:
            return jsonify({'error': 'Missing "symptoms" key in JSON input'}), 400

        symptoms = data['symptoms']

        # Initialize feature vector
        input_vector = [0] * len(cleaned_symptoms_list)

        # Set index to 1 for each reported symptom
        for symptom in symptoms:
            symptom = symptom.strip().lower()
            if symptom in cleaned_symptoms_list:
                index = cleaned_symptoms_list.index(symptom)
                input_vector[index] = 1

        # Convert to numpy array
        features = np.array(input_vector).reshape(1, -1)

        # Predict disease
        prediction = model.predict(features)[0]

        return jsonify({
            'input_symptoms': symptoms,
            'prediction': str(prediction)
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
