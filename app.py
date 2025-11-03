from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("random_forest_model.pkl")

# -------------------------------
# Symptom list (132)
# -------------------------------
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

# -------------------------------
# Disease → Treatment mapping
# -------------------------------
treatments = {
    "Fungal infection": "Apply antifungal cream, keep area dry, wear loose clothing.",
    "Allergy": "Avoid allergen, take antihistamines, consult doctor if severe.",
    "GERD": "Avoid spicy food, eat small meals, take antacids or PPIs.",
    "Chronic cholestasis": "Eat low-fat meals, avoid alcohol, consult gastroenterologist.",
    "Drug Reaction": "Stop suspected drug, consult doctor, take antihistamines if mild.",
    "Peptic ulcer diseae": "Avoid NSAIDs, take proton pump inhibitors, eat bland food.",
    "AIDS": "Antiretroviral therapy (ART), maintain healthy diet, avoid infections.",
    "Diabetes": "Monitor blood sugar, regular exercise, diabetic diet, insulin if prescribed.",
    "Gastroenteritis": "Stay hydrated, eat light food, take oral rehydration solution (ORS).",
    "Bronchial Asthma": "Use inhaler, avoid allergens, take bronchodilators as prescribed.",
    "Hypertension": "Low salt diet, regular exercise, take antihypertensive medicines.",
    "Migraine": "Rest in dark room, stay hydrated, take prescribed pain relievers.",
    "Cervical spondylosis": "Physical therapy, neck exercises, pain management.",
    "Paralysis (brain hemorrhage)": "Physiotherapy, medications, and medical supervision.",
    "Jaundice": "Adequate rest, hydration, avoid alcohol and fatty food.",
    "Malaria": "Antimalarial drugs, hydration, rest, avoid mosquito bites.",
    "Chicken pox": "Rest, calamine lotion for itching, avoid scratching.",
    "Dengue": "Hydration, rest, monitor platelet count, avoid NSAIDs.",
    "Typhoid": "Antibiotics, hydration, eat soft food, maintain hygiene.",
    "hepatitis A": "Rest, hydration, avoid alcohol, eat healthy food.",
    "Hepatitis B": "Antiviral medications, avoid alcohol, monitor liver function.",
    "Hepatitis C": "Antiviral therapy, rest, avoid alcohol and drugs toxic to liver.",
    "Hepatitis D": "Avoid alcohol, rest, consult hepatologist for management.",
    "Hepatitis E": "Rest, proper hydration, avoid alcohol and fatty foods.",
    "Alcoholic hepatitis": "Stop alcohol completely, nutritious diet, doctor supervision.",
    "Tuberculosis": "6-month antibiotic therapy (DOTS), maintain good nutrition.",
    "Common Cold": "Rest, fluids, steam inhalation, OTC cold medicine if needed.",
    "Pneumonia": "Antibiotics if bacterial, rest, fluids, consult doctor.",
    "Dimorphic hemmorhoids(piles)": "Fiber-rich diet, warm baths, topical creams.",
    "Heart attack": "Immediate medical attention, aspirin, lifestyle modification.",
    "Varicose veins": "Avoid standing long, compression stockings, exercise.",
    "Hypothyroidism": "Thyroxine medication, regular thyroid check-up.",
    "Hyperthyroidism": "Antithyroid drugs, beta blockers, avoid caffeine.",
    "Hypoglycemia": "Eat frequent small meals, carry glucose tablets.",
    "Osteoarthristis": "Weight management, physiotherapy, pain relief medication.",
    "Arthritis": "Anti-inflammatory drugs, physiotherapy, balanced diet.",
    "(vertigo) Paroymsal  Positional Vertigo": "Vestibular exercises, avoid sudden head movements.",
    "Acne": "Use mild cleanser, topical creams, avoid oily foods.",
    "Urinary tract infection": "Antibiotics, hydration, maintain good hygiene.",
    "Psoriasis": "Moisturizers, topical steroids, phototherapy if severe.",
    "Impetigo": "Topical antibiotics, maintain hygiene, avoid scratching."
}

@app.route('/')
def home():
    return "🩺 HealthSnap Symptom Prediction API with Treatments is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'symptoms' not in data:
            return jsonify({'error': 'Missing \"symptoms\" key in JSON input'}), 400

        symptoms = data['symptoms']

        # Binary input vector
        input_vector = [0] * len(cleaned_symptoms_list)
        for s in symptoms:
            s = s.strip().lower()
            if s in cleaned_symptoms_list:
                idx = cleaned_symptoms_list.index(s)
                input_vector[idx] = 1

        features = np.array(input_vector).reshape(1, -1)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = model.classes_

            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = []
            for i in top_indices:
                disease = classes[i]
                confidence = round(probs[i] * 100, 2)
                treatment = treatments.get(disease, "No specific treatment available. Consult a doctor.")
                top_predictions.append({
                    "disease": disease,
                    "confidence": confidence,
                    "treatment": treatment
                })

            return jsonify({
                "input_symptoms": symptoms,
                "top_predictions": top_predictions
            })
        else:
            # Fallback
            pred = model.predict(features)[0]
            return jsonify({
                "input_symptoms": symptoms,
                "prediction": pred,
                "treatment": treatments.get(pred, "Consult doctor.")
            })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


if __name__ == '__main__':
    app.run(debug=True)
