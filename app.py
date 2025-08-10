# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = os.path.join('models','model.joblib')
model = joblib.load(MODEL_PATH)

# Adjust feature order to match training
FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = {f: float(request.form.get(f, 0)) for f in FEATURES}
        X = pd.DataFrame([data], columns=FEATURES)
        proba = model.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)
        return render_template('index.html', probability=f"{proba:.3f}", prediction=pred, inputs=data)
    return render_template('index.html')
def predict():
    data = {f: float(request.form.get(f, 0)) for f in FEATURES}
    X = pd.DataFrame([data], columns=FEATURES)
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    return render_template('index.html', probability=f"{proba:.3f}", prediction=pred, inputs=data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    content = request.json
    # Expect either a dict mapping features, or a list of lists
    if isinstance(content, dict):
        X = pd.DataFrame([content], columns=FEATURES)
    else:
        return jsonify({"error":"invalid payload"}), 400
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    return jsonify({'prediction': pred, 'probability': float(proba)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)