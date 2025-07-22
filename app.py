from flask import Flask, request, jsonify
import joblib
import numpy as np 

app = Flask(__name__)
model = joblib.load("modelo_svm.pkl")

@app.route("/")
def home():
    return "API de previsão de doença cardiaca (SVM)"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get.json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()
    return jsonify ({"prediction": int(prediction), "probability": probability})
