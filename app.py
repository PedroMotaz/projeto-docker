from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("modelo_svm.pkl")

# Mapeamentos para conversão de texto para número
sex_map = {"M": 1, "F": 0}
cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
angina_map = {"N": 0, "Y": 1}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recebe os dados do formulário
        data = request.form.to_dict()

        # Converte para DataFrame
        input_data = pd.DataFrame([{
            "Age": int(data["Age"]),
            "Sex": sex_map[data["Sex"]],
            "ChestPainType": cp_map[data["ChestPainType"]],
            "RestingBP": int(data["RestingBP"]),
            "Cholesterol": int(data["Cholesterol"]),
            "FastingBS": int(data["FastingBS"]),
            "RestingECG": ecg_map[data["RestingECG"]],
            "MaxHR": int(data["MaxHR"]),
            "ExerciseAngina": angina_map[data["ExerciseAngina"]],
            "Oldpeak": float(data["Oldpeak"]),
            "ST_Slope": slope_map[data["ST_Slope"]],
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        resultado = "Risco de doença cardíaca detectado!" if prediction == 1 else "Sem risco de doença cardíaca."
        return f"<h2>{resultado}</h2><p>Probabilidade: {proba:.2%}</p>"

    except Exception as e:
        return f"Erro na previsão: {e}"
