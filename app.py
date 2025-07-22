import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


model = joblib.load("modelo_svm.pkl")
feature_columns = joblib.load("colunas.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("formulario.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.form.to_dict()
        df = pd.DataFrame([data])

       
        df_encoded = pd.get_dummies(df, drop_first=True)

        
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        
        df_encoded = df_encoded[feature_columns]

        # Faz a previsão
        prediction = model.predict(df_encoded)[0]
        proba = model.predict_proba(df_encoded)[0][1]

        return jsonify({
            "previsao": int(prediction),
            "probabilidade": round(float(proba), 2)
        })
    except Exception as e:
        return jsonify({"erro": f"Erro na previsão: {str(e)}"})

