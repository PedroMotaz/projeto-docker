import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


model = joblib.load("modelo_svm.pkl")


feature_columns = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M', 'Sex_F',
    'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_N', 'ExerciseAngina_Y',
    'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

      
        df = pd.DataFrame([data])

       
        df_encoded = pd.get_dummies(df)

        
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

      
        df_encoded = df_encoded[feature_columns]

      
        prediction = model.predict(df_encoded)[0]
        proba = model.predict_proba(df_encoded)[0][1]

        return jsonify({
            "previsao": int(prediction),
            "probabilidade": round(float(proba), 2)
        })
    except Exception as e:
        return jsonify({"erro": f"Erro na previsão: {str(e)}"})
