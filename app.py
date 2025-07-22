from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

app = Flask(__name__)


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model = SVC(probability=True)
model.fit(X_train, y_train)
joblib.dump(model, 'svm_model.pkl')

svm_model = joblib.load('svm_model.pkl')

@app.route("/")
def home():
    return "API SVM Iris ativa"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = svm_model.predict(features)
        prob = svm_model.predict_proba(features).tolist()
        return jsonify({
            "prediction": int(prediction[0]),
            "probs": prob
        })

    except Exception as e:
        return jsonify ({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)