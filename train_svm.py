import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib


df = pd.read_csv("heart.csv")


df_encoded = pd.get_dummies(df)


X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(probability=True)
model.fit(X_train, y_train)


joblib.dump(model, "modelo_svm.pkl")


joblib.dump(X.columns.tolist(), "colunas.pkl")
