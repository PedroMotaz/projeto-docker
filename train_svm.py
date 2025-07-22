import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib 

from sklearn.datasets import load_heart_disease
data = load_heart_disease()
X, y = data.data, data.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "modelo_svm.pkl")