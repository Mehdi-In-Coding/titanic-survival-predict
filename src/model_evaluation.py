import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model_path, X_val, y_val):
    """
    Charge un modèle et l'évalue sur des données de validation.

    Arguments:
    model_path -- Chemin du modèle à charger
    X_val -- Features de validation
    y_val -- Labels de validation
    """
    model = joblib.load(model_path)
    y_pred = model.predict(X_val)

    print(f"✅ Évaluation du modèle {model_path.split('/')[-1]} :")
    print(f"Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print("Classification Report :")
    print(classification_report(y_val, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_val, y_pred))
