import joblib
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def evaluate_model(model_path, X_val, y_val):
    """
    Charge et évalue un modèle sauvegardé sur un ensemble de validation.

    Args:
        model_path (str): Chemin du modèle sauvegardé.
        X_val (pd.DataFrame): Caractéristiques de validation.
        y_val (pd.Series): Étiquettes de validation.
    
    Returns:
        dict: Résultats de l'évaluation.
    """
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)

        print(f"Modèle: {model_path}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)

        return {"accuracy": acc, "report": report}

    except FileNotFoundError:
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return None
