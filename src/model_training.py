import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_models(train_df, model_dir="models"):
    """
    Entraîne les modèles de Machine Learning sur les données du Titanic et les sauvegarde.

    Args:
        train_df (pd.DataFrame): Données d'entraînement.
        model_dir (str): Répertoire où sauvegarder les modèles.
    
    Returns:
        dict: Dictionnaire contenant les modèles entraînés.
    """
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError("train_df doit être un DataFrame valide.")

    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création du dossier si inexistant
    os.makedirs(model_dir, exist_ok=True)

    # Entraînement des modèles
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Sauvegarde des modèles
    joblib.dump(logreg, os.path.join(model_dir, "logreg.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

    print("Modèles entraînés et sauvegardés avec succès !")

    return {"logreg": logreg, "rf": rf}


if __name__ == "__main__":
    train, _ = preprocess_data("../data/train.csv", "../data/test.csv")
    train_models(train)
