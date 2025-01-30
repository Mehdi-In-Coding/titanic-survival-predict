import os
import pandas as pd
import joblib
from src.model_training import train_models
from sklearn.metrics import accuracy_score

def test_train_models():
    """Test que l'entraînement des modèles fonctionne et sauvegarde correctement les fichiers"""
    train_df = pd.read_csv("data/train.csv")
    train_models(train_df)
    
    # Vérifier que les modèles sont bien sauvegardés
    assert os.path.exists("models/logreg.pkl"), "Modèle logreg non sauvegardé"
    assert os.path.exists("models/random_forest.pkl"), "Modèle RandomForest non sauvegardé"
    
    # Vérifier que les modèles peuvent être rechargés
    logreg = joblib.load("models/logreg.pkl")
    rf = joblib.load("models/random_forest.pkl")
    
    # Vérifier qu'ils peuvent faire des prédictions
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']
    
    logreg_pred = logreg.predict(X)
    rf_pred = rf.predict(X)
    
    # Vérifier que la précision n'est pas nulle
    assert accuracy_score(y, logreg_pred) > 0, "Logistic Regression donne une précision nulle"
    assert accuracy_score(y, rf_pred) > 0, "RandomForest donne une précision nulle"
