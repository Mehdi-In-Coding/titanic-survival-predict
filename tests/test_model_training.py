import os
from src.model_training import train_models
import pandas as pd

def test_train_models():
    train_df = pd.read_csv("data/train.csv")
    train_models(train_df)
    assert os.path.exists("models/logreg.pkl"), "Modèle logreg non sauvegardé"
    assert os.path.exists("models/random_forest.pkl"), "Modèle random forest non sauvegardé"
