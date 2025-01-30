import os
import pytest
import pandas as pd
from src.model_training import train_models
from src.data_preprocessing import preprocess_data


def test_train_models():
    train_df, _ = preprocess_data("data/train.csv", "data/test.csv")

    models = train_models(train_df, model_dir="models_test")
    
    assert os.path.exists("models_test/logreg.pkl"), "Modèle logreg non sauvegardé"
    assert os.path.exists("models_test/random_forest.pkl"), "Modèle RandomForest non sauvegardé"
