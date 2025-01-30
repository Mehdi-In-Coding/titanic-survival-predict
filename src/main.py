from data_preprocessing import preprocess_data
from model_training import train_models
import pandas as pd

# Charger les données
train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")

# Entraîner les modèles
train_models(train_df)
