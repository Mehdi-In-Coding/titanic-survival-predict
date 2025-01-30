from data_preprocessing import preprocess_data
from model_training import train_models
import os

if __name__ == "__main__":
    print("Début du pipeline Titanic...")

    # Charger et prétraiter les données
    train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")

    if train_df is not None:
        # Entraîner les modèles
        models = train_models(train_df)

        print("Pipeline terminé avec succès !")
    else:
        print("Erreur dans le chargement des données.")
