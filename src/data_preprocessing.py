import pandas as pd
import numpy as np


def preprocess_data(train_path, test_path):
    """
    Charge et prétraite les données Titanic en nettoyant les valeurs manquantes et encodant les variables catégoriques.
    
    Args:
        train_path (str): Chemin du fichier CSV des données d'entraînement.
        test_path (str): Chemin du fichier CSV des données de test.

    Returns:
        pd.DataFrame, pd.DataFrame: Données d'entraînement et de test prétraitées.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Gestion des valeurs manquantes
        train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
        test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

        # Encodage des variables catégoriques
        train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
        test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

        return train_df, test_df

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
        return None, None
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return None, None


if __name__ == "__main__":
    train, test = preprocess_data("../data/train.csv", "../data/test.csv")
    print(train.head())
