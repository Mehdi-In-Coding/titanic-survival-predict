import pandas as pd
import numpy as np

def preprocess_data(train_path, test_path):
    """
    Prétraitement des données : gestion des valeurs manquantes, encodage, création de nouvelles features.
    
    Arguments:
    train_path -- Chemin vers le fichier CSV des données d'entraînement
    test_path -- Chemin vers le fichier CSV des données de test
    
    Retourne:
    train_df, test_df -- DataFrames prétraités
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remplissage des valeurs manquantes
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # Encodage des variables catégoriques
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Création de nouvelles features
    for df in [train_df, test_df]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Suppression des colonnes inutiles
    train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    return train_df, test_df
