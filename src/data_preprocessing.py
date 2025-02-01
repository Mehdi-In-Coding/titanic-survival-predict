# partie 3 
#             data_preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data(train_path, test_path):
    """
    Charge les données, applique le prétraitement et retourne les DataFrames transformés.
    """

    # Charger les fichiers CSV
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    ### Vérification initiale ###
    assert train_df.shape[0] > 0, "Le fichier train est vide"
    assert test_df.shape[0] > 0, "Le fichier test est vide"
    
    print("✔️ Chargement des données réussi")

    # Étape 1 : Gestion des valeurs manquantes
    train_df['Age'] = train_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    test_df['Age'] = test_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

    assert train_df['Age'].isnull().sum() == 0, "Il reste des valeurs manquantes dans Age (train)"
    assert test_df['Age'].isnull().sum() == 0, "Il reste des valeurs manquantes dans Age (test)"
    
    print("✔️ Remplissage des valeurs manquantes d'Age réussi")

    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    test_df['Fare'] = test_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

    assert train_df['Embarked'].isnull().sum() == 0, "Il reste des valeurs manquantes dans Embarked (train)"
    assert test_df['Fare'].isnull().sum() == 0, "Il reste des valeurs manquantes dans Fare (test)"
    
    print("✔️ Remplissage des valeurs manquantes d'Embarked et Fare réussi")

    # Encodage des variables catégoriques
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    assert 'Sex' in train_df.columns and train_df['Sex'].dtype == np.int64, "Problème d'encodage de Sex"
    assert 'Embarked' in train_df.columns and train_df['Embarked'].dtype == np.int64, "Problème d'encodage de Embarked"

    print("✔️ Encodage des variables catégoriques réussi")

    # Création de nouvelles variables
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
    test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

    assert 'FamilySize' in train_df.columns, "Colonne FamilySize absente"
    assert 'IsAlone' in train_df.columns, "Colonne IsAlone absente"

    print("✔️ Création des nouvelles caractéristiques réussie")

    # Supprimer les colonnes inutiles
    train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    print("✔️ Suppression des colonnes inutiles réussie")

    return train_df, test_df


if __name__ == "__main__":
    # Exécution du script en standalone avec assertions
    train_data_path = "/content/drive/My Drive/Titanic-Survival-Predict-main/train_cleaned.csv"
    test_data_path = "/content/drive/My Drive/Titanic-Survival-Predict-main/test_cleaned.csv"

    train_df, test_df = preprocess_data(train_data_path, test_data_path)

    print("✅ Prétraitement des données terminé avec succès !")
