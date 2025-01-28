import pandas as pd
import numpy as np

def preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Gestion des valeurs manquantes
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

    # Encodage des variables catégoriques
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    return train_df, test_df
