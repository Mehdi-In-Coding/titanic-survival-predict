import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    """Test que le preprocessing nettoie bien les données et encode les variables catégoriques"""
    train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")
    
    # Vérifier qu'il n'y a plus de valeurs manquantes
    assert train_df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes dans train_df"
    assert test_df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes dans test_df"
    
    # Vérifier que les colonnes attendues sont bien présentes
    expected_columns = ["Sex", "Age", "Fare", "Pclass", "Embarked", "FamilySize", "Title", "Deck"]
    for col in expected_columns:
        assert col in train_df.columns, f"Colonne {col} absente dans train_df"
        assert col in test_df.columns, f"Colonne {col} absente dans test_df"
    
    # Vérifier que les colonnes encodées sont bien transformées
    assert train_df["Sex"].isin([0, 1]).all(), "Sex n'est pas encodé correctement"
    assert train_df["Embarked"].isin([0, 1, 2]).all(), "Embarked n'est pas encodé correctement"
    
    # Vérifier que AgeGroup et FareBin sont bien catégorisés
    assert train_df["AgeGroup"].dtype == "category", "AgeGroup n'est pas une catégorie"
    assert train_df["FareBin"].dtype == "category", "FareBin n'est pas une catégorie"
    
    # Vérifier si les valeurs numériques sont bien transformées
    assert train_df["FamilySize"].min() >= 1, "FamilySize a des valeurs invalides"

def test_preprocess_data_with_missing_values():
    """Test si le preprocessing gère correctement les valeurs manquantes"""
    sample_data = {
        "Pclass": [1, 3, 2],
        "Sex": ["male", "female", "male"],
        "Age": [np.nan, 22, np.nan],
        "Fare": [np.nan, 7.25, 10.5],
        "Embarked": ["S", np.nan, "Q"],
        "SibSp": [1, 0, 2],
        "Parch": [0, 1, 1],
        "Cabin": ["C85", "", "B20"],
        "Name": ["John Doe", "Jane Smith", "Robert Brown"],
        "Ticket": ["12345", "54321", "67890"]
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Sauvegarder en CSV temporaire
    sample_df.to_csv("data/sample_test.csv", index=False)
    
    # Appliquer la transformation
    processed_df, _ = preprocess_data("data/sample_test.csv", "data/sample_test.csv")
    
    # Vérifier que les valeurs NaN ont bien été remplacées
    assert processed_df["Age"].isnull().sum() == 0, "Age n'a pas été rempli"
    assert processed_df["Fare"].isnull().sum() == 0, "Fare n'a pas été rempli"
    assert processed_df["Embarked"].isnull().sum() == 0, "Embarked n'a pas été rempli"
