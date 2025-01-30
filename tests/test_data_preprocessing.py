import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data


def test_preprocess_data():
    train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")
    
    assert isinstance(train_df, pd.DataFrame), "Le train_df doit être un DataFrame"
    assert isinstance(test_df, pd.DataFrame), "Le test_df doit être un DataFrame"
    assert train_df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes dans train_df"
    assert "Sex" in train_df.columns, "Colonne Sex absente après encodage"
