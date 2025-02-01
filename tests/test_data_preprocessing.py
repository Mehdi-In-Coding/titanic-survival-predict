import sys
sys.path.append('/content/titanic-survival-predict/src')

import pandas as pd
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")
    assert train_df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes"
    assert "Sex" in train_df.columns, "Colonne Sex absente"


