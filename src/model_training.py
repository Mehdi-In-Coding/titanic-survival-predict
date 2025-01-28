import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_models(train_df):
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement des modèles
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Sauvegarde des modèles
    joblib.dump(logreg, "models/logreg.pkl")
    joblib.dump(rf, "models/random_forest.pkl")

    print("Modèles sauvegardés !")
