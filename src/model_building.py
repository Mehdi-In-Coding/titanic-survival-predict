#   partie 4

#         model_building.py 


import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Charger les données prétraitées
train_df = pd.read_csv('/content/drive/My Drive/Titanic-Survival-Predict-main/train_cleaned.csv')

# Vérification des données
assert 'Survived' in train_df.columns, "La colonne 'Survived' est absente"

# Séparation des caractéristiques (X) et de la cible (y)
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

# Vérification que les données sont correctes
assert not X.isnull().sum().any(), "NA sont pas un problemes pour les graphiques (on les hide)"
assert set(y.unique()).issubset({0, 1}), "y doit contenir uniquement des valeurs binaires (0 ou 1)"

# Division des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

assert X_train.shape[0] > 0 and X_val.shape[0] > 0, "Les ensembles d'entraînement et de validation ne doivent pas être vides"

# Fonction d'évaluation des modèles
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"Précision : {accuracy:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(y_val, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))
    
    return model

# Création des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP Classifier": MLPClassifier(alpha=0.06, hidden_layer_sizes=(50, 50), learning_rate_init=0.03, max_iter=158),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, enable_categorical=True, eval_metric='logloss', random_state=42)
}

# Entraînement et évaluation
trained_models = {}
for name, model in models.items():
    trained_models[name] = evaluate_model(model, X_train, y_train, X_val, y_val, name)

# Sauvegarde des modèles
models_path = "/content/drive/MyDrive/Titanic-Survival-Predict-main/models"
os.makedirs(models_path, exist_ok=True)

for name, model in trained_models.items():
    model_filename = os.path.join(models_path, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_filename)
    print(f"Modèle {name} sauvegardé dans {model_filename}")

print("\n✅ Tous les modèles ont été entraînés et sauvegardés avec succès !")
