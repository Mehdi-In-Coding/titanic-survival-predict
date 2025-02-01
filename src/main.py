# main.py


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Définition des chemins
MODELS_PATH = "/content/drive/MyDrive/Titanic-Survival-Predict-main/models"

# Chargement des données
train_df = pd.read_csv('/content/drive/My Drive/Titanic-Survival-Predict-main/train_cleaned.csv')
test_df = pd.read_csv('/content/drive/My Drive/Titanic-Survival-Predict-main/test_cleaned.csv')

# Vérification des données
assert train_df.shape[0] > 0, "Le dataset d'entraînement est vide"
assert test_df.shape[0] > 0, "Le dataset de test est vide"

# Préparation des données
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement des modèles
print("\n--- Entraînement des modèles de base ---")
logreg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
mlp = MLPClassifier(alpha=0.06, hidden_layer_sizes=(50, 50), learning_rate_init=0.03, max_iter=158).fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, enable_categorical=True, eval_metric='logloss', random_state=42).fit(X_train, y_train)

# Vérification des modèles
assert logreg, "Erreur lors de l'entraînement de la Régression Logistique"
assert rf, "Erreur lors de l'entraînement du RandomForest"
assert mlp, "Erreur lors de l'entraînement du MLP"
assert xgb_model, "Erreur lors de l'entraînement de XGBoost"

# Évaluation des modèles
print("\n--- Évaluation des modèles de base ---")
for model, name in zip([logreg, rf, mlp, xgb_model], ["Logistic Regression", "RandomForest", "MLP", "XGBoost"]):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = classification_report(y_val, y_pred, output_dict=True)["macro avg"]["f1-score"]
    print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# Sauvegarde des modèles
print("\n--- Sauvegarde des modèles ---")
os.makedirs(MODELS_PATH, exist_ok=True)
joblib.dump(logreg, f"{MODELS_PATH}/logreg.pkl")
joblib.dump(rf, f"{MODELS_PATH}/random_forest.pkl")
joblib.dump(mlp, f"{MODELS_PATH}/mlp.pkl")
joblib.dump(xgb_model, f"{MODELS_PATH}/xgboost.pkl")

print(f"Les modèles ont été sauvegardés dans {MODELS_PATH}")

# Chargement des modèles sauvegardés pour validation
print("\n--- Chargement des modèles sauvegardés ---")
logreg_loaded = joblib.load(f"{MODELS_PATH}/logreg.pkl")
rf_loaded = joblib.load(f"{MODELS_PATH}/random_forest.pkl")
mlp_loaded = joblib.load(f"{MODELS_PATH}/mlp.pkl")
xgb_loaded = joblib.load(f"{MODELS_PATH}/xgboost.pkl")

assert logreg_loaded, "Erreur lors du chargement de la Régression Logistique"
assert rf_loaded, "Erreur lors du chargement du RandomForest"
assert mlp_loaded, "Erreur lors du chargement du MLP"
assert xgb_loaded, "Erreur lors du chargement du XGBoost"

print("Les modèles ont été correctement chargés.")

# Passage à la Partie 9 (IA Explicable)
print("\n--- Exécution de la Partie 9: IA Explicable (XAI) ---")
os.system("python partie9_xai.py")
         
