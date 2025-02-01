# partie 7


#         load_models.py

import joblib
import os

# Définir les chemins des modèles
logreg_model_path = "/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/logistic_regression_model.pkl"
rf_model_path = "/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/random_forest_model.pkl"
xgb_model_path = "/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/xgboost_model.pkl"

# Vérifier l'existence des fichiers avant chargement
assert os.path.exists(logreg_model_path), f"Erreur: Le fichier {logreg_model_path} n'existe pas !"
assert os.path.exists(rf_model_path), f"Erreur: Le fichier {rf_model_path} n'existe pas !"
assert os.path.exists(xgb_model_path), f"Erreur: Le fichier {xgb_model_path} n'existe pas !"

# Charger les modèles
final_logreg = joblib.load(logreg_model_path)
final_rf = joblib.load(rf_model_path)
final_xgb = joblib.load(xgb_model_path)

# Vérifier que les objets chargés sont bien des modèles
assert hasattr(final_logreg, "predict"), "Erreur: final_logreg n'est pas un modèle valide !"
assert hasattr(final_rf, "predict"), "Erreur: final_rf n'est pas un modèle valide !"
assert hasattr(final_xgb, "predict"), "Erreur: final_xgb n'est pas un modèle valide !"

print(f"✔ Modèle de régression logistique chargé depuis {logreg_model_path}")
print(f"✔ Modèle RandomForest chargé depuis {rf_model_path}")
print(f"✔ Modèle XGBoost chargé depuis {xgb_model_path}")
