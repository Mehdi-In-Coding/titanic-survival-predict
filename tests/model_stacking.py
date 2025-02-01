# partie 8


# model_stacking.py



import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement des modèles entraînés
logreg_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/logistic_regression_model.pkl'
rf_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/random_forest_model.pkl'
xgb_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/xgboost_model.pkl'

final_logreg = joblib.load(logreg_model_path)
final_rf = joblib.load(rf_model_path)
final_xgb = joblib.load(xgb_model_path)

# Assertions pour vérifier que les modèles sont bien chargés
assert final_logreg is not None, "Le modèle Logistic Regression n'a pas été chargé."
assert final_rf is not None, "Le modèle Random Forest n'a pas été chargé."
assert final_xgb is not None, "Le modèle XGBoost n'a pas été chargé."

# Chargement des données d'entraînement et de validation
X_train = pd.read_csv('/content/drive/MyDrive/Titanic-Survival-Predict-main/X_train.csv')
X_val = pd.read_csv('/content/drive/MyDrive/Titanic-Survival-Predict-main/X_val.csv')
y_train = pd.read_csv('/content/drive/MyDrive/Titanic-Survival-Predict-main/y_train.csv').values.ravel()
y_val = pd.read_csv('/content/drive/MyDrive/Titanic-Survival-Predict-main/y_val.csv').values.ravel()

# Assertions pour vérifier que les données sont bien chargées
assert X_train.shape[0] > 0, "X_train est vide."
assert X_val.shape[0] > 0, "X_val est vide."
assert y_train.shape[0] > 0, "y_train est vide."
assert y_val.shape[0] > 0, "y_val est vide."

# Étape 1 : Générer des prédictions des modèles de base sur l'ensemble d'entraînement
print("\n--- Génération des prédictions des modèles de base sur l'ensemble d'entraînement ---")
train_pred_logreg = final_logreg.predict(X_train)
train_pred_rf = final_rf.predict(X_train)
train_pred_xgb = final_xgb.predict(X_train)

# Créer un DataFrame avec ces prédictions
train_meta_features = pd.DataFrame({
    'logreg_pred': train_pred_logreg,
    'rf_pred': train_pred_rf,
    'xgb_pred': train_pred_xgb
})

print("Prédictions des modèles de base générées avec succès.")

# Étape 2 : Entraîner le Modèle Meta sur ces Prédictions
print("\n--- Entraînement du modèle Meta (Régression Logistique) ---")
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(train_meta_features, y_train)
print("Modèle Meta entraîné avec succès.")

# Étape 3 : Générer des prédictions des modèles de base sur l'ensemble de validation
print("\n--- Génération des prédictions des modèles de base sur l'ensemble de validation ---")
val_pred_logreg = final_logreg.predict(X_val)
val_pred_rf = final_rf.predict(X_val)
val_pred_xgb = final_xgb.predict(X_val)

# Créer un DataFrame avec ces prédictions
val_meta_features = pd.DataFrame({
    'logreg_pred': val_pred_logreg,
    'rf_pred': val_pred_rf,
    'xgb_pred': val_pred_xgb
})

print("Prédictions des modèles de base sur l'ensemble de validation générées avec succès.")

# Étape 4 : Utiliser le Modèle Meta pour Prédire sur l'Ensemble de Validation
print("\n--- Prédiction avec le modèle Meta ---")
y_pred_stack = meta_model.predict(val_meta_features)

# Étape 5 : Évaluer les Performances
print("\n--- Évaluation des Performances du Modèle Empilé ---")
stack_accuracy = accuracy_score(y_val, y_pred_stack)
print(f"Accuracy du modèle empilé : {stack_accuracy:.4f}")

print("Matrice de confusion du modèle empilé :")
print(confusion_matrix(y_val, y_pred_stack))

print("\nRapport de classification du modèle empilé :")
print(classification_report(y_val, y_pred_stack))

# Optionnel : Sauvegarder le Modèle Meta
stack_meta_model_path = "/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/stacking_meta_model.pkl"
joblib.dump(meta_model, stack_meta_model_path)
print(f"\nLe modèle empilé (meta) est bien sauvegardé dans {stack_meta_model_path}")
