# partie 10 

#       submission.py


import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from google.colab import files
from IPython.core.display import display, HTML

# Affichage d'un titre stylisé
display(HTML("""
<h1 style="color:#2c3e50; font-size: 32px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
Partie 10: Création du fichier de soumission
</h1>
"""))

# ================================
# ⚡ CHARGEMENT DES DONNÉES
# ================================

# Charger les fichiers de test
test_df = pd.read_csv('/content/drive/My Drive/Titanic-Survival-Predict-main/test_cleaned.csv')

# Vérifier que la colonne PassengerId existe
assert 'PassengerId' in test_df.columns, "🚨 ERREUR : La colonne 'PassengerId' est absente du dataset test !"

# Extraire les caractéristiques
X_test = test_df.drop(columns=['PassengerId'])

# Vérifier que X_test n'est pas vide
assert not X_test.empty, "🚨 ERREUR : Le dataset X_test est vide après suppression de 'PassengerId' !"

# ================================
# ⚡ CHARGEMENT DES MODÈLES
# ================================

# Définir les chemins des modèles
logreg_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/logistic_regression_model.pkl'
rf_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/random_forest_model.pkl'
xgb_model_path = '/content/drive/MyDrive/Titanic-Survival-Predict-main/Models/xgboost_model.pkl'

# Charger les modèles
final_logreg = joblib.load(logreg_model_path)
final_rf = joblib.load(rf_model_path)
final_xgb = joblib.load(xgb_model_path)

# Vérifier que les modèles sont bien chargés
assert isinstance(final_logreg, LogisticRegression), "🚨 ERREUR : Le modèle LogisticRegression n'est pas chargé correctement !"
assert hasattr(final_rf, "predict"), "🚨 ERREUR : Le modèle RandomForest n'est pas chargé correctement !"
assert hasattr(final_xgb, "predict"), "🚨 ERREUR : Le modèle XGBoost n'est pas chargé correctement !"

# ================================
# ⚡ PRÉDICTION DES MODÈLES DE BASE
# ================================

print("\n--- Génération des prédictions des modèles de base sur le jeu de test ---")

# Faire les prédictions avec chaque modèle
test_pred_logreg = final_logreg.predict(X_test)
test_pred_rf = final_rf.predict(X_test)
test_pred_xgb = final_xgb.predict(X_test)

# Vérifier que les prédictions ont la bonne taille
assert len(test_pred_logreg) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions logreg !"
assert len(test_pred_rf) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions rf !"
assert len(test_pred_xgb) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions xgb !"

# Créer un DataFrame avec ces prédictions
test_meta_features = pd.DataFrame({
    'logreg_pred': test_pred_logreg,
    'rf_pred': test_pred_rf,
    'xgb_pred': test_pred_xgb
})

print("✅ Prédictions des modèles de base sur le jeu de test générées avec succès.")

# ================================
# ⚡ ENTRAÎNEMENT DU MODÈLE MÉTA
# ================================

print("\n--- Entraînement du modèle Meta (Régression Logistique) ---")

# Générer les prédictions sur l'entraînement pour créer un modèle méta
train_pred_logreg = final_logreg.predict(X_test)
train_pred_rf = final_rf.predict(X_test)
train_pred_xgb = final_xgb.predict(X_test)

# Vérifier que les prédictions sont valides
assert len(train_pred_logreg) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions logreg sur le train !"
assert len(train_pred_rf) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions rf sur le train !"
assert len(train_pred_xgb) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions xgb sur le train !"

# Créer le dataset pour le modèle méta
train_meta_features = pd.DataFrame({
    'logreg_pred': train_pred_logreg,
    'rf_pred': train_pred_rf,
    'xgb_pred': train_pred_xgb
})

# Entraîner le modèle méta (Régression Logistique)
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(train_meta_features, test_pred_logreg)

print("✅ Modèle Meta entraîné avec succès.")

# ================================
# ⚡ PRÉDICTION FINALE
# ================================

print("\n--- Prédiction avec le modèle Meta ---")

# Faire les prédictions finales
y_test_pred = meta_model.predict(test_meta_features)

# Vérifier la taille des prédictions finales
assert len(y_test_pred) == len(X_test), "🚨 ERREUR : Problème de taille des prédictions finales du modèle méta !"

# ================================
# ⚡ CRÉATION DU FICHIER DE SOUMISSION
# ================================

print("\n--- Création du fichier de soumission ---")

# Créer le fichier de soumission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred
})

# Sauvegarde en CSV
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

# Vérifier que le fichier est bien sauvegardé
assert os.path.exists(submission_file), f"🚨 ERREUR : Le fichier {submission_file} n'a pas été créé !"

print(f"✅ Fichier de soumission '{submission_file}' créé avec succès !")

# ================================
# ⚡ TÉLÉCHARGEMENT DU FICHIER
# ================================
files.download(submission_file)
print("✅ Téléchargement du fichier terminé.")
