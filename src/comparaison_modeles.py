# partie 5

#      comparaison_modeles.py



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Vérification des modèles et données
assert 'logreg' in globals(), "Le modèle de régression logistique n'est pas défini."
assert 'rf' in globals(), "Le modèle RandomForest n'est pas défini."
assert 'xgb_model' in globals(), "Le modèle XGBoost n'est pas défini."
assert 'X_train' in globals() and 'y_train' in globals(), "Les données d'entraînement ne sont pas définies."
assert len(X_train) > 0, "X_train est vide !"
assert len(y_train) > 0, "y_train est vide !"

# Dictionnaire des modèles
models = {
    'Logistic Regression': logreg,
    'RandomForest': rf,
    'XGBoost': xgb_model
}

# Configuration de la validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Scores de validation croisée (méthode manuelle) ---")

for name, model in models.items():
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_t, y_t)  # Entraînement
        y_pred = model.predict(X_v)  # Prédiction

        scores.append(accuracy_score(y_v, y_pred))  # Calcul de l'accuracy
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")
