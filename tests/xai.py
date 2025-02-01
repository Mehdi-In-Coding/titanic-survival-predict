# partie 9 

#       xai.py



import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Assurer que les modèles et les données existent
assert 'logreg' in globals(), "Le modèle de régression logistique (logreg) n'est pas défini."
assert 'rf' in globals(), "Le modèle RandomForest (rf) n'est pas défini."
assert 'mlp' in globals(), "Le modèle MLP (mlp) n'est pas défini."
assert 'X_train' in globals(), "X_train n'est pas défini."
assert 'X_val' in globals(), "X_val n'est pas défini."
assert 'y_val' in globals(), "y_val n'est pas défini."

print("Toutes les assertions sont validées. Exécution du code XAI...")

# ---- Explication avec SHAP pour RandomForest ----
explainer = shap.Explainer(rf)
shap_values = explainer(X_train)

# Visualisation SHAP pour la première observation
plt.title("Graphique en cascade SHAP pour la première observation")
shap.plots.waterfall(shap_values[0, :, 1])
plt.show()

# ---- Analyse des coefficients pour la régression logistique ----
coefficients = logreg.coef_[0]
feature_names = X_train.columns
coef_feature_pairs = sorted(zip(coefficients, feature_names), key=lambda x: abs(x[0]), reverse=True)

# Sélection des 10 principales caractéristiques
sorted_coefficients, sorted_feature_names = zip(*coef_feature_pairs[:10])
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_coefficients, color='skyblue')
plt.xlabel('Valeur des coefficients')
plt.ylabel('Nom des caractéristiques')
plt.title('Top 10 des coefficients de la régression logistique')
plt.gca().invert_yaxis()
plt.show()

# ---- Analyse des importances des caractéristiques pour RandomForest ----
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Sélection des 10 principales caractéristiques
top_features = feature_importance_df.head(10)
plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='lightgreen')
plt.xlabel('Importance des caractéristiques')
plt.ylabel('Nom des caractéristiques')
plt.title('Top 10 des caractéristiques les plus importantes (RandomForest)')
plt.gca().invert_yaxis()
plt.show()

# ---- Importances par permutation pour MLP ----
perm_importance = permutation_importance(mlp, X_val, y_val, n_repeats=10, random_state=42)
feature_importances = perm_importance.importances_mean
sorted_idx = feature_importances.argsort()[::-1]

# Sélection des 10 principales caractéristiques
top_features = [X_train.columns[i] for i in sorted_idx[:10]]
top_importances = feature_importances[sorted_idx[:10]]
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color='cornflowerblue')
plt.xlabel("Importance des caractéristiques")
plt.ylabel("Nom des caractéristiques")
plt.title("Top 10 importances des caractéristiques par permutation")
plt.gca().invert_yaxis()
plt.show()

print("Analyse XAI terminée avec succès !")
