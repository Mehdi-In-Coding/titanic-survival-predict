#     partie 2 :
#     eda_analysis.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

# Chargement des données (à modifier si nécessaire)
train_path = "/content/drive/My Drive/Titanic-Survival-Predict-main/train_cleaned.csv"
test_path = "/content/drive/My Drive/Titanic-Survival-Predict-main/test_cleaned.csv"

# Lecture des fichiers CSV
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Assertions pour vérifier que les datasets sont bien chargés
assert isinstance(train_df, pd.DataFrame), "Erreur: train_df n'est pas un DataFrame"
assert isinstance(test_df, pd.DataFrame), "Erreur: test_df n'est pas un DataFrame"

# Vérification des colonnes essentielles
required_columns = ["Survived", "Pclass", "Age"]
for col in required_columns:
    assert col in train_df.columns, f"Erreur: La colonne {col} est absente du dataset d'entraînement"

# Vérification des valeurs manquantes
assert train_df.isnull().sum().sum() < len(train_df), "Erreur: Trop de valeurs manquantes dans train_df"
assert test_df.isnull().sum().sum() < len(test_df), "Erreur: Trop de valeurs manquantes dans test_df"

# Affichage du titre HTML
display(HTML("""
<h1 style="color:#2c3e50; font-size: 32px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
Partie 2: Analyse Exploratoire des Données (EDA)
</h1>
"""))

# Configuration des styles pour les graphiques
plt.style.use('ggplot')
sns.set_context('notebook')

# Étape 1 : Visualisation de la répartition de la variable 'Survived'
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Survived', palette='Set2')
plt.title('Répartition de la variable Survived')
plt.show()

# Étape 2 : Distribution des classes de passagers (Pclass)
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Pclass', palette='Set3')
plt.title('Répartition des classes de passagers (Pclass)')
plt.show()

# Étape 3 : Distribution de l'âge par rapport à la survie
plt.figure(figsize=(10, 6))
sns.histplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins=20, color='green', label='Survécu', kde=True)
sns.histplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins=20, color='red', label='Non survécu', kde=True)
plt.title('Distribution des âges par rapport à la survie')
plt.legend()
plt.show()

# Étape 4 : Matrice de corrélation des variables numériques
numeric_features = train_df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()

# Vérification finale des dimensions des datasets
assert train_df.shape[0] > 0, "Erreur: train_df est vide"
assert test_df.shape[0] > 0, "Erreur: test_df est vide"

print("\n✅ Analyse exploratoire des données terminée avec succès !")
