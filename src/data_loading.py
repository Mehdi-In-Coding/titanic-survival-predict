#  partie1
#        partie_1_data_loading.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from IPython.core.display import display, HTML

# Définir les styles visuels
plt.style.use('ggplot')
sns.set_context('notebook')

# 📌 Affichage du titre de la section

# 🚀 Monter Google Drive
drive.mount('/content/drive')

# 🔍 Chemins des fichiers
TRAIN_PATH = "/content/drive/My Drive/Titanic-Survival-Predict-main/train_cleaned.csv"
TEST_PATH = "/content/drive/My Drive/Titanic-Survival-Predict-main/test_cleaned.csv"

# 📂 Vérifier si les fichiers existent
assert os.path.exists(TRAIN_PATH), f"❌ Le fichier {TRAIN_PATH} est introuvable."
assert os.path.exists(TEST_PATH), f"❌ Le fichier {TEST_PATH} est introuvable."

# 📥 Charger les données
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ✅ Vérification que les DataFrames ne sont pas vides
assert not train_df.empty, "❌ Le DataFrame train_df est vide."
assert not test_df.empty, "❌ Le DataFrame test_df est vide."

# 🔎 Affichage des informations du jeu d'entraînement
print("\n📊 Training Set Information:")
print(train_df.info())

# 🔎 Affichage des informations du jeu de test
print("\n📊 Test Set Information:")
print(test_df.info())

# 🧐 Vérification des valeurs manquantes
print("\n🔎 Valeurs manquantes dans le jeu d'entraînement :")
print(train_df.isnull().sum())

print("\n🔎 Valeurs manquantes dans le jeu de test :")
print(test_df.isnull().sum())

# ✅ Vérification
assert train_df.isnull().sum().sum() < 100, "⚠️ Trop de valeurs manquantes dans train_df."
assert test_df.isnull().sum().sum() < 100, "⚠️ Trop de valeurs manquantes dans test_df."

# 🔍 Aperçu des premières lignes
print("\n📌 Les 5 premières lignes du jeu d'entraînement :")
print(train_df.head())

print("\n📌 Les 5 premières lignes du jeu de test :")
print(test_df.head())

# 🔎 Exploration des données - Distribution de la variable cible
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Survived', palette='Set2')
plt.title('Répartition de la variable Survived')
plt.show()

# 🔎 Distribution des classes de passagers (Pclass)
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Pclass', palette='Set3')
plt.title('Répartition des classes de passagers (Pclass)')
plt.show()

# 🔎 Distribution de l'âge par rapport à la survie
plt.figure(figsize=(10, 6))
sns.histplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins=20, color='green', label='Survécu', kde=True)
sns.histplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins=20, color='red', label='Non survécu', kde=True)
plt.title('Distribution des âges par rapport à la survie')
plt.legend()
plt.show()

# 🔎 Matrice de corrélation pour vérifier les relations entre les variables numériques
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()

print("\n✅ Partie 1: Data Loading et Exploration des Données terminée avec succès !")
