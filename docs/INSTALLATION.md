Installation et Exécution du Projet

Ce guide vous aidera à installer et exécuter le projet Titanic Survival Prediction sur votre machine locale.

📌 Prérequis

Avant de commencer, assurez-vous d'avoir :

Python 3.8+ installé

Git installé

pip (Python package manager) installé

Un compte Kaggle pour télécharger les données

🛠️ Étape 1 : Cloner le Dépôt

git clone https://github.com/votre-github/titanic-survival-predict.git
cd titanic-survival-predict

📦 Étape 2 : Créer un Environnement Virtuel

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

📥 Étape 3 : Installer les Dépendances

pip install -r requirements.txt

📊 Étape 4 : Télécharger les Données

Rendez-vous sur Kaggle - Titanic Dataset.

Téléchargez train.csv et test.csv.

Placez ces fichiers dans data/ du projet.

🚀 Étape 5 : Exécuter le Projet

python src/main.py

🛠️ Étape 6 : Tester le Projet