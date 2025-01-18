# **Prédiction de Survie sur le Titanic**

Ce dépôt propose une solution structurée et modulaire pour résoudre le problème classique de prédiction de survie des passagers du Titanic. Le projet met en œuvre les meilleures pratiques d’ingénierie logicielle tout en exploitant l’apprentissage automatique pour prédire la survie des passagers en fonction de caractéristiques telles que l’âge, le genre et la classe de billet.

---

## **Objectif du Projet**

L’objectif de ce projet est de refactoriser un notebook existant sur la prédiction de survie du Titanic en un code Python prêt pour la production. Les points clés incluent :

### **Mise en œuvre des bonnes pratiques d’ingénierie logicielle :**
- Scripts Python modulaires.
- Conformité au style **PEP 8**.
- Tests unitaires pour les fonctions principales.

### **Construction d’un pipeline d’apprentissage automatique :**
- Prétraitement des données.
- Entraînement et évaluation des modèles prédictifs.

### **Collaboration et intégration CI/CD :**
- Utilisation de **Git** et **GitHub** pour le contrôle de version.
- Configuration de workflows automatisés avec **GitHub Actions** pour le linting, les tests et le déploiement.

Ce projet respecte les exigences académiques de l'IUT Paris Cité, avec un accent sur le travail en équipe, la modularité et la maintenabilité.

---

## **Instructions d’Installation**

### **Étape 1 : Cloner le Dépôt**
```bash
git clone https://github.com/votre-repo/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### **Étape 2 : Installer les Dépendances**
Assurez-vous d’avoir Python 3.8+ installé. Installez les librairies nécessaires :**
```bash
pip install -r requirements.txt
```

### **Étape 3 : Exécuter les Scripts**
Lancez les scripts dans l’ordre suivant :

**Prétraitement des données :**
```bash
python src/data_preprocessing.py
```

**Entraînement des modèles :**
```bash
python src/model_training.py
```

**Evaluation des modèles :**
```bash
python src/model_evaluation.py
```

---

## **Résultats Clés**

| **Model**             | **Accuracy** | **F1-Score** |
|-----------------------|--------------|--------------|
| Logistic Regression   | 00.00%      | 0.00        |
| RandomForest          | 00.00%      | 0.00         |
| MultiLayer Perceptron | 00.00%      | 0.00        |
| XGBoost               | TBD         | TBD          |
| **Stacked Model**     | **00.00%**  | **0.00**     |

---

IA Explicable (XAI)
Techniques utilisées pour interpréter les modèles :

Valeurs SHAP : Analyse de l’impact des caractéristiques sur les prédictions.
Importances des caractéristiques : Mise en évidence des variables les plus significatives.
Permutation Importance : Évaluation de la pertinence des variables par permutation.
Pipeline CI/CD
Le dépôt intègre des workflows GitHub Actions pour :

Linting : Vérifie la qualité du code avec Flake8 et Black.
Tests : Exécute les tests unitaires via Pytest.
Simulation de déploiement : Containerisation optionnelle avec Docker.
Améliorations Futures
Intégration de méthodes avancées d’ensembles pour de meilleures performances.
Enrichissement des caractéristiques avec des données externes.
Déploiement du modèle en tant qu’API avec Flask ou FastAPI.
Comment Contribuer
Contributions bienvenues ! Pour participer :

Forkez le dépôt.
Créez une branche pour vos modifications.
Soumettez une pull request pour révision.
Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.

