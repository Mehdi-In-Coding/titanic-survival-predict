# partie 6

#             otpimisation.py



import joblib
from google.colab import files
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import pandas as pd

# Vérifier que les données existent
assert 'X_train' in globals(), "X_train n'est pas défini"
assert 'y_train' in globals(), "y_train n'est pas défini"

HO_TUNING = True  # Active l'optimisation manuelle

if HO_TUNING:
    print("\n--- Optimisation des hyperparamètres : Régression Logistique ---")
    logreg_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [200, 500, 1000]
    }
    
    logreg_grid = GridSearchCV(LogisticRegression(random_state=42), logreg_params, cv=5, scoring='accuracy')
    logreg_grid.fit(X_train, y_train)
    
    print(f"Meilleurs hyperparamètres pour la régression logistique : {logreg_grid.best_params_}")
    best_logreg = logreg_grid.best_estimator_
    
    logreg_model_path = "/content/logistic_regression_model.pkl"
    joblib.dump(best_logreg, logreg_model_path)
    print(f"Modèle Regression Logistique sauvegardé : {logreg_model_path}")
    files.download(logreg_model_path)

if HO_TUNING:
    print("\n--- Optimisation des hyperparamètres : RandomForest ---")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'bootstrap': [True]
    }
    
    X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42)
    rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=rf_params, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
    rf_random.fit(X_train_sampled, y_train_sampled)
    
    print(f"Meilleurs hyperparamètres pour RandomForest : {rf_random.best_params_}")
    best_rf = rf_random.best_estimator_
    
    rf_model_path = "/content/random_forest_model.pkl"
    joblib.dump(best_rf, rf_model_path)
    print(f"Modèle RandomForest sauvegardé : {rf_model_path}")
    files.download(rf_model_path)

if HO_TUNING:
    print("\n--- Optimisation des hyperparamètres : XGBoost ---")
    xgb_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    best_score = 0.0
    best_params = None
    for combo in (dict(zip(xgb_params.keys(), values)) for values in itertools.product(*xgb_params.values())):
        model = xgb.XGBClassifier(enable_categorical=True, eval_metric='logloss', random_state=42, **combo)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        if score > best_score:
            best_score = score
            best_params = combo
    
    print(f"Meilleurs hyperparamètres pour XGBoost : {best_params}")
    best_xgb = xgb.XGBClassifier(enable_categorical=True, eval_metric='logloss', random_state=42, **best_params)
    best_xgb.fit(X_train, y_train)
    
    xgb_model_path = "/content/xgboost_model.pkl"
    joblib.dump(best_xgb, xgb_model_path)
    print(f"Modèle XGBoost sauvegardé : {xgb_model_path}")
    files.download(xgb_model_path)
