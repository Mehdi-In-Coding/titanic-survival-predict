import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def train_models(train_df, optimize=False):
    """
    Entraînement des modèles ML et sauvegarde des modèles entraînés.
    
    Arguments:
    train_df -- DataFrame contenant les données prétraitées
    optimize -- Si True, recherche d'hyperparamètres avec GridSearchCV
    
    Retourne:
    logreg, rf, xgb_model -- Modèles entraînés
    """
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📢 Entraînement du modèle de Régression Logistique...")
    logreg = LogisticRegression(max_iter=1000)
    if optimize:
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        logreg = GridSearchCV(logreg, param_grid, cv=5)
    logreg.fit(X_train, y_train)

    print("📢 Entraînement du modèle RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print("📢 Entraînement du modèle XGBoost...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    print("✅ Modèles entraînés et sauvegardés !")
    joblib.dump(logreg, "models/logreg.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")

    return logreg, rf, xgb_model
