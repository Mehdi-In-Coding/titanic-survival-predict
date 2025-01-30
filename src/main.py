import argparse
from data_preprocessing import preprocess_data
from model_training import train_models
from model_evaluation import evaluate_model
import pandas as pd

def main(optimize=False):
    print("📢 Chargement et prétraitement des données...")
    train_df, test_df = preprocess_data("data/train.csv", "data/test.csv")

    print("📢 Entraînement des modèles...")
    logreg, rf, xgb_model = train_models(train_df, optimize=optimize)

    print("📢 Évaluation des modèles...")
    X_val = train_df.drop(columns=['Survived'])
    y_val = train_df['Survived']
    evaluate_model("models/logreg.pkl", X_val, y_val)
    evaluate_model("models/random_forest.pkl", X_val, y_val)
    evaluate_model("models/xgboost.pkl", X_val, y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Activer l'optimisation des hyperparamètres")
    args = parser.parse_args()
    main(optimize=args.optimize)
