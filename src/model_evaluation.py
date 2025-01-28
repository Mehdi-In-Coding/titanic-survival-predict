import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, X_val, y_val):
    model = joblib.load(model_path)
    y_pred = model.predict(X_val)
    
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
