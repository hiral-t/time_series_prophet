import joblib
import mlflow

def save_model(model, filename):
    """
    Save the trained model to a file using joblib.
    """
    joblib.dump(model, filename)
    mlflow.sklearn.log_model(model, "model")

