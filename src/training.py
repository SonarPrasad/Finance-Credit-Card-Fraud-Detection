import os
import numpy as np
import mlflow
import mlflow.sklearn
import time
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
LOCAL_MODEL_DIR = "./models/"
MLFLOW_TRACKING_URI = './mlflow'
TRAIN_DATA_PATH = './data/processed/train.npz'

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_data():
    """Load the training data."""
    logging.info("Loading training data...")
    data = np.load(TRAIN_DATA_PATH)
    logging.info("Data loaded successfully.")
    return data['X'], data['y']

def get_models():
    """Return a dictionary of models and their hyperparameter grids."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear'),
            "params": {"C": [0.01, 0.1, 1, 10]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": randint(50, 200),
                "max_depth": [5, 10, None],
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 20)
            },
            "search_method": RandomizedSearchCV
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss'),
            "params": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100], "max_depth": [3, 5]},
            "search_method": GridSearchCV
        }
    }

def train_model(model_name, model_info, X_train, y_train):
    """Train a model using hyperparameter tuning and log parameters/metrics."""
    with mlflow.start_run(run_name=model_name):
        model = model_info['model']
        param_grid = model_info['params']
        search_method = model_info.get('search_method', GridSearchCV)

        # Measure training time
        start_time = time.time()
        search = search_method(model, param_grid, cv=3, scoring='roc_auc')
        search.fit(X_train, y_train)
        training_time = time.time() - start_time

        best_model_candidate = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("Training_Time_seconds", training_time)

        # Evaluate the model
        metrics = evaluate_model(best_model_candidate, X_train, y_train)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model size and inference time
        model_size, inference_time = log_model_metrics(best_model_candidate, X_train)

        return best_model_candidate, search.best_score_, training_time, model_size, inference_time, metrics

def evaluate_model(model, X, y):
    """Evaluate model using various metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc_roc = roc_auc_score(y, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    auc_pr = auc(recall, precision)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    log_loss_val = log_loss(y, y_pred_proba)

    return {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "Accuracy": accuracy,
        "F1_Score": f1,
        "Log_Loss": log_loss_val
    }

def log_model_metrics(model, X_train):
    """Log model size and inference time."""
    model_size = len(pickle.dumps(model)) / 1024  # in KB
    inference_start = time.time()
    model.predict_proba(X_train[:10])  # Small sample prediction
    inference_time = (time.time() - inference_start) / 10  # Average time per sample
    mlflow.log_metric("Model_Size_KB", model_size)
    mlflow.log_metric("Inference_Time_seconds_sample", inference_time)
    return model_size, inference_time

def save_model_locally(model, model_name):
    """Save the best model locally."""
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)
    model_path = os.path.join(LOCAL_MODEL_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved locally at {model_path}")

def setup_experiment(experiment_name):
    """Set up or get an MLflow experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.info(f"Experiment '{experiment_name}' not found. Creating a new experiment.")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        logging.info(f"Using existing experiment '{experiment_name}'.")
        experiment_id = experiment.experiment_id
    return experiment_id

def main():
    """Run the training and evaluation pipeline."""
    experiment_name = "Fraud Detection Experiment"  # Set your experiment name
    experiment_id = setup_experiment(experiment_name)  # Ensure the experiment exists
    mlflow.set_experiment(experiment_name)

    X_train, y_train = load_data()
    models = get_models()

    best_model_info = {"score": -float("inf")}  # Initialize best model info

    for model_name, model_info in models.items():
        logging.info(f"Training {model_name}...")
        best_model, cv_auc_roc, training_time, model_size, inference_time, metrics = train_model(
            model_name, model_info, X_train, y_train)

        # Update best model based on custom scoring
        score = cv_auc_roc - (training_time / 1000) - (model_size / 1000) - (inference_time * 100)
        if score > best_model_info["score"]:
            best_model_info.update({
                "model_name": model_name,
                "model": best_model,
                "cv_auc_roc": cv_auc_roc,
                "training_time": training_time,
                "inference_time": inference_time,
                "model_size": model_size,
                "metrics": metrics
            })

        logging.info(f"Training {model_name} - Completed")

    # Log and save the best model
    logging.info("\nBest Model Selected:")
    for key, value in best_model_info.items():
        if key != "model":
            logging.info(f"{key}: {value}")

    save_model_locally(best_model_info["model"], best_model_info["model_name"])
    mlflow.sklearn.log_model(best_model_info["model"], artifact_path="best_model")

if __name__ == "__main__":
    main()