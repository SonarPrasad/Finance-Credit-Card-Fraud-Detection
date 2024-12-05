import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)

def load_best_model(experiment_name):
    """
    Load the model saved during training based on the logged artifact.
    This assumes the model was already selected as the best during training.
    """
    logging.info("Setting MLflow experiment and loading the model...")
    
    # Get experiment details
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    experiment_id = experiment.experiment_id
    logging.info(f"Using Experiment ID: {experiment_id}")
    
    # Search for the best run in this experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time desc"]  # We are just interested in the most recent run
    )
    if runs.empty:
        logging.error("No runs found in the MLflow experiment. Ensure training has been logged properly.")
        raise ValueError("No suitable runs found in MLflow.")
    
    # Retrieve the latest run's details
    latest_run = runs.iloc[0]
    model_uri = latest_run.artifact_uri
    logging.info(f"Raw model URI: {model_uri}")
    
    # If the URI starts with 'file:', remove it to get a proper file path
    if model_uri.startswith("file:"):
        model_uri = model_uri[5:]  # Remove 'file:' prefix
    
    # Normalize the path
    model_uri = os.path.normpath(os.path.join(model_uri, "best_model"))
    logging.info(f"Normalized model path: {model_uri}")
    
    # Verify if the model path exists
    if not os.path.exists(model_uri):
        logging.error(f"Model path does not exist: {model_uri}")
        raise FileNotFoundError(f"Model not found at path: {model_uri}")
    
    # Load the model
    model = mlflow.sklearn.load_model(model_uri)
    return model, latest_run

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset.
    Computes AUC-ROC and AUC-PR metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    auc_roc_test = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr_test = auc(recall, precision)

    return auc_roc_test, auc_pr_test, y_pred_proba


def plot_curves(y_test, y_pred_proba):
    """
    Plot Precision-Recall and ROC curves.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(12, 5))
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.', label='AUC-PR')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label='AUC-ROC')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

def log_metrics_to_mlflow(auc_roc_test, auc_pr_test):
    """
    Log evaluation metrics to MLflow.
    """
    mlflow.log_metric("AUC_ROC_Test", auc_roc_test)
    mlflow.log_metric("AUC_PR_Test", auc_pr_test)
    logging.info(f"AUC-ROC (Test): {auc_roc_test:.4f}")
    logging.info(f"AUC-PR (Test): {auc_pr_test:.4f}")

def main():
    """
    Main function to load the best model and evaluate it on the test set.
    """
    mlflow.set_tracking_uri('./mlflow')
    experiment_name = "Fraud Detection Experiment"

    logging.info("Loading test data...")
    test_data_path = './data/processed/test.npz'
    data = np.load(test_data_path)
    X_test, y_test = data['X'], data['y']

    try:
        model, latest_run = load_best_model(experiment_name)
    except ValueError as e:
        logging.error(f"Error loading the model: {e}")
        return

    auc_roc_test, auc_pr_test, y_pred_proba = evaluate_model(model, X_test, y_test)
    log_metrics_to_mlflow(auc_roc_test, auc_pr_test)

    # Plot the Precision-Recall and ROC curves
    logging.info("Generating evaluation plots...")
    plot_curves(y_test, y_pred_proba)

    logging.info("Model evaluation complete.")
    logging.info(f"AUC-ROC (Test): {auc_roc_test:.4f}")
    logging.info(f"AUC-PR (Test): {auc_pr_test:.4f}")


if __name__ == "__main__":
    main()