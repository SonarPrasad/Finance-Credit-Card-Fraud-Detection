import mlflow
import mlflow.sklearn
import os
from flask import Flask, request, jsonify
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Set the tracking URI for MLflow
mlflow.set_tracking_uri('./mlflow')
EXPERIMENT_NAME = "Fraud Detection Experiment"

def load_best_model(experiment_name):
    """
    Load the best model saved during training based on the logged artifact.
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

# Load the model at application start
model = None
try:
    model, _ = load_best_model(EXPERIMENT_NAME)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/')
def home():
    return """
    <html>
        <head>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                }
                .message {
                    font-weight: bold;
                    font-size: 24px;
                    margin: 10px;
                }
            </style>
        </head>
        <body>
            <div class="message">
                Welcome to the Fraud Detection API.
            </div>
            <div class="message">
                Use the /predict endpoint to get predictions.
            </div>
        </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        data = request.get_json()
        features = data['features']  # Expecting a list of features
        prediction = model.predict([features]).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



from waitress import serve

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)