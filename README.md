# Fraud Detection System

## Overview

The Fraud Detection System is a machine learning project aimed at identifying fraudulent transactions in a dataset. This project utilizes various classification algorithms, including Logistic Regression, Random Forest, and XGBoost, to predict whether a transaction is legitimate or fraudulent. The system is built using Python and leverages MLflow for tracking experiments and managing models.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

## Features

- Multiple machine learning models for fraud detection.
- Hyperparameter tuning using Grid Search and Randomized Search.
- Logging of metrics and parameters using MLflow.
- Evaluation of model performance using various metrics (AUC-ROC, F1 Score, etc.).
- RESTful API for making predictions on new data.

## Technologies Used

- Python 3.13.0
- MLflow
- Scikit-learn
- XGBoost
- Flask
- NumPy
- Pandas
- Matplotlib
- Gunicorn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up MLflow tracking URI:
   ```bash
   export MLFLOW_TRACKING_URI=./mlflow  # On Windows use `set MLFLOW_TRACKING_URI=./mlflow`
   ```

## Usage

1. **Data Preprocessing**: Before training the model, ensure that the data is preprocessed. You can run the preprocessing script:
   ```bash
   python src/preprocessing.py
   ```

2. **Train the Model**: To train the model, run:
   ```bash
   python src/training.py
   ```

3. **Evaluate the Model**: After training, you can evaluate the model using:
   ```bash
   python src/evaluate.py
   ```

4. **Run the API**: To start the Flask API, run:
   ```bash
   python src/app.py
   ```

## Data

The dataset used for this project is a credit card transaction dataset, which contains features related to transactions and a label indicating whether the transaction is fraudulent or not. The data should be placed in the `./data/processed/` directory.

## Model Training

The training script (`src/training.py`) loads the training data, trains multiple models, and logs the results using MLflow. The best model is selected based on a custom scoring metric that considers AUC-ROC, training time, model size, and inference time.

## Model Evaluation

The evaluation script (`src/evaluate.py`) loads the best model and evaluates it on a test dataset. It computes various metrics, including AUC-ROC and AUC-PR, and logs these metrics to MLflow.

## API

The project includes a RESTful API built with Flask. The API has the following endpoints:

- **GET /**: Returns a welcome message.
- **POST /predict**: Accepts a JSON payload with features and returns the prediction.

### Example Request
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [value1, value2, ...]}'
   ```


### Example Response

    ```json
        {
        "prediction": [0] // 0 for legitimate, 1 for fraudulent
        }
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.