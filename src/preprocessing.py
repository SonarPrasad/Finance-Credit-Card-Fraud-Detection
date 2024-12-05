import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Constants
RAW_DATA_PATH = "data/creditcard.csv"
PROCESSED_DATA_PATH = "data/processed/"
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "train.npz")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "test.npz")

# Set up logging
logging.basicConfig(level=logging.INFO)

def check_and_create_directory(path: str):
    """Ensure the directory exists, create it if not."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory {path} created.")
    else:
        logging.info(f"Directory {path} already exists.")

def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """Load raw dataset from the given path."""
    try:
        data = pd.read_csv(raw_data_path)
        logging.info(f"Raw data loaded from {raw_data_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_data_path}")
        raise

def preprocess_features_and_target(data: pd.DataFrame):
    """Separate features and target variable."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    logging.info("Features and target separated.")
    return X, y

def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Apply SMOTE to balance the class distribution in the training set."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info("Class imbalance handled using SMOTE.")
    return X_resampled, y_resampled

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, features_to_scale: list) -> tuple:
    """Scale specified features using StandardScaler."""
    scaler = StandardScaler()
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
    logging.info(f"Features {features_to_scale} scaled.")
    return X_train, X_test

def save_processed_data(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """Save the processed data to the disk."""
    np.savez_compressed(TRAIN_DATA_PATH, X=X_train, y=y_train)
    np.savez_compressed(TEST_DATA_PATH, X=X_test, y=y_test)
    logging.info(f"Processed data saved to {PROCESSED_DATA_PATH}")

def preprocess_data():
    """Complete preprocessing pipeline: load, process, and save the data."""
    # Load the raw data
    try:
        raw_data = load_raw_data(RAW_DATA_PATH)
    except FileNotFoundError:
        return

    # Split features and target
    X, y = preprocess_features_and_target(raw_data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data split into training and testing sets.")

    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    # Scale features 'Amount' and 'Time'
    X_train_resampled, X_test = scale_features(X_train_resampled, X_test, features_to_scale=['Amount', 'Time'])

    # Ensure the processed data directory exists
    check_and_create_directory(PROCESSED_DATA_PATH)

    # Save the processed data
    save_processed_data(X_train_resampled, y_train_resampled, X_test, y_test)

# Entry point
if __name__ == "__main__":
    preprocess_data()