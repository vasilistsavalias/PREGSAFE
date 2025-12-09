# backend/train_smote_predictor.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import logging
from pathlib import Path

# --- Configuration ---
# Define paths relative to the script's location
BASE_DIR = Path(__file__).parent
INPUT_DATA_PATH = BASE_DIR.parent / "ml_pipeline/data/dimitris_train_sets/Train_set_1.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models/smote_predictor.pkl"
TARGET_VARIABLE = "GDM01"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_model():
    """
    Loads the training data, applies SMOTE to balance the classes,
    trains a RandomForest classifier, and saves the trained model.
    """
    logging.info("--- Starting SMOTE-based RandomForest model training ---")

    # 1. Load Data
    logging.info(f"Loading training data from: {INPUT_DATA_PATH}")
    try:
        train_data = pd.read_csv(INPUT_DATA_PATH)
    except FileNotFoundError:
        logging.error(f"CRITICAL: Input data file not found at {INPUT_DATA_PATH}. Please ensure the path is correct.")
        return

    # Prepare data for SMOTE
    X = train_data.drop(TARGET_VARIABLE, axis=1)
    y = train_data[TARGET_VARIABLE]

    # Log original class distribution
    original_dist = y.value_counts()
    logging.info(f"Original class distribution:\n{original_dist}")

    # 2. Apply SMOTE
    logging.info("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Log balanced class distribution
    balanced_dist = y_resampled.value_counts()
    logging.info(f"New balanced class distribution:\n{balanced_dist}")

    # 3. Train Random Forest Model
    logging.info("Training RandomForest model on the balanced data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)
    logging.info("Model training complete.")

    # 4. Save the Model
    # Ensure the output directory exists
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving trained model to: {MODEL_OUTPUT_PATH}")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
