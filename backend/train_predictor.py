# backend/train_predictor.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# --- Configuration ---
SYNTHESIZER_PATH = "models/synthesizer.pkl"
PREDICTOR_OUTPUT_PATH = "models/predictor.pkl"
INPUT_DATA_PATH = "../ml_pipeline/data/dimitris_train_sets/Train_set_1.csv"
TARGET_VARIABLE = "GDM01"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_predictor():
    """
    Trains a prediction model on a synthetically balanced dataset and saves it.
    """
    logging.info("--- Starting Python Predictor Model Training ---")

    # 1. Load original data and synthesizer
    logging.info(f"Loading original data from: {INPUT_DATA_PATH}")
    original_data = pd.read_csv(INPUT_DATA_PATH)
    
    logging.info(f"Loading CTGAN synthesizer from: {SYNTHESIZER_PATH}")
    synthesizer = joblib.load(SYNTHESIZER_PATH)

    # 2. Generate synthetic data
    # We'll generate enough samples to balance the classes.
    num_minority = original_data[TARGET_VARIABLE].value_counts().min()
    num_majority = original_data[TARGET_VARIABLE].value_counts().max()
    num_to_generate = num_majority - num_minority
    
    logging.info(f"Generating {num_to_generate} synthetic samples for the minority class...")
    # We assume the synthesizer was trained on the minority class, so we can sample directly.
    synthetic_data = synthesizer.sample(num_rows=num_to_generate)
    synthetic_data.columns = original_data.columns # Ensure column names match
    
    # Clean the synthetic data: drop NaNs and ensure correct types
    synthetic_data.dropna(inplace=True)
    synthetic_data[TARGET_VARIABLE] = synthetic_data[TARGET_VARIABLE].astype(int)
    logging.info(f"Generated and cleaned {len(synthetic_data)} synthetic samples.")
    
    # Combine original and synthetic data, resetting index for clean merge
    balanced_data = pd.concat([original_data.reset_index(drop=True), synthetic_data.reset_index(drop=True)], ignore_index=True)
    logging.info("Dataset successfully balanced.")
    logging.info(f"Original class distribution:\n{original_data[TARGET_VARIABLE].value_counts()}")
    logging.info(f"New balanced class distribution:\n{balanced_data[TARGET_VARIABLE].value_counts()}")

    # 3. Prepare data for training
    X = balanced_data.drop(TARGET_VARIABLE, axis=1)
    y = balanced_data[TARGET_VARIABLE]
    
    # Split for a quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Train a RandomForestClassifier
    logging.info("Training RandomForestClassifier model...")
    predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    predictor.fit(X_train, y_train)
    
    logging.info("Model training complete.")

    # 5. Evaluate the model (optional, but good practice)
    y_pred = predictor.predict(X_test)
    logging.info("--- Model Evaluation Report ---")
    logging.info("\n" + classification_report(y_test, y_pred))
    logging.info("-----------------------------")

    # 6. Save the trained predictor model
    logging.info(f"Saving trained predictor to: {PREDICTOR_OUTPUT_PATH}")
    joblib.dump(predictor, PREDICTOR_OUTPUT_PATH)
    logging.info("Predictor saved successfully.")


if __name__ == "__main__":
    train_and_save_predictor()
