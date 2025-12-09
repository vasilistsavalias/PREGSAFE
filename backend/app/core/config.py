# backend/app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Environment Setup ---
# Determine the absolute path to the project root directory.
# This allows for robust path creation throughout the application.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# --- Model Paths ---
# Define absolute paths to the model files to prevent ambiguity.
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "preprocessor.pkl")
SYNTHESIZER_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "synthesizer.pkl")
# Define the absolute path to the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Path to the trained model file
PREDICTOR_PATH = BASE_DIR / "backend" / "models" / "final_predictor.pkl"
SMOTE_PREDICTOR_PATH = BASE_DIR / "backend" / "models" / "smote_predictor.pkl"


# --- CORS Configuration ---
# A list of origins that are allowed to make cross-origin requests.
# Using a wildcard "*" is convenient for development but should be
# restricted in a production environment.
ALLOWED_ORIGINS = ["*"]

# --- Database Configuration ---
# Use the DATABASE_URL from the environment variable if it exists (for Render/production).
# Otherwise, fall back to the local SQLite database for development.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(PROJECT_ROOT, 'database', 'database.db')}"
)