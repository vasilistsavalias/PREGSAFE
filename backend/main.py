import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Adjust imports to the new structure
from app.core.config import ALLOWED_ORIGINS
from app.api.endpoints import predictions
from alembic.config import Config
from alembic import command

# --- Application Setup ---
app = FastAPI(
    title="PREGSAFE API",
    description="API for GDM Prediction using CTGAN and SMOTE models.",
    version="1.0.0",
)

# --- CORS Middleware (MUST be added before routes) ---
# Get the frontend URL from an environment variable for production.
FRONTEND_URL = os.environ.get("FRONTEND_URL")

# Define the origins that are allowed to make cross-origin requests.
# We include localhost for local development.
origins = [
    "http://localhost:3000",
]

# If a production frontend URL is provided, add it to the list.
if FRONTEND_URL:
    origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=200)
def health_check():
    """Endpoint for health checks."""
    return {"status": "ok"}

# --- API Router ---
# Include the router from our endpoints module
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
