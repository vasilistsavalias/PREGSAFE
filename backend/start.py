import subprocess
import os
import sys
from pathlib import Path

def main():
    port = os.environ.get('PORT', '10000') # Default to Render's port
    
    print(f"Starting Uvicorn server on host 0.0.0.0:{port}...")

    # Run Alembic Migrations before starting the server
    try:
        print("Running database migrations...")
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
        print("Migrations applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Database migration failed: {e}")
        # We continue even if migrations fail, as the DB might be fine or this might be a restart
    
    # Use subprocess.run to execute uvicorn, which will block until terminated.
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", port
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Uvicorn server failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Uvicorn server shutting down.")

if __name__ == "__main__":
    main()