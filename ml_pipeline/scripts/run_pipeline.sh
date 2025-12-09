#!/bin/bash
# run_pipeline.sh
# Orchestrates the full GDM Pipeline (Folds + Final)

SMOKE_TEST=false
TUNE=false

# Parse arguments
for arg in "$@"
do
    case $arg in
        --smoke-test)
        SMOKE_TEST=true
        shift
        ;;
        --tune)
        TUNE=true
        shift
        ;;
    esac
done

ARGS_LIST=""
if [ "$SMOKE_TEST" = true ]; then
    ARGS_LIST="$ARGS_LIST --smoke-test"
    echo "--- RUNNING PIPELINE IN SMOKE TEST MODE ---"
fi

if [ "$TUNE" = true ]; then
    ARGS_LIST="$ARGS_LIST --tune"
    echo "--- TUNING ENABLED: Will run Optuna Tuning Stage for Folds ---"
else
    echo "--- RUNNING PIPELINE IN FULL PRODUCTION MODE (Baseline Generation) ---"
fi

# Set Project Root (one level up from scripts -> ml_pipeline/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure dependencies are installed (Verbose, Fail fast)
echo "Ensuring package is installed..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: 'pip install -e .' failed. Check your environment/requirements."
    exit 1
fi

# Force PYTHONPATH to include src just in case
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run Python Orchestrator
echo "Starting main.py..."
python3 main.py $ARGS_LIST

if [ $? -ne 0 ]; then
    echo "Pipeline Failed!"
    exit 1
fi

echo "Pipeline Completed Successfully!"
