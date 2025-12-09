#!/bin/bash

# This script resets the project to a clean state by deleting all generated files.

echo "====================================================="
echo " GDM Project Cleaner"
echo "====================================================="

FORCE=false
if [[ "$1" == "--force" ]]; then
    FORCE=true
fi

# --- Confirmation ---
if [ "$FORCE" = false ]; then
    read -p "This will permanently delete the /outputs and /logs directories. Are you sure? (y/n) " -n 1 -r
    echo    # move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Operation cancelled."
        exit 1
    fi
fi

# --- Deletion ---
echo "Deleting /outputs directory..."
rm -rf outputs

echo "Deleting /logs directory..."
rm -rf logs

echo "Deleting temporary Python cache files..."
find . -type d -name "__pycache__" -exec rm -r {} +

echo "====================================================="
echo " Project has been cleaned."
echo "====================================================="
