#!/bin/bash

# Initialize DVC without SCM, force if already exists
dvc init --no-scm -f

# Add Google Drive remote, force if already exists
dvc remote add -d myremote gdrive://1CF-7XIaupXxt3EREqhwSOUI0gEvVZAoH -f

# Configure Google Drive remote with service account
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote --local gdrive_service_account_json /app/gdrive_service_account.json

echo "DVC initialization and configuration completed."

# Start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

# Wait for a few seconds to ensure MLflow server is up and running
sleep 10

# Run the model training script
python /app/src/train_model.py
