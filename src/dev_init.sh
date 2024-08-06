#!/bin/bash

# Initialize DVC without SCM
dvc init --no-scm

# Add Google Drive remote
dvc remote add -d myremote gdrive://1CF-7XIaupXxt3EREqhwSOUI0gEvVZAoH

# Configure Google Drive remote
dvc remote modify myremote gdrive_use_service_account false
dvc remote modify myremote --local gdrive_client_id 42411527554-l23lu38n6teaeuc9rl6ja2c6jpt6ekqj.apps.googleusercontent.com
dvc remote modify myremote --local gdrive_client_secret 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

echo "DVC initialization and configuration completed."

# Start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
