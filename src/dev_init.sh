#!/bin/bash

# Decode the service account JSON file from the environment variable and save it to a file
echo $GDRIVE_SERVICE_ACCOUNT_JSON | base64 --decode > gdrive_service_account.json

# Ensure the service account JSON file exists
if [ ! -f "gdrive_service_account.json" ]; then
    echo "Service account JSON file (gdrive_service_account.json) not found!"
    exit 1
fi

# Initialize DVC without SCM
dvc init --no-scm -f

# Add Google Drive remote, force if already exists
dvc remote add -d myremote gdrive://1CF-7XIaupXxt3EREqhwSOUI0gEvVZAoH -f

# Configure Google Drive remote with service account
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote --local gdrive_service_account_json_file_path gdrive_service_account.json

echo "DVC initialization and configuration completed."

# Start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns --host 0.0.0.0 --port 5000 &

# Pull the latest dataset from DVC
dvc pull breast_cancer_data.csv.dvc

echo "Pulled the latest dataset from DVC."

# Push:
# dvc add breast_cancer_data.csv
# dvc push
