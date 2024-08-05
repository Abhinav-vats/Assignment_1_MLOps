import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os
from src.utils import save_object


# Step 1: Set the MLflow tracking URI
mlflow.set_tracking_uri("/Users/Tarak/Personal/mlruns")

# Step 2: Create the mlruns directory if it does not exist
if not os.path.exists("/Users/Tarak/Personal/mlruns"):
    os.makedirs("/Users/Tarak/Personal/mlruns")

# Added for Test commit again

# Step 3: Create or get default experiment
experiment_name = "Default"
mlflow.set_experiment(experiment_name)

# Step 4: Train a sample model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

save_object(
                file_path=os.path.join('artifact', 'model.pkl'),
                obj=model
            )

# Step 5: Log the model using MLflow
with mlflow.start_run() as run:
    # Log parameters (if any)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log metrics (if any)
    mlflow.log_metric("accuracy", model.score(X, y))

    print(f"Model logged successfully with run ID: {run.info.run_id}")
