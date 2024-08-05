import os
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import save_object

# Step 1: Set the MLflow tracking URI
mlflow.set_tracking_uri("/Users/Tarak/Personal/mlruns")

# Step 2: Create the mlruns directory if it does not exist
if not os.path.exists("/Users/Tarak/Personal/mlruns"):
    os.makedirs("/Users/Tarak/Personal/mlruns")

# Create or get default experiment
experiment_name = "Default"
mlflow.set_experiment(experiment_name)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Save the dataset to a CSV file
data_path = "breast_cancer_data.csv"
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv(data_path, index=False)

def train_and_log_model(n_estimators, max_depth):
    # Train a sample model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs('artifact', exist_ok=True)
    save_object(
        file_path=os.path.join('artifact', f'model_{n_estimators}_{max_depth}.pkl'),
        obj=model
    )

    # Log the model using MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Evaluate and log metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Model logged successfully with run ID: {run.info.run_id}")

# Train and log multiple models with different parameters
train_and_log_model(n_estimators=10, max_depth=2)
train_and_log_model(n_estimators=50, max_depth=5)
train_and_log_model(n_estimators=100, max_depth=10)

# Initialize DVC and add the dataset
if not os.path.exists('.dvc'):
    os.system('dvc init --no-scm')

# Add the dataset to DVC
os.system(f'dvc add {data_path}')
os.system('dvc push')

# Add the model artifacts to DVC
os.system('dvc add artifact/')
os.system('dvc push')
