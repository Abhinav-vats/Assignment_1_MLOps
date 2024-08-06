import os
import mlflow
import pandas as pd
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from flask import Flask, request, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

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

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth, random_state=42)
    score = cross_val_score(model, X, y, n_jobs=-1, cv=3).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters: ", best_params)

def train_and_log_model(params):
    # Train a sample model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs('artifact', exist_ok=True)
    model_path = os.path.join('artifact', 'model.pkl')
    joblib.dump(model, model_path)

    # Log the model using MLflow
    with mlflow.start_run() as run:
        # Log parameters
        # mlflow.log_params(params)

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

    return model_path

model_path = train_and_log_model(best_params)

# Flask app for serving the model
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        model = joblib.load(model_path)
        prediction = model.predict([data['features']])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
