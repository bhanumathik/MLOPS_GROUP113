# prompt: deploy the model

import mlflow
from mlflow.tracking import MlflowClient

# Replace with your actual MLflow tracking URI if needed
mlflow.set_tracking_uri("http://localhost:5000")  # Or your remote tracking server URI

# Get the latest run information
client = MlflowClient()
experiment = client.get_experiment_by_name("Twitter Sentiment Analysis")
runs = client.search_runs(experiment_ids=experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
latest_run = runs[0]

# Get the model URI
model_uri = f"runs:/{latest_run.info.run_id}/twitter_sentiment_model"

# Deploy the model (example using mlflow models serve)
!mlflow models serve -m $model_uri --host 0.0.0.0 --port 1234

print(f"Model deployed from run ID: {latest_run.info.run_id}")
print(f"Model URI: {model_uri}")
print("Access the model at: http://0.0.0.0:1234")