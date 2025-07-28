import mlflow

"""
Registers the best model from a specific MLflow run for the Iris dataset.
"""

# Replace with your actual best run ID
run_id = "a655669fbbbb41f19a83f4348731e531"
model_uri = f"runs:/{run_id}/model"

# Register the model to MLflow Model Registry
mlflow.register_model(model_uri, "IrisBestModel")
