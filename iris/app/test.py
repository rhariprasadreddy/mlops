import mlflow

# This is the run ID of the best model
run_id = "a655669fbbbb41f19a83f4348731e531"

# Register the model with a name
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "IrisBestModel")
