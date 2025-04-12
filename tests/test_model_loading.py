import pytest
import mlflow
from mlflow import MlflowClient
import os
import json
import dagshub
import logging

# Init DagsHub
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/swiggy-delivery-time-prediction.mlflow')

def model_load_information(file_path):
    with open(file_path,'rb') as f:
        run_info=json.load(f)
    return run_info


def get_root_directory() -> str:
    """Get the root directory (one level up from this script's location)."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    root_dir = os.path.abspath(os.path.join(current_dir, '..')) 

    return root_dir

 
root_path = get_root_directory()
file_path = os.path.join(root_path, 'run_information.json')
model_name=model_load_information(file_path=file_path)['model_name']


@pytest.mark.parametrize(argnames="model_name, stage",
                         argvalues=[(model_name, "Staging")])
def test_load_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None
    
    assert latest_version is not None, f"No model at {stage} stage"
    
    # load the model
    model_path = f"models:/{model_name}/{stage}"

    # load the latest model from model registry
    model = mlflow.sklearn.load_model(model_path)
    
    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")
