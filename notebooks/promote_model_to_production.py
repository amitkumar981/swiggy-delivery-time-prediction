import mlflow
import json
import os
from mlflow import MlflowClient
import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)

#set tracking uri
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


root_dir=get_root_directory()

file_path=os.path.join(root_dir,'run_information.json')
model_name=model_load_information(file_path)['model_name']


stage='staging'

client=MlflowClient()
latest_versions=client.get_latest_versions(name=model_name,stages=[stage])
model_latest_version=latest_versions[0].version

#promote model
promote_stage='production'

client.transition_model_version_stage(name=model_name,version=model_latest_version,
                                      stage=promote_stage,archive_existing_versions=True)



