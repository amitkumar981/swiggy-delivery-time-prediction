#importing libraries

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
import pickle
import json
import logging
import os


#configure logging
logger=logging.getLogger('register_model')
logger.setLevel(logging.DEBUG)

#configure consolw handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add console handler to logger
logger.addHandler(file_handler)

#set formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)

 #set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/swiggy-delivery-time-prediction.mlflow')



logger = logging.getLogger('model_evaluation')

def load_model_info(model_path):
    """
    Load model run information from a JSON file.

    Args:
        model_path (str): Path to the JSON file containing run information.

    Returns:
        dict: Dictionary containing run_id, artifact_path, and model_name.
    """
    try:
        logger.info(f"Attempting to load model info from: {model_path}")
        with open(model_path, "r") as f:
            run_info = json.load(f)
        logger.info("Model info loaded successfully.")
        return run_info
    except FileNotFoundError:
        logger.error(f"Model info file not found at: {model_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {model_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading model info from {model_path}: {e}")
        raise

def ensure_directory_exists(directory: str):
    """Ensure a directory exists; if not, create it."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error ensuring directory exists {directory}: {e}")
        raise

def get_root_directory() -> str:
    """Get the root directory (one level up from this script's location)."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '../..'))  
        logger.debug(f"Current directory: {current_dir}")
        logger.debug(f"Resolved root directory: {root_dir}")
        return root_dir
    except Exception as e:
        logger.error('Error getting root directory: %s', e)
        return None
    
def main():
    root_dir=get_root_directory()

    #load model info
    model_info=load_model_info(os.path.join(root_dir,'run_information.json'))

    #get run id and model name from model info
    run_id=model_info['run_id']
    model_name=model_info['model_name']

    #model  to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"

    #register model version

    model_version=mlflow.register_model(model_uri=model_registry_path,name=model_name)

    #get the model version
    registered_model_version=model_version.version
    registered_model_name=model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")


    #update the model version
    client=MlflowClient()
    client.transition_model_version_stage(name=registered_model_name,version=registered_model_version,
                                          stage='staging')

if __name__=="__main__":
    main()

    
    
        
