import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression


import logging
import pickle
import os
import yaml

#configure logging
logger=logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

#configure console handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add handler to logger
logger.addHandler(file_handler)

#configure formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

def load_data(file_path):
    """
    Load a single pickle file.

   
    """
    try:
        logger.info(f"Loading data from: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded data successfully from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file at {file_path}: {e}")
        raise

def load_params(params_path: str):
    """
    Load parameters from a pickle file.
    """
    try:
        logger.info(f"Loading parameters from: {params_path}")
        with open(params_path, 'rb') as f:
            params = yaml.safe_load(f)
        logger.info("Parameters loaded successfully.")
        return params

    except FileNotFoundError as e:
        logger.error(f"Parameter file not found at {params_path}: {e}")
        raise


def train_model(model, x_train, y_train):
    """
    Train the model using the provided training data.

    """
    logger.info("Starting model training...")
    model.fit(x_train, y_train)
    logger.info("Model training completed.")
    return model


def save_model(model, save_path):
    """
    Save the trained model to a file using pickle.

    """
    try:
        logger.info(f"Saving model to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def save_transformer(transformer, save_path):
    """
    Save the fitted transformer to a file using pickle.

       """
    try:
        logger.info(f"Saving transformer to {save_path}...")
        with open(save_path, 'wb') as f:  
            pickle.dump(transformer, f)
        logger.info("Transformer saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save transformer: {e}")
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
    
    #get the root directory
    root_dir=get_root_directory()

    #load params from root directory

    params=load_params(os.path.join(root_dir,'params.yaml'))
    rf_params=params['model_building']['rf_params']
    lgbm_params=params['model_building']['lgbm_params']

    
    # Load x_train and y_train separately
    x_train_path = os.path.join(root_dir, 'data','processed','x_train_trans.pkl')
    y_train_path = os.path.join(root_dir, 'data','processed','y_train.pkl')

    x_train = load_data(x_train_path)
    y_train = load_data(y_train_path)


    rf_regressor=RandomForestRegressor(**rf_params)
    lgbm_reg=LGBMRegressor(**lgbm_params)

    transformer=PowerTransformer()
    #meta model
    lr=LinearRegression()

    stacking_reg=StackingRegressor(estimators=[('rf',rf_regressor),('lgbm_reg',lgbm_reg)],final_estimator=lr)

    model=TransformedTargetRegressor(regressor=stacking_reg,transformer=transformer)
    model.fit(x_train,y_train)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_model(model,os.path.join(current_dir,'model.pkl'))
    
    save_transformer(transformer,os.path.join(current_dir,'transformer.pkl'))

if __name__ == "__main__":
    main()

    #









        


