# -------------------- Import Necessary Libraries --------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import yaml
import os

# -------------------- Logging Setup --------------------

logger = logging.getLogger('Data_preparation')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Avoid adding duplicate handlers in case of reruns
if not logger.hasHandlers():
    logger.addHandler(handler)

# -------------------- Function to Load Dataset --------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from the specified CSV file path"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f'Data loaded successfully from {data_path}')
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
    except Exception as e:
        logger.exception(f"Error while loading data from {data_path}")
    return pd.DataFrame()

# -------------------- Function to Load Parameters --------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""
    try:
        logger.info(f"Loading parameters from: {params_path}")
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info(f"Parameters loaded successfully from: {params_path}")
            return params
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
    except Exception as e:
        logger.exception(f"Error while loading parameters from {params_path}")

# -------------------- Function to Split Dataset --------------------

def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    """Split the dataset into training and testing sets."""
    try:
        logger.info("Splitting data into train and test sets")
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        logger.info("Data split successfully")
        return train_data, test_data
    except Exception as e:
        logger.exception("Error while splitting data")

# -------------------- Function to Save Dataset --------------------

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_path: Path):
    """Save train_data and test_data to specified path"""
    try:
        logger.info(f"Saving data to directory: {save_path}")
        train_data.to_csv(save_path / 'train_data.csv', index=False)
        test_data.to_csv(save_path / 'test_data.csv', index=False)
        logger.info("Train and test data saved successfully")
    except Exception as e:
        logger.exception('Error while saving datasets')

# -------------------- Utility Functions --------------------

def ensure_directory_exists(directory: Path):
    """Ensure a directory exists; if not, create it."""
    try:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error ensuring directory exists {directory}: {e}")
        raise

def get_root_directory() -> Path:
    """Get the root directory (one level up from this script's location)."""
    try:
        current_dir = Path(__file__).resolve().parent.parent
        root_dir = current_dir.parent
        logger.debug(f"Current directory: {current_dir}")
        logger.debug(f"Resolved root directory: {root_dir}")
        return root_dir
    except Exception as e:
        logger.error('Error getting root directory: %s', e)
        raise

# -------------------- Main Function --------------------

def main():
    try:
        # Get current and root directories
        current_dir = Path(__file__).resolve().parent
        root_dir = get_root_directory()
        print(root_dir)

        # Define paths
        data_path = root_dir/ 'data' / 'cleaned' / 'swiggy_cleaned.csv'
        params_path = root_dir / 'params.yaml'
        save_path = root_dir / 'data' / 'interim'

        # Load data
        data = load_data(str(data_path))
        if data.empty:
            logger.warning("Data is empty. Exiting.")
            return

        # Load parameters
        params = load_params(str(params_path))
        if not params or 'data_preparation' not in params:
            logger.error("Invalid or missing parameters in YAML. Exiting.")
            return

        test_size = params['data_preparation'].get('test_size', 0.2)
        random_state = params['data_preparation'].get('random_state', 42)

        # Split data
        train_data, test_data = split_data(data, test_size=test_size, random_state=random_state)

        # Ensure save directory exists
        ensure_directory_exists(save_path)

        # Save datasets
        save_data(train_data, test_data, save_path)

    except Exception as e:
        logger.exception('Error during the data preparation process')

# -------------------- Entry Point --------------------

if __name__ == '__main__':
    main()
     
        

        
















     





    

        



        
    


