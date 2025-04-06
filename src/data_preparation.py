# -------------------- Import Necessary Libraries --------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import yaml

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
        train_data.to_csv(save_path / 'train_data.csv',index=False)
        test_data.to_csv(save_path / 'test_data.csv',index=False)
        logger.info("Train and test data saved successfully")
    except Exception as e:
        logger.exception('Error while saving datasets')

# -------------------- Main Function --------------------

def main():
    try:
        # Define root project path
        root_path = Path(r"C:\Users\redhu\swiggy-delivery-time-prediction").resolve()

        # Define paths
        data_path = root_path / 'data' / 'cleaned' / 'swiggy_cleaned.csv'  # <-- FIXED: actual file, not just folder
        params_path = root_path / 'params.yaml'
        save_path = root_path / 'data' / 'interim'  # <-- FIXED: typo "imterim" to "interim"

        # Load data
        data = load_data(data_path)
        if data.empty:
            logger.warning("Data is empty. Exiting.")
            return

        # Load parameters
        params = load_params(params_path)
        test_size = params['data_preparation']['test_size']
        random_state = params['data_preparation']['random_state']

        # Split data
        train_data, test_data = split_data(data, test_size=test_size, random_state=random_state)

        # Create save directory if not exist
        save_path.mkdir(exist_ok=True, parents=True)

        # Save datasets
        save_data(train_data, test_data, save_path)

    except Exception as e:
        logger.exception('Error during the data preparation process')

# -------------------- Entry Point --------------------

if __name__ == '__main__':
    main()  # <-- FIXED: added missing function call















     





    

        



        
    


