import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn import set_config
import pathlib

import logging
import pickle
import yaml
import os

# Set sklearn transformer output to pandas DataFrame
set_config(transform_output='pandas')

# Numerical and categorical columns
num_cols = ['person_age', 'person_ratings', 'pickup_time_in_minutes', 'distance']
num_cat_cols = ['weather_condition', 'order_type', 'vehicle_type', 'multiple_deliveries', 'city', 'festival',
                'city_name', 'day_of_week', 'time_slot']
ordinal_cat_cols = ['traffic_density', 'distance_type']
traffic_order = ['Low ', 'Medium ', 'High ', 'Jam ']
distance_order = ['short', 'medium', 'long', 'very long']

target_column = ['time_taken']

# Logger configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV data from path."""
    try:
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
    except Exception as e:
        logger.exception(f"Error while loading data from {data_path}")
    return pd.DataFrame()


def load_params(param_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(param_path) as file:
            params = yaml.safe_load(file)
            logger.debug(f"Parameters loaded from: {param_path}")
            return params
    except Exception as e:
        logger.error(f"Error loading parameters from {param_path}: {e}")
        raise


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values and log the change in size."""
    logger.info(f"Original dataset shape: {data.shape}")
    df_dropped = data.dropna()
    logger.info(f"Shape after dropping missing values: {df_dropped.shape}")
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        logger.warning("Data had missing values which were dropped.")
    return df_dropped


def apply_transformation(train_data: pd.DataFrame):
    """Apply transformation using ColumnTransformer and return X, y."""
    logger.info("Starting transformation process...")

    # Drop target column to get features
    x_train = train_data.drop(columns=target_column)
    y_train = train_data[target_column].values.ravel()

    preprocessor = ColumnTransformer(transformers=[
        ('scaling', MinMaxScaler(), num_cols),
        ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), num_cat_cols),
        ('ordinal_encoding', OrdinalEncoder(categories=[traffic_order, distance_order]), ordinal_cat_cols)
    ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=-1)

    x_train_trans = preprocessor.fit_transform(x_train)
    logger.info("Transformation completed successfully.")

    return x_train_trans, y_train, preprocessor


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
    logger.info("Starting data preprocessing pipeline...")

    # Paths
    root_dir = get_root_directory()
    param_path = os.path.join(root_dir, 'params.yaml')
    train_path = os.path.join(root_dir, 'data/interim/train_data.csv')
    processed_data_dir = os.path.join(root_dir, 'data/processed')

    # Load params and data
    params = load_params(param_path)
    train_data = load_data(train_path)

    # Drop missing values
    train_data = drop_missing_values(train_data)

    # Transform the data
    x_train_trans, y_train, preprocessor = apply_transformation(train_data)

    # Save preprocessed data and preprocessor
    ensure_directory_exists(processed_data_dir)

    # Save transformed features and target
    with open(os.path.join(processed_data_dir, 'x_train_trans.pkl'), 'wb') as f:
        pickle.dump(x_train_trans, f)
    with open(os.path.join(processed_data_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(processed_data_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

    logger.info("Preprocessing complete. Data and preprocessor saved successfully.")


if __name__ == "__main__":
    main()

          

          
    



    
    


    





    




