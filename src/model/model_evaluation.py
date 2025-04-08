#importing libraries

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
import logging
import yaml
import pickle
import mlflow
import json

import os

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)


#configure logger
logger=logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# configure console_handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add console_handler to logger
logger.addHandler(file_handler)

#set formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

#create a function to load data
def load_data(data_path):
    """
    Load a single pickle file.

   
    """
    try:
        logger.info(f"Loading data from: {data_path}")
        data=pd.read_csv(data_path)
        logger.info(f"Original dataset shape: {data.shape}")
        data.dropna(inplace=True)
        logger.info(f"Shape after dropping missing values: {data.shape}")
        logger.info(f"Loaded data successfully from: {data_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file at {data_path}: {e}")
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
def make_X_and_y(data: pd.DataFrame, target_column: str):
    """
    Splits a DataFrame into features (X) and target variable (y).

    Args:
        data (pd.DataFrame): The dataset containing features and target.
        target_column (str): Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tuple containing features (X) and target (y).
    """
    try:
        
        logging.info(f"Creating feature matrix X and target vector y using target column: '{target_column}'")

        # Drop the target column to create feature matrix X
        x = data.drop(columns=[target_column])

        # Select the target column to create target vector y
        y = data[target_column]

        logging.info(f"Features and target split successfully. X shape: {x.shape}, y shape: {y.shape}")
        return x, y

    except KeyError as e:
        logging.error(f"Column '{target_column}' not found in the DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while splitting features and target: {e}")
        raise

#create a function to load preprocessor
def load_preprocessor(preprocessor_path: str):
    """
    Load a preprocessor object from a pickle file.

    Args:
        preprocessor_path (str): Path to the pickle file containing the preprocessor.

    Returns:
        Object: Loaded preprocessor object.
    """
    try:
        logging.info(f"Attempting to load preprocessor from: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logging.info("Preprocessor loaded successfully.")
        return preprocessor
    except FileNotFoundError:
        logging.error(f"File not found: {preprocessor_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the preprocessor: {e}")
        raise

def load_model(model_path: str):
    """
    Load a machine learning model from a pickle file.

    Args:
        model_path (str): Path to the pickle file containing the model.

    Returns:
        object: The loaded model object.
    """
    try:
        logging.info(f"Attempting to load model from: {model_path}")
        with open(model_path, 'rb') as f: 
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"File not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        raise

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)

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

    #set tracking uri
    mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/swiggy-delivery-time-prediction.mlflow')

    #set experiment
    mlflow.set_experiment('dvc-pipeline-run')
    TARGET='time_taken'

    with mlflow.start_run() as run:
         # set tags
        mlflow.set_tag("model","Food Delivery Time Regressor")

        #get root directory
        root_dir=get_root_directory()

         #load data
        test_data=load_data(os.path.join(root_dir,'data','interim','test_data.csv'))
        train_data=load_data(os.path.join(root_dir,'data','interim','train_data.csv'))
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        # Log parameters
        for key, value in params.items():
           logger.info(f'Logging parameter: {key} = {value}')
           mlflow.log_param(key, value)
           logger.info('All parameters logged successfully.')

        preprocessor=load_preprocessor(os.path.join(root_dir,'data','processed','preprocessor.pkl'))
        
        #load_model
        model=load_model(os.path.join(root_dir,'src','model','model.pkl'))

        #split data in x and y
        x_train,y_train=make_X_and_y(train_data,target_column=TARGET)
        x_test,y_test=make_X_and_y(test_data,target_column=TARGET)

       #apply transformation
        logger.info('apply transformation')
        x_train_trans=preprocessor.fit_transform(x_train)
        x_test_trans=preprocessor.transform(x_test)
        logger.info('transformation completed')

        transformer=PowerTransformer()

        #make prediction
        y_train_pred=model.predict(x_train_trans)
        y_test_pred=model.predict(x_test_trans)

           # calculate the train and test mae
        train_mae = mean_absolute_error(y_train,y_train_pred)
        test_mae = mean_absolute_error(y_test,y_test_pred)
        logger.info("error calculated")
    
        # calculate the r2 scores
        train_r2 = r2_score(y_train,y_train_pred)
        test_r2 = r2_score(y_test,y_test_pred)
        logger.info("r2 score calculated")
    
    # calculate cross val scores
        cv_scores = cross_val_score(model,
                                    x_train_trans,
                                    y_train,
                                    cv=5,
                                    scoring="neg_mean_absolute_error",
                                    n_jobs=-1)
        logger.info("cross validation complete")
        
        # mean cross val score
        mean_cv_score = -(cv_scores.mean())

        # log metrics
        mlflow.log_metric("train_mae",train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2",train_r2)
        mlflow.log_metric("test_r2",test_r2)
        mlflow.log_metric("mean_cv_score",-(cv_scores.mean()))

            # log individual cv scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})
            
            # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)
        
        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()
            
        logger.info("Mlflow logging complete and model logged")
            
        # get the run id 
        run_id = run.info.run_id
        model_name = "delivery_time_pred_model"
        
    

        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")
            
            # model signature
        sample_input = x_train.sample(20, random_state=42)
        sample_transformed = preprocessor.transform(sample_input)
        model_signature = mlflow.models.infer_signature(model_input=sample_transformed,
                                                    model_output=model.predict(sample_transformed))

            
            # log the final model
        mlflow.sklearn.log_model(model,"delivery_time_pred_model",signature=model_signature)

        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()
            
        logger.info("Mlflow logging complete and model logged")
            
        # get the run id 
        run_id = run.info.run_id
        model_name = "delivery_time_pred_model"

        # save the model info
        save_json_path = os.path.join(root_dir , "run_information.json")
        save_model_info(save_json_path=save_json_path,
                        run_id=run_id,
                        artifact_path=artifact_uri,
                        model_name=model_name)
        logger.info("Model Information saved")
if __name__ == "__main__":
    main()
    
 


    


    


        

      


        






        





    








    

