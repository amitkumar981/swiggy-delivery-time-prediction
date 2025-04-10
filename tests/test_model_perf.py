import pytest
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
import dagshub
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer

# Init DagsHub + MLflow
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/swiggy-delivery-time-prediction.mlflow')
client = MlflowClient()

def model_load_information(file_path):
    with open(file_path, 'r') as f:
        run_info = json.load(f)
    return run_info

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    return root_dir

# Load run info
root_path = get_root_directory()
file_path = os.path.join(root_path, 'run_information.json')
run_info = model_load_information(file_path)

model_name = run_info['model_name']
stage = 'staging'

# Load model
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

def load_preprocessor(preprocessor_path:str):
    with open(preprocessor_path,'rb') as f:
        preprocessor=pickle.load(f)
        return preprocessor
    


preprocessor_path=os.path.join(root_path,'data','processed','preprocessor.pkl')
data_path=os.path.join(root_path,'data','interim','test_data.csv')

#load preprocessor
preprocessor=load_preprocessor(preprocessor_path)
transformer=PowerTransformer()

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])


@pytest.mark.parametrize(argnames="model_pipe, data_path, threshold_error",
                        argvalues=[(model_pipe, data_path,5)])

def test_model_performance(model_pipe,data_path,threshold_error):

    #load data set
    df=pd.read_csv(data_path)

    # drop missing values
    df.dropna(inplace=True)

    #split x and y
    x=df.drop(columns=['time_taken'])
    y=df['time_taken']

    #get predictions
    y_pred=model_pipe.predict(x)

    #calculate error
    mean_error=mean_absolute_error(y_pred,y)

       # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The {model_name} model passed the performance test")




















