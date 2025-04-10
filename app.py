import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from notebooks.data_cleaning_utils_py import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='swiggy-delivery-time-prediction', mlflow=True)

 #set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/swiggy-delivery-time-prediction.mlflow')

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        return run_info
        
    
def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

num_cols = ['person_age', 'person_ratings', 'pickup_time_in_minutes', 'distance']
num_cat_cols = ['weather_condition', 'order_type', 'vehicle_type', 'multiple_deliveries', 'city', 'festival',
                'city_name', 'day_of_week', 'time_slot']
ordinal_cat_cols = ['traffic_density', 'distance_type']
traffic_order = ['medium', 'high', 'jam', 'low']
distance_order = ['short', 'medium', 'long', 'very long']

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "staging"

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)



# load the preprocessor
preprocessor_path = "data/processed/preprocessor.pkl"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])
# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data).dropna()
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions
   
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
   
    




