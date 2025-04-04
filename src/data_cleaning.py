# importing libraries

import numpy as np
import pandas as pd
from pathlib import Path
import logging

import logging

#create logger
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.DEBUG)

#configure console handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

#add handler to logger 
logger.addHandler(handler)

#create formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#add formatter to handler
handler.setFormatter(formatter)



def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df  # Ensure the dataframe is returned
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return pd.DataFrame()  # Return an empty DataFrame instead of None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def change_column_name(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(str.lower, axis=1).rename(
        columns={
            'delivery_person_id': 'delivery_ID',
            'delivery_person_age': 'person_age',
            'delivery_person_ratings': 'person_ratings',
            'restaurant_latitude': 'latitude',
            'restaurant_longitude': 'longitude',
            'order_date': 'order_date',
            'time_orderd': 'order_time',
            'time_order_picked': 'order_picked_time',
            'weatherconditions': 'weather_condition',
            'road_traffic_density': 'traffic_density',
            'vehicle_condition': 'vehicle_condition',
            'type_of_order': 'order_type',
            'type_of_vehicle': 'vehicle_type',
            'festival': 'festival',
            'city': 'city',
            'time_taken(min)': 'time_taken'
        }
    )

def assign_time_slot(order_hour):
    """Categorize order time into different time slots."""
    if pd.isna(order_hour):
        return np.nan  
    if 6 <= order_hour < 12:
        return "Morning"
    elif 12 <= order_hour < 17:
        return "Afternoon"
    elif 17 <= order_hour < 20:
        return "Evening"
    else:
        return "Night"

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the given DataFrame."""
    logger.info("Starting data cleaning process")
    
    lower_age_indexes = data[data['person_age'].astype(float) < 18].index.to_list()
    high_rating_indexes = data[data['person_ratings'].astype(float) > 5].index.to_list()
    
    data = data.copy()
    
    data.replace(['NaN ', ' nan', '', 'NaN', 'None', ' null'], np.nan, inplace=True)
    data['person_age'] = data['person_age'].astype(float)
    data['person_ratings'] = data['person_ratings'].astype(float)
    
    if 'weather_condition' in data.columns:
        data['weather_condition'] = (
            data['weather_condition']
            .astype(str)
            .str.replace('conditions', '', regex=False)
            .str.lower()
            .str.strip()
            .replace(['nan', '', ' nan', 'NaN', 'None', 'null'], np.nan)
        )
    
    data['city_name'] = data['delivery_ID'].str.split('RES').str[0]
    
    data.drop(index=lower_age_indexes, inplace=True)
    data.drop(index=high_rating_indexes, inplace=True)
    
    data['latitude'] = data['latitude'].astype(float).abs()
    data['longitude'] = data['longitude'].astype(float).abs()
    data['delivery_location_latitude'] = data['delivery_location_latitude'].astype(float).abs()
    data['delivery_location_longitude'] = data['delivery_location_longitude'].astype(float).abs()
    
    data['order_date'] = pd.to_datetime(data['order_date'], dayfirst=True)
    
    data['day'] = data['order_date'].dt.day
    data['month'] = data['order_date'].dt.month
    data['day_of_week'] = data['order_date'].dt.day_name()
    
    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    data['order_time'] = pd.to_datetime(data['order_time'], format="mixed")
    data['order_picked_time'] = pd.to_datetime(data['order_picked_time'], format="mixed")
    
    data['pickup_time_in_minutes'] = (data['order_picked_time'] - data['order_time']).dt.seconds / 60
    
    data['order_time_hr'] = data['order_time'].dt.hour
    data['time_slot'] = data['order_time_hr'].apply(assign_time_slot)
    
    if 'order_time' in data.columns:
        data['order_time'] = data['order_time'].dt.strftime('%H:%M:%S')
    if 'order_picked_time' in data.columns:
        data['order_picked_time'] = data['order_picked_time'].dt.strftime('%H:%M:%S')
    
    for col in ['vehicle_type', 'festival', 'city']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.rstrip().str.lower()
    
    if 'multiple_deliveries' in data.columns:
        data['multiple_deliveries'] = data['multiple_deliveries'].astype(float)
    
    if 'time_taken' in data.columns:
        data['time_taken'] = data['time_taken'].astype(str).str.split().str[1]
    
    data.drop(columns=['id', 'delivery_ID', 'order_date', 'order_time', 'order_picked_time'], inplace=True)
    
    logger.info("Data cleaning process completed")
    return data

def clean_lat_long(data: pd.DataFrame, threshold=1):
    logger.info("Cleaning latitude and longitude values")
    location_subset = data.loc[:, ['latitude', 'longitude', 'delivery_location_latitude', 'delivery_location_longitude']]
    location_columns = location_subset.columns.tolist()
    
    cleaned_data = data.assign(**{
        col: (
            np.where(data[col] < threshold, np.nan, data[col].values)
        )
        for col in location_columns
    })
    
    logger.info("Latitude and longitude values cleaned successfully")
    return cleaned_data

def calculate_haversine_distance(data):
    logger.info("Calculating haversine distance")
    location_subset = data.loc[:, ['latitude', 'longitude', 'delivery_location_latitude', 'delivery_location_longitude']]
    location_columns = location_subset.columns.tolist()
    lat1 = data[location_columns[0]]
    lon1 = data[location_columns[1]]
    lat2 = data[location_columns[2]]
    lon2 = data[location_columns[3]]
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c
    
    logger.info("Haversine distance calculation completed")
    return data.assign(distance=distance)
    
def assign_distance_type(distance):
    """Categorize distance into different ranges: short, medium, long, very long."""
    if pd.isna(distance):
        return np.nan  # Return NaN for missing values
    if not isinstance(distance, (int, float)):  # Ensure distance is a number
        return np.nan
    if distance < 0:
        return "invalid"  # Handling negative distances
    if 0 <= distance < 5:
        return "short"
    elif 5 <= distance < 10:
        return "medium"
    elif 10 <= distance < 15:
        return "long"
    else:
        return "very long"
    


def perform_data_cleaning(data: pd.DataFrame, saved_data_path: str) -> None:
    logger.info("Starting full data processing pipeline")

    # Ensure column names are renamed first
    data = change_column_name(data)

    cleaned_data = (data.pipe(data_cleaning)
                    .pipe(clean_lat_long)
                    .pipe(calculate_haversine_distance)
                    .assign(distance_type=lambda df: df["distance"].apply(assign_distance_type)))

    cleaned_data.to_csv(saved_data_path, index=False)
    logger.info(f"Cleaned data saved to {saved_data_path}")

    
if __name__ == "__main__":
    # root path
    root_path = Path(r"C:\Users\redhu\swiggy_delivery_time_prediction\swiggy-delivery-time-prediction").resolve()
    print("Project Root:", root_path)
   
    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exits
    cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
    # cleaned data file name
    cleaned_data_filename = "swiggy_cleaned.csv"
    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
    # data load path
    data_load_path = root_path / "data" / "raw" /'swiggy.csv'
    
    # load the data
    df = load_data(data_load_path)
    logger.info("Data read successfully")

     # clean the data and save
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
    logger.info("Data cleaned and saved")
    
   




    






