# Importing necessary libraries
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os

# Configure logging
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.DEBUG)

# Console handler setup
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter for console logs
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)  # Add the handler to the logger


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from: {data_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return pd.DataFrame()
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
            'Delivery_location_latitude':'delivery_location_latitude',
            'Delivery_location_longitude':'delivery_location_longitude',
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
        return np.nan  # Return NaN for missing values
    if 6 <= order_hour < 12:
        return "Morning"
    elif 12 <= order_hour < 17:
        return "Afternoon"
    elif 17 <= order_hour < 20:
        return "Evening"
    else:
        return "Night"

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
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
    if 'traffic_density' in data.columns:
        data['traffic_density'] = data['traffic_density'].astype(str).str.lower().str.strip()
    return data

def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_subset=data.loc[:,['latitude','longitude','delivery_location_latitude','delivery_location_longitude']]
    location_columns = location_subset.columns.tolist()

    return (
        data
        .assign(**{
            col: (
                np.where(data[col] < threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )

def calculate_haversine_distance(data):
    location_subset=data.loc[:,['latitude','longitude','delivery_location_latitude','delivery_location_longitude']]
    location_columns = location_subset.columns.tolist()
    lat1 = data[location_columns[0]]
    lon1 = data[location_columns[1]]
    lat2 = data[location_columns[2]]
    lon2 = data[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        data.assign(
            distance = distance)
    )

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

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=[
        'id', 'delivery_ID', 'order_date', 'order_time', 'order_picked_time',
        'latitude', 'longitude', 'delivery_location_latitude', 'delivery_location_longitude',
        'order_time_hr', 'day', 'month'
    ], errors='ignore')

def perform_data_cleaning(data: pd.DataFrame, save_path: str):
    """Run full data cleaning pipeline and save cleaned data."""
    try:
        logger.info("Starting data cleaning process...")
        cleaned_data = (
            data.pipe(change_column_name)
                .pipe(data_cleaning)
                .pipe(clean_lat_long)
                .pipe(calculate_haversine_distance)
                .assign(distance_type=lambda df: df["distance"].apply(assign_distance_type))
                .pipe(drop_columns)
        )
        cleaned_data.to_csv(save_path, index=False)
        logger.info(f"Cleaned data saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise
    
    
def ensure_directory_exists(directory: Path) -> None:
    """Ensure that the output directory exists, create if not."""
    try:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error ensuring directory exists: {directory} - {e}")
        raise


def get_root_directory() -> Path:
    """Get the root directory (two levels up from the current script)."""
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent.parent
        logger.debug(f"Current directory: {current_dir}")
        logger.debug(f"Resolved root directory: {root_dir}")
        return root_dir
    except Exception as e:
        logger.error(f"Error getting root directory: {e}")
        raise


def main():
    """Main pipeline for data loading, cleaning, and saving."""
    logger.info("Starting Swiggy delivery time data cleaning pipeline...")

    try:
        root_path=get_root_directory()

        # Step 2: Define file paths
        data_load_path = root_path/'data'/'raw'/'swiggy.csv'
        cleaned_data_save_dir = root_path/'data'/'cleaned'
        ensure_directory_exists(cleaned_data_save_dir)
        cleaned_data_save_path = cleaned_data_save_dir / "swiggy_cleaned.csv"

        # Step 3: Load the data
        data = load_data(str(data_load_path))
        if data.empty:
            logger.warning("No data to clean. Exiting pipeline.")
            return

        logger.info("Data successfully loaded. Starting cleaning...")

        # Step 4: Clean and save the data
        perform_data_cleaning(data, str(cleaned_data_save_path))
        logger.info("Pipeline executed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed due to: {e}")


if __name__ == "__main__":
    main()




    
   




    






