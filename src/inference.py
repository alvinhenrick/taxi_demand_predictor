from datetime import datetime, timedelta
from pdb import set_trace as stop
import os

import requests
import geopandas as gpd
import hopsworks
import pandas as pd
import numpy as np
# from dotenv import load_dotenv

import src.config as config

# load key-value pairs from .env file as environment variables
# load_dotenv()

# HOPSWORKS_PROJECT_NAME = 'paulescu'
# # FEATURE_GROUP_NAME = 'taxi_demand_time_series_hourly_fg'
# # FEATURE_GROUP_VERSION = 11
# FEATURE_VIEW_NAME = 'taxi_demand_time_series_hourly_fv'
# FEATURE_VIEW_VERSION = 11
# MODEL_NAME = 'lightgbm_regressor_taxi_demand_next_hour'
# MODEL_VERSION = 5

project = hopsworks.login(
    project=config.HOPSWORKS_PROJECT_NAME,
    api_key_value=config.HOPSWORKS_API_KEY
)
feature_store = project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results


def load_batch_of_features_from_store(
    current_date: datetime,    
) -> pd.DataFrame:

    n_features = 24*7*4

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == n_features*len(location_ids), "Time-series data is not complete"

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    print(f'{ts_data=}')

    # transpose time-series data as a feature vector, for each `location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    
    # print(f'{features=}')

    return features
    

def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model