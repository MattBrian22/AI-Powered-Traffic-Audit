import requests
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from dotenv import load_dotenv
import numpy as np
import pickle

# API Configuration
load_dotenv()
API_KEY = os.getenv("VIVACITY_API_KEY")
API_URL = "https://api.vivacitylabs.com/countline/counts"
HEADERS = {
    "Accept": "application/json",
    "x-vivacity-api-key": API_KEY
}

# Set start and end date
start_date = datetime.today() - timedelta(days=7)
end_date = datetime.today()

# Align the start and end dates to full 24-hour periods
start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=0)

# Convert datetime to ISO format for API query
start_date_iso = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
end_date_iso = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

# Input for countline_ids (comma-separated)
countline_ids = "22668,304,305,307,484,485"

# Classes to fetch data for
classes = ["car", "bus", "truck", "cyclist", "pedestrian"]

# Query Parameters for API request
params = {
    "fill_zeros": "true",  
    "countline_ids": countline_ids,
    "from": start_date_iso,
    "to": end_date_iso,
    "classes": ",".join(classes),
    "time_bucket": "24h"
}

# Function to fetch and process data
def fetch_data():
    response = requests.get(API_URL, headers=HEADERS, params=params)
    data = response.json()

    records_list = []
    for countline_id, records in data.items():
        for record in records:
            from_time = datetime.fromisoformat(record["from"].replace("Z", "+00:00"))
            to_time = datetime.fromisoformat(record["to"].replace("Z", "+00:00"))
            
            inbound_total = sum(record["anti_clockwise"].values())
            outbound_total = sum(record["clockwise"].values())
            
            records_list.append({
                "countline_id": countline_id,
                "timestamp_from": from_time,
                "timestamp_to": to_time,
                "inbound": inbound_total,
                "outbound": outbound_total
            })

    traffic_df = pd.DataFrame(records_list)
    return traffic_df

# Feature engineering and preparing data
def process_data(traffic_df):
    traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek
    X = traffic_df[['inbound', 'day_of_week']]
    y = traffic_df[['inbound', 'outbound']]
    return X, y

# Train a stacking model
def train_stacking_model(X_train, y_train):
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    meta_model = LinearRegression()

    stacking_model = StackingRegressor(
        estimators=[('xgb', xgb_model), ('rf', rf_model)], 
        final_estimator=meta_model
    )
    
    # Wrap with MultiOutputRegressor to handle multiple outputs (inbound, outbound)
    model = MultiOutputRegressor(stacking_model)
    model.fit(X_train, y_train)

    with open('stacking_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    return model

# Main function to execute training
def main():
    traffic_df = fetch_data()
    X, y = process_data(traffic_df)
    train_size = int(len(traffic_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train and save the model
    model = train_stacking_model(X_train, y_train)

    # Optionally, calculate and print MAE, RMSE for the test set
    y_pred = model.predict(X_test)
    mae_inbound = mean_absolute_error(y_test['inbound'], y_pred[:, 0])
    rmse_inbound = np.sqrt(mean_squared_error(y_test['inbound'], y_pred[:, 0]))

    mae_outbound = mean_absolute_error(y_test['outbound'], y_pred[:, 1])
    rmse_outbound = np.sqrt(mean_squared_error(y_test['outbound'], y_pred[:, 1]))

    print(f"MAE Inbound: {mae_inbound}")
    print(f"RMSE Inbound: {rmse_inbound}")
    print(f"MAE Outbound: {mae_outbound}")
    print(f"RMSE Outbound: {rmse_outbound}")

if __name__ == '__main__':
    main()