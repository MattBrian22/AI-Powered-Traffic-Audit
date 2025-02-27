import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import numpy as np
import pickle

# # API Configuration
# Load environment variables
load_dotenv()
API_KEY = os.getenv("VIVACITY_API_KEY")
API_URL = "https://api.vivacitylabs.com/countline/counts"
HEADERS = {
    "Accept": "application/json",
    "x-vivacity-api-key": API_KEY
}

# Set start and end date
start_date = datetime.today() - timedelta(days=92)
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

# Set the time bucket to 24 hours
time_bucket = "24h"

# Query Parameters for API request (including time_bucket)
params = {
    "fill_zeros": "true",  
    "countline_ids": countline_ids,
    "from": start_date_iso,
    "to": end_date_iso,
    "classes": ",".join(classes),
    "time_bucket": time_bucket
}

# Function to fetch and process data
def fetch_data():
    # Send GET request to API
    response = requests.get(API_URL, headers=HEADERS, params=params)

    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print(f"Response Text: {response.text}")
    else:
        data = response.json()
        print("Data fetched successfully!")

        # Process the JSON response into a DataFrame
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

        # Remove rows where inbound or outbound count is zero
        traffic_df = traffic_df[(traffic_df['inbound'] > 0) & (traffic_df['outbound'] > 0)]

        # # Feature Engineering
        # traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek
        # traffic_df['hour'] = traffic_df['timestamp_from'].dt.hour

        # # Prepare features (X) and target (y)
        # X = traffic_df[['inbound', 'day_of_week', 'hour']]
        # y = traffic_df[['inbound', 'outbound']]  # Multi-output regression

        # # Feature Engineering
        # traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek

        # # Prepare features (X) and target (y)
        # X = traffic_df[['inbound', 'day_of_week']]  # Removed 'hour' column
        # y = traffic_df[['inbound', 'outbound']]  # Multi-output regression


        # # Split data into training and test sets
        # train_size = int(len(traffic_df) * 0.8)
        # X_train, X_test = X[:train_size], X[train_size:]
        # y_train, y_test = y[:train_size], y[train_size:]

        # Example for US holidays (adjust for your country)
        UK_holidays = holidays.UK(years=[2024])  # Get US holidays for the year 2024
        holiday_dates = list(UK_holidays.keys())

        # Feature Engineering: Adding seasonal features
        traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek
        traffic_df['month'] = traffic_df['timestamp_from'].dt.month
        traffic_df['week_of_year'] = traffic_df['timestamp_from'].dt.isocalendar().week
        traffic_df['is_holiday'] = traffic_df['timestamp_from'].dt.date.isin(holiday_dates)

        # Prepare features (X) and target (y)
        X = traffic_df[['inbound', 'day_of_week', 'month', 'week_of_year', 'is_holiday']]
        y = traffic_df[['inbound', 'outbound']]  # Multi-output regression

        # Train/Test split
        train_size = int(len(traffic_df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]


        # Create the XGBoost and RandomForest models
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)


        # Create a Voting Regressor combining both models
        voting_model = VotingRegressor([('xgb', xgb_model), ('rf', rf_model)])

        # Use MultiOutputRegressor for multi-output regression
        model = MultiOutputRegressor(voting_model)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate MAE, RMSE, and percentage error for both inbound and outbound predictions
        mae_inbound = mean_absolute_error(y_test['inbound'], y_pred[:, 0])
        rmse_inbound = np.sqrt(mean_squared_error(y_test['inbound'], y_pred[:, 0]))

        mae_outbound = mean_absolute_error(y_test['outbound'], y_pred[:, 1])
        rmse_outbound = np.sqrt(mean_squared_error(y_test['outbound'], y_pred[:, 1]))

        perc_error_inbound = np.mean(np.abs((y_test['inbound'] - y_pred[:, 0]) / y_test['inbound']) * 100)
        perc_error_outbound = np.mean(np.abs((y_test['outbound'] - y_pred[:, 1]) / y_test['outbound']) * 100)

        # Display MAE, RMSE, and percentage errors
        print(f"Mean Absolute Error (MAE) for Inbound: {mae_inbound:.2f}")
        print(f"Root Mean Squared Error (RMSE) for Inbound: {rmse_inbound:.2f}")
        print(f"Percentage Error for Inbound: {perc_error_inbound:.2f}%")

        print(f"Mean Absolute Error (MAE) for Outbound: {mae_outbound:.2f}")
        print(f"Root Mean Squared Error (RMSE) for Outbound: {rmse_outbound:.2f}")
        print(f"Percentage Error for Outbound: {perc_error_outbound:.2f}%")

        # Save the trained model to disk
        with open('model_v5.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)

# Fetch data and train the model
fetch_data()

