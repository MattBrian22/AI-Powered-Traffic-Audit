# import streamlit as st
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import pickle
# from geopy.geocoders import Nominatim
# import os
# from dotenv import load_dotenv


# # Load the trained machine learning model
# with open('model_v5.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # API Configuration for Vivacity Labs
# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("VIVACITY_API_KEY")
# API_URL = "https://api.vivacitylabs.com/countline/counts"
# HEADERS = {
#     "Accept": "application/json",
#     "x-vivacity-api-key": API_KEY
# }

# # Streamlit App Title
# st.title("Real-Time Traffic Counts Dashboard")

# # Sidebar Inputs for Parameters
# st.sidebar.header("Filter Options")
# start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=30))
# end_date = st.sidebar.date_input("End Date", value=datetime.today())
# time_bucket = st.sidebar.selectbox("Select Time Bucket", ["24h", "1h", "30m", "5m", "15m", "30m", "1m", "10m"])
# countline_ids = st.sidebar.text_input("Countline IDs (comma-separated)", "22668,304,305,307,484,485")

# # Fetch and Process Data
# def fetch_data():
#     start_date_iso = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
#     end_date_iso = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
#     params = {
#         "fill_zeros": "true",
#         "countline_ids": countline_ids,
#         "from": start_date_iso,
#         "to": end_date_iso,
#         "time_bucket": time_bucket,
#         "classes": "car,bus,truck,cyclist,pedestrian"
#     }
#     response = requests.get(API_URL, headers=HEADERS, params=params)
#     data = response.json()

#     records_list = []
#     for countline_id, records in data.items():
#         for record in records:
#             from_time = datetime.fromisoformat(record["from"].replace("Z", "+00:00"))
#             to_time = datetime.fromisoformat(record["to"].replace("Z", "+00:00"))
#             inbound_total = sum(record["anti_clockwise"].values())
#             outbound_total = sum(record["clockwise"].values())

#             records_list.append({
#                 "countline_id": countline_id,
#                 "timestamp_from": from_time,
#                 "timestamp_to": to_time,
#                 "inbound": inbound_total,
#                 "outbound": outbound_total
#             })

#     traffic_df = pd.DataFrame(records_list)
#     return traffic_df


# # Process the data and predict using the ML model
# def predict_traffic(traffic_df):
#     # Feature engineering (excluding 'hour' column)
#     traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek

#     # Prepare features (X), excluding 'hour'
#     X = traffic_df[['inbound', 'day_of_week']]  # 'hour' is no longer part of the features
    
#     # Make predictions using the model
#     predictions = model.predict(X)
    
#     # Add predictions to the dataframe
#     traffic_df['predicted_inbound'] = predictions[:, 0]
#     traffic_df['predicted_outbound'] = predictions[:, 1]
    
#     return traffic_df



# # Display the results in Streamlit
# if st.sidebar.button("Fetch Traffic Data"):
#     traffic_df = fetch_data()
#     traffic_df = predict_traffic(traffic_df)

#     st.write("### Traffic Data with Predictions")
#       # Remove rows where inbound or outbound count is zero
#     traffic_df = traffic_df[(traffic_df['inbound'] > 0) & (traffic_df['outbound'] > 0)]
#     st.write(traffic_df)

#     # Display graphical predictions vs actual
#     st.subheader("Predictions vs Actual")
#     st.line_chart(traffic_df[['timestamp_from', 'inbound', 'predicted_inbound']].set_index('timestamp_from'))
#     st.line_chart(traffic_df[['timestamp_from', 'outbound', 'predicted_outbound']].set_index('timestamp_from'))


import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from dotenv import load_dotenv
import holidays
import numpy as np

# Load the trained machine learning model
with open('model_v5.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# API Configuration for Vivacity Labs
load_dotenv()
API_KEY = os.getenv("VIVACITY_API_KEY")
API_URL = "https://api.vivacitylabs.com/countline/counts"
HEADERS = {
    "Accept": "application/json",
    "x-vivacity-api-key": API_KEY
}

# Streamlit App Title
st.title("AI-Powered Real Time Traffic Audit")

# Sidebar Inputs for Parameters
st.sidebar.header("Filter Options")
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=92))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
time_bucket = st.sidebar.selectbox("Select Time Bucket", ["24h", "1h", "30m", "5m", "15m", "30m", "1m", "10m"])
countline_ids = st.sidebar.text_input("Countline IDs (comma-separated)", "22668,304,305,307,484,485")

# Fetch and Process Data
def fetch_data():
    start_date_iso = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_iso = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    params = {
        "fill_zeros": "true",
        "countline_ids": countline_ids,
        "from": start_date_iso,
        "to": end_date_iso,
        "time_bucket": time_bucket,
        "classes": "car,bus,truck,cyclist,pedestrian"
    }
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


# Process the data and predict using the ML model
def predict_traffic(traffic_df):
    # Feature engineering
    traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek
    traffic_df['month'] = traffic_df['timestamp_from'].dt.month
    traffic_df['week_of_year'] = traffic_df['timestamp_from'].dt.isocalendar().week

    # Adding holiday feature
    UK_holidays = holidays.UK(years=[2024])  # Adjust for your country if needed
    holiday_dates = list(UK_holidays.keys())
    traffic_df['is_holiday'] = traffic_df['timestamp_from'].dt.date.isin(holiday_dates)

    # Prepare features (X)
    X = traffic_df[['inbound', 'day_of_week', 'month', 'week_of_year', 'is_holiday']]

    # Make predictions using the model
    predictions = model.predict(X)
    
    # Add predictions to the dataframe
    traffic_df['predicted_inbound'] = predictions[:, 0]
    traffic_df['predicted_outbound'] = predictions[:, 1]
    
    return traffic_df


# Display the results in Streamlit
if st.sidebar.button("Fetch Traffic Data"):
    traffic_df = fetch_data()
    traffic_df = predict_traffic(traffic_df)

    st.write("### Traffic Data with Predictions")
    # Remove rows where inbound or outbound count is zero
    traffic_df = traffic_df[(traffic_df['inbound'] > 0) & (traffic_df['outbound'] > 0)]
    st.write(traffic_df)

    # Display graphical predictions vs actual
    st.subheader("Predictions vs Actual")
    st.line_chart(traffic_df[['timestamp_from', 'inbound', 'predicted_inbound']].set_index('timestamp_from'))
    st.line_chart(traffic_df[['timestamp_from', 'outbound', 'predicted_outbound']].set_index('timestamp_from'))
