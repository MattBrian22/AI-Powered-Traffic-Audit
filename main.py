import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from dotenv import load_dotenv
import numpy as np

# API Configuration
load_dotenv()
API_KEY = os.getenv("VIVACITY_API_KEY")
API_URL = "https://api.vivacitylabs.com/countline/counts"
HEADERS = {
    "Accept": "application/json",
    "x-vivacity-api-key": API_KEY
}

# Streamlit App Title
st.title("Real-Time Traffic Counts Dashboard")

# Sidebar Inputs for Parameters
st.sidebar.header("Filter Options")

# Date input for start and end date
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Convert datetime to ISO format for API query
start_date_iso = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
end_date_iso = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

# Time bucket input (e.g., 24h, 1h, 30m, etc.)
time_bucket = st.sidebar.selectbox(
    "Select Time Bucket",
    ["24h", "1h", "30m", "5m", "15m", "30m", "1m", "10m"]
)

# Input for countline_ids (comma-separated)
countline_ids = st.sidebar.text_input("Countline IDs (comma-separated)", "22668,304,305,307,484,485")

# Display the selected options
st.sidebar.write(f"Start Date: {start_date}")
st.sidebar.write(f"End Date: {end_date}")
st.sidebar.write(f"Time Bucket: {time_bucket}")
st.sidebar.write(f"Countline IDs: {countline_ids}")

# Query Parameters for API request
params = {
    "fill_zeros": "true",  
    "countline_ids": countline_ids,
    "from": start_date_iso,
    "to": end_date_iso,
    "time_bucket": time_bucket,
    "classes": "car,bus,truck,cyclist,pedestrian"
}

# Function to fetch and process data
def fetch_data():
    with st.spinner("Fetching data..."):
        # Send GET request to API
        response = requests.get(API_URL, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
            st.write(f"Response Text: {response.text}")
        else:
            data = response.json()
            st.success("Data fetched successfully!")

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
            return traffic_df

# Load the trained stacking model
with open('stacking_model.pkl', 'rb') as model_file:
    stacking_model = pickle.load(model_file)

# Feature engineering and prediction
def predict_traffic(traffic_df):
    traffic_df['day_of_week'] = traffic_df['timestamp_from'].dt.dayofweek
    X = traffic_df[['inbound', 'day_of_week']]  # Features from data
    
    # Make predictions
    predictions = stacking_model.predict(X)
    
    # Add predictions to the dataframe
    traffic_df['predicted_inbound'] = predictions[:, 0]
    traffic_df['predicted_outbound'] = predictions[:, 1]
    
    return traffic_df

# Main function to execute Streamlit
if st.sidebar.button("Fetch Traffic Data"):
    traffic_df = fetch_data()
    
    if traffic_df is not None:
        traffic_df = predict_traffic(traffic_df)
        st.write("### Traffic Data with Predictions")
        st.write(traffic_df)

        # Display graphical predictions vs actual
        st.subheader("Predictions vs Actual")
        st.line_chart(traffic_df[['timestamp_from', 'inbound', 'predicted_inbound']].set_index('timestamp_from'))
        st.line_chart(traffic_df[['timestamp_from', 'outbound', 'predicted_outbound']].set_index('timestamp_from'))
