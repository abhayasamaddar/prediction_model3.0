import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import requests
import json
import urllib.parse

warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Supabase configuration
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

# Blynk API configuration
BLYNK_API_TOKEN = "pbHd8QA0u4enaLQZHhQwqoHN0rKMXsK7"
BLYNK_UPDATE_BASE_URL = "https://blynk.cloud/external/api/update"
BLYNK_GET_BASE_URL = "https://blynk.cloud/external/api/get"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=300)
def load_data():
    try:
        supabase = init_supabase()
        response = supabase.table('airquality').select('*').execute()
        
        if not response.data:
            st.error("No data found in the database.")
            return pd.DataFrame()
            
        df = pd.DataFrame(response.data)
        
        # Convert columns to appropriate data types
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at')
        
        # Convert numeric columns
        numeric_cols = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with forward fill and then backward fill
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # If there are still missing values, fill with column mean
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def test_blynk_connection():
    """Test if Blynk API is accessible and check available pins"""
    try:
        # Test with virtual pin V0 (most common)
        test_url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&V0=1"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            return True, "Connection successful - V0 is available"
        else:
            # Try with lowercase v0
            test_url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&v0=1"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                return True, "Connection successful - v0 is available"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"

def get_available_blynk_pins():
    """Try to determine which virtual pins are available"""
    available_pins = []
    
    # Test common virtual pins (V0 through V9)
    for pin in range(10):
        try:
            test_url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&V{pin}=0"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                available_pins.append(f"V{pin}")
                
            # Also test lowercase
            test_url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&v{pin}=0"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                available_pins.append(f"v{pin}")
        except:
            continue
    
    return available_pins

def send_to_blynk_simple(predictions_data, selected_targets, models_to_send):
    """Send prediction data to Blynk using virtual pins"""
    try:
        if models_to_send and len(models_to_send) > 0:
            primary_model = models_to_send[0]
            
            # Get first hour predictions
            blynk_data = {}
            for target in selected_targets:
                if (primary_model in predictions_data and 
                    target in predictions_data[primary_model] and 
                    len(predictions_data[primary_model][target]) > 0):
                    value = predictions_data[primary_model][target][0]
                    if not np.isnan(value):
                        blynk_data[target] = float(value)
            
            # Map our parameters to virtual pins
            # You'll need to adjust this mapping based on your Blynk project setup
            pin_mapping = {
                'temperature': 'V0',
                'humidity': 'V1', 
                'co2': 'V2',
                'co': 'V3',
                'pm25': 'V4',
                'pm10': 'V5',
                'rain': 'V6',
                'light': 'V7'
            }
            
            # Add default values for missing parameters
            if 'rain' not in blynk_data:
                blynk_data['rain'] = 0
            if 'light' not in blynk_data:
                blynk_data['light'] = 0
            
            success_count = 0
            total_params = len(blynk_data)
            
            st.write("📤 Attempting to send data to Blynk...")
            
            for param, value in blynk_data.items():
                pin_name = pin_mapping.get(param)
                if pin_name:
                    try:
                        # Format: https://blynk.cloud/external/api/update?token=YOUR_TOKEN&V0=25.6
                        url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&{pin_name}={value}"
                        st.write(f"Trying: {url}")
                        
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            success_count += 1
                            st.success(f"✅ Sent {param} ({value}) to {pin_name}")
                        else:
                            st.error(f"❌ Failed to send {param} to {pin_name}: HTTP {response.status_code}")
                            st.write(f"Response: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error sending {param}: {str(e)}")
                else:
                    st.warning(f"⚠️ No pin mapping for {param}")
            
            if success_count > 0:
                st.success(f"✅ Successfully sent {success_count}/{total_params} parameters to Blynk!")
                return True
            else:
                st.error("❌ Failed to send any parameters to Blynk")
                return False
        else:
            st.warning("No valid prediction data to send to Blynk")
            return False
            
    except Exception as e:
        st.error(f"❌ Error sending data to Blynk: {e}")
        return False

def send_to_blynk_batch(predictions_data, selected_targets, models_to_send):
    """Send all data as a batch to a single virtual pin as JSON"""
    try:
        if models_to_send and len(models_to_send) > 0:
            primary_model = models_to_send[0]
            
            # Create a comprehensive data structure
            blynk_payload = {
                "timestamp": datetime.now().isoformat(),
                "model_used": primary_model,
                "current_predictions": {},
                "hourly_predictions": []
            }
            
            # Add current (first hour) predictions
            for target in selected_targets:
                if (primary_model in predictions_data and 
                    target in predictions_data[primary_model] and 
                    len(predictions_data[primary_model][target]) > 0):
                    value = predictions_data[primary_model][target][0]
                    if not np.isnan(value):
                        blynk_payload["current_predictions"][target] = float(value)
            
            # Add hourly predictions for next 24 hours
            for hour in range(24):
                hour_data = {"hour": hour + 1}
                for target in selected_targets:
                    if (primary_model in predictions_data and 
                        target in predictions_data[primary_model] and 
                        len(predictions_data[primary_model][target]) > hour):
                        value = predictions_data[primary_model][target][hour]
                        if not np.isnan(value):
                            hour_data[target] = float(value)
                blynk_payload["hourly_predictions"].append(hour_data)
            
            # Add default values
            if 'rain' not in blynk_payload["current_predictions"]:
                blynk_payload["current_predictions"]['rain'] = 0
            if 'light' not in blynk_payload["current_predictions"]:
                blynk_payload["current_predictions"]['light'] = 0
            
            # Convert to JSON and send to V0
            json_data = json.dumps(blynk_payload)
            
            # Try different virtual pins
            test_pins = ['V0', 'v0', 'V1', 'v1', 'V2', 'v2']
            
            for pin in test_pins:
                try:
                    url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&{pin}={urllib.parse.quote(json_data)}"
                    st.write(f"Trying to send batch data to {pin}...")
                    
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        st.success(f"✅ Successfully sent batch data to {pin}!")
                        st.json(blynk_payload)
                        return True
                    else:
                        st.write(f"❌ Failed to send to {pin}: HTTP {response.status_code}")
                except Exception as e:
                    st.write(f"❌ Error sending to {pin}: {str(e)}")
            
            st.error("❌ Failed to send batch data to any virtual pin")
            return False
        else:
            st.warning("No valid prediction data to send to Blynk")
            return False
            
    except Exception as e:
        st.error(f"❌ Error sending batch data to Blynk: {e}")
        return False

def send_to_blynk_individual_pins(predictions_data, selected_targets, models_to_send):
    """Send each parameter to individual virtual pins"""
    try:
        if models_to_send and len(models_to_send) > 0:
            primary_model = models_to_send[0]
            
            # Get first hour predictions
            blynk_data = {}
            for target in selected_targets:
                if (primary_model in predictions_data and 
                    target in predictions_data[primary_model] and 
                    len(predictions_data[primary_model][target]) > 0):
                    value = predictions_data[primary_model][target][0]
                    if not np.isnan(value):
                        blynk_data[target] = float(value)
            
            # Add default values
            if 'rain' not in blynk_data:
                blynk_data['rain'] = 0
            if 'light' not in blynk_data:
                blynk_data['light'] = 0
            
            success_count = 0
            total_params = len(blynk_data)
            
            st.write("📤 Testing individual virtual pins...")
            
            # Test each parameter with multiple pin options
            for param, value in blynk_data.items():
                sent = False
                
                # Try different pin naming conventions
                pin_options = [
                    f"V{i}" for i in range(10)  # V0, V1, V2, ...
                ] + [
                    f"v{i}" for i in range(10)  # v0, v1, v2, ...
                ]
                
                for pin in pin_options:
                    try:
                        url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&{pin}={value}"
                        response = requests.get(url, timeout=5)
                        
                        if response.status_code == 200:
                            success_count += 1
                            st.success(f"✅ Sent {param} ({value}) to {pin}")
                            sent = True
                            break  # Stop trying other pins for this parameter
                        elif response.status_code == 400:
                            continue  # Try next pin
                        else:
                            st.write(f"❌ {pin}: HTTP {response.status_code}")
                    except:
                        continue
                
                if not sent:
                    st.error(f"❌ Could not send {param} to any virtual pin")
            
            if success_count > 0:
                st.success(f"✅ Successfully sent {success_count}/{total_params} parameters!")
                return True
            else:
                st.error("❌ Failed to send any parameters")
                return False
        else:
            st.warning("No valid prediction data to send to Blynk")
            return False
            
    except Exception as e:
        st.error(f"❌ Error testing pins: {e}")
        return False

# ... (keep all the existing ML functions: create_features, prepare_lstm_data, train_random_forest, etc.)
# [Include all the ML functions from the previous code here - they remain unchanged]

def main():
    st.set_page_config(page_title="Air Quality Prediction", layout="wide")
    
    st.title("🌤️ Air Quality Prediction Dashboard")
    st.markdown("""
    This app predicts future values of PM2.5, PM10, CO2, CO, Temperature, and Humidity using multiple machine learning models.
    **All data is used for training** and continuous predictions are shown in hourly and weekly plots.
    """)
    
    # Blynk API Configuration
    st.sidebar.header("🌐 Blynk API Configuration")
    enable_blynk = st.sidebar.checkbox("Enable Blynk API Integration", value=True)
    
    if enable_blynk:
        st.sidebar.info("Blynk API is configured with your token")
        
        # Test Blynk connection
        if st.sidebar.button("Test Blynk Connection"):
            with st.sidebar:
                with st.spinner('Testing Blynk connection...'):
                    success, message = test_blynk_connection()
                    if success:
                        st.success("Blynk connection successful!")
                        st.write(message)
                        
                        # Show available pins
                        with st.spinner('Checking available pins...'):
                            available_pins = get_available_blynk_pins()
                            if available_pins:
                                st.success(f"Available pins: {', '.join(available_pins)}")
                            else:
                                st.warning("No virtual pins found. You may need to create them in your Blynk project.")
                    else:
                        st.error(f"Blynk connection failed: {message}")
        
        # Blynk send method selection
        blynk_send_method = st.sidebar.radio(
            "Blynk Send Method:",
            ["Simple Virtual Pins", "Batch JSON Data", "Auto-Detect Pins"],
            index=0,
            help="Choose how to send data to Blynk"
        )
    
    # Load data and run ML pipeline (keep all the existing code from here)
    # [Include all the existing main function code for data loading, ML training, etc.]
    
    # In the Blynk sending section, replace with:
    if enable_blynk and future_predictions:
        st.subheader("🌐 Sending to Blynk API")
        with st.spinner('Sending prediction data to Blynk...'):
            if blynk_send_method == "Simple Virtual Pins":
                success = send_to_blynk_simple(future_predictions, selected_targets, list(all_models.keys()))
            elif blynk_send_method == "Batch JSON Data":
                success = send_to_blynk_batch(future_predictions, selected_targets, list(all_models.keys()))
            else:  # Auto-Detect Pins
                success = send_to_blynk_individual_pins(future_predictions, selected_targets, list(all_models.keys()))
            
            if success:
                st.success("✅ Data successfully sent to Blynk API!")
