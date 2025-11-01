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
    """Test if Blynk API is accessible"""
    try:
        # Test with a simple value
        test_url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&v0=test"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            return True, "Connection successful"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"

def send_simple_to_blynk(predictions_data, selected_targets, models_to_send):
    """Send simplified prediction data to Blynk API"""
    try:
        if models_to_send and len(models_to_send) > 0:
            primary_model = models_to_send[0]
            
            # Get first hour predictions only (simpler approach)
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
                blynk_data['rain'] = 1
            if 'light' not in blynk_data:
                blynk_data['light'] = 10
            
            # Send each parameter individually
            success_count = 0
            total_params = len(blynk_data)
            
            for param, value in blynk_data.items():
                try:
                    # Use proper URL encoding
                    url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&{param}={value}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        success_count += 1
                        st.write(f"✅ Sent {param}: {value}")
                    else:
                        st.write(f"❌ Failed to send {param}: HTTP {response.status_code}")
                except Exception as e:
                    st.write(f"❌ Error sending {param}: {str(e)}")
            
            if success_count == total_params:
                st.success(f"✅ Successfully sent {success_count}/{total_params} parameters to Blynk!")
                return True
            elif success_count > 0:
                st.warning(f"⚠️ Partially sent {success_count}/{total_params} parameters to Blynk")
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

def send_individual_hours_to_blynk(predictions_data, selected_targets, models_to_send):
    """Send each hour's data to separate virtual pins"""
    try:
        if models_to_send and len(models_to_send) > 0:
            primary_model = models_to_send[0]
            success_count = 0
            
            for hour in range(24):
                hour_data = {}
                
                # Get predictions for this hour
                for target in selected_targets:
                    if (primary_model in predictions_data and 
                        target in predictions_data[primary_model] and 
                        len(predictions_data[primary_model][target]) > hour):
                        value = predictions_data[primary_model][target][hour]
                        if not np.isnan(value):
                            hour_data[target] = float(value)
                
                # Add default values
                if 'rain' not in hour_data:
                    hour_data['rain'] = 1
                if 'light' not in hour_data:
                    hour_data['light'] = 10
                
                # Send as JSON to virtual pin for this hour
                json_data = json.dumps(hour_data)
                encoded_json = urllib.parse.quote(json_data)
                
                url = f"{BLYNK_UPDATE_BASE_URL}?token={BLYNK_API_TOKEN}&v{hour}={encoded_json}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    success_count += 1
                    st.write(f"✅ Sent hour {hour+1} to v{hour}")
                else:
                    st.write(f"❌ Failed to send hour {hour+1}: HTTP {response.status_code}")
            
            if success_count == 24:
                st.success(f"✅ All 24 hours sent to Blynk!")
                return True
            elif success_count > 0:
                st.warning(f"⚠️ Sent {success_count}/24 hours to Blynk")
                return True
            else:
                st.error("❌ Failed to send any hours to Blynk")
                return False
        else:
            st.warning("No valid prediction data to send to Blynk")
            return False
            
    except Exception as e:
        st.error(f"❌ Error sending individual hours to Blynk: {e}")
        return False

# ... (keep all the other existing functions exactly as they were - create_features, prepare_lstm_data, train_random_forest, etc.)

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
                    else:
                        st.error(f"Blynk connection failed: {message}")
    
    blynk_send_method = st.sidebar.radio(
        "Blynk Send Method:",
        ["Simple Parameters", "Individual Hours (v0-v23)"],
        index=0,
        help="Simple: Send current hour as individual parameters. Individual: Send each hour to separate virtual pins."
    )
    
    # Load data
    with st.spinner('Loading data from Supabase...'):
        df = load_data()
    
    if df.empty:
        st.error("No data loaded. Please check your Supabase connection and ensure the 'airquality' table exists.")
        return
    
    # Show data quality information
    st.header("📊 Data Quality Check")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        missing_data = df[['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].isna().sum().sum()
        st.metric("Missing Values", missing_data)
    
    with col3:
        completeness = (1 - missing_data / (len(df) * 6)) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col4:
        date_range = f"{df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)
    
    # Show data preview
    if st.checkbox("Show raw data preview"):
        st.dataframe(df.tail(10))
    
    # Show data statistics
    if st.checkbox("Show data statistics"):
        st.subheader("Data Statistics")
        st.dataframe(df[['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].describe())
    
    # Show missing data details
    if st.checkbox("Show missing data details"):
        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame({
            'Column': ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10'],
            'Missing Count': [df[col].isna().sum() for col in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']],
            'Missing Percentage': [df[col].isna().sum() / len(df) * 100 for col in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']]
        })
        st.dataframe(missing_df)
    
    st.sidebar.header("Model Configuration")
    
    # Target selection
    target_columns = ['pm25', 'pm10', 'co2', 'co', 'temperature', 'humidity']
    selected_targets = st.sidebar.multiselect(
        "Select targets to predict:",
        target_columns,
        default=target_columns
    )
    
    if not selected_targets:
        st.warning("Please select at least one target variable to predict.")
        return
    
    # Model selection
    models_to_train = st.sidebar.multiselect(
        "Select models to train:",
        ['Random Forest', 'XGBoost', 'SVM', 'LSTM'],
        default=['Random Forest', 'XGBoost']
    )
    
    if not models_to_train:
        st.warning("Please select at least one model to train.")
        return
    
    # Feature engineering
    st.header("🔧 Feature Engineering")
    
    if len(df) < 2:
        st.error("Not enough data for feature engineering. Need at least 2 records.")
        return
        
    with st.spinner('Creating features...'):
        df_eng = create_features(df, selected_targets, n_lags=2)
    
    if len(df_eng) == 0:
        st.error("""
        No data available after feature engineering. This usually happens when:
        1. There are too many missing values in your data
        2. The data is too short for the requested lag features
        3. There are issues with the data types
        
        Please check your data quality and try again.
        """)
        return
    
    st.success(f"Feature engineering completed! Created {len(df_eng)} samples with {len(df_eng.columns)} features.")
    
    # Prepare data for traditional ML models
    feature_cols = [col for col in df_eng.columns if col not in ['id', 'created_at'] + selected_targets]
    
    if not feature_cols:
        st.error("No features generated. Check your data.")
        return
        
    X = df_eng[feature_cols].values
    y = df_eng[selected_targets].values
    
    # Split data for initial evaluation (but we'll use all data for final training)
    test_size = 0.2
    
    if len(X) < 2:
        st.error("Not enough data for training. Need at least 2 samples after feature engineering.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    
    # Model training section
    st.header("🤖 Model Training & Evaluation")
    st.info("⚠️ All available data is used for training the models")
    
    all_models = {}
    all_predictions = {}
    all_scores = {}
    
    # Model name to key mapping
    model_keys = {
        'Random Forest': 'RF',
        'XGBoost': 'XGB', 
        'SVM': 'SVM',
        'LSTM': 'LSTM'
    }
    
    # Train selected models using all data
    for model_name in models_to_train:
        st.subheader(f"{model_name} Model")
        model_key = model_keys[model_name]
        
        try:
            with st.spinner(f'Training {model_name} with all data...'):
                if model_name == 'Random Forest':
                    models, predictions, scores = train_random_forest(X_train, X_test, y_train, y_test, selected_targets)
                    all_models[model_key] = models
                    all_predictions[model_key] = predictions
                    all_scores[model_key] = scores
                    
                elif model_name == 'XGBoost':
                    models, predictions, scores = train_xgboost(X_train, X_test, y_train, y_test, selected_targets)
                    all_models[model_key] = models
                    all_predictions[model_key] = predictions
                    all_scores[model_key] = scores
                    
                elif model_name == 'SVM':
                    model, predictions, scores, scaler_X, scaler_y = train_svm(X_train, X_test, y_train, y_test, selected_targets)
                    all_models[model_key] = (model, scaler_X, scaler_y)
                    all_predictions[model_key] = {col: predictions[:, i] for i, col in enumerate(selected_targets)}
                    all_scores[model_key] = scores
                    
                elif model_name == 'LSTM':
                    # Prepare LSTM data
                    sequence_length = min(5, len(df_eng) // 3)
                    if sequence_length < 2:
                        st.warning("Not enough data for LSTM. Skipping LSTM training.")
                        continue
                        
                    X_lstm, y_lstm = prepare_lstm_data(df_eng, selected_targets, sequence_length)
                    
                    if len(X_lstm) == 0:
                        st.warning("Not enough sequences for LSTM training. Skipping LSTM.")
                        continue
                    
                    # Split LSTM data for initial setup
                    split_idx = int(len(X_lstm) * (1 - test_size))
                    X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
                    y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
                    
                    model, predictions, scores, history, scaler_X, scaler_y = train_lstm(
                        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, selected_targets
                    )
                    all_models[model_key] = (model, scaler_X, scaler_y)
                    all_predictions[model_key] = {col: predictions[:, i] for i, col in enumerate(selected_targets)}
                    all_scores[model_key] = scores
            
            # Display scores for this model
            if model_key in all_scores:
                scores_df = pd.DataFrame(all_scores[model_key]).T
                scores_df.columns = ['RMSE', 'R² Score']
                formatted_scores = scores_df.copy()
                for col in formatted_scores.columns:
                    formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                st.dataframe(formatted_scores)
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Model comparison
    if len(all_scores) > 1:
        st.header("📈 Model Comparison")
        
        comparison_data = []
        for model_key, scores_dict in all_scores.items():
            for target in selected_targets:
                if target in scores_dict:
                    comparison_data.append({
                        'Model': model_key,
                        'Target': target,
                        'RMSE': scores_dict[target]['rmse'],
                        'R2_Score': scores_dict[target]['r2']
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_rmse = go.Figure()
            for model in comparison_df['Model'].unique():
                model_data = comparison_df[comparison_df['Model'] == model]
                fig_rmse.add_trace(go.Bar(
                    name=model,
                    x=model_data['Target'],
                    y=model_data['RMSE']
                ))
            
            fig_rmse.update_layout(
                title="RMSE Comparison by Model and Target",
                xaxis_title="Target Variable",
                yaxis_title="RMSE",
                barmode='group'
            )
            
            st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Future prediction with plots
    st.header("🔮 Continuous Future Predictions")
    
    if st.button("Generate Future Predictions") and all_models:
        try:
            with st.spinner('Generating future predictions...'):
                # Generate predictions for the next 168 hours (1 week)
                future_predictions = generate_future_predictions(
                    all_models, df_eng, feature_cols, selected_targets, hours=168
                )
                
                # Create prediction plots
                plots = create_prediction_plots(df, future_predictions, selected_targets, list(all_models.keys()))
                
                # Display plots
                for target, fig in plots.items():
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction table for the next 24 hours
                st.subheader("📋 Prediction Values for Next 24 Hours")
                prediction_data = []
                
                for hour in range(24):
                    hour_data = {'Hour': f"Hour {hour+1}"}
                    for model_key in all_models.keys():
                        if model_key in future_predictions:
                            for target in selected_targets:
                                value = future_predictions[model_key][target][hour]
                                if np.isnan(value):
                                    hour_data[f"{model_key}_{target}"] = "N/A"
                                else:
                                    hour_data[f"{model_key}_{target}"] = f"{value:.4f}"
                    prediction_data.append(hour_data)
                
                if prediction_data:
                    pred_df = pd.DataFrame(prediction_data)
                    st.dataframe(pred_df)
                    
                    # Send to Blynk if enabled
                    if enable_blynk and future_predictions:
                        st.subheader("🌐 Sending to Blynk API")
                        with st.spinner('Sending prediction data to Blynk...'):
                            if blynk_send_method == "Simple Parameters":
                                success = send_simple_to_blynk(future_predictions, selected_targets, list(all_models.keys()))
                            else:
                                success = send_individual_hours_to_blynk(future_predictions, selected_targets, list(all_models.keys()))
                            
                            if success:
                                st.success("✅ Data successfully sent to Blynk API!")
                    
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")

if __name__ == "__main__":
    main()
