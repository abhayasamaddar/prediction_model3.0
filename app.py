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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

def create_correlation_heatmap(df):
    """Create correlation heatmap for air quality parameters"""
    # Select only numeric columns for correlation
    numeric_cols = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    corr_df = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_df.round(3).values,
        texttemplate="%{text}",
        hoverinfo="text",
        hovertemplate="Correlation between %{x} and %{y}: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Correlation Heatmap of Air Quality Parameters",
        xaxis_title="Parameters",
        yaxis_title="Parameters",
        width=700,
        height=600
    )
    
    return fig, corr_df

def create_features(df, target_columns, n_lags=3):
    """Create lag features for time series prediction with better handling"""
    df_eng = df.copy()
    
    # Determine safe number of lags based on data length
    safe_n_lags = min(n_lags, len(df_eng) - 1)
    if safe_n_lags < 1:
        st.warning("Not enough data for lag features. Using basic features only.")
        safe_n_lags = 0
    
    # Create lag features only if we have enough data
    for col in target_columns:
        for lag in range(1, safe_n_lags + 1):
            df_eng[f'{col}_lag_{lag}'] = df_eng[col].shift(lag)
    
    # Add time-based features
    df_eng['hour'] = df_eng['created_at'].dt.hour
    df_eng['day_of_week'] = df_eng['created_at'].dt.dayofweek
    df_eng['month'] = df_eng['created_at'].dt.month
    df_eng['day_of_year'] = df_eng['created_at'].dt.dayofyear
    
    # Add rolling statistics (if enough data)
    if len(df_eng) > 5:
        for col in target_columns:
            df_eng[f'{col}_rolling_mean_3'] = df_eng[col].rolling(window=3, min_periods=1).mean()
            df_eng[f'{col}_rolling_std_3'] = df_eng[col].rolling(window=3, min_periods=1).std()
    
    # Fill NaN values created by lag features and rolling stats
    numeric_columns = df_eng.select_dtypes(include=[np.number]).columns
    df_eng[numeric_columns] = df_eng[numeric_columns].fillna(method='bfill').fillna(method='ffill')
    
    # If there are still missing values, fill with column mean
    for col in numeric_columns:
        if df_eng[col].isna().any():
            df_eng[col] = df_eng[col].fillna(df_eng[col].mean())
    
    # Drop rows that still have NaN values (should be very few if any)
    initial_count = len(df_eng)
    df_eng = df_eng.dropna()
    final_count = len(df_eng)
    
    if initial_count != final_count:
        st.warning(f"Dropped {initial_count - final_count} rows with missing values after feature engineering.")
    
    return df_eng

def prepare_lstm_data(df, target_columns, sequence_length=10):
    """Prepare data for LSTM model with safe sequence length"""
    features = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    X, y = [], []
    
    # Use safe sequence length
    safe_sequence_length = min(sequence_length, len(df) - 1)
    if safe_sequence_length < 2:
        return np.array([]), np.array([])
    
    for i in range(safe_sequence_length, len(df)):
        X.append(df[features].iloc[i-safe_sequence_length:i].values)
        y.append(df[target_columns].iloc[i].values)
    
    return np.array(X), np.array(y)

def train_lstm(X_train, X_test, y_train, y_test, target_columns):
    """Train LSTM model using all data"""
    # Use all data for training
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_all_reshaped = X_all.reshape(-1, X_all.shape[2])
    
    X_all_scaled = scaler_X.fit_transform(X_all_reshaped).reshape(X_all.shape)
    y_all_scaled = scaler_y.fit_transform(y_all)
    
    # Build LSTM model
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X_all.shape[1], X_all.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16),
        Dense(len(target_columns))
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        X_all_scaled, y_all_scaled,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # Make predictions on all data
    pred_scaled = model.predict(X_all_scaled, verbose=0)
    predictions = scaler_y.inverse_transform(pred_scaled)
    
    scores = {}
    for i, col in enumerate(target_columns):
        scores[col] = {
            'rmse': np.sqrt(mean_squared_error(y_all[:, i], predictions[:, i])),
            'r2': r2_score(y_all[:, i], predictions[:, i])
        }
    
    return model, predictions, scores, history, scaler_X, scaler_y

def generate_lstm_future_predictions(model_data, df_eng, selected_targets, hours=168):
    """Generate future predictions for the next hours using LSTM"""
    try:
        model, scaler_X, scaler_y = model_data
        sequence_length = model.input_shape[1]
        
        future_preds = {target: [] for target in selected_targets}
        
        # Get the last sequence
        last_sequence = df_eng[['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].tail(sequence_length).values
        
        for hour in range(hours):
            sequence_scaled = scaler_X.transform(last_sequence.reshape(-1, 6)).reshape(1, sequence_length, 6)
            pred_scaled = model.predict(sequence_scaled, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0]
            
            for i, target in enumerate(selected_targets):
                future_preds[target].append(pred[i])
            
            # Update the sequence for next prediction
            new_row = pred.copy()
            last_sequence = np.vstack([last_sequence[1:], new_row])
        
        return future_preds
        
    except Exception as e:
        st.error(f"Error in LSTM prediction: {e}")
        # Return empty predictions if error occurs
        return {target: [] for target in selected_targets}

def create_prediction_plots(df, future_predictions, selected_targets):
    """Create hourly and weekly prediction plots"""
    
    # Generate future timestamps
    last_timestamp = df['created_at'].iloc[-1]
    if future_predictions and selected_targets and selected_targets[0] in future_predictions:
        future_hours = len(future_predictions[selected_targets[0]])
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, future_hours + 1)]
    else:
        future_timestamps = []
    
    plots = {}
    
    for target in selected_targets:
        # Create subplots for actual vs predicted
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Hourly Predictions - {target}', f'Weekly Overview - {target}'),
            vertical_spacing=0.1
        )
        
        # Plot 1: Hourly predictions (first 24 hours)
        # Actual data (last 24 hours if available)
        display_hours = min(24, len(df))
        if display_hours > 0:
            last_actual = df.tail(display_hours)
            fig.add_trace(
                go.Scatter(
                    x=last_actual['created_at'],
                    y=last_actual[target],
                    mode='lines+markers',
                    name='Actual (Recent)',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Future predictions (first 24 hours)
        if target in future_predictions and future_predictions[target]:
            pred_values = future_predictions[target][:24]
            pred_timestamps = future_timestamps[:24]
            
            # Filter out NaN values
            valid_indices = [j for j, val in enumerate(pred_values) if not np.isnan(val)]
            if valid_indices:
                valid_pred = [pred_values[j] for j in valid_indices]
                valid_times = [pred_timestamps[j] for j in valid_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=valid_times,
                        y=valid_pred,
                        mode='lines+markers',
                        name='Predicted LSTM',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Weekly overview (all 168 hours)
        # Recent actual data
        display_week = min(168, len(df))
        if display_week > 0:
            last_week_actual = df.tail(display_week)
            fig.add_trace(
                go.Scatter(
                    x=last_week_actual['created_at'],
                    y=last_week_actual[target],
                    mode='lines',
                    name='Actual (Recent)',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Future predictions (all 168 hours)
        if target in future_predictions and future_predictions[target]:
            pred_values = future_predictions[target]
            
            # Filter out NaN values
            valid_indices = [j for j, val in enumerate(pred_values) if not np.isnan(val)]
            if valid_indices:
                valid_pred = [pred_values[j] for j in valid_indices]
                valid_times = [future_timestamps[j] for j in valid_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=valid_times,
                        y=valid_pred,
                        mode='lines',
                        name='Predicted LSTM',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=800,
            title_text=f"LSTM Prediction Analysis for {target}",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text=target, row=1, col=1)
        fig.update_yaxes(title_text=target, row=2, col=1)
        
        plots[target] = fig
    
    return plots

def create_lstm_prediction_json(future_predictions, selected_targets):
    """Create JSON format of LSTM prediction data"""
    try:
        # Create comprehensive data structure with all 24 hours
        prediction_json = {
            "timestamp": datetime.now().isoformat(),
            "model_used": "LSTM",
            "predictions_24h": []
        }
        
        # Add predictions for all 24 hours
        for hour in range(24):
            hour_data = {
                "hour": hour + 1,
                "predictions": {}
            }
            
            # Get predictions for all targets for this hour
            for target in selected_targets:
                if (target in future_predictions and 
                    len(future_predictions[target]) > hour):
                    value = future_predictions[target][hour]
                    if not np.isnan(value):
                        hour_data["predictions"][target] = float(value)
            
            # Add default values if missing
            if 'rain' not in hour_data["predictions"]:
                hour_data["predictions"]['rain'] = 0
            if 'light' not in hour_data["predictions"]:
                hour_data["predictions"]['light'] = 0
            
            prediction_json["predictions_24h"].append(hour_data)
        
        return prediction_json
    except Exception as e:
        st.error(f"Error creating JSON: {e}")
        return None

def download_json(data, filename="lstm_prediction_data.json"):
    """Create download button for JSON data"""
    json_str = json.dumps(data, indent=2)
    st.download_button(
        label="📥 Download LSTM Prediction Data as JSON",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

def main():
    st.set_page_config(page_title="LSTM Air Quality Prediction", layout="wide")
    
    st.title("🧠 LSTM Air Quality Prediction Dashboard")
    st.markdown("""
    This app predicts future values of PM2.5, PM10, CO2, CO, Temperature, and Humidity using **LSTM neural networks**.
    **All data is used for training** and continuous predictions are shown in hourly and weekly plots.
    """)
    
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
    
    # Correlation Analysis Section
    st.header("📈 Correlation Analysis")
    
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap of Air Quality Parameters")
        
        # Create correlation heatmap
        corr_fig, corr_matrix = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Display correlation insights
        st.subheader("📋 Correlation Insights")
        
        # Find strong correlations (absolute value > 0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'Parameter 1': corr_matrix.columns[i],
                        'Parameter 2': corr_matrix.columns[j],
                        'Correlation': f"{corr_value:.3f}",
                        'Strength': 'Strong Positive' if corr_value > 0 else 'Strong Negative'
                    })
        
        if strong_correlations:
            st.write("**Strong Correlations (|r| > 0.7):**")
            strong_corr_df = pd.DataFrame(strong_correlations)
            st.dataframe(strong_corr_df)
        else:
            st.info("No strong correlations (|r| > 0.7) found between parameters.")
        
        # Display correlation matrix as table
        if st.checkbox("Show Correlation Matrix Table"):
            st.subheader("Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
    
    st.sidebar.header("LSTM Configuration")
    
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
    
    # LSTM parameters
    st.sidebar.subheader("LSTM Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", min_value=3, max_value=24, value=6, 
                                       help="Number of previous time steps to use for prediction")
    
    # Feature engineering
    st.header("🔧 Feature Engineering")
    
    if len(df) < 2:
        st.error("Not enough data for feature engineering. Need at least 2 records.")
        return
        
    with st.spinner('Creating features for LSTM...'):
        df_eng = create_features(df, selected_targets, n_lags=2)
    
    if len(df_eng) == 0:
        st.error("No data available after feature engineering.")
        return
    
    st.success(f"Feature engineering completed! Created {len(df_eng)} samples with {len(df_eng.columns)} features.")
    
    # Prepare LSTM data
    st.header("🧠 LSTM Model Training")
    
    with st.spinner('Preparing LSTM data...'):
        X_lstm, y_lstm = prepare_lstm_data(df_eng, selected_targets, sequence_length)
    
    if len(X_lstm) == 0:
        st.error("Not enough sequences for LSTM training. Please increase sequence length or add more data.")
        return
    
    st.info(f"Prepared {len(X_lstm)} sequences for LSTM training")
    
    # Split LSTM data for training
    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
    
    # Train LSTM model
    with st.spinner('Training LSTM model...'):
        try:
            model, predictions, scores, history, scaler_X, scaler_y = train_lstm(
                X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, selected_targets
            )
            
            # Display LSTM model scores
            st.subheader("LSTM Model Performance")
            scores_df = pd.DataFrame(scores).T
            scores_df.columns = ['RMSE', 'R² Score']
            formatted_scores = scores_df.copy()
            for col in formatted_scores.columns:
                formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            st.dataframe(formatted_scores)
            
            # Store the trained model
            lstm_model_data = (model, scaler_X, scaler_y)
            
        except Exception as e:
            st.error(f"Error training LSTM model: {str(e)}")
            return
    
    # Future prediction with plots
    st.header("🔮 LSTM Future Predictions")
    
    if st.button("Generate LSTM Future Predictions"):
        try:
            with st.spinner('Generating future predictions with LSTM...'):
                # Generate predictions for the next 168 hours (1 week)
                future_predictions = generate_lstm_future_predictions(
                    lstm_model_data, df_eng, selected_targets, hours=168
                )
                
                # Create prediction plots
                plots = create_prediction_plots(df, future_predictions, selected_targets)
                
                # Display plots
                for target, fig in plots.items():
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction table for the next 24 hours
                st.subheader("📋 LSTM Prediction Values for Next 24 Hours")
                prediction_data = []
                
                for hour in range(24):
                    hour_data = {'Hour': f"Hour {hour+1}"}
                    for target in selected_targets:
                        if target in future_predictions and len(future_predictions[target]) > hour:
                            value = future_predictions[target][hour]
                            if np.isnan(value):
                                hour_data[target] = "N/A"
                            else:
                                hour_data[target] = f"{value:.4f}"
                        else:
                            hour_data[target] = "N/A"
                    prediction_data.append(hour_data)
                
                if prediction_data:
                    pred_df = pd.DataFrame(prediction_data)
                    st.dataframe(pred_df)
                    
                    # Create JSON data
                    st.subheader("📄 LSTM JSON Data Format")
                    prediction_json = create_lstm_prediction_json(future_predictions, selected_targets)
                    
                    if prediction_json:
                        # Display JSON data
                        st.json(prediction_json)
                        
                        # Download JSON button
                        download_json(prediction_json, "lstm_air_quality_predictions.json")
                        
                        st.success("✅ LSTM predictions generated successfully!")
                        st.info("You can download the JSON data using the button above.")
                    
        except Exception as e:
            st.error(f"Error generating LSTM predictions: {str(e)}")

if __name__ == "__main__":
    main()
