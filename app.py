import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
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

def generate_future_predictions(models, df_eng, feature_cols, selected_targets, hours=168):
    """Generate future predictions for the next hours"""
    future_predictions = {}
    
    for model_key, model_data in models.items():
        future_preds = {target: [] for target in selected_targets}
        
        try:
            if model_key == 'LSTM':
                model, scaler_X, scaler_y = model_data
                sequence_length = model.input_shape[1]
                
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
                
        except Exception as e:
            st.warning(f"Error in prediction for {model_key}: {e}")
            # Fill with simple trend if prediction fails
            for target in selected_targets:
                if len(future_preds[target]) < hours:
                    last_value = df_eng[target].iloc[-1]
                    # Simple linear trend
                    for i in range(len(future_preds[target]), hours):
                        future_preds[target].append(last_value * (1 + 0.01 * i))
        
        future_predictions[model_key] = future_preds
    
    return future_predictions

def create_prediction_plots(df, future_predictions, selected_targets, models_to_plot):
    """Create hourly and weekly prediction plots"""
    
    # Generate future timestamps
    last_timestamp = df['created_at'].iloc[-1]
    future_hours = len(list(future_predictions.values())[0][selected_targets[0]])
    future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, future_hours + 1)]
    
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
        
        # Future predictions for each model (first 24 hours)
        colors = ['red', 'green', 'orange', 'purple']
        for i, model_key in enumerate(models_to_plot):
            if model_key in future_predictions and target in future_predictions[model_key]:
                pred_values = future_predictions[model_key][target][:24]
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
                            name=f'Predicted {model_key}',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
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
        
        # Future predictions for each model (all 168 hours)
        for i, model_key in enumerate(models_to_plot):
            if model_key in future_predictions and target in future_predictions[model_key]:
                pred_values = future_predictions[model_key][target]
                
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
                            name=f'Predicted {model_key}',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(
            height=800,
            title_text=f"Prediction Analysis for {target}",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text=target, row=1, col=1)
        fig.update_yaxes(title_text=target, row=2, col=1)
        
        plots[target] = fig
    
    return plots

def main():
    st.set_page_config(page_title="Air Quality Prediction", layout="wide")
    
    st.title("🌤️ Air Quality Prediction Dashboard")
    st.markdown("""
    This app predicts future values of PM2.5, PM10, CO2, CO, Temperature, and Humidity using LSTM model.
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
    
    # Show missing data details
    if st.checkbox("Show missing data details"):
        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame({
            'Column': ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10'],
            'Missing Count': [df[col].isna().sum() for col in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']],
            'Missing Percentage': [df[col].isna().sum() / len(df) * 100 for col in ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']]
        })
        st.dataframe(missing_df)
    
    st.sidebar.header("Configuration")
    
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
    
    # Model selection - only LSTM available
    models_to_train = ['LSTM']
    
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
        
        # Show diagnostic information
        st.subheader("Diagnostic Information")
        st.write(f"Original data shape: {df.shape}")
        st.write(f"Columns in original data: {list(df.columns)}")
        st.write(f"Data types: {df.dtypes}")
        
        # Check for any rows with all NaN values
        all_nan_rows = df[['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].isna().all(axis=1).sum()
        st.write(f"Rows with all NaN values: {all_nan_rows}")
        
        return
    
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
    st.info("⚠️ All available data is used for training the LSTM model")
    
    all_models = {}
    all_predictions = {}
    all_scores = {}
    
    # Model name to key mapping
    model_keys = {
        'LSTM': 'LSTM'
    }
    
    # Train LSTM model
    for model_name in models_to_train:
        st.subheader(f"{model_name} Model")
        model_key = model_keys[model_name]
        
        try:
            with st.spinner(f'Training {model_name} with all data...'):
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
                # Format the scores properly
                formatted_scores = scores_df.copy()
                for col in formatted_scores.columns:
                    formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                st.dataframe(formatted_scores)
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            continue
    
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
                                # Handle NaN values
                                if np.isnan(value):
                                    hour_data[f"{model_key}_{target}"] = "N/A"
                                else:
                                    hour_data[f"{model_key}_{target}"] = f"{value:.4f}"
                    prediction_data.append(hour_data)
                
                if prediction_data:
                    pred_df = pd.DataFrame(prediction_data)
                    st.dataframe(pred_df)
                    
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
