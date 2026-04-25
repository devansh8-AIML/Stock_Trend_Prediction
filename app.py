import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Add these imports

# Try to import TensorFlow with error handling
model_loaded = False
try:
    import tensorflow as tf
    from keras.models import load_model
    
    # Check if model file exists
    model_path = "Stock Predictions Model Hybrid.keras"
    if os.path.exists(model_path):
        model = load_model(model_path)
        model_loaded = True
        st.sidebar.success("✓ Model loaded successfully!")
    else:
        st.sidebar.warning(f"Model file '{model_path}' not found.")
        model_loaded = False
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model_loaded = False

# --- Start of the Streamlit App ---
st.title("📈 Stock Market Predictor")
st.markdown("Enter a stock symbol to view data and predictions.")

# Sidebar for user inputs
stock = st.sidebar.text_input('Enter Stock Symbol', 'AAPL').upper()
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))

# Main app
try:
    # Download stock data
    with st.spinner('Fetching stock data...'):
        data = yf.download(stock, start=start_date, end=end_date)
    
    if data.empty:
        st.warning("No data found. Please try a different stock symbol.")
    else:
        # Display data
        st.subheader(f'{stock} Stock Data (Last 10 Days)')
        st.dataframe(data.tail(10))
        
        # Price chart
        st.subheader('Price Chart')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Prediction section
        if model_loaded:
            st.subheader('Price Prediction')
            
            # Prepare data for prediction
            price_data = data[['Close']].values
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(price_data)
            
            # Create sequences for prediction
            sequence_length = 100
            X = []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
            
            if len(X) > 0:
                X = np.array(X)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Make predictions
                with st.spinner('Making predictions...'):
                    predictions = model.predict(X)
                    predictions = scaler.inverse_transform(predictions)
                
                # Create DataFrame for actual vs predicted
                actual = price_data[sequence_length:]
                result_df = pd.DataFrame({
                    'Actual': actual.flatten(),
                    'Predicted': predictions.flatten()
                }, index=data.index[sequence_length:])
                
                # Plot results
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(result_df.index, result_df['Actual'], label='Actual Price', linewidth=2, color='blue')
                ax2.plot(result_df.index, result_df['Predicted'], label='Predicted Price', linewidth=2, linestyle='--', color='red')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Price ($)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Show latest prediction
                latest_actual = result_df['Actual'].iloc[-1]
                latest_predicted = result_df['Predicted'].iloc[-1]
                
                st.metric(
                    label="Latest Prediction",
                    value=f"${latest_predicted:.2f}",
                    delta=f"Actual: ${latest_actual:.2f}"
                )
                
                # Calculate accuracy metrics (ADDED SECTION)
                st.subheader("Model Performance Metrics")
                
                # Prepare data for metrics calculation
                y_actual = result_df['Actual'].values
                y_pred = result_df['Predicted'].values
                
                # Calculate metrics
                mse = mean_squared_error(y_actual, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred)
                
                # Calculate MAPE and accuracy
                eps = 1e-8  # Small value to avoid division by zero
                mape = np.mean(np.abs((y_actual - y_pred) / np.where(np.abs(y_actual) < eps, eps, y_actual))) * 100
                accuracy = 100 - mape
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("MAE", f"{mae:.2f}")
                    st.metric("R² Score", f"{r2:.4f}")
                
                with col2:
                    st.metric("MAPE", f"{mape:.2f}%")
                    st.metric("Accuracy (100 - MAPE)", f"{accuracy:.2f}%")
                
            else:
                st.warning("Not enough data to make predictions. Need at least 100 days of historical data.")
        else:
            st.info("Model not loaded. Predictions unavailable.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Stock predictions are based on historical data and are not guaranteed to be accurate.")