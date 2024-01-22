# gold_price_visualization.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

def calculate_rsi(data, window=14):
    close_price = data['Close']
    delta = close_price.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def market_status(rsi_value):
    if rsi_value > 70:
        return 'Overbought'
    elif rsi_value < 30:
        return 'Oversold'
    else:
        return 'Neutral'

def visualize_gold_price(input_date, days_ranges=[180, 365], window_size=60):
    model = load_model('gold_price_prediction_model.h5')
    st.write("Model loaded successfully.")

    # Separate the plotting logic into two independent plots
    for days_range in days_ranges:
        plot_start_date = input_date - pd.Timedelta(days=days_range // 2)
        plot_end_date = input_date + pd.Timedelta(days=days_range // 2)

        try:
            # Retrieve historical data for the specified date range
            plot_data = yf.download('GC=F', start=plot_start_date, end=plot_end_date)

            # Check if there is enough data for scaling
            if len(plot_data) >= window_size:
                # Extract the close prices for the specified date range
                input_data = plot_data['Close'][-window_size:].values.reshape(-1, 1)

                scaler = MinMaxScaler()
                scaler.fit(plot_data['Close'].values.reshape(-1, 1))

                # Initialize close_price variable
                close_price = plot_data['Close'].iloc[-1]

                # Function to preprocess input data and make predictions
                def predict_gold_price(input_data):
                    input_data = scaler.transform(input_data.reshape(-1, 1))
                    input_data = np.reshape(input_data, (1, window_size, 1))
                    predicted_price = model.predict(input_data)
                    predicted_price = scaler.inverse_transform(predicted_price)
                    return predicted_price[0, 0]

                # Make predictions for the input date and all days in the days_ranges period
                predictions = []
                for i in range(days_range + 1):
                    predictions.append(predict_gold_price(input_data))
                    input_data = np.append(input_data[1:], predictions[-1]) 

                # Calculate the percentage difference for the last day of days_range
                last_day_difference = ((predictions[-1] - close_price) / close_price) * 100

                last_day_direction = 'Up' if last_day_difference > 0 else 'Down'
                # Get the last date in the plot_data
                last_day_date = plot_data.index[-1]  
                
                future_dates = [input_date + timedelta(days=i) for i in range(days_range + 1)]
                
                st.write(f"The price is expected to go {last_day_direction} by {abs(last_day_difference):.2f}%")
                # st.write(f"Predicted price on {last_day_date.strftime('%Y-%m-%d')}: USD {predictions[-1]:.2f}")
                # st.write(f"Predicted price after {days_range} days on {future_dates[-1].strftime('%Y-%m-%d')} is USD {close_price:.2f}")

                # Gold Price and LSTM Prediction Plot 
                fig1, ax1 = plt.subplots(figsize=(10, 6))

                ax1.plot(plot_data.index, plot_data['Close'], label='Gold Price', linestyle='-', color='blue')
                ax1.plot(future_dates, predictions, color='orange', linestyle='-', label='Gold Price Prediction')
                ax1.axvline(x=input_date, color='red', linestyle='--')

                ax1.scatter(input_date, predictions[0], color='green')
                ax1.text(input_date, predictions[0], f'USD {predictions[0]:.2f}', ha='left',
                         va='bottom', color='green')
                ax1.scatter(input_date, plot_data['Close'].iloc[-1], color='red')
                ax1.axvline(x=input_date, color='red', linestyle='-')
                ax1.text(input_date, close_price, f'USD {close_price:.2f} ', ha='right',
                         va='bottom', color='red')

                last_day_annotation = f'{future_dates[-1].strftime("%Y-%m-%d")}\nUSD {predictions[-1]:.2f} ({last_day_difference:.2f}%)'
                ax1.scatter(future_dates[-1], predictions[-1], color='purple', marker='^')
                ax1.text(future_dates[-1], predictions[-1], last_day_annotation, ha='right',
                         va='bottom', color='purple')

                ax1.set_title(f'Gold Price Data Around {input_date.strftime("%Y-%m-%d")} ({days_range} Days)')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Gold Price')
                ax1.legend()
                ax1.grid(True)

                # RSI Plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                rsi_data = calculate_rsi(plot_data)
                ax2.plot(plot_data.index, rsi_data, label='RSI', linestyle='-', color='purple')
                ax2.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
                ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')

                ax2.axvline(x=input_date, color='red', linestyle='-')

                market_status_value = market_status(rsi_data.iloc[-1])
                ax2.set_title(f'Market Status: {market_status_value}')
                ax2.set_ylabel('RSI')
                ax2.legend()

                plt.tight_layout()

                # Display the plots
                st.pyplot(fig1)
                st.pyplot(fig2)
                
                st.markdown("---")
                
            else:
                st.write(f"Insufficient historical data for scaling in the {days_range}-day plot.")

        except Exception as e:
            st.write(f"Failed to download data for the {days_range}-day plot: {e}")

