import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf

import yfinance as yf
from datetime import datetime

# Download gold data
ticker = 'GC=F'
end_date = datetime.today().strftime('%Y-%m-%d')
gold_data = yf.download(ticker, start='2004-01-01', end=end_date)
goldDF = pd.DataFrame(gold_data).reset_index()

def eda():
    summary_stats = goldDF.describe()
    st.write("### Summary Statistics:")
    st.write(summary_stats)

    mean_difference = (goldDF['Adj Close'] - goldDF['Close']).mean()
    std_difference = (goldDF['Adj Close'] - goldDF['Close']).std()
    st.write(f"Mean Difference: {mean_difference:.2f} USD")
    st.write(f"Standard Deviation of Difference: {std_difference:.2f} USD")

    # Plotting Adjusted Close Price vs. Close Price
    st.write("### Adjusted Close Price vs. Closed Market Price")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(goldDF['Date'], goldDF['Adj Close'], label='Adjusted Close Price', linestyle='-')
    ax.plot(goldDF['Date'], goldDF['Close'], label='Close Price', linestyle='-')
    ax.legend()
    ax.set_title('Adjusted Close Price vs. Closed Market Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    # Standard Deviation of DataFrame and Standard Scaling
    st.write("### Standard Deviation of DataFrame and Standard Scaling")
    st.write(goldDF.std())

    scaler = StandardScaler()
    X_scaler = scaler.fit_transform(goldDF[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    st.write(X_scaler.std())

    st.write("### Correlation Matrix")
    goldDF_corr = goldDF.corr()
    st.write(goldDF_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(goldDF_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    st.pyplot(plt)

    st.write("### Gold Price Open Market vs. Closed Market")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(goldDF['Date'], goldDF['Open'], label='Open', linestyle='-', color='blue')
    ax.plot(goldDF['Date'], goldDF['Close'], label='Close', linestyle='-', color='red')
    ax.legend()
    ax.set_title('Gold Price Open Market vs. Closed Market')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    # Plotting Gold Price Open Market vs. Closed Market for Year 2023
    st.write("### Gold Price Open Market vs. Closed Market - Year 2023")
    year2023 = goldDF[goldDF['Date'].dt.year == 2023]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(year2023['Date'], year2023['Open'], label='Open', linestyle='-', color='blue')
    ax.plot(year2023['Date'], year2023['Close'], label='Close', linestyle='-', color='red')
    ax.legend()
    ax.set_title('Gold Price Open Market vs. Closed Market - Year 2023')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    # Candlestick Chart using mplfinance
    st.write("### Candlestick Chart 2023")
    Fdf = goldDF.copy()
    Fdf.set_index('Date', inplace=True)
    dec_2023 = Fdf.loc['2023-01-01':'2023-12-31']
    fig_candle, ax_candle = mpf.plot(dec_2023, type='candle', mav=(3, 6, 9), figratio=(12, 8), volume=True,
                      title='Candle Chart 2023', show_nontrading=True, returnfig=True)

    fig_candle.savefig('candlestick_chart.png')

    st.image('candlestick_chart.png')

    # Volume Distribution - Top 5 Volume Days - December 2023
    st.write("### Volume Distribution - Top 5 Volume Days - December 2023")
    goldDF['Date'] = pd.to_datetime(goldDF['Date'])
    dec_2023_data = goldDF[(goldDF['Date'].dt.year == 2023) & (goldDF['Date'].dt.month == 12)]
    top_5_volume_dec_2023 = dec_2023_data.sort_values(by='Volume', ascending=False).head()
    fig_volume, ax_volume = plt.subplots(figsize=(12, 8))
    ax_volume.bar(top_5_volume_dec_2023['Date'], top_5_volume_dec_2023['Volume'], color='blue', alpha=0.7)
    ax_volume.legend(['Volume'])
    ax_volume.set_title('Volume Distribution - Top 5 Volume Days - December 2023')
    ax_volume.set_xlabel('Date')
    ax_volume.set_ylabel('Volume')

    fig_volume.savefig('volume_distribution.png')

    st.image('volume_distribution.png')
    
    
