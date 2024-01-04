import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

import yfinance as yf
from datetime import datetime

def data():
    st.title('Gold Futures Data Analysis')

    ticker = 'GC=F'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Get the latest data for gold futures
    gold_data = yf.download(ticker, start='2004-01-01', end=end_date)

    goldDF = pd.DataFrame(gold_data).reset_index()

    # Display the raw data using Streamlit
    st.markdown('### Gold Price History Data')
    st.dataframe(goldDF)
    
    # Create two columns
    data_types_col, missing_values_col = st.columns([2, 1])

    # Column for Data Types
    with data_types_col:
        st.write('### Data Types of Columns')
        st.write(goldDF.dtypes)
        st.write('### Summary Statistics')
        st.write(goldDF.describe())
        
    # Column for Missing Values
    with missing_values_col:
        st.write('### Missing Values')
        st.write(goldDF.isnull().sum())

        # Handling missing values
        st.write('#### Handling Missing Values')
        st.write('Before handling missing values:', goldDF.shape)
        goldDF = goldDF.dropna()
        st.write('After handling missing values:', goldDF.shape)
    
    st.markdown('## Exploratory Data Analysis (EDA)')

