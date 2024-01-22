import streamlit as st
import pandas as pd
from PIL import Image
from data import data
from eda import eda
from machine_learning import machine_learning
from predict_demo import visualize_gold_price
def home():
    menu = ['Home', 'Data', 'Exploratory Data Analysis', 'Machine Learning','Gold Price Prediction']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""#### Name: LEE KAI CHUN""")
            st.markdown("""#### Project Title: Stock Market Trends with Time Series Analysis""")
            st.markdown("""#### Date: 22 JANUARY 2023""")
        with col2:
            img = Image.open('./kai chun.JPG')
            st.image(img)
            
        st.markdown("""#### The purpose of doing this project""")
        st.write("The purpose of creating this is to gain a better understanding of price fluctuations. By utilizing deep learning to forecast future price trends, it allows me to optimize my investment allocation. The goal is to enhance returns through a combination of personal analysis and predictions from this model, thereby reducing risks associated with investments.")
        st.markdown("---")
        st.markdown("""##### Framework: Jupyter Notebooks and Streamlit""")
        st.markdown("""##### Import Library""")
        img_ImportLibrary = Image.open('./Import Library.JPG')
        st.image(img_ImportLibrary)

    elif choice == 'Data':
        data()
    elif choice == 'Exploratory Data Analysis':
        eda()   
    elif choice == 'Machine Learning':
        machine_learning() 
    elif choice == 'Gold Price Prediction':
        input_date = st.date_input("Enter the date to visualize:", pd.to_datetime("2024-01-01"))
        visualize_gold_price(input_date)

home()
