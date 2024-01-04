import streamlit as st
from PIL import Image
from data import data
from machine_learning import machine_learning

def home():
    menu = ['Home', 'Data', 'Machine Learning Prediction']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.title('DS Project')
        # img = Image.open('./data/heart.png')
        # st.image(img)
        st.write("""This is my DS Project""")

    elif choice == 'Data':
        data()
    else:
        machine_learning()

home()