import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def machine_learning():
    st.write("## Machine Learning")
    st.markdown("---")
    st.write("#### RandomForestClassifier")
    img_random_forest = Image.open('./random_forest.png')
    st.image(img_random_forest)
    st.write("#### Support Vector Machine (SVM)")
    img_svm = Image.open('./svm.jpeg')
    st.image(img_svm)
    st.write("#### GradientBoostingClassifier")
    img_gradient_boosting_classifier = Image.open('./gradient_boosting_classifier.png')
    st.image(img_gradient_boosting_classifier)
    st.write("#### Logistic Regression")
    img_logistic_regression = Image.open('./logistic_regression.png')
    st.image(img_logistic_regression)
    st.write("#### Long Short-Term Memory (LSTM)")
    img_lstm = Image.open('./lstm.png')
    st.image(img_lstm)
    st.markdown("---")
    st.markdown("#### All Model Accuracy")
    img_accuracy = Image.open('./accuracy.png')
    st.image(img_accuracy)
    
    