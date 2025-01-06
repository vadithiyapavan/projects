#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 09:43:59 2021

@author: shubham
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

# Load the pre-trained classifier
pickle_in = open("model_poly.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome ALL"

def predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    # Perform prediction using the loaded model
    prediction = classifier.predict([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    print(prediction)
    return prediction

def main():
    st.title("Bankruptcy Detector")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detector ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for features
    industrial_risk = st.text_input("Industrial Risk (numeric value)", "Type Here")
    management_risk = st.text_input("Management Risk (numeric value)", "Type Here")
    financial_flexibility = st.text_input("Financial Flexibility (numeric value)", "Type Here")
    credibility = st.text_input("Credibility (numeric value)", "Type Here")
    competitiveness = st.text_input("Competitiveness (numeric value)", "Type Here")
    operating_risk = st.text_input("Operating Risk (numeric value)", "Type Here")
    
    result = ""
    if st.button("Predict"):
        try:
            # Convert input strings to numeric values
            industrial_risk = float(industrial_risk)
            management_risk = float(management_risk)
            financial_flexibility = float(financial_flexibility)
            credibility = float(credibility)
            competitiveness = float(competitiveness)
            operating_risk = float(operating_risk)
            
            # Perform prediction
            result = predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
            result = "Bankruptcy Risk: {}".format(result[0])  # Assuming classifier returns a single value
        except ValueError:
            result = "Invalid input! Please enter numeric values for all fields."
    
    st.success(result)
    
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
