
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

os.chdir(r'H:\My Drive\IBM Data Science G2\03- Machine Learning')
st.title('Car Price Prediction App')

model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

X = pd.read_csv('cleaned_data.csv')


# brand = st.selectbox( 'Brand', X['Brand'].unique())
# model = st.selectbox( 'Model', X['Model'].unique())
# kilometers_driven = st.number_input( 'Kilometers Driven', min_value = X['Kilometers_Driven'].min(), max_value = X['Kilometers_Driven'].max())
# fuel_type = st.selectbox( 'Fuel Type', X['Fuel_Type'].unique())
# transmission = st.selectbox('Transmission',  X['Transmission'].unique())
# owner_type = st.selectbox( 'Owner Type', X['Owner_Type'].unique())
# engine = st.number_input( 'Engine', min_value = X['Engine'].min(), max_value = X['Engine'].max())
# power = st.number_input('Power', min_value = X['Power'].min(), max_value = X['Power'].max())
# seats = st.number_input( 'Seats', min_value = 1, max_value = 10, step = 1)
# location = st.selectbox( 'Location', X['Location'].unique())
# mileage = st.number_input( 'Mileage', min_value = X['Mileage(kmpl)'].min(), max_value = X['Mileage(kmpl)'].max())
# year = st.number_input( 'Year', min_value = 1998, max_value = 2019)
# age = 2022 - year
new_data = X.iloc[:1]
# new_data = pd.DataFrame( {
#     'Location': location,
#     'Kilometers_Driven': kilometers_driven,
#     'Fuel_Type': fuel_type,
#     'Transmission': transmission,
#     'Owner_Type': owner_type,
#     'Engine': engine,
#     'Power': power,
#     'Seats': seats,
#     'Brand': brand,
#     'Model': model,
#     'Age': age,
#     'Mileage(kmpl)': mileage
# }, index = [0])

data_prep = preprocessor.transform( new_data )
pred = model.predict( data_prep )

st.write( 'The predicted price of the car is', pred[0] )
