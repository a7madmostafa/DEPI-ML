
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Car Price Prediction App')

reg = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('cleaned_data.csv')


brand = st.selectbox( 'Brand', df['Brand'].unique())
model = st.selectbox( 'Model', df['Model'].unique())
kilometers_driven = st.number_input( 'Kilometers Driven', min_value = df['Kilometers_Driven'].min(), max_value = df['Kilometers_Driven'].max())
fuel_type = st.selectbox( 'Fuel Type', df['Fuel_Type'].unique())
transmission = st.selectbox('Transmission',  df['Transmission'].unique())
owner_type = st.selectbox( 'Owner Type', df['Owner_Type'].unique())
engine = st.number_input( 'Engine', min_value = df['Engine'].min(), max_value = df['Engine'].max())
power = st.number_input('Power', min_value = df['Power'].min(), max_value = df['Power'].max())
seats = st.number_input( 'Seats', min_value = df['Seats'].min(), max_value = df['Seats'].max())
location = st.selectbox( 'Location', df['Location'].unique())
mileage = st.number_input( 'Mileage', min_value = df['Mileage(kmpl)'].min(), max_value = df['Mileage(kmpl)'].max())
year = st.number_input( 'Year', min_value = 1998, max_value = 2019)
age = 2022 - year
new_data = pd.DataFrame( {
    'Location': location,
    'Kilometers_Driven': kilometers_driven,
    'Fuel_Type': fuel_type,
    'Transmission': transmission,
    'Owner_Type': owner_type,
    'Engine': engine,
    'Power': power,
    'Seats': seats,
    'Brand': brand,
    'Model': model,
    'Age': age,
    'Mileage(kmpl)': mileage
}, index = [0])

pred = reg.predict( new_data )
price = np.exp( pred )
st.write( 'The predicted price of the car is', price[0].round(2) )