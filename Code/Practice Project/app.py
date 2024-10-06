import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.title('Employee Attrition Prediction')


data = pd.read_csv('data_cleaned.csv')

age = st.number_input('Age', min_value=data.Age.min(),  max_value=data.Age.max())

gender = st.selectbox('Gender', data.Gender.unique())

monthly_income = st.number_input('Monthly Income', min_value=data['Monthly Income'].min(),  max_value=data['Monthly Income'].max())

work_life_balance = st.selectbox('Work-Life Balance', data['Work-Life Balance'].unique())

job_satisfaction = st.selectbox('Job Satisfaction', data['Job Satisfaction'].unique())

performance_rating = st.selectbox('Performance Rating', data['Performance Rating'].unique())

number_of_promotions = st.selectbox('Number of Promotions', np.arange(5))

overtime = st.selectbox('Overtime', data['Overtime'].unique())

distance_from_home = st.number_input('Distance from Home', min_value=data['Distance from Home'].min(),  max_value=data['Distance from Home'].max())

education_level = st.selectbox('Education Level', data['Education Level'].unique())

marital_status = st.selectbox('Marital Status', data['Marital Status'].unique())

number_of_dependents = st.selectbox('Number of Dependents', np.arange(7))

job_level = st.selectbox('Job Level', data['Job Level'].unique())

remote_work = st.selectbox('Remote Work', data['Remote Work'].unique())

company_reputation = st.selectbox('Company Reputation', data['Company Reputation'].unique())

def categorize_dependents(x):
    if x <= 3:
        return 'low'
    else:
        return 'high'
    
def categorize_promotions(x):
    if x <= 2:
        return 'low'
    else:
        return 'high'
    
number_of_dependents = categorize_dependents(number_of_dependents)
number_of_promotions = categorize_promotions(number_of_promotions)


data = pd.DataFrame({'Age': [age],
                     'Gender': [gender],
                     'Monthly Income': [monthly_income],
                     'Work-Life Balance': [work_life_balance],
                     'Job Satisfaction': [job_satisfaction],
                     'Performance Rating': [performance_rating],
                     'Number of Promotions': [number_of_promotions],
                     'Overtime': [overtime],
                     'Distance from Home': [distance_from_home],
                     'Education Level': [education_level],
                     'Marital Status': [marital_status],
                     'Number of Dependents': [number_of_dependents],
                     'Job Level': [job_level],
                     'Remote Work': [remote_work],
                     'Company Reputation': [company_reputation]
                    })

prep = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


X_prep = prep.transform(data)
y_pred = model.predict(X_prep)

st.success('Predicted Attrition: {}'.format(y_pred[0]))

