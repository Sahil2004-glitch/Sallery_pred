
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for features
rating = st.slider('Rating', min_value=1.0, max_value=5.0, value=3.5, step=0.1)
company_name = st.number_input('Company Name (Encoded)', min_value=0, value=100)
job_title = st.number_input('Job Title (Encoded)', min_value=0, value=50)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=10)
location = st.number_input('Location (Encoded)', min_value=0, value=0)
employment_status = st.number_input('Employment Status (Encoded)', min_value=0, value=1)
job_roles = st.number_input('Job Roles (Encoded)', min_value=0, value=0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[rating, company_name, job_title, salaries_reported, location, employment_status, job_roles]],
                            columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'The predicted salary is: {prediction:,.2f}')
