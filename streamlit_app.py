import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

rating = st.slider('Rating', 1.0, 5.0, 3.5, 0.1)
company_name = st.number_input('Company Name (Encoded)', 0, 100)
job_title = st.number_input('Job Title (Encoded)', 0, 50)
salaries_reported = st.number_input('Salaries Reported', 1, 10)
location = st.number_input('Location (Encoded)', 0, 0)
employment_status = st.number_input('Employment Status (Encoded)', 0, 1)
job_roles = st.number_input('Job Roles (Encoded)', 0, 0)

input_data = pd.DataFrame([[rating, company_name, job_title, salaries_reported, location, employment_status, job_roles]],
                          columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

if st.button('Predict Salary'):
    if model:
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted Salary: ₹ {prediction:,.2f}')
        except Exception as e:
            st.error(f"Prediction error: {e}")
