import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model safely
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('best_model.pkl'):
            st.error("❌ Model file 'best_model.pkl' not found. Please upload it.")
            return None
        
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

st.title('💰 Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields
rating = st.slider('Rating', 1.0, 5.0, 3.5, 0.1)
company_name = st.number_input('Company Name (Encoded)', 0, 100)
job_title = st.number_input('Job Title (Encoded)', 0, 50)
salaries_reported = st.number_input('Salaries Reported', 1, 10)
location = st.number_input('Location (Encoded)', 0, 0)
employment_status = st.number_input('Employment Status (Encoded)', 0, 1)
job_roles = st.number_input('Job Roles (Encoded)', 0, 0)

# Create DataFrame
input_data = pd.DataFrame(
    [[rating, company_name, job_title, salaries_reported, location, employment_status, job_roles]],
    columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
)

# Prediction
if st.button('Predict Salary'):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'💰 Predicted Salary: ₹ {prediction:,.2f}')
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("⚠️ Model is not loaded properly.")
