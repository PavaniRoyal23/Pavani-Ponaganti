import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = 'Trained_model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Function to make predictions
def predict(features):
    prediction = loaded_model.predict([features])
    return prediction[0]

# Streamlit app
st.title('Bankruptcy Prediction App')

st.write("""
## Predict the probability of a business going bankrupt
""")

# Input features
st.header('Input Features')

# Create input fields for each feature
industrial_risk = st.number_input('Industrial Risk')
management_risk = st.number_input('Management Risk')
financial_flexibility = st.number_input('Financial Flexibility')
credibility = st.number_input('Credibility')
competitiveness = st.number_input('Competitiveness')
operating_risk = st.number_input('Operating Risk')

# Collect the input features into a numpy array
features = np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk])

# Make prediction
if st.button('Predict'):
    result = predict(features)
    st.write(f'The prediction is: {result}')
