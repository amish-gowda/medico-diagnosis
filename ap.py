import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Helper function to load models
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file not found: {file_path}")
        return None

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
models = {
    "diabetes": load_model(f"{working_dir}/saved_models/diabetes_model.sav"),
    "heart": load_model(f"{working_dir}/saved_models/heart_disease_model.sav"),
    "parkinsons": load_model(f"{working_dir}/saved_models/parkinsons_model.sav"),
}

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Medico Diagnosis System',
        ['Heart Disease Prediction', 'Diabetes Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['heart', 'activity', 'person'],
        default_index=0,
    )

# Function for prediction logic
def make_prediction(model, inputs, feature_names):
    try:
        user_input = [float(inputs[feature]) for feature in feature_names]
        prediction = model.predict([user_input])
        return prediction[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None
    
def st_inputs_render(feature_data):

    inputs = {}

    # Iterate through the feature data
    for feature, data in feature_data.items():
        # Handle the case for dropdown inputs (options are required)
        if data["options_req"]:
            sel_option = st.selectbox(
                feature.replace('_', ' ').capitalize(),  # Capitalize the feature name for display
                options=data["options"],  # Provide the options from JSON
                help=data["descp"]  # Display the description as help
            )
            inputs[feature] = data["options"].index(sel_option)
        else:            
            # If no max range is provided, just accept any number
            inputs[feature] = st.number_input(
                feature.replace('_', ' ').capitalize(),  # Capitalize the feature name for display
                value=0.0,  # Default value
                help=data["descp"]  # Display the description as help
            )        
    return inputs

# Load Json File 
def load_json(json_file):
    with open(f"{working_dir}/{json_file}", 'r') as f:
        feature_data = json.load(f)
    return feature_data


# Heart Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    feature_data = load_json("heart_input.json")

    # Render inputs of feature data
    inputs = st_inputs_render(feature_data=feature_data)    

    if st.button('Heart Disease Test Result'):
        prediction = make_prediction(models['heart'], inputs, feature_data)
        if prediction is not None:
            st.success("The person has heart disease" if prediction == 1 else "The person does not have heart disease")


# Parkinson's Prediction
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    
    feature_data = load_json("parkinsons_input.json")

    # Render inputs of feature data
    inputs = st_inputs_render(feature_data=feature_data)
    
    if st.button("Parkinson's Test Result"):
        prediction = make_prediction(models['parkinsons'], inputs, feature_data)
        if prediction is not None:
            st.success("The person has Parkinson's disease" if prediction == 1 else "The person does not have Parkinson's disease")

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    feature_data = load_json("diabetes_input.json")

    # Render inputs of feature data
    inputs = st_inputs_render(feature_data=feature_data)

    if st.button('Diabetes Test Result'):
        prediction = make_prediction(models['diabetes'], inputs, feature_data)
        if prediction is not None:
            st.success("The person is diabetic" if prediction == 1 else "The person is not diabetic")
