import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Set page configuration
st.set_page_config(page_title="💼 Employee Salary Prediction", layout="centered")

# Adaptive neutral styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .info-box {
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            background-color: rgba(120, 120, 120, 0.1);
            border: 1px solid rgba(120, 120, 120, 0.2);
        }
        ul {
            margin: 0.5rem 0;
            padding-left: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Salary Prediction App")
st.markdown("### Enter Employee Details Below to Predict Salary Category:")

# UI Inputs
with st.container():
    age = st.slider("📅 Age", 17, 75, 30)
    workclass = st.selectbox("🏢 Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
        'State-gov', 'Others'
    ])
    fnlwgt = st.number_input("📊 Fnlwgt", min_value=10000, max_value=1000000, value=100000)
    educational_num = st.slider("🎓 Educational-num", 5, 16, 10)
    st.caption("ℹ️ Higher numbers represent higher education levels (e.g., 9 = High School Graduate, 13 = Bachelor's, 16 = Professional Degree)")

    
    
    # Added missing fields
    marital_status = st.selectbox("💍 Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
        'Married-spouse-absent', 'Married-AF-spouse'
    ])
    
    occupation = st.selectbox("🔧 Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces', 'Others'
    ])
    
    relationship = st.selectbox("👨‍👩‍👧‍👦 Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    
    race = st.selectbox("🌍 Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    
    gender = st.selectbox("⚧ Gender", ['Male', 'Female'])
    
    capital_gain = st.number_input("💰 Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("📉 Capital Loss", min_value=0, max_value=4356, value=0)
    
    hours_per_week = st.slider("⏰ Hours per week", 1, 99, 40)
    native_country = st.selectbox("🌎 Native Country", [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
        'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
        'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
        'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
        'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands', 'Others-country'
    ])

# Create input dictionary with all required features
input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

input_df = pd.DataFrame([input_dict])

# --- Encoders ---
workclass_encoder = LabelEncoder()
workclass_encoder.classes_ = np.array(['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Others'])

marital_status_encoder = LabelEncoder()
marital_status_encoder.classes_ = np.array(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])

occupation_encoder = LabelEncoder()
occupation_encoder.classes_ = np.array(['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Others'])

relationship_encoder = LabelEncoder()
relationship_encoder.classes_ = np.array(['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])

race_encoder = LabelEncoder()
race_encoder.classes_ = np.array(['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])

gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['Female', 'Male'])

native_country_encoder = LabelEncoder()
native_country_encoder.classes_ = np.array([
    'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia', 'Others-country'
])

# Encode categorical variables
input_df['workclass'] = workclass_encoder.transform([input_df['workclass'][0]])
input_df['marital-status'] = marital_status_encoder.transform([input_df['marital-status'][0]])
input_df['occupation'] = occupation_encoder.transform([input_df['occupation'][0]])
input_df['relationship'] = relationship_encoder.transform([input_df['relationship'][0]])
input_df['race'] = race_encoder.transform([input_df['race'][0]])
input_df['gender'] = gender_encoder.transform([input_df['gender'][0]])
input_df['native-country'] = native_country_encoder.transform([input_df['native-country'][0]])

# --- Prediction ---
if st.button("🔍 Predict Salary Category"):
    try:
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
            <div class="info-box">
                <h4>📢 <b>Predicted Salary Category:</b> <span>{prediction}</span></h4>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all input values are valid and try again.")