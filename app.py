import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from fpdf import FPDF
from gtts import gTTS
import os

# Load the dataset
try:
    df = pd.read_csv("synthetic_health_data.csv")
except FileNotFoundError:
    st.error("Error: synthetic_health_data.csv not found. Please ensure the file is in the same directory as the script, or provide the correct path.")
    st.stop()

# Preprocessing
df = df.fillna("Unknown")
s = df['Symptoms'].str.split(',').explode()
df = df.join(pd.crosstab(s.index, s))
df = df.drop('Symptoms', axis=1)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Gender', 'Medical_History', 'Diagnosis', 'Data_Source', 'Restricted_Fields','Access_Level','Medications','Treatment_Plan']
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Prepare data for model training
X = df.drop(['Patient_ID','Data_Source', 'Diagnosis', 'Medical_History', 'Patient_Query', 'LLM_Response', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Cholesterol', 'Blood_Sugar','Medications','Treatment_Plan'], axis=1)

# Prepare target variables
y_diagnosis = df['Diagnosis']
y_data_source = df['Data_Source']
y_medications = df['Medications']
y_treatment_plan = df['Treatment_Plan']

# Split data
X_train, X_test, y_train_diagnosis, y_test_diagnosis, y_train_data_source, y_test_data_source, y_train_medications, y_test_medications, y_train_treatment_plan, y_test_treatment_plan = train_test_split(
    X, y_diagnosis, y_data_source, y_medications, y_treatment_plan, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

def train_model(y_train, y_test):
    grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_

models = {
    'Diagnosis': train_model(y_train_diagnosis, y_test_diagnosis),
    'Data_Source': train_model(y_train_data_source, y_test_data_source),
    'Medications': train_model(y_train_medications, y_test_medications),
    'Treatment_Plan': train_model(y_train_treatment_plan, y_test_treatment_plan)
}

st.title("Comprehensive Health Outcome Predictor")

# Collect user input
name = st.text_input("Enter your Name (Optional)")
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
gender_text = st.selectbox("Select Gender", options=label_encoders['Gender'].inverse_transform(df['Gender'].unique()), index=0)
symptoms_selected = st.multiselect("Select Symptoms", options=df.columns[11:])
access_level_text = st.selectbox("Select Access Level", options=label_encoders['Access_Level'].inverse_transform(df['Access_Level'].unique()), index=0)
restricted_fields_text = st.selectbox("Select Restricted Fields", options=label_encoders['Restricted_Fields'].inverse_transform(df['Restricted_Fields'].unique()), index=0)

def generate_report(name, age, diagnosis, data_source, medications, treatment_plan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Health Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Diagnosis: {diagnosis}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Data Source: {data_source}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Medications: {medications}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Treatment Plan: {treatment_plan}", ln=True)
    pdf_file = "health_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

def generate_audio(text):
    tts = gTTS(text)
    audio_file = "health_report.mp3"
    tts.save(audio_file)
    return audio_file

if st.button("Predict All"):
    diagnosis, data_source, medications, treatment_plan = models['Diagnosis'].predict([X_test_scaled[0]])[0], models['Data_Source'].predict([X_test_scaled[0]])[0], models['Medications'].predict([X_test_scaled[0]])[0], models['Treatment_Plan'].predict([X_test_scaled[0]])[0]
    st.subheader(f"Predicted Diagnosis: {diagnosis}")
    st.subheader(f"Predicted Data Source: {data_source}")
    st.subheader(f"Predicted Medications: {medications}")
    st.subheader(f"Predicted Treatment Plan: {treatment_plan}")
    
    report_file = generate_report(name, age, diagnosis, data_source, medications, treatment_plan)
    st.download_button("Download Report", report_file, "health_report.pdf")
    
    audio_text = f"Your predicted diagnosis is {diagnosis}. Recommended medications: {medications}. Suggested treatment plan: {treatment_plan}."
    audio_file = generate_audio(audio_text)
    st.audio(audio_file)
    st.download_button("Download Audio", audio_file, "health_report.mp3")
