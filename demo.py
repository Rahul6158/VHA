import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from fpdf import FPDF
from googletrans import Translator

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Encode categorical features
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Feature Engineering: Symptoms handling
df['symptoms'] = df['symptoms'].fillna('').apply(lambda x: x.split(','))
all_symptoms = list(set([s for sublist in df['symptoms'] for s in sublist]))
for symptom in all_symptoms:
    df[symptom] = df['symptoms'].apply(lambda x: 1 if symptom in x else 0)
df.drop(columns=['symptoms'], inplace=True)

# Define features and targets
X = df.drop(columns=['diagnosis', 'medications', 'treatment_plan'])
y_diagnosis = df['diagnosis']
y_medications = df['medications']
y_treatment = df['treatment_plan']

# Split dataset
X_train, X_test, y_d_train, y_d_test = train_test_split(X, y_diagnosis, test_size=0.2, random_state=42)
X_train, X_test, y_m_train, y_m_test = train_test_split(X, y_medications, test_size=0.2, random_state=42)
X_train, X_test, y_t_train, y_t_test = train_test_split(X, y_treatment, test_size=0.2, random_state=42)

# Model training with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
svm_diagnosis = GridSearchCV(SVC(), param_grid, cv=3)
svm_diagnosis.fit(X_train, y_d_train)
svm_medications = GridSearchCV(SVC(), param_grid, cv=3)
svm_medications.fit(X_train, y_m_train)
svm_treatment = GridSearchCV(SVC(), param_grid, cv=3)
svm_treatment.fit(X_train, y_t_train)

# Save models
joblib.dump(svm_diagnosis, "svm_diagnosis.pkl")
joblib.dump(svm_medications, "svm_medications.pkl")
joblib.dump(svm_treatment, "svm_treatment.pkl")

# Load models
svm_diagnosis = joblib.load("svm_diagnosis.pkl")
svm_medications = joblib.load("svm_medications.pkl")
svm_treatment = joblib.load("svm_treatment.pkl")

# Multilingual Support
translator = Translator()
languages = {"English": "en", "Telugu": "te", "Hindi": "hi"}

def translate_text(text, lang):
    return translator.translate(text, dest=lang).text

# Streamlit App
st.title("Health Diagnosis Prediction")
language = st.selectbox("Select Language", list(languages.keys()))

def get_prediction(features):
    diagnosis = svm_diagnosis.predict([features])[0]
    medications = svm_medications.predict([features])[0]
    treatment = svm_treatment.predict([features])[0]
    return diagnosis, medications, treatment

st.sidebar.header("Enter Patient Details")
user_data = []
for col in X.columns:
    user_data.append(st.sidebar.text_input(col, "0"))
user_data = np.array(user_data, dtype=float)

if st.sidebar.button("Predict"):
    diagnosis, medications, treatment = get_prediction(user_data)
    st.write("Diagnosis:", translate_text(diagnosis, languages[language]))
    st.write("Medications:", translate_text(medications, languages[language]))
    st.write("Treatment Plan:", translate_text(treatment, languages[language]))

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Health Report", ln=True, align='C')
    pdf.cell(200, 10, f"Diagnosis: {diagnosis}", ln=True)
    pdf.cell(200, 10, f"Medications: {medications}", ln=True)
    pdf.cell(200, 10, f"Treatment Plan: {treatment}", ln=True)
    pdf.output("Health_Report.pdf")
    st.success("Report Generated: Health_Report.pdf")
