import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random
from gtts import gTTS
import base64
from fpdf import FPDF
import datetime

# Load the dataset
try:
    df = pd.read_csv("synthetic_health_data.csv")
except FileNotFoundError:
    st.error("Error: synthetic_health_data.csv not found. Please ensure the file is in the same directory as the script, or provide the correct path.")
    st.stop()

# Preprocessing
df = df.fillna("Unknown")

# Explode the symptoms column so we can treat each one separately
s = df['Symptoms'].str.split(',').explode()
df = df.join(pd.crosstab(s.index, s))
df = df.drop('Symptoms', axis=1)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Gender', 'Medical_History', 'Diagnosis', 'Data_Source', 'Restricted_Fields', 'Access_Level', 'Medications', 'Treatment_Plan']
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Prepare data for model training
X = df.drop(['Patient_ID', 'Data_Source', 'Diagnosis', 'Medical_History', 'Patient_Query', 'LLM_Response', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Cholesterol', 'Blood_Sugar', 'Medications', 'Treatment_Plan'], axis=1)

# Prepare target variables
y_diagnosis = df['Diagnosis']
y_medications = df['Medications']
y_treatment_plan = df['Treatment_Plan']

# Split data
X_train, X_test, y_train_diagnosis, y_test_diagnosis, y_train_medications, y_test_medications, y_train_treatment_plan, y_test_treatment_plan = train_test_split(
    X, y_diagnosis, y_medications, y_treatment_plan, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for SVM
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Train SVM model for Diagnosis
grid_search_diagnosis = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
grid_search_diagnosis.fit(X_train_scaled, y_train_diagnosis)
best_model_diagnosis = grid_search_diagnosis.best_estimator_

# Train SVM model for Medications
grid_search_medications = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
grid_search_medications.fit(X_train_scaled, y_train_medications)
best_model_medications = grid_search_medications.best_estimator_

# Train SVM model for Treatment Plan
grid_search_treatment_plan = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
grid_search_treatment_plan.fit(X_train_scaled, y_train_treatment_plan)
best_model_treatment_plan = grid_search_treatment_plan.best_estimator_

# Store the models
models = {
    'Diagnosis': best_model_diagnosis,
    'Medications': best_model_medications,
    'Treatment_Plan': best_model_treatment_plan
}

# Language selection
language = st.sidebar.selectbox("Select Language", options=["English", "Telugu", "Hindi"], index=0)

# Hardcoded translations
translations = {
    "English": {
        "title": "Comprehensive Health Outcome Predictor",
        "model_evaluation": "Model Evaluation",
        "show_evaluation_metrics": "Show Model Evaluation Metrics",
        "about": "About",
        "info": "This app predicts health outcomes based on the synthetic dataset 'synthetic_health_data.csv'. This is just a demonstration.",
        "warning": "Disclaimer: This is a demonstration using synthetic data. Do not use these results for actual medical decisions. Always consult a healthcare professional for any medical advice.",
        "enter_age": "Enter Age",
        "select_gender": "Select Gender",
        "select_symptoms": "Select Symptoms",
        "select_access_level": "Select Access Level",
        "select_restricted_fields": "Select Restricted Fields",
        "predict_all": "Predict All",
        "predicted_diagnosis": "Predicted Diagnosis",
        "predicted_medications": "Predicted Medications",
        "predicted_treatment_plan": "Predicted Treatment Plan",
        "accuracy_for_diagnosis": "Accuracy for Diagnosis",
        "accuracy_for_medications": "Accuracy for Medications",
        "accuracy_for_treatment_plan": "Accuracy for Treatment Plan",
        "generate_report": "Generate Report",
        "enter_name": "Enter your name",
        "download_report": "Download PDF Report",
        "date": "Date"
    },
    "Telugu": {
        "title": "వ్యాప్త ఆరోగ్య ఫలితాన్వేషి",
        "model_evaluation": "మోడల్ మూల్యాంకనం",
        "show_evaluation_metrics": "మోడల్ మూల్యాంకన మెట్రిక్స్ చూపించు",
        "about": "గురించి",
        "info": "ఈ అనువర్తనం 'synthetic_health_data.csv' అనే కృత్రిమ డేటాసెట్ ఆధారంగా ఆరోగ్య ఫలితాలను అంచనా వేస్తుంది. ఇది కేవలం ఒక డెమోను.",
        "warning": "సూచన: ఇది కృత్రిమ డేటాను ఉపయోగించి ఒక డెమోను. నిజమైన వైద్య నిర్ణయాలకు ఈ ఫలితాలను ఉపయోగించవద్దు. ఏదైనా వైద్య సలహా కోసం ఎల్లప్పుడూ ఒక ఆరోగ్య సంరక్షణ నిపుణుడిని సంప్రదించండి.",
        "enter_age": "వయస్సును నమోదు చేయండి",
        "select_gender": "లింగాన్ని ఎంచుకోండి",
        "select_symptoms": "లక్షణాలను ఎంచుకోండి",
        "select_access_level": "ప్రవేశ స్థాయిని ఎంచుకోండి",
        "select_restricted_fields": "ఆంక్షలు ఉన్న రంగాలను ఎంచుకోండి",
        "predict_all": "అన్ని అంచనా వేయండి",
        "predicted_diagnosis": "అంచనా విధానం",
        "predicted_medications": "అంచనా మందులు",
        "predicted_treatment_plan": "అంచనా చికిత్స ప్రణాళిక",
        "accuracy_for_diagnosis": "విధానం ఖచ్చితత్వం",
        "accuracy_for_medications": "మందుల ఖచ్చితత్వం",
        "accuracy_for_treatment_plan": "చికిత్స ప్రణాళిక ఖచ్చితత్వం",
        "generate_report": "రిపోర్ట్ జనరేట్ చేయండి",
        "enter_name": "మీ పేరు నమోదు చేయండి",
        "download_report": "PDF రిపోర్ట్ డౌన్లోడ్ చేయండి",
        "date": "తేదీ"
    },
    "Hindi": {
        "title": "व्यापक स्वास्थ्य परिणाम पूर्वानुमानक",
        "model_evaluation": "मॉडल मूल्यांकन",
        "show_evaluation_metrics": "मॉडल मूल्यांकन मीट्रिक दिखाएं",
        "about": "के बारे में",
        "info": "यह ऐप सिंथेटिक डेटा सेट 'synthetic_health_data.csv' के आधार पर स्वास्थ्य परिणामों की भविष्यवाणी करता है। यह सिर्फ एक डेमो है।",
        "warning": "अस्वीकरण: यह एक डेमो है जो सिंथेटिक डेटा का उपयोग कर रहा है। वास्तविक चिकित्सा निर्णयों के लिए इन परिणामों का उपयोग न करें। किसी भी चिकित्सा सलाह के लिए हमेशा एक स्वास्थ्य देखभाल पेशेवर से परामर्श करें।",
        "enter_age": "आयु दर्ज करें",
        "select_gender": "लिंग चुनें",
        "select_symptoms": "लक्षण चुनें",
        "select_access_level": "प्रवेश स्तर चुनें",
        "select_restricted_fields": "प्रतिबंधित क्षेत्र चुनें",
        "predict_all": "सभी की भविष्यवाणी करें",
        "predicted_diagnosis": "अनुमानित निदान",
        "predicted_medications": "अनुमानित दवाएं",
        "predicted_treatment_plan": "अनुमानित उपचार योजना",
        "accuracy_for_diagnosis": "निदान के लिए सटीकता",
        "accuracy_for_medications": "दवाओं के लिए सटीकता",
        "accuracy_for_treatment_plan": "उपचार योजना के लिए सटीकता",
        "generate_report": "रिपोर्ट जनरेट करें",
        "enter_name": "अपना नाम दर्ज करें",
        "download_report": "PDF रिपोर्ट डाउनलोड करें",
        "date": "तारीख"
    }
}

def translate_text(key, lang):
    return translations[lang].get(key, key)

# Streamlit app
st.title(translate_text("title", language))

# Sidebar for Model Evaluation
with st.sidebar:
    st.header(translate_text("model_evaluation", language))
    show_evaluation = st.checkbox(translate_text("show_evaluation_metrics", language), value=False)

    st.header(translate_text("about", language))
    st.info(translate_text("info", language))
    st.warning(translate_text("warning", language))

# Input fields - all
name = st.text_input(translate_text("enter_name", language))  # Ask for the user's name
age = st.number_input(translate_text("enter_age", language), min_value=0, max_value=120, value=30)
gender_text = st.selectbox(translate_text("select_gender", language), options=label_encoders['Gender'].inverse_transform(df['Gender'].unique()), index=0)
symptoms_selected = st.multiselect(translate_text("select_symptoms", language), options=df.columns[11:])
access_level_text = st.selectbox(translate_text("select_access_level", language), options=label_encoders['Access_Level'].inverse_transform(df['Access_Level'].unique()), index=0)
restricted_fields_text = st.selectbox(translate_text("select_restricted_fields", language), options=label_encoders['Restricted_Fields'].inverse_transform(df['Restricted_Fields'].unique()), index=0)

# Generate audio file
        audio_file = generate_audio_file(diagnosis, medications, treatment_plan, language)
        st.audio(audio_file, format='audio/mp3')

        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, txt="Health Prediction Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Predicted Diagnosis:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"{diagnosis}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Predicted Medications:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"{medications}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Predicted Treatment Plan:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"{treatment_plan}")
        pdf.ln(10)

        pdf_file = "health_prediction_report.pdf"
        pdf.output(pdf_file)

        # Provide download link for the PDF
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Download Health Report",
                data=file,
                file_name=pdf_file,
                mime="application/pdf"
            )
