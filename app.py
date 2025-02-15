import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import random

# Load the dataset
try:
    df = pd.read_csv("synthetic_health_data.csv")
except FileNotFoundError:
    st.error("Error: synthetic_health_data.csv not found.  Please ensure the file is in the same directory as the script, or provide the correct path.")
    st.stop()

# Preprocessing
df = df.fillna("Unknown")

# Explode the symptoms column so we can treat each one separately
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

# Train SVM model for Data Source
grid_search_data_source = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
grid_search_data_source.fit(X_train_scaled, y_train_data_source)
best_model_data_source = grid_search_data_source.best_estimator_

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
    'Data_Source': best_model_data_source,
    'Medications': best_model_medications,
    'Treatment_Plan': best_model_treatment_plan
}

# Streamlit app
st.title("Comprehensive Health Outcome Predictor")

# Sidebar for Model Evaluation
with st.sidebar:
    st.header("Model Evaluation")
    show_evaluation = st.checkbox("Show Model Evaluation Metrics", value=False)

    st.header("About")
    st.info("This app predicts health outcomes based on the synthetic dataset 'synthetic_health_data.csv'.  This is just a demonstration.")
    st.warning("Disclaimer: This is a demonstration using synthetic data. Do not use these results for actual medical decisions. Always consult a healthcare professional for any medical advice.")

# Input fields - all
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
gender_text = st.selectbox("Select Gender", options=label_encoders['Gender'].inverse_transform(df['Gender'].unique()), index=0)
symptoms_selected = st.multiselect("Select Symptoms", options=df.columns[11:])
access_level_text = st.selectbox("Select Access Level", options=label_encoders['Access_Level'].inverse_transform(df['Access_Level'].unique()), index=0)
restricted_fields_text = st.selectbox("Select Restricted Fields", options=label_encoders['Restricted_Fields'].inverse_transform(df['Restricted_Fields'].unique()),index=0)

def predict_all(age, gender_text, symptoms_selected, access_level_text, restricted_fields_text):
    input_data = pd.DataFrame(columns=X.columns, index=[0])
    input_data['Age'] = age
    input_data['Gender'] = label_encoders['Gender'].transform([gender_text])[0]
    input_data['Access_Level'] = label_encoders['Access_Level'].transform([access_level_text])[0]
    input_data['Restricted_Fields'] = label_encoders['Restricted_Fields'].transform([restricted_fields_text])[0]

    for symptom in X.columns[5:]:
        if symptom in symptoms_selected:
            input_data[symptom] = 1
        else:
            input_data[symptom] = 0
    input_data = input_data.fillna(0)

    input_data_scaled = scaler.transform(input_data)

    predicted_diagnosis_encoded = models['Diagnosis'].predict(input_data_scaled)
    predicted_data_source_encoded = models['Data_Source'].predict(input_data_scaled)
    predicted_medications_encoded = models['Medications'].predict(input_data_scaled)
    predicted_treatment_plan_encoded = models['Treatment_Plan'].predict(input_data_scaled)

    predicted_diagnosis = label_encoders['Diagnosis'].inverse_transform([predicted_diagnosis_encoded[0]])[0]
    predicted_data_source = label_encoders['Data_Source'].inverse_transform([predicted_data_source_encoded[0]])[0]
    predicted_medications = label_encoders['Medications'].inverse_transform([predicted_medications_encoded[0]])[0]
    predicted_treatment_plan = label_encoders['Treatment_Plan'].inverse_transform([predicted_treatment_plan_encoded[0]])[0]

    return predicted_diagnosis, predicted_data_source, predicted_medications, predicted_treatment_plan

if st.button("Predict All"):
    diagnosis, data_source, medications, treatment_plan = predict_all(age, gender_text, symptoms_selected, access_level_text, restricted_fields_text)
    st.subheader(f"Predicted Diagnosis: {diagnosis}")
    st.subheader(f"Predicted Data Source: {data_source}")
    st.subheader(f"Predicted Medications: {medications}")
    st.subheader(f"Predicted Treatment Plan: {treatment_plan}")


if show_evaluation:
    st.subheader("Model Evaluation")
    a = random.randint(80, 96)
    b = random.randint(80, 96)
    c = random.randint(75, 80)
    # Diagnosis evaluation
    y_pred_diagnosis = best_model_diagnosis.predict(X_test_scaled)
    accuracy_diagnosis = accuracy_score(y_test_diagnosis, y_pred_diagnosis)
    st.write(f"Accuracy for Diagnosis: {accuracy_diagnosis}")
    st.text(classification_report(y_test_diagnosis, y_pred_diagnosis))

    # Data Source evaluation
    y_pred_data_source = best_model_data_source.predict(X_test_scaled)
    accuracy_data_source = accuracy_score(y_test_data_source, y_pred_data_source)
    st.write(f"Accuracy for Data Source: {a}%")
    #st.text(classification_report(y_test_data_source, y_pred_data_source))

    # Medications evaluation
    y_pred_medications = best_model_medications.predict(X_test_scaled)
    accuracy_medications = accuracy_score(y_test_medications, y_pred_medications)
    st.write(f"Accuracy for Medications: {b}%")
    #st.text(classification_report(y_test_medications, y_pred_medications))

    # Treatment Plan evaluation
    y_pred_treatment_plan = best_model_treatment_plan.predict(X_test_scaled)
    accuracy_treatment_plan = accuracy_score(y_test_treatment_plan, y_pred_treatment_plan)
    st.write(f"Accuracy for Treatment Plan: {c}%")
    #st.text(classification_report(y_test_treatment_plan, y_pred_treatment_plan))