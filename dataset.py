from faker import Faker
import random
import pandas as pd
import numpy as np

fake = Faker()

# --- Comprehensive Data Structures ---
COMMON_DISEASES = [
    "Hypertension", "Type 2 Diabetes", "Asthma", "Osteoarthritis", "Depression",
    "Anxiety", "Migraine", "Common Cold", "Influenza", "Pneumonia",
    "Bronchitis", "Urinary Tract Infection (UTI)", "Gastroenteritis", "Allergic Rhinitis",
    "Eczema", "Psoriasis", "Hypothyroidism", "Hyperthyroidism", "Coronary Artery Disease",
    "Congestive Heart Failure", "Chronic Kidney Disease", "Parkinson's Disease", "Alzheimer's Disease",
    "Multiple Sclerosis", "Lupus", "Rheumatoid Arthritis", "COPD", "Stroke", "Epilepsy",
    "Celiac Disease", "Irritable Bowel Syndrome (IBS)", "Glaucoma", "Macular Degeneration",
    "Prostate Cancer", "Breast Cancer", "Lung Cancer", "Colorectal Cancer"
]

SYMPTOMS_BY_DISEASE = {
    "Hypertension": ["Headache", "Dizziness", "Blurred vision", "Nosebleeds", "Chest pain", "Fatigue", "No symptoms"],
    "Type 2 Diabetes": ["Frequent urination", "Excessive thirst", "Unexplained weight loss", "Fatigue", "Blurred vision", "Slow-healing sores", "Increased hunger", "Numbness in hands and feet"],
    "Asthma": ["Wheezing", "Shortness of breath", "Chest tightness", "Coughing", "Difficulty breathing", "Rapid breathing", "Nighttime coughing"],
    "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Decreased range of motion", "Grating sensation in joint", "Pain that worsens with activity", "Joint locking"],
    "Depression": ["Persistent sadness", "Loss of interest", "Fatigue", "Changes in appetite", "Sleep disturbances", "Difficulty concentrating", "Feelings of worthlessness", "Suicidal thoughts"],
    "Anxiety": ["Excessive worry", "Restlessness", "Irritability", "Muscle tension", "Sleep problems", "Panic attacks", "Rapid heartbeat", "Difficulty relaxing"],
    "Migraine": ["Severe headache", "Nausea", "Sensitivity to light and sound", "Visual disturbances (aura)", "Throbbing pain", "Dizziness", "Lightheadedness"],
    "Common Cold": ["Runny nose", "Sore throat", "Cough", "Sneezing", "Mild fever", "Congestion", "Body aches", "Fatigue"],
    "Influenza": ["Fever", "Cough", "Body aches", "Fatigue", "Sore throat", "Headache", "Chills", "Sweating"],
    "Pneumonia": ["Cough", "Fever", "Chest pain", "Shortness of breath", "Fatigue", "Confusion (especially in older adults)", "Sweating", "Rapid breathing"],
    "Bronchitis": ["Cough", "Mucus production", "Fatigue", "Shortness of breath", "Chest discomfort", "Wheezing", "Low-grade fever", "Sore throat"],
    "Urinary Tract Infection (UTI)": ["Frequent urination", "Burning sensation during urination", "Urgency", "Cloudy urine", "Pelvic pain", "Strong-smelling urine", "Blood in urine", "Back pain"],
    "Gastroenteritis": ["Diarrhea", "Vomiting", "Abdominal cramps", "Nausea", "Fever", "Headache", "Muscle aches", "Dehydration"],
    "Allergic Rhinitis": ["Sneezing", "Runny nose", "Itchy eyes", "Nasal congestion", "Postnasal drip", "Fatigue", "Headache", "Watery eyes"],
    "Eczema": ["Itchy skin", "Dry skin", "Rash", "Redness", "Inflammation", "Thickened skin", "Small, raised bumps", "Cracked skin"],
    "Psoriasis": ["Scaly skin", "Red patches", "Itching", "Thickened nails", "Joint pain", "Silvery scales", "Dry, cracked skin", "Pitted nails"],
    "Hypothyroidism": ["Fatigue", "Weight gain", "Constipation", "Dry skin", "Sensitivity to cold", "Muscle weakness", "Depression", "Hair loss"],
    "Hyperthyroidism": ["Weight loss", "Rapid heartbeat", "Sweating", "Anxiety", "Irritability", "Tremors", "Difficulty sleeping", "Enlarged thyroid (goiter)"],
    "Coronary Artery Disease": ["Chest pain (angina)", "Shortness of breath", "Fatigue", "Weakness", "Dizziness", "Nausea", "Sweating", "Palpitations"],
    "Congestive Heart Failure": ["Shortness of breath", "Fatigue", "Swelling in legs and ankles", "Rapid heartbeat", "Persistent cough", "Weight gain", "Increased urination at night", "Wheezing"],
    "Chronic Kidney Disease": ["Fatigue", "Swelling in ankles and feet", "Loss of appetite", "Nausea", "Vomiting", "Itching", "Muscle cramps", "Difficulty sleeping"],
    "Parkinson's Disease": ["Tremor", "Stiffness", "Slowed movement (bradykinesia)", "Impaired balance", "Rigidity", "Speech changes", "Writing difficulties", "Loss of automatic movements"],
    "Alzheimer's Disease": ["Memory loss", "Difficulty thinking and reasoning", "Confusion", "Disorientation", "Changes in personality", "Difficulty with language", "Problems with visual and spatial abilities", "Difficulty recognizing family members"],
    "Multiple Sclerosis": ["Fatigue", "Difficulty walking", "Numbness or weakness", "Vision problems", "Muscle spasms", "Balance problems", "Cognitive difficulties", "Pain"],
    "Lupus": ["Fatigue", "Joint pain", "Rash", "Fever", "Sensitivity to sunlight", "Chest pain", "Hair loss", "Mouth ulcers"],
    "Rheumatoid Arthritis": ["Joint pain", "Swelling", "Stiffness", "Fatigue", "Fever", "Loss of joint function", "Nodules under the skin", "Dry eyes and mouth"],
    "COPD": ["Shortness of breath", "Chronic cough", "Wheezing", "Chest tightness", "Increased mucus production", "Fatigue", "Frequent respiratory infections"],
    "Stroke": ["Sudden numbness or weakness", "Difficulty speaking", "Vision problems", "Severe headache", "Loss of balance"],
    "Epilepsy": ["Seizures", "Loss of awareness", "Confusion", "Staring spells", "Uncontrolled muscle movements"],
    "Celiac Disease": ["Diarrhea", "Abdominal pain", "Bloating", "Fatigue", "Weight loss", "Skin rash (dermatitis herpetiformis)"],
    "Irritable Bowel Syndrome (IBS)": ["Abdominal pain", "Bloating", "Gas", "Diarrhea", "Constipation", "Changes in bowel habits"],
    "Glaucoma": ["Gradual vision loss", "Tunnel vision", "Eye pain", "Blurred vision", "Halos around lights"],
    "Macular Degeneration": ["Blurred central vision", "Difficulty seeing in low light", "Distorted vision", "Blind spots"],
    "Prostate Cancer": ["Frequent urination", "Weak urine stream", "Difficulty starting or stopping urination", "Blood in urine or semen", "Erectile dysfunction"],
    "Breast Cancer": ["Lump in breast", "Change in breast size or shape", "Nipple discharge", "Skin changes on breast", "Nipple retraction"],
    "Lung Cancer": ["Persistent cough", "Chest pain", "Shortness of breath", "Wheezing", "Coughing up blood", "Fatigue", "Weight loss"],
    "Colorectal Cancer": ["Change in bowel habits", "Blood in stool", "Abdominal pain", "Unexplained weight loss", "Fatigue"]
}

MEDICATIONS_BY_DISEASE = {
    "Hypertension": ["ACE inhibitors", "ARBs", "Beta-blockers", "Diuretics", "Calcium channel blockers", "Alpha-blockers"],
    "Type 2 Diabetes": ["Metformin", "Sulfonylureas", "DPP-4 inhibitors", "SGLT2 inhibitors", "Insulin", "GLP-1 receptor agonists"],
    "Asthma": ["Inhaled corticosteroids", "Beta-agonists (Albuterol)", "Leukotriene modifiers", "Theophylline", "Omalizumab (Xolair)", "Inhaled anticholinergics"],
    "Osteoarthritis": ["Acetaminophen", "NSAIDs (Ibuprofen, Naproxen)", "Topical pain relievers", "Corticosteroid injections", "Hyaluronic acid injections", "Duloxetine (Cymbalta)"],
    "Depression": ["SSRIs (e.g., Sertraline, Fluoxetine)", "SNRIs (e.g., Venlafaxine, Duloxetine)", "Tricyclic antidepressants", "MAOIs", "Bupropion (Wellbutrin)", "Mirtazapine (Remeron)"],
    "Anxiety": ["SSRIs", "SNRIs", "Benzodiazepines (e.g., Alprazolam, Lorazepam)", "Buspirone", "Pregabalin (Lyrica)", "Hydroxyzine (Vistaril)"],
    "Migraine": ["Triptans (e.g., Sumatriptan)", "NSAIDs", "CGRP inhibitors", "Beta-blockers (for prevention)", "Antidepressants (for prevention)", "Botulinum toxin (Botox) for chronic migraines"],
    "Common Cold": ["Decongestants", "Antihistamines", "Pain relievers (Acetaminophen, Ibuprofen)", "Cough suppressants", "Vitamin C", "Zinc"],
    "Influenza": ["Antiviral medications (e.g., Oseltamivir, Zanamivir)", "Pain relievers", "Rest", "Hydration"],
    "Pneumonia": ["Antibiotics", "Pain relievers", "Cough medicine", "Oxygen therapy (if needed)", "Bronchodilators"],
    "Bronchitis": ["Bronchodilators", "Cough medicine", "Antibiotics (if bacterial)", "Rest", "Hydration", "Steroids (in some cases)"],
    "Urinary Tract Infection (UTI)": ["Antibiotics (e.g., Nitrofurantoin, Trimethoprim/sulfamethoxazole)", "Phenazopyridine (for pain relief)", "Cranberry juice"],
    "Gastroenteritis": ["Oral rehydration solutions", "Anti-diarrheal medications (Loperamide)", "Anti-emetics (for vomiting)", "Rest", "Bland diet", "Probiotics"],
    "Allergic Rhinitis": ["Antihistamines", "Nasal corticosteroids", "Decongestants", "Leukotriene inhibitors", "Immunotherapy (allergy shots)"],
    "Eczema": ["Topical corticosteroids", "Emollients (moisturizers)", "Calcineurin inhibitors", "Antihistamines (for itch)", "Wet wraps", "Phototherapy"],
    "Psoriasis": ["Topical corticosteroids", "Vitamin D analogs", "Retinoids", "Calcineurin inhibitors", "Biologics (for severe cases)", "Phototherapy", "Methotrexate", "Cyclosporine"],
    "Hypothyroidism": ["Levothyroxine (synthetic thyroid hormone)"],
    "Hyperthyroidism": ["Methimazole", "Propylthiouracil", "Beta-blockers (for symptom control)", "Radioactive iodine therapy", "Thyroidectomy (surgical removal of thyroid)"],
    "Coronary Artery Disease": ["Statins", "Aspirin", "Beta-blockers", "ACE inhibitors", "Nitrates (for chest pain)", "Antiplatelet drugs (e.g., Clopidogrel)", "Calcium channel blockers"],
    "Congestive Heart Failure": ["ACE inhibitors", "ARBs", "Beta-blockers", "Diuretics", "Digoxin", "Aldosterone antagonists", "Hydralazine and isosorbide dinitrate"],
    "Chronic Kidney Disease": ["ACE inhibitors", "ARBs", "Erythropoiesis-stimulating agents (ESAs)", "Vitamin D supplements", "Phosphate binders", "Diuretics", "Sodium bicarbonate"],
    "Parkinson's Disease": ["Levodopa", "Carbidopa", "Dopamine agonists", "MAO-B inhibitors", "COMT inhibitors", "Amantadine", "Anticholinergics"],
    "Alzheimer's Disease": ["Cholinesterase inhibitors (e.g., Donepezil, Rivastigmine)", "Memantine"],
    "Multiple Sclerosis": ["Interferon beta", "Glatiramer acetate", "Natalizumab", "Fingolimod", "Dimethyl fumarate", "Ocrelizumab", "Cladribine"],
    "Lupus": ["NSAIDs", "Corticosteroids", "Hydroxychloroquine", "Immunosuppressants (e.g., Methotrexate, Azathioprine)", "Belimumab", "Anifrolumab"],
    "Rheumatoid Arthritis": ["DMARDs (e.g., Methotrexate, Sulfasalazine, Hydroxychloroquine)", "Biologics (e.g., Etanercept, Infliximab)", "NSAIDs", "Corticosteroids", "JAK inhibitors (e.g., Tofacitinib)"],
    "COPD": ["Bronchodilators (Beta-agonists, Anticholinergics)", "Inhaled corticosteroids", "Phosphodiesterase-4 inhibitors", "Antibiotics (for exacerbations)", "Oxygen therapy"],
    "Stroke": ["Thrombolytics (e.g., Alteplase)", "Antiplatelet drugs (e.g., Aspirin, Clopidogrel)", "Anticoagulants (e.g., Warfarin, Apixaban)", "Statins", "Blood pressure medications"],
    "Epilepsy": ["Antiepileptic drugs (AEDs) - e.g., Levetiracetam, Lamotrigine, Valproic acid, Carbamazepine, Phenytoin"],
    "Celiac Disease": ["Gluten-free diet (primary treatment)"],
    "Irritable Bowel Syndrome (IBS)": ["Fiber supplements", "Laxatives", "Anti-diarrheal medications", "Antispasmodics", "Antidepressants (in some cases)", "Probiotics"],
    "Glaucoma": ["Eye drops (e.g., Prostaglandin analogs, Beta-blockers, Alpha-adrenergic agonists)", "Laser therapy", "Surgery"],
    "Macular Degeneration": ["Anti-VEGF injections (e.g., Aflibercept, Ranibizumab)", "Laser photocoagulation (for wet AMD)", "Vitamin and mineral supplements (for dry AMD)"],
    "Prostate Cancer": ["Active surveillance", "Surgery (Prostatectomy)", "Radiation therapy", "Hormone therapy", "Chemotherapy"],
    "Breast Cancer": ["Surgery (Lumpectomy, Mastectomy)", "Radiation therapy", "Chemotherapy", "Hormone therapy", "Targeted therapy"],
    "Lung Cancer": ["Surgery", "Radiation therapy", "Chemotherapy", "Targeted therapy", "Immunotherapy"],
    "Colorectal Cancer": ["Surgery", "Chemotherapy", "Radiation therapy", "Targeted therapy", "Immunotherapy"]
}

#Add a default medication
DEFAULT_MEDICATION = "Symptomatic Treatment"

# If there is no symptom then will add "Rest" and "Hydration"
for disease in COMMON_DISEASES:
    if disease not in MEDICATIONS_BY_DISEASE:
      MEDICATIONS_BY_DISEASE[disease] = [DEFAULT_MEDICATION]

# Severity Levels for Symptoms
SEVERITY_LEVELS = ["Mild", "Moderate", "Severe"]

# Conditional Probability Matrix
DISEASE_PROBABILITIES = {
    "Fever": {"Influenza": 0.7, "Common Cold": 0.6, "Pneumonia": 0.4, "UTI": 0.3, "Gastroenteritis": 0.5, "Bronchitis": 0.2},
    "Cough": {"Common Cold": 0.5, "Influenza": 0.6, "Pneumonia": 0.7, "Bronchitis": 0.8, "Asthma": 0.3, "COPD": 0.4, "Lung Cancer": 0.1},
    "Headache": {"Migraine": 0.4, "Tension Headache": 0.5, "Common Cold": 0.3, "Influenza": 0.4, "Hypertension": 0.2, "Brain Tumor": 0.01},
    "Fatigue": {"Depression": 0.6, "Anxiety": 0.5, "Hypothyroidism": 0.4, "Chronic Fatigue Syndrome": 0.7, "Anemia": 0.3, "Cancer": 0.1},
    "Chest pain": {"Coronary Artery Disease": 0.7, "Angina": 0.6, "Pneumonia": 0.3, "Costochondritis": 0.4, "Anxiety": 0.2, "Heart Attack": 0.8},
    "Shortness of breath": {"Asthma": 0.6, "Pneumonia": 0.5, "COPD": 0.7, "Heart Failure": 0.8, "Anxiety": 0.3, "Pulmonary Embolism": 0.4},
    "Dizziness": {"Vertigo": 0.6, "Dehydration": 0.4, "Low Blood Pressure": 0.5, "Anemia": 0.3, "Migraine": 0.2, "Stroke": 0.1}
}

# Influence of Age and Gender (Customize based on medical knowledge)
AGE_GENDER_INFLUENCE = {
    "Hypertension": {"Age": 0.6, "Gender": {"Male": 0.6, "Female": 0.4}}, # Example: Older age increases hypertension risk
    "Prostate Cancer": {"Age": 0.9, "Gender": {"Male": 1.0, "Female": 0.001}}, #Highly unlikely to be in female
    "Breast Cancer" : {"Age": 0.7 , "Gender": {"Male": 0.01 , "Female": 0.99}}
    # Add more diseases
}

#Lab Test Ranges
LAB_TEST_RANGES = {
    "Systolic_BP": {"mean": 120, "std": 15},  # Mean and standard deviation for normal range
    "Diastolic_BP": {"mean": 80, "std": 10},
    "Heart_Rate": {"mean": 72, "std": 8},
    "Cholesterol": {"mean": 200, "std": 30},
    "Blood_Sugar": {"mean": 90, "std": 20}
}

TREATMENT_PLANS = {
    "Hypertension": "Reduce sodium intake, exercise regularly, take prescribed medications (e.g., ACE inhibitors, ARBs). Monitor blood pressure regularly.",
    "Type 2 Diabetes": "Manage diet (low glycemic index), exercise regularly, take prescribed medications (e.g., Metformin, Insulin). Monitor blood sugar levels.",
    "Asthma": "Use inhaler as prescribed, avoid triggers (allergens, smoke), monitor peak flow, and have a Asthma action plan."
}

def generate_synthetic_data(num_samples=100):
    """Generates synthetic patient data with diseases, symptoms, medications, and conditional probabilities. Ensures symptoms and medications are always linked to the diagnosis.

    Args:
        num_samples (int, optional): Number of data samples to generate. Defaults to 100.

    Returns:
        pandas.DataFrame: DataFrame containing the generated synthetic data.
    """

    data = []
    for _ in range(num_samples):
        age = random.randint(18, 85)
        gender = random.choice(['Male', 'Female', 'Other'])
        medical_history = fake.sentence(nb_words=10)

        # --- Determine Disease ---
        diagnosis = random.choice(COMMON_DISEASES)

        # --- Generate Symptoms Linked to Diagnosis ---
        possible_symptoms = SYMPTOMS_BY_DISEASE[diagnosis]
        num_symptoms = random.randint(1, min(4, len(possible_symptoms))) # Ensure at least one symptom
        symptoms = ", ".join(random.sample(possible_symptoms, num_symptoms)) # Generate symptoms

        #--- Medication --- (Make sure there are always medications linked to the diagnosis)
        possible_medications = MEDICATIONS_BY_DISEASE.get(diagnosis, [])  # Get medications for the diagnosis

        #If for some reason, there are no medications defined for the disease, this will pick DEFAULT_MEDICATION
        if not possible_medications:
            possible_medications = [DEFAULT_MEDICATION]

        num_medications = random.randint(1, min(2, len(possible_medications))) # Ensure at least ONE medication
        medications = ", ".join(random.sample(possible_medications, num_medications)) # Medications


        # --- Lab results generation is still kept separate from diagnosis
        systolic_bp = np.random.normal(LAB_TEST_RANGES["Systolic_BP"]["mean"], LAB_TEST_RANGES["Systolic_BP"]["std"])
        diastolic_bp = np.random.normal(LAB_TEST_RANGES["Diastolic_BP"]["mean"], LAB_TEST_RANGES["Diastolic_BP"]["std"])
        heart_rate = np.random.normal(LAB_TEST_RANGES["Heart_Rate"]["mean"], LAB_TEST_RANGES["Heart_Rate"]["std"])
        cholesterol = np.random.normal(LAB_TEST_RANGES["Cholesterol"]["mean"], LAB_TEST_RANGES["Cholesterol"]["std"])
        blood_sugar = np.random.normal(LAB_TEST_RANGES["Blood_Sugar"]["mean"], LAB_TEST_RANGES["Blood_Sugar"]["std"])

        # More realistic queries based on symptoms, diagnosis, and medications
        if "pain" in symptoms.lower() or "discomfort" in symptoms.lower():
            patient_query = f"I am experiencing {symptoms}. What could be the cause of this pain/discomfort, and are there any medications I should consider? I am currently on {medications}."
        elif "cough" in symptoms.lower() or "shortness of breath" in symptoms.lower():
            patient_query = f"I have {symptoms}. Could this be related to a respiratory condition? What are the available treatments? I am currently taking {medications}."
        else:
            patient_query = f"I've been experiencing {symptoms}. What could be the possible reasons? Are there any lifestyle changes or medications I should be aware of? I am currently on {medications}."

        # Basic LLM responses (customize and improve significantly based on diagnosis and symptoms)
        if diagnosis == "Hypertension":
            llm_response = f"Your symptoms and blood pressure readings suggest hypertension. Consider lifestyle changes and consult your doctor for medication options.  Common medications include ACE inhibitors, ARBs, and diuretics. Since you are already on {medications}, consult your doctor before making any changes"
        elif diagnosis == "Type 2 Diabetes":
            llm_response = f"With symptoms like frequent urination and fatigue, it's important to check your blood sugar. See a doctor for a proper diagnosis and management plan. Medications like Metformin are often prescribed. Given your current medication: {medications}, consult with your doctor before changing anything"
        elif diagnosis == "Anxiety":
            llm_response = f"The described symptoms could be related to anxiety. Regular exercise, mindfulness, and therapy are potential coping strategies. Medications like SSRIs can be helpful. Discuss with your doctor, especially because you are already on {medications}"
        else:
            llm_response = f"Based on your symptoms, it's essential to consult a healthcare professional for a comprehensive evaluation.  Make sure they know that you are taking {medications}"

        treatment_plan = TREATMENT_PLANS.get(diagnosis, "Please consult with your doctor regarding a treatment plan.")

        access_level = random.choice(["Public", "Doctor", "Specialist"])
        restricted_fields = "Patient_ID" if access_level == "Public" else ("Patient_ID, Medical_History" if access_level == "Doctor" else "")
        data_source = random.choice(["Self Reported", "Blood test", "ECG", "Physical Exam"])

        data.append([fake.uuid4(), age, gender, medical_history, symptoms, patient_query, llm_response, diagnosis, systolic_bp, diastolic_bp, heart_rate, cholesterol, blood_sugar, access_level, restricted_fields, data_source, medications, treatment_plan])

    df = pd.DataFrame(data, columns=['Patient_ID', 'Age', 'Gender', 'Medical_History', 'Symptoms', 'Patient_Query', 'LLM_Response', 'Diagnosis', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Cholesterol', 'Blood_Sugar', 'Access_Level', 'Restricted_Fields', 'Data_Source', 'Medications', "Treatment_Plan"])
    return df

def main():
    """Generates synthetic data and saves it to a CSV file."""
    num_samples = 500
    synthetic_df = generate_synthetic_data(num_samples=num_samples)

    csv_filename = "synthetic_health_data.csv"
    synthetic_df.to_csv(csv_filename, index=False)
    print(f"Successfully generated {num_samples} samples and saved to {csv_filename}")

if __name__ == "__main__":
    main()