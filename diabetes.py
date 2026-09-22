import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Page Configuration
st.set_page_config(page_title="Diabetes Risk Assessment Application", page_icon="🩺", layout="wide")

# Load, Preprocess, and Train Models
@st.cache_resource
def load_and_train_models():
    df = pd.read_csv("diabetes.csv")

    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())

    X_raw = df.iloc[:, 0:8].values
    Y = df.iloc[:, 8].values

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_raw)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=0)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, criterion='entropy', random_state=0),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, solver='lbfgs', random_state=0),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=9, weights='uniform', metric='minkowski'),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=4, random_state=0)
    }

    trained_models = {}
    model_scores = {}
    model_reports = {}
    model_cms = {}

    for name, model in models.items():
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        trained_models[name] = model

        model_scores[name] = round(accuracy_score(Y_test, preds) * 100, 2)
        model_reports[name] = classification_report(Y_test, preds, output_dict=True)
        model_cms[name] = confusion_matrix(Y_test, preds)

    return df, sc, trained_models, models, model_scores, model_reports, model_cms, X_test, Y_test


df, scaler, trained_models, models, model_scores, model_reports, model_cms, X_test, Y_test = load_and_train_models()

# Helper function
def get_conditional_mean(feature_name, user_age):
    subset = df[(df['Age'] >= user_age - 5) & (df['Age'] <= user_age + 5)]
    if len(subset) > 5:
        return subset[feature_name].mean()
    else:
        return df[feature_name].mean()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🩺 Dynamic Risk & Wellness Hub", "📊 Data Analysis & Visuals", "📈 Algorithm Comparison"])

# --- PATIENT PROFILE ---
st.sidebar.markdown("---")
st.sidebar.title("Patient Profile")

age = st.sidebar.number_input("Age (years)", 18, 100, int(df['Age'].mean()))
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, int(df['Pregnancies'].mean()))
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 70.0, float(df['BMI'].mean()), step=0.01)

# --- AGE-BASED ESTIMATED BIOMARKERS ---
def compute_estimates():
    return (
        get_conditional_mean('Glucose', age),
        get_conditional_mean('BloodPressure', age),
        get_conditional_mean('SkinThickness', age),
        get_conditional_mean('Insulin', age),
        get_conditional_mean('DiabetesPedigreeFunction', age)
    )

est_glu, est_bp, est_skin, est_ins, est_dpf = compute_estimates()

# --- SESSION STATE INITIALIZATION ---
if "glucose" not in st.session_state:
    st.session_state.glucose = int(est_glu)
if "bp" not in st.session_state:
    st.session_state.bp = int(est_bp)
if "skin" not in st.session_state:
    st.session_state.skin = float(est_skin)
if "ins" not in st.session_state:
    st.session_state.ins = int(est_ins)
if "dpf" not in st.session_state:
    st.session_state.dpf = float(est_dpf)

# --- RESET BUTTON ---
if st.sidebar.button("↺ Reset Optional Biomarkers"):
    st.session_state.glucose = int(est_glu)
    st.session_state.bp = int(est_bp)
    st.session_state.skin = float(est_skin)
    st.session_state.ins = int(est_ins)
    st.session_state.dpf = float(est_dpf)

# --- ADVANCED BIOMARKERS (FIXED BUFFERING) ---
with st.sidebar.expander("Advanced Biomarkers (Optional)", expanded=False):
    st.number_input("Glucose (mg/dL)", 50, 300, key="glucose")
    st.number_input("Blood Pressure (mm Hg)", 60, 180, key="bp")
    st.number_input("Skin Thickness (mm)", 0.0, 100.0, key="skin")
    st.number_input("Insulin (mlU/L)", 0, 500, key="ins")
    st.number_input("Pedigree Score", 0.0, 2.5, key="dpf", step=0.01)

# Use session_state values
glucose = st.session_state.glucose
blood_pressure = st.session_state.bp
skin_thickness = st.session_state.skin
insulin = st.session_state.ins
dpf = st.session_state.dpf

# --- MODEL CONFIGURATION ---
st.sidebar.markdown("---")
st.sidebar.title("Model Configuration")

best_model_name = max(model_scores, key=model_scores.get)
selected_model_name = st.sidebar.selectbox(
    "Prediction Algorithm",
    options=list(models.keys()),
    index=list(models.keys()).index(best_model_name)
)
active_model = trained_models[selected_model_name]

# --- PAGE 1: RISK HUB ---
if page == "🩺 Dynamic Risk & Wellness Hub":
    st.title("🩺 Dynamic Diabetes Risk & Lifestyle Roadmap")
    st.write(f"Active Prediction Engine: **{selected_model_name}** ({model_scores[selected_model_name]}% accuracy)")

    # Prediction
    user_input_raw = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    selected_user_features = scaler.transform(user_input_raw)
    prediction_proba = active_model.predict_proba(selected_user_features)
    risk_score = round(prediction_proba[0][1] * 100, 2)

    col1, col2 = st.columns(2)

    # --- CLINICAL PROFILE ---
    with col1:
        st.subheader("Complete Clinical Profile")

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", age)
        c2.metric("Pregnancies", pregnancies)
        c3.metric("BMI", round(bmi, 1))

        c4, c5, c6 = st.columns(3)
        c4.metric("Glucose", round(glucose, 1))
        c5.metric("Blood Pressure", round(blood_pressure, 1))
        c6.metric("Skin Thickness", round(skin_thickness, 1))

        c7, c8 = st.columns(2)
        c7.metric("Insulin", round(insulin, 1))
        c8.metric("DPF Score", round(dpf, 3))

    # --- RISK STRATIFICATION ---
    with col2:
        st.subheader("Risk Stratification Assessment")

        if risk_score < 30:
            risk_tier = "🟢 Low Risk (Optimal Metabolic Health)"
            st.success(risk_tier)
        elif 30 <= risk_score < 65:
            risk_tier = "🟡 Moderate Risk (Pre-diabetic Tendencies Detected)"
            st.warning(risk_tier)
        else:
            risk_tier = "🔴 High Risk (Elevated Diabetic Indicators)"
            st.error(risk_tier)

        st.write(f"**Calculated Diabetic Probability:** {risk_score}%")
        st.progress(int(risk_score))

        st.caption(
            f"Model Confidence — Non-Diabetic: {round(prediction_proba[0][0]*100, 1)}% | "
            f"Diabetic: {round(prediction_proba[0][1]*100, 1)}%"
        )

    # --- CLINICAL REFERENCE ---
    st.markdown("---")
    st.subheader("📋 Clinical Reference Standards Evaluation")
    reference_data = pd.DataFrame({
        'Biomarker': ['Glucose (mg/dL)', 'BMI (kg/m²)', 'Blood Pressure (mm Hg)', 'Insulin (mlU/L)'],
        'Your Value': [round(glucose, 1), round(bmi, 1), round(blood_pressure, 1), round(insulin, 1)],
        'Clinical Reference Range': [
            'Normal: 70 - 99 mg/dL',
            'Normal: 18.5 - 24.9',
            'Normal: 90 - 120 mm Hg',
            'Normal: 2.5 - 25.0'
        ],
        'Status': [
            'Elevated' if glucose > 99 else ('Low' if glucose < 70 else 'Optimal'),
            'Overweight/Obese' if bmi > 24.9 else ('Low' if bmi < 18.5 else 'Optimal'),
            'Elevated' if blood_pressure > 120 else ('Low' if blood_pressure < 90 else 'Optimal'),
            'Elevated' if insulin > 25 else ('Low' if insulin < 2.5 else 'Optimal')
        ]
    })
    reference_data.index = reference_data.index + 1
    st.dataframe(reference_data, use_container_width=True)

    # --- POPULATION BENCHMARK ---
    st.markdown("---")
    st.subheader("📊 Population Benchmark Comparison (Glucose)")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5))
    sns.kdeplot(df[df['Outcome'] == 0]['Glucose'], label='Non-Diabetic Cohort', fill=True, ax=ax_dist, color='#2b5c8f', alpha=0.4)
    sns.kdeplot(df[df['Outcome'] == 1]['Glucose'], label='Diabetic Cohort', fill=True, ax=ax_dist, color='#d9534f', alpha=0.4)
    ax_dist.axvline(glucose, color='#28a745', linestyle='--', linewidth=2.5, label=f'Patient Level ({round(glucose, 1)} mg/dL)')
    ax_dist.set_title("")  # Removed title
    ax_dist.set_xlabel("Glucose (mg/dL)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(loc='upper right')
    sns.despine(ax=ax_dist)
    st.pyplot(fig_dist)

    # --- LOCAL EXPLAINABILITY / CLINICAL INSIGHTS ---
    st.markdown("---")
    st.subheader("🔍 Clinical Feature Contribution Breakdown")
    explainer_col1, explainer_col2 = st.columns(2)
    with explainer_col1:
        glu_status = "Elevated" if glucose > 99 else ("Low" if glucose < 70 else "Optimal")
        ins_status = "Elevated" if insulin > 25 else ("Low" if insulin < 2.5 else "Optimal")
        st.markdown(f"""
        * **Glucose Impact:** Your glucose level ({round(glucose, 1)} mg/dL) is evaluated as **{glu_status}**. Glucose remains the primary biological driver for diabetes risk scoring.
        * **Insulin & Pancreatic Workload:** Your reading ({round(insulin, 1)} mlU/L) is **{ins_status}**. High levels indicate your pancreas is working in overdrive to force glucose into resistant cells.
        """)
    with explainer_col2:
        bmi_status = "Overweight/Obese" if bmi > 24.9 else ("Low" if bmi < 18.5 else "Optimal")
        bp_status = "Elevated" if blood_pressure > 120 else ("Low" if blood_pressure < 90 else "Optimal")
        dpf_status = "Elevated Genetic Predisposition" if dpf > 0.5 else "Standard Baseline Risk"
        st.markdown(f"""
        * **BMI Status:** Your body mass index ({round(bmi, 1)} kg/m²) falls into the **{bmi_status}** category relative to standard metabolic thresholds.
        * **Blood Pressure:** Your reading ({round(blood_pressure, 1)} mm Hg) is evaluated as **{bp_status}**, tracking concurrent cardiovascular strain.
        * **Genetic Risk (DPF):** Your pedigree score ({round(dpf, 3)}) indicates a **{dpf_status}** based on familial diabetes patterns.
        """)

    # --- LIFESTYLE ROADMAP ---
    st.markdown("---")
    st.subheader("🌱 Tailored Lifestyle & Action Roadmap")

    if risk_score < 30:
        st.success("Your metabolic profile is healthy. Maintain and protect it.")
        st.markdown(
            """
### 🥗 Nutrition  
- Balanced diet with lean proteins, healthy fats, and fiber  
- Limit refined sugars and ultra-processed foods  

### 🚶 Activity  
- At least **150 minutes of aerobic exercise per week**  
- Light movement after meals  

### 🧘 Wellness  
- Hydration: 2.5–3L/day  
- Annual health screenings  
"""
        )
        roadmap_text = "Healthy metabolic profile — maintain balanced diet, exercise, hydration, and annual screenings."

    elif risk_score < 65:
        st.warning("You show early signs of insulin resistance.")
        st.markdown(
            """
### 🥗 Nutrition  
- Low GI diet  
- Replace refined carbs with whole grains  

### 🚶 Activity  
- 15-minute walk after meals  
- Aim for 5–7% weight reduction  

### 😴 Lifestyle  
- 7–8 hours of sleep  
"""
        )
        roadmap_text = "Moderate risk — low GI diet, post-meal walks, weight optimization, sleep."

    else:
        st.error("High probability suggests immediate clinical follow-up.")
        st.markdown(
            """
### 🩺 Medical  
- Schedule HbA1c test  
- Consult a clinician  

### 🥗 Nutrition  
- Remove added sugars entirely  

### 🏋️ Activity  
- Cardio + strength training  

### 📊 Monitoring  
- Track glucose regularly  
"""
        )
        roadmap_text = "High risk — medical consultation, strict diet, structured exercise, glucose monitoring."

    # --- DOWNLOAD REPORT ---
    st.markdown("---")
    st.subheader("📥 Export Clinical Report")

    report_content = f"""
DIABETES RISK ASSESSMENT REPORT
===============================
Generated via Diabetes Risk Assessment Application Powered By AI Models
Model: {selected_model_name} ({model_scores[selected_model_name]}% accuracy)
Risk Tier: {risk_tier}
Risk Probability: {risk_score}%

PATIENT PROFILE
---------------
Age: {age}
Pregnancies: {pregnancies}
BMI: {round(bmi, 2)}
Glucose: {round(glucose, 2)}
Blood Pressure: {round(blood_pressure, 2)}
Skin Thickness: {round(skin_thickness, 2)}
Insulin: {round(insulin, 2)}
DPF: {round(dpf, 3)}

RECOMMENDATIONS
---------------
{roadmap_text}
===============================
Please review this automated assessment with a certified healthcare professional for formal diagnostic paneling (such as an HbA1c blood test).
"""

    st.download_button(
        label="📄 Download Patient Clinical Summary (.txt)",
        data=report_content,
        file_name=f"diabetes_report_age_{age}_bmi_{int(bmi)}.txt",
        mime="text/plain"
    )

# --- PAGE 2: DATA ANALYSIS ---
elif page == "📊 Data Analysis & Visuals":
    st.title("📊 Exploratory Data Analysis")
    st.write("Deep dive into the underlying distributions and patterns within the full diabetes dataset.")

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outcome Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x='Outcome', data=df, palette='coolwarm', ax=ax)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

# --- PAGE 3: ALGORITHM COMPARISON ---
elif page == "📈 Algorithm Comparison":
    st.title("📈 Model Benchmarking & Metrics")
    st.write(f"Comparative accuracy evaluation calculated dynamically from live model evaluations (Active Engine: **{selected_model_name}**).")
    
    score_df = (
        pd.DataFrame(list(model_scores.items()), columns=['Algorithm', 'Accuracy (%)'])
        .sort_values(by='Accuracy (%)', ascending=False)
        .reset_index(drop=True)
    )
    score_df.index = score_df.index + 1
    st.dataframe(score_df, use_container_width=True)

    st.subheader("Accuracy Comparison")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    bars = ax_bar.bar(score_df['Algorithm'], score_df['Accuracy (%)'],
                      color=['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1'])
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_xlabel("Algorithm")
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("Algorithm Performance")

    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center')

    st.pyplot(fig_bar)

    st.subheader(f"Detailed Evaluation Reports — {selected_model_name}")
    col_rep, col_cm = st.columns(2)
    with col_rep:
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(model_reports[selected_model_name]).transpose())
    with col_cm:
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            model_cms[selected_model_name], 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=True, 
            vmin=0, 
            vmax=100, 
            ax=ax_cm, 
            annot_kws={"size": 14}
        )
        ax_cm.set_xlabel("Predicted Label", fontsize=11)
        ax_cm.set_ylabel("True Label", fontsize=11)
        ax_cm.set_title(f"Confusion Matrix ({selected_model_name})", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_cm)

    st.subheader(f"Feature Importance — {selected_model_name}")
    current_model = models[selected_model_name]
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']

    if hasattr(current_model, 'feature_importances_'):
        importances = current_model.feature_importances_
    elif hasattr(current_model, 'coef_'):
        importances = np.abs(current_model.coef_[0])
    else:
        perm = permutation_importance(active_model, X_test, Y_test, n_repeats=10, random_state=0)
        importances = perm.importances_mean

    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(
        by='Importance', ascending=False
    )

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='magma', ax=ax_imp)
    ax_imp.set_title(f"Feature Importance — {selected_model_name}")
    ax_imp.set_xlabel("Importance")
    ax_imp.set_ylabel("")
    st.pyplot(fig_imp)
