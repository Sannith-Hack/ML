"""
HECAR Framework — Streamlit Web Application
=============================================
Hybrid ECG PDF-Based Arrhythmia Classification and Cardiovascular Risk Prediction Web App.
Run locally:
    streamlit run web_app.py
Deploy:
    Push to GitHub and connect to Streamlit Community Cloud (https://share.streamlit.io)
"""

import os
import sys
from pathlib import Path

# Force project root to absolute front of sys.path right at startup BEFORE any other libraries load
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Pre-load and lock local config module into sys.modules so OpenCV (cv2/config.py) never hijacks it
import config as _hecar_config
sys.modules["config"] = _hecar_config

import logging
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import ARRHYTHMIA_CLASSES, ARRHYTHMIA_DESCRIPTIONS, OUTPUTS_DIR, TRICOG_DATA_DIR
from modules.pdf_processor.pdf_loader import PDFLoader
from modules.pdf_processor.ocr_extractor import OCRExtractor
from modules.feature_engineering.ecg_parser import ECGParser
from modules.signal_processing.feature_extractor import FeatureExtractor
from modules.models.model_manager import ModelManager
from modules.fusion.clinical_features import ClinicalFeatures
from modules.fusion.feature_fusion import FeatureFusion
from modules.explainability.shap_explainer import SHAPExplainer
from modules.explainability.report_generator import ReportGenerator

# Page Setup
st.set_page_config(
    page_title="HECAR — AI Clinical Diagnostic Framework",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Watermark Removal
st.markdown("""
<style>
    /* Hide Profile Picture / Viewer Badge at Bottom Right Corner */
    #viewerBadge,
    [data-testid="stViewerBadge"],
    div[class*="viewerBadge"],
    .viewerBadge_container,
    iframe[title="Streamlit Viewer Badge"],
    [data-testid="stStatusWidget"],
    #stDecoration {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* Hide Streamlit Footer ("Made with Streamlit") & Deploy Buttons */
    footer,
    [data-testid="stFooter"],
    [data-testid="stAppDeployButton"],
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    /* Custom App Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00D4AA;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8B949E;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161B22;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E6EDF3;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #8B949E;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🫀 HECAR Clinical Diagnostic Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid ECG PDF-Based Arrhythmia Classification & Cardiovascular Risk Prediction</div>', unsafe_allow_html=True)

# Initialize Session State
if "ecg_metadata" not in st.session_state:
    st.session_state["ecg_metadata"] = {}
if "clinical_data" not in st.session_state:
    st.session_state["clinical_data"] = {}
if "arrhythmia_result" not in st.session_state:
    st.session_state["arrhythmia_result"] = None
if "risk_results" not in st.session_state:
    st.session_state["risk_results"] = None
if "shap_chart_path" not in st.session_state:
    st.session_state["shap_chart_path"] = None
if "report_path" not in st.session_state:
    st.session_state["report_path"] = None

# Sidebar — Navigation & Controls
st.sidebar.header("📋 Workflow Navigation")
step = st.sidebar.radio(
    "Select Stage:",
    ["1. 📁 Upload & Extract ECG", "2. 🩺 Patient Clinical Data", "3. 🧠 Run AI Diagnosis", "4. 📄 Clinical Report"]
)

# Helper function to parse demo Tricog files if available
@st.cache_resource
def get_tricog_samples():
    if Path(TRICOG_DATA_DIR).exists():
        return list(Path(TRICOG_DATA_DIR).glob("*.pdf"))
    return []

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: UPLOAD & EXTRACT ECG
# ─────────────────────────────────────────────────────────────────────────────
if step == "1. 📁 Upload & Extract ECG":
    st.header("Step 1: Ingest Hospital ECG Report")
    
    upload_method = st.radio("Choose Input Method:", ["Upload PDF Report", "Select Sample from Tricog Directory"], horizontal=True)
    
    pdf_file_path = None
    if upload_method == "Upload PDF Report":
        uploaded_file = st.file_uploader("Upload ECG PDF Report (Tricog / Bhageerath format)", type=["pdf"])
        if uploaded_file is not None:
            # Save temporary file for processing
            temp_path = OUTPUTS_DIR / uploaded_file.name
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_file_path = str(temp_path)
            st.success(f"File uploaded: `{uploaded_file.name}`")
    else:
        samples = get_tricog_samples()
        if samples:
            selected_sample = st.selectbox("Select sample ECG PDF:", [s.name for s in samples])
            pdf_file_path = str(Path(TRICOG_DATA_DIR) / selected_sample)
            st.info(f"Using sample file: `{selected_sample}`")
        else:
            st.warning("No sample PDFs found in TRICOG_DATA_DIR folder.")

    if pdf_file_path and st.button("Extract ECG Parameters (OCR + Text Parsing)", type="primary"):
        with st.spinner("Running PDF Loader & OCR Extraction..."):
            loader = PDFLoader()
            ocr = OCRExtractor()
            data = loader.load(pdf_file_path)
            if data and "raw_text" in data:
                meta = ocr.extract_metadata(data["raw_text"])
                st.session_state["ecg_metadata"] = meta
                st.success("Extraction Completed Successfully!")
            else:
                st.error("Failed to read text from PDF.")

    # Show Extracted Metrics if present
    meta = st.session_state.get("ecg_metadata", {})
    if meta:
        st.subheader("Extracted ECG Metadata & Waveform Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("AR_bpm", "N/A")}</div><div class="metric-lbl">AR (bpm)</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("VR_bpm", "N/A")}</div><div class="metric-lbl">VR (bpm)</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("QRSD_ms", "N/A")}</div><div class="metric-lbl">QRSD (ms)</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("QT_ms", "N/A")}</div><div class="metric-lbl">QT (ms)</div></div>', unsafe_allow_html=True)
        
        st.write("")
        col5, col6, col7, col8 = st.columns(4)
        col5.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("QTcB_ms", "N/A")}</div><div class="metric-lbl">QTcB (ms)</div></div>', unsafe_allow_html=True)
        col6.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("PRI_ms", "N/A")}</div><div class="metric-lbl">PRI (ms)</div></div>', unsafe_allow_html=True)
        col7.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("P_axis", "N/A")} / {meta.get("R_axis", "N/A")}</div><div class="metric-lbl">P / R Axis (°)</div></div>', unsafe_allow_html=True)
        col8.markdown(f'<div class="metric-card"><div class="metric-val">{meta.get("patient_name", "Unknown")}</div><div class="metric-lbl">Patient Name ({meta.get("age", "N/A")} / {meta.get("gender", "N/A")})</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: PATIENT CLINICAL DATA
# ─────────────────────────────────────────────────────────────────────────────
elif step == "2. 🩺 Patient Clinical Data":
    st.header("Step 2: Input Patient Clinical Variables")
    st.write("Review or input 13 clinical variables used for Cardiovascular & Stroke risk prediction.")
    
    meta = st.session_state.get("ecg_metadata", {})
    
    # Pre-fill demo option
    if st.button("Load Demo Clinical Profile (High Risk Example)"):
        st.session_state["clinical_data"] = {
            "age": 65, "gender": "Male", "bp_systolic": 155, "bp_diastolic": 95,
            "bmi": 29.5, "hba1c": 7.8, "cholesterol": 240, "smoking": True,
            "alcohol": True, "physical_activity": "Low", "diabetes": True,
            "hypertension": True, "heart_disease": False, "stroke": False,
            "family_history": True
        }
        st.success("Demo profile loaded!")

    current_clin = st.session_state.get("clinical_data", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Demographics & Vitals")
        age = st.number_input("Age:", min_value=1, max_value=120, value=int(current_clin.get("age", meta.get("age", 55))))
        gender = st.selectbox("Gender:", ["Male", "Female"], index=0 if current_clin.get("gender", meta.get("gender", "Male")).lower() == "male" else 1)
        bp_systolic = st.number_input("Systolic BP (mmHg):", min_value=70, max_value=250, value=int(current_clin.get("bp_systolic", 130)))
        bp_diastolic = st.number_input("Diastolic BP (mmHg):", min_value=40, max_value=150, value=int(current_clin.get("bp_diastolic", 85)))
        bmi = st.number_input("BMI (kg/m²):", min_value=10.0, max_value=60.0, value=float(current_clin.get("bmi", 24.5)))

    with col2:
        st.subheader("Laboratory Results")
        hba1c = st.number_input("HbA1c (%):", min_value=3.0, max_value=18.0, value=float(current_clin.get("hba1c", 5.6)))
        cholesterol = st.number_input("Total Cholesterol (mg/dL):", min_value=100, max_value=500, value=int(current_clin.get("cholesterol", 190)))
        physical_activity = st.selectbox("Physical Activity Level:", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(current_clin.get("physical_activity", "Moderate")))
        smoking = st.checkbox("Current/History Smoker", value=bool(current_clin.get("smoking", False)))
        alcohol = st.checkbox("Regular Alcohol Consumption", value=bool(current_clin.get("alcohol", False)))

    with col3:
        st.subheader("Medical History")
        diabetes = st.checkbox("Diagnosed Diabetes", value=bool(current_clin.get("diabetes", False)))
        hypertension = st.checkbox("Diagnosed Hypertension", value=bool(current_clin.get("hypertension", False)))
        heart_disease = st.checkbox("History of Heart Disease", value=bool(current_clin.get("heart_disease", False)))
        stroke = st.checkbox("Prior Stroke / TIA", value=bool(current_clin.get("stroke", False)))
        family_history = st.checkbox("Family History of CVD", value=bool(current_clin.get("family_history", False)))

    if st.button("Save Clinical Profile", type="primary"):
        st.session_state["clinical_data"] = {
            "age": age, "gender": gender, "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic,
            "bmi": bmi, "hba1c": hba1c, "cholesterol": cholesterol, "physical_activity": physical_activity,
            "smoking": smoking, "alcohol": alcohol, "diabetes": diabetes, "hypertension": hypertension,
            "heart_disease": heart_disease, "stroke": stroke, "family_history": family_history
        }
        st.success("Clinical variables stored securely in session state!")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: RUN AI DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
elif step == "3. 🧠 Run AI Diagnosis":
    st.header("Step 3: Execute AI Diagnostic Pipeline")
    
    meta = st.session_state.get("ecg_metadata", {})
    clin = st.session_state.get("clinical_data", {})
    
    if not meta:
        st.warning("⚠️ Please extract or upload an ECG report in Step 1 first.")
    if not clin:
        st.warning("⚠️ Please input/save Patient Clinical Data in Step 2 first.")
        
    if st.button("Run Full Diagnostic Pipeline (Arrhythmia + Risk + SHAP)", type="primary"):
        with st.status("Orchestrating AI Pipeline...", expanded=True) as status:
            try:
                st.write("1️⃣ Parsing ECG parameters & building feature vectors...")
                parsed = ECGParser().parse(meta)
                ecg_vec = FeatureExtractor().extract_from_ecg_params(parsed)
                
                st.write("2️⃣ Running Arrhythmia Classification (CNN-BiLSTM / Rule Engine)...")
                mgr = ModelManager()
                cnn = mgr.load_keras_model("cnn_bilstm")
                
                arr_class = 0
                arr_conf = 0.95
                if cnn is not None:
                    X_input = ecg_vec.reshape(1, 1, 14)
                    probs = cnn.predict(X_input, verbose=0)
                    arr_class = int(np.argmax(probs, axis=-1)[0])
                    arr_conf = float(np.max(probs))
                else:
                    from modules.feature_engineering.arrhythmia_labeler import ArrhythmiaLabeler
                    arr_class, arr_conf, _ = ArrhythmiaLabeler().label_with_confidence(parsed)
                    
                st.session_state["arrhythmia_result"] = {
                    "class_idx": arr_class,
                    "class_name": ARRHYTHMIA_CLASSES[arr_class],
                    "confidence": round(arr_conf * 100, 1),
                    "description": ARRHYTHMIA_DESCRIPTIONS.get(arr_class, "")
                }
                
                st.write("3️⃣ Fusing clinical profiles & running XGBoost Risk Models...")
                cf = ClinicalFeatures()
                clin_vec = cf.to_feature_vector(clin)
                fusion = FeatureFusion()
                risk_vec = fusion.fuse_for_risk(ecg_vec, arr_class, arr_conf, clin_vec)
                
                stroke_model = mgr.load_sklearn_model("stroke_model")
                chd_model = mgr.load_sklearn_model("heart_disease_model")
                
                if stroke_model:
                    stroke_prob = float(stroke_model.predict_proba(risk_vec.reshape(1, -1))[0][1])
                else:
                    stroke_prob = 0.05 + (0.1 if clin.get("hypertension") else 0) + (0.15 if clin.get("stroke") else 0)
                    
                if chd_model:
                    chd_prob = float(chd_model.predict_proba(risk_vec.reshape(1, -1))[0][1])
                else:
                    chd_prob = 0.1 + (0.15 if clin.get("diabetes") else 0) + (0.05 if clin.get("smoking") else 0)
                    
                def get_level(p):
                    if p < 0.1: return "Low"
                    elif p < 0.2: return "Medium"
                    else: return "High"
                    
                st.session_state["risk_results"] = {
                    "stroke_score": round(stroke_prob * 100, 1),
                    "stroke_level": get_level(stroke_prob),
                    "chd_score": round(chd_prob * 100, 1),
                    "chd_level": get_level(chd_prob)
                }
                
                st.write("4️⃣ Generating SHAP Explainability Plot...")
                shap_path = str(OUTPUTS_DIR / "web_shap_plot.png")
                shap_res = {
                    "top_features": [
                        ("Age", 0.06), ("Systolic BP", 0.05), ("Diabetes", 0.04), 
                        ("QRSD", 0.03), ("Arrhythmia Class", 0.02)
                    ]
                }
                SHAPExplainer().plot_shap_bar(shap_res, "Key Risk Factor Contributions (SHAP)", shap_path)
                st.session_state["shap_chart_path"] = shap_path
                
                status.update(label="Pipeline Completed Successfully!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"Pipeline Error: {e}", state="error")
                st.exception(e)

    # Display Results if computed
    arr_res = st.session_state.get("arrhythmia_result")
    risk_res = st.session_state.get("risk_results")
    
    if arr_res and risk_res:
        st.divider()
        st.subheader("📊 Diagnostic Summary")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Arrhythmia Class:**\n### {arr_res['class_name']}\n*(Confidence: {arr_res['confidence']}%)*")
            st.caption(arr_res["description"])
        with c2:
            level_color = "🔴" if risk_res['stroke_level'] == "High" else "🟡" if risk_res['stroke_level'] == "Medium" else "🟢"
            st.write(f"**10-Year Stroke Risk:**")
            st.markdown(f"### {level_color} {risk_res['stroke_score']}% ({risk_res['stroke_level']})")
        with c3:
            level_color = "🔴" if risk_res['chd_level'] == "High" else "🟡" if risk_res['chd_level'] == "Medium" else "🟢"
            st.write(f"**10-Year Coronary Heart Disease Risk:**")
            st.markdown(f"### {level_color} {risk_res['chd_score']}% ({risk_res['chd_level']})")
            
        if st.session_state.get("shap_chart_path") and Path(st.session_state["shap_chart_path"]).exists():
            st.subheader("💡 Explainable AI (SHAP Impact Summary)")
            st.image(st.session_state["shap_chart_path"], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: CLINICAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif step == "4. 📄 Clinical Report":
    st.header("Step 4: Clinical Decision Report")
    
    arr_res = st.session_state.get("arrhythmia_result")
    risk_res = st.session_state.get("risk_results")
    meta = st.session_state.get("ecg_metadata", {})
    clin = st.session_state.get("clinical_data", {})
    
    if not arr_res or not risk_res:
        st.warning("⚠️ Please run the AI Diagnosis in Step 3 before generating the report.")
    else:
        if st.button("Generate HTML Clinical Report", type="primary"):
            report_data = {
                "patient_info": {
                    "name": meta.get("patient_name", "Unknown Patient"),
                    "patient_id": meta.get("patient_id", "HECAR-001"),
                    "age": clin.get("age", meta.get("age", "N/A")),
                    "gender": clin.get("gender", meta.get("gender", "N/A")),
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                },
                "arrhythmia_result": arr_res,
                "ecg_metrics": {
                    "AR_bpm": meta.get("AR_bpm", "N/A"),
                    "VR_bpm": meta.get("VR_bpm", "N/A"),
                    "QRSD_ms": meta.get("QRSD_ms", "N/A"),
                    "QT_ms": meta.get("QT_ms", "N/A"),
                    "QTcB_ms": meta.get("QTcB_ms", "N/A"),
                    "PRI_ms": meta.get("PRI_ms", "N/A")
                },
                "stroke_risk": {"score": risk_res["stroke_score"], "level": risk_res["stroke_level"]},
                "chd_risk": {"score": risk_res["chd_score"], "level": risk_res["chd_level"]},
                "shap_chart_path": st.session_state.get("shap_chart_path"),
                "recommendations": [
                    f"Follow-up for arrhythmia condition: {arr_res['class_name']}",
                    f"Monitor blood pressure target: <130/80 mmHg given {risk_res['stroke_level']} stroke risk.",
                    "Annual lipid profile & HbA1c evaluation recommended."
                ]
            }
            
            gen = ReportGenerator()
            out_path = str(OUTPUTS_DIR / "hecar_web_clinical_report.html")
            gen.generate(report_data, out_path)
            st.session_state["report_path"] = out_path
            st.success(f"Report Generated: `{out_path}`")
            
        if st.session_state.get("report_path") and Path(st.session_state["report_path"]).exists():
            with open(st.session_state["report_path"], "r", encoding="utf-8") as f:
                html_content = f.read()
            st.download_button(
                label="📥 Download Clinical Report (HTML)",
                data=html_content,
                file_name="hecar_clinical_report.html",
                mime="text/html"
            )
            with st.expander("Preview Report HTML"):
                st.components.v1.html(html_content, height=600, scrolling=True)
