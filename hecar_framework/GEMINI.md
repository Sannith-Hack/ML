# HECAR Framework — Project Documentation & Architecture
=========================================================

## Overview
**HECAR** (Hybrid ECG PDF-Based Arrhythmia Classification and Cardiovascular Risk Prediction Framework) is a production-ready, multi-modal AI clinical diagnostic suite designed to replace legacy single-signal heart monitoring systems. It delivers an end-to-end diagnostic pipeline across two powerful interfaces:
1. **Streamlit Web & Mobile Application (`web_app.py`)**: Hosted on Streamlit Community Cloud (`sirwork.streamlit.app`) with responsive top-level horizontal navigation, medical glassmorphism UI/UX, and atomic state synchronization.
2. **CustomTkinter Desktop Application (`main.py`)**: A modern 8-screen dark-mode desktop GUI for local workstation deployments.

---

## Core Capabilities & Diagnostic Pipeline

The HECAR framework executes a 6-stage clinical workflow:
1. **PDF Report Ingestion & Digital Extraction:** Accepts uploaded patient ECG reports (`.pdf` in Tricog or Bhageerath format) or allows instant selection from **32 real clinical hospital demo samples** inside `ECG-REPORTS-DATA-TRICOG/`.
2. **OCR & Waveform Parameter Parsing:** Utilizes `pdfplumber` and `pytesseract` to extract structured cardiac parameters (`AR_bpm`, `VR_bpm`, `QRSD_ms`, `QT_ms`, `QTcB_ms`, `PRI_ms`, and `P/R Axis`).
3. **Primary Arrhythmia Classification (`CNN-BiLSTM`):** Fuses 14-channel signal feature vectors through a deep Convolutional Bi-Directional LSTM neural network (`model/saved/cnn_bilstm.keras`) to classify into **7 distinct cardiac conditions**.
4. **Clinical Variable & Risk Factor Fusion:** Integrates extracted ECG metrics with **13 patient clinical variables** (`Age`, `Gender`, `Systolic/Diastolic BP`, `BMI`, `HbA1c`, `Total Cholesterol`, `Activity`, `Smoking`, `Alcohol`, `Diabetes`, `Hypertension`, `CHD History`, `Stroke History`, and `Family History`).
5. **Multi-Modal Prognostic Risk Scoring (`XGBoost`):** Computes **10-Year Ischemic Stroke Risk** (`Low`, `Medium`, `High` tiers) and **10-Year Coronary Heart Disease (CAD) Risk** (`Low`, `Medium`, `High` tiers).
6. **Explainable AI (`SHAP`) & Report Generation:** Generates SHAP feature attribution bar charts explaining exact risk factor contributions, and exports standardized publication-ready HTML medical reports (`HECAR_Clinical_Report.html`).

---

## Technology Stack

| Component | Selected Technology | Version / Details | Rationale |
| :--- | :--- | :--- | :--- |
| **Web & Mobile Interface** | `Streamlit` | 1.40+ | Powers `web_app.py` with responsive top horizontal navigation (`st.radio`), custom CSS glassmorphism, and instant cloud deployment. |
| **Desktop Interface** | `CustomTkinter` | 5.2+ | Powers `main.py` with an 8-screen modern dark-mode desktop GUI. |
| **PDF & OCR Engine** | `pdfplumber` + `pytesseract` | Latest | High-precision text and OCR digital extraction from hospital PDF reports. |
| **Deep Learning Engine** | `TensorFlow` (`keras`) | 2.16+ / 2.20 | Convolutional BiLSTM (`CNN-BiLSTM`) architecture for temporal ECG feature modeling. |
| **Risk Classifier** | `XGBoost` + `scikit-learn` | 2.0+ / 1.4+ | Gradient boosted trees for tabular multi-modal clinical risk prediction. |
| **Explainable AI** | `SHAP` | 0.44+ | Shapley Additive exPlanations (`SHAPExplainer`) for clinical transparency. |
| **Data Processing** | `pandas`, `numpy`, `scipy` | 2.0+, 1.26+, 1.12+ | Core numerical modeling, signal normalization, and feature vector construction. |
| **Report Engine** | `Jinja2` + `matplotlib` | 3.1+, 3.8+ | Dynamic HTML medical report rendering and SHAP attribution plot generation. |

---

## Project Structure & Module Organization

```
hecar_framework/
│
├── web_app.py                     # Streamlit Web & Mobile Application (4-Stage Responsive Suite)
├── main.py                        # CustomTkinter Desktop Application (8-Screen Local GUI)
├── config.py                      # Global paths, hyperparameters, arrhythmia class definitions
├── requirements.txt               # Full project dependencies
├── run.bat / run_streamlit.bat    # Windows batch launchers for desktop and web apps
├── GEMINI.md                      # Comprehensive technical project documentation
│
├── data/
│   ├── ecg_pdfs/                  # Patient uploaded ECG PDF files
│   ├── clinical/                  # Patient clinical variables storage
│   ├── processed/                 # Extracted feature DataFrames and tensors
│   └── outputs/                   # Generated HTML reports, SHAP plots, execution logs
│
├── model/
│   ├── saved/                     # Persisted model weights (`cnn_bilstm.keras`, `xgboost_risk.pkl`)
│   └── history/                   # Epoch training loss/accuracy history JSONs
│
└── modules/
    ├── pdf_processor/             # `PDFLoader` and `OCRExtractor` (Tricog/Bhageerath parsers)
    ├── signal_processing/         # Noise filtering, baseline wander removal, normalization
    ├── feature_engineering/       # `ECGParser`, `ArrhythmiaLabeler`, `FeatureExtractor`
    ├── models/                    # `ModelManager`, CNN-BiLSTM architecture, XGBoost wrappers
    ├── fusion/                    # `ClinicalFeatures` and `FeatureFusion` vector concatenation
    ├── explainability/            # `SHAPExplainer` bar plotting and `ReportGenerator` (HTML)
    ├── evaluation/                # Confusion matrices, ROC/PR curves, metric calculators
    └── gui/                       # CustomTkinter screens (`App`, `UploadFrame`, `ResultsFrame`, etc.)
```

---

## Streamlit Web Application Workflow (`web_app.py`)

The Streamlit web suite is structured into **4 Synchronized Diagnostic Stages** navigated via a top horizontal stepper bar or in-page action buttons:

```
[Stage 1: Upload & Extract ECG] ➔ [Stage 2: Patient Clinical Data] ➔ [Stage 3: Run AI Diagnosis] ➔ [Stage 4: Clinical Report]
```

### 📍 Stage 1: Upload & Extract ECG (`1. 📁 Upload & Extract ECG`)
- Input options: Upload a live patient PDF or pick from **32 deterministic sorted clinical samples (`get_tricog_samples()`)**.
- Executes `PDFLoader` + `OCRExtractor` upon clicking **`⚡ Run AI Parameter Extraction`**.
- Displays digital waveform metrics inside glowing medical cards (`AR`, `VR`, `QRSD`, `QT`, `QTcB`, `PRI`, `Axis`, `Patient Profile`).

### 📍 Stage 2: Patient Clinical Data (`2. 🩺 Patient Clinical Data`)
- Interactive form across 3 categorized columns (`Demographics & Vitals`, `Lab Panels & Habits`, `Diagnosed Medical History`).
- **Instant Profile Presets (`load_preset`):** 1-click buttons (`🔴 Load High-Risk CAD/Stroke Profile`, `🟡 Load Moderate Risk Profile`, `🟢 Load Normal Healthy Profile`) immediately populate all 13 fields via atomic callbacks.

### 📍 Stage 3: Run AI Diagnosis (`3. 🧠 Run AI Diagnosis`)
- Executes the unified AI pipeline with live status tracking (`st.status`):
  1. Constructs the 14-channel normalized ECG feature vector.
  2. Runs CNN-BiLSTM classification (`arrhythmia_result`).
  3. Fuses ECG + clinical variables across XGBoost models (`stroke_model`, `heart_disease_model`).
  4. Generates SHAP explainability bar plots (`web_shap_plot.png`).
- Displays a glowing primary diagnosis banner, risk tier scorecards (`🔴 High`, `🟡 Medium`, `🟢 Low`), and feature attribution visuals.

### 📍 Stage 4: Clinical Report (`4. 📄 Clinical Report`)
- Compiles the complete diagnostic assessment into `HECAR_Clinical_Report.html` (`ReportGenerator`).
- Offers instant 1-click download (`st.download_button`) with clinical care recommendations and consultation guidelines.

---

## Arrhythmia Classification Categories & Clinical Rules

The framework classifies patients into one of 7 diagnostic classes based on waveform metrics and neural network confidence:

| Class Index | Arrhythmia Condition | Diagnostic / Threshold Rule | Clinical Significance |
| :---: | :--- | :--- | :--- |
| **0** | **Normal Sinus Rhythm** | $60 \le \text{AR} \le 100 \text{ bpm}$, normal intervals | Healthy conduction; standard reference baseline. |
| **1** | **Sinus Tachycardia** | $\text{AR} > 100 \text{ bpm}$ | Elevated heart rate; potential stress, fever, or ischemia. |
| **2** | **Sinus Bradycardia** | $\text{AR} < 60 \text{ bpm}$ | Slow conduction; common in athletes or hypothyroidism. |
| **3** | **Bundle Branch Block** | $\text{QRSD} > 100 \text{ ms}$ | Delayed ventricular depolarization; intraventricular block. |
| **4** | **Long QT Syndrome** | $\text{QTcB} > 460 \text{ ms (F)} / 440 \text{ ms (M)}$ | Repolarization delay; risk of Torsades de Pointes. |
| **5** | **1st Degree AV Block** | $\text{PRI} > 200 \text{ ms}$ | Atrioventricular node conduction delay. |
| **6** | **Abnormal Electrical Axis** | P/R Axis outside $-30^\circ \text{ to } +90^\circ$ | Ventricular hypertrophy or hemiblock indication. |

---

## Key Technical & Architectural Engineering Breakthroughs

### 1. Atomic Navigation & State Callbacks (`on_click` Architecture)
To eliminate `StreamlitAPIException: st.session_state.top_workflow_stage cannot be modified after the widget is instantiated`, all navigation (`go_step2`, `go_step3`, `go_step4`), presets (`apply_preset`), and session resets (`reset_session`) are bound directly to **Streamlit `on_click` pre-execution callbacks**. This ensures session variables and widget state keys (`clin_age`, `clin_sys`, etc.) are atomically updated *before* any UI widgets render.

### 2. Module Pre-Loading Lock (`sys.modules["config"] = _hecar_config`)
OpenCV (`cv2`) internally bundles a sub-package named `cv2.config`. When importing `cv2` inside sub-modules, Python's module resolution could accidentally override the root `config.py`. To make deployment bulletproof on Linux/Streamlit Cloud, `web_app.py` forces `PROJECT_ROOT` to the front of `sys.path` and pre-loads `import config as _hecar_config; sys.modules["config"] = _hecar_config` right at startup.

### 3. Parent-Window Cross-Origin DOM Destroyer
Streamlit Community Cloud wraps applications inside an `iframe` with bottom viewer/profile badges and right deploy toolbars. `web_app.py` injects a targeted JavaScript block (`hecar-watermark-destroyer`) that traverses `window.parent.document` and `window.top.document` to cleanly hide `stToolbar`, `stAppDeployButton`, and `viewerBadge` containers while preserving full pointer interactions for BaseWeb dropdown menus (`st.selectbox` popovers) and status progress widgets (`st.status`).

### 4. Deterministic Sample Sorting on Linux (`sorted(Path.glob)`)
On Linux file systems, `glob("*.pdf")` returns paths in arbitrary directory hash order. `get_tricog_samples()` wraps `Path.glob` inside `sorted(...)` and caches the filenames (`@st.cache_data`), guaranteeing that option lists and widget keys (`tricog_pdf_selectbox`) never shift between reruns.

---

## Deployment & Local Launch Instructions

### 🌐 Running Locally (Streamlit Web App)
```bash
# Activate virtual environment
venv\Scripts\activate

# Launch Streamlit web interface
streamlit run hecar_framework/web_app.py
```
*Access in browser at `http://localhost:8501`.*

### 🖥️ Running Locally (CustomTkinter Desktop App)
```bash
# Launch desktop GUI
python hecar_framework/main.py
```
*Or double-click `run.bat` on Windows.*

### ☁️ Production Cloud Deployment
The web suite (`web_app.py`) is continuously deployed on **Streamlit Community Cloud**:
- **App URL:** `https://sirwork.streamlit.app`
- **Git Repository:** `https://github.com/Sannith-Hack/ML.git` (`master` branch)
- **Python Runtime:** Python 3.12
