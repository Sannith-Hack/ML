# HECAR Framework — Project Documentation

## Overview
**HECAR** (Hybrid ECG PDF-Based Arrhythmia Classification and Cardiovascular Risk Prediction Framework) is a production-ready AI diagnostic system designed to replace older legacy systems. It provides an end-to-end pipeline:

1. Ingests **hospital ECG PDF reports** (Tricog/Bhageerath format)
2. Extracts structured ECG parameters via OCR + text parsing
3. Classifies **7 arrhythmia types** using a **CNN-BiLSTM** deep learning model
4. Fuses ECG features with **13 patient clinical variables**
5. Predicts **Stroke Risk** and **Coronary Heart Disease Risk** via XGBoost
6. Generates **SHAP-explained** clinical decision reports
7. Presents everything in a **CustomTkinter** dark-mode GUI

---

## Tech Stack Selected

We evaluated the requirements and selected the following state-of-the-art tech stack for production:

| Component | Selected Technology | Rationale |
|-----------|----------------------|-----------|
| **GUI Framework** | `CustomTkinter` | Provides a modern, dark-mode, premium desktop UI, vastly superior to standard Tkinter. |
| **PDF Extraction** | `pdfplumber` + `pytesseract` | Reliable text and OCR extraction for structured reports like Tricog PDFs. |
| **Deep Learning** | `TensorFlow 2.20` (`keras`) | Best for building the CNN-BiLSTM architecture for temporal feature modeling. |
| **Risk Prediction** | `XGBoost` | State-of-the-art for tabular clinical data classification (Stroke & CHD risk). |
| **Explainable AI** | `SHAP` | Industry standard for visualizing model feature importance in clinical settings. |
| **Data Processing** | `pandas` + `numpy` + `scipy` | Core libraries for signal processing and feature engineering. |
| **Reporting** | `Jinja2` | Allows generating dynamic, styled HTML clinical reports. |

---

## Project Structure

```
hecar_framework/
├── main.py                        # Entry point
├── config.py                      # All configs, paths, hyperparams
├── requirements.txt               # Dependencies
├── run.bat                        # Windows launcher
│
├── data/
│   ├── ecg_pdfs/                  # Uploaded ECG PDFs
│   ├── clinical/                  # Patient clinical CSVs
│   ├── processed/                 # Feature DataFrames
│   └── outputs/                   # Reports, logs, plots
│
├── model/
│   ├── saved/                     # .keras, .pkl model files
│   └── history/                   # Training history JSON
│
├── modules/
│   ├── pdf_processor/             # PDF loading, OCR, waveform extraction
│   ├── signal_processing/         # Noise filter, normalizer, segmenter
│   ├── feature_engineering/       # ECG parser, labeler, dataset builder
│   ├── models/                    # CNN-BiLSTM, XGBoost, ModelManager
│   ├── fusion/                    # Clinical features + feature fusion
│   ├── explainability/            # SHAP + HTML report generator
│   ├── evaluation/                # Metrics + visualizations
│   └── gui/                       # CustomTkinter app + 8 screens
│
└── tests/                         # Unit tests for all modules
```

---

## ECG Data Format (Tricog PDF)
The system trains on Tricog ECG PDF reports located in the `ECG-REPORTS-DATA-TRICOG` folder. These reports contain textual metrics:
```
Patient Name: <name>
Age / Gender: <age>/<gender>
Patient ID: <id>  Reported At: N/A
AR: <n>bpm  VR: <n>bpm  QRSD: <n>ms  QT: <n>ms  QTcB: <n>ms  PRI: <n>ms  P-R-T: <n>° <n>° <n>°
```

---

## Arrhythmia Classes & Rules

| Class | Label | Detection Rule |
|-------|-------|---------------|
| 0 | Normal Sinus Rhythm | All params normal |
| 1 | Sinus Tachycardia | AR > 100 bpm |
| 2 | Sinus Bradycardia | AR < 60 bpm |
| 3 | Bundle Branch Block | QRSD > 100 ms |
| 4 | Long QT Syndrome | QTcB > 460ms(F)/440ms(M) |
| 5 | 1st Degree AV Block | PRI > 200 ms |
| 6 | Abnormal Axis | P/R axis out of range |

---

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
# or double-click run.bat
```
