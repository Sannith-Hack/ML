"""
HECAR Framework — Centralized Configuration
============================================
All paths, hyperparameters, class definitions, and theme settings.
Import this module everywhere instead of hardcoding values.
"""

import logging
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR          = PROJECT_ROOT / "data"
ECG_PDFS_DIR      = DATA_DIR / "ecg_pdfs"
CLINICAL_DIR      = DATA_DIR / "clinical"
PROCESSED_DIR     = DATA_DIR / "processed"
OUTPUTS_DIR       = DATA_DIR / "outputs"

MODEL_DIR         = PROJECT_ROOT / "model"
MODEL_SAVED_DIR   = MODEL_DIR / "saved"
MODEL_HISTORY_DIR = MODEL_DIR / "history"

TRICOG_DATA_DIR = (
    PROJECT_ROOT.parent / "ECG-REPORTS-DATA-TRICOG"
)

# ── Ensure all directories exist ─────────────────────────────────────────────
for _dir in [
    ECG_PDFS_DIR, CLINICAL_DIR, PROCESSED_DIR, OUTPUTS_DIR,
    MODEL_SAVED_DIR, MODEL_HISTORY_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Arrhythmia Classes ────────────────────────────────────────────────────────
ARRHYTHMIA_CLASSES: dict = {
    0: "Normal Sinus Rhythm",
    1: "Sinus Tachycardia",
    2: "Sinus Bradycardia",
    3: "Bundle Branch Block",
    4: "Long QT Syndrome",
    5: "1st Degree AV Block",
    6: "Abnormal Axis",
}

ARRHYTHMIA_DESCRIPTIONS: dict = {
    0: "Heart rhythm is normal. All ECG parameters are within reference ranges.",
    1: "Resting heart rate exceeds 100 bpm. May indicate fever, anxiety, dehydration, or cardiac pathology.",
    2: "Resting heart rate below 60 bpm. May be normal in athletes or indicate conduction issues.",
    3: "QRS complex is prolonged (>100 ms), indicating delayed ventricular conduction.",
    4: "Corrected QT interval is prolonged, increasing risk of ventricular arrhythmias.",
    5: "PR interval exceeds 200 ms, indicating delayed AV node conduction.",
    6: "Electrical axis of the heart is outside normal range, suggesting structural changes.",
}

ARRHYTHMIA_SEVERITY: dict = {
    0: "Normal",
    1: "Moderate",
    2: "Mild",
    3: "High",
    4: "High",
    5: "Moderate",
    6: "Moderate",
}

NUM_CLASSES = len(ARRHYTHMIA_CLASSES)

# ── Arrhythmia Classification Rules ─────────────────────────────────────────
ARRHYTHMIA_RULES = {
    "AR_HIGH_THRESHOLD":       100,
    "AR_LOW_THRESHOLD":         60,
    "QRSD_THRESHOLD":          100,
    "QTCB_FEMALE_THRESHOLD":   460,
    "QTCB_MALE_THRESHOLD":     440,
    "PRI_THRESHOLD":           200,
    "P_AXIS_MIN":                0,
    "P_AXIS_MAX":               90,
    "R_AXIS_MIN":              -30,
    "R_AXIS_MAX":               90,
}

# ── ECG Feature Names ────────────────────────────────────────────────────────
ECG_TEXT_FEATURES = [
    "AR_bpm", "VR_bpm", "QRSD_ms", "QT_ms", "QTcB_ms", "PRI_ms",
    "P_axis", "R_axis", "T_axis",
    "age", "gender_encoded",
    "qrs_wide", "pr_long", "qtc_prolonged",
]
ECG_FEATURE_DIM = len(ECG_TEXT_FEATURES)  # 14

# ── Clinical Feature Names ────────────────────────────────────────────────────
CLINICAL_FEATURES = [
    "age", "gender_encoded", "bp_systolic", "bp_diastolic",
    "hba1c", "cholesterol", "bmi",
    "smoking", "alcohol", "physical_activity_encoded",
    "diabetes", "hypertension", "heart_disease_history",
    "stroke_history", "family_history",
]
CLINICAL_FEATURE_DIM = len(CLINICAL_FEATURES)  # 15

# ── CNN-BiLSTM Hyperparameters ────────────────────────────────────────────────
MODEL_CONFIG = {
    "input_shape":      (1, ECG_FEATURE_DIM),
    "cnn_filters_1":    32,
    "cnn_filters_2":    64,
    "cnn_kernel_size":  3,
    "lstm_units_1":     64,
    "lstm_units_2":     32,
    "dense_units":      64,
    "dropout_rate":     0.3,
    "num_classes":      NUM_CLASSES,
    "epochs":           100,
    "batch_size":       16,
    "learning_rate":    0.001,
    "validation_split": 0.2,
    "patience":         15,
}

# ── XGBoost Risk Model Config ─────────────────────────────────────────────────
XGBOOST_CONFIG = {
    "n_estimators":      200,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "logloss",
    "random_state":      42,
}

# ── Data Augmentation ─────────────────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "target_per_class": 20,
    "noise_std":        0.02,
    "random_seed":      42,
}

# ── Model File Names ──────────────────────────────────────────────────────────
MODEL_NAMES = {
    "cnn_bilstm":       "cnn_bilstm",
    "stroke":           "stroke_model",
    "heart_disease":    "heart_disease_model",
    "scaler":           "feature_scaler",
    "label_encoder":    "label_encoder",
    "training_history": "cnn_bilstm_history",
}

# ── Application ───────────────────────────────────────────────────────────────
APP_TITLE   = "HECAR — Cardiac AI Diagnostic Framework"
APP_VERSION = "1.0.0"
APP_WIDTH   = 1400
APP_HEIGHT  = 900

# ── GUI Theme ─────────────────────────────────────────────────────────────────
THEME = {
    "bg_primary":       "#0D1117",
    "bg_secondary":     "#161B22",
    "bg_card":          "#1C2128",
    "bg_sidebar":       "#010409",
    "accent_primary":   "#00D4AA",
    "accent_secondary": "#1F6FEB",
    "text_primary":     "#E6EDF3",
    "text_secondary":   "#8B949E",
    "text_muted":       "#484F58",
    "success":          "#3FB950",
    "warning":          "#D29922",
    "danger":           "#F85149",
    "border":           "#30363D",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ── ECG Normal Ranges (for GUI color-coding) ──────────────────────────────────
ECG_NORMAL_RANGES = {
    "AR_bpm":   (60,   100),
    "VR_bpm":   (60,   100),
    "QRSD_ms":  (60,   100),
    "QT_ms":    (350,  440),
    "QTcB_ms":  (360,  460),
    "PRI_ms":   (120,  200),
    "P_axis":   (0,    90),
    "R_axis":   (-30,  90),
    "T_axis":   (-10,  90),
}
