# Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network
==========================================================================

## Project Overview

This project implements an automated system for detecting **cardiac arrhythmia** from ECG signal data using deep learning models — specifically **LSTM (Long Short-Term Memory)** and **CNN (Convolutional Neural Network)**. The application provides a **Tkinter-based GUI** for an interactive, step-by-step workflow: loading data, preprocessing, training/loading models, and visualizing performance metrics.

The dataset used is the **MIT-BIH Arrhythmia Dataset**, and the system classifies ECG records into **7 distinct cardiac conditions**.

---

## 🚀 Evolution to the Next-Generation Suite (`hecar_framework/`)

While this root project (`Main.py`) serves as the foundational academic baseline comparing raw tabular ECG feature classification across LSTM and CNN architectures, the framework has evolved into the **HECAR Clinical Diagnostic Suite** (`hecar_framework/`), a production-grade, multi-modal medical application:
- **Streamlit Web & Mobile Application (`hecar_framework/web_app.py`)**: Hosted on Streamlit Community Cloud (`https://sirwork.streamlit.app`) with responsive top horizontal navigation, medical glassmorphism dark-mode styling, and atomic `on_click` state callbacks.
- **CustomTkinter Desktop Suite (`hecar_framework/main.py`)**: An 8-screen dark-mode desktop GUI for local hospital workstations.
- **PDF & OCR Ingestion:** Instead of requiring raw CSV uploads (`arrhythmia.csv`), the modern suite directly ingests hospital ECG PDF reports (Tricog / Bhageerath format), performing OCR text parsing (`pdfplumber` + `pytesseract`) to extract digital waveform parameters (`AR`, `VR`, `QRSD`, `QT`, `QTcB`, `PRI`, and `Axis`).
- **Multi-Modal Clinical Fusion & Risk Prognosis:** Fuses ECG parameters with **13 patient clinical variables** (`Age`, `Gender`, `Blood Pressure`, `Cholesterol`, `HbA1c`, `Habits`, and `Medical History`) across an **XGBoost** risk engine to compute **10-Year Ischemic Stroke Risk** and **10-Year Coronary Heart Disease (CAD) Risk**, complete with **SHAP explainability plots** and downloadable **HTML clinical reports**.

> **Note:** For full architectural documentation of the modern multi-modal suite, please see [hecar_framework/GEMINI.md](file:///D:/User/Desktop/Sir%20Work%202.0/hecar_framework/GEMINI.md).

---

## Project Structure (Baseline MIT-BIH Application)

```
project-root/
│
├── Main.py                  # Main baseline application entry point (Tkinter GUI + ML pipeline)
├── requirements.txt         # Python dependencies (pip install commands)
├── run.bat                  # Windows batch script to launch the app
├── output.html              # Generated performance comparison table (HTML)
├── GEMINI.md                # Project documentation (this file)
│
├── Dataset/
│   └── arrhythmia.csv       # MIT-BIH Arrhythmia Dataset (CSV format, ~4.4 MB)
│
├── model/                   # Saved model artifacts (auto-generated after training)
│   ├── lstm_model.json          # LSTM model architecture (JSON)
│   ├── lstm_model_weights.h5    # LSTM trained weights (H5)
│   ├── lstm_history.pckl        # LSTM training history (pickle)
│   ├── cnn_model.json           # CNN model architecture (JSON)
│   ├── cnn_model_weights.h5     # CNN trained weights (H5)
│   ├── cnn_history.pckl         # CNN training history (pickle)
│   └── pca.txt                  # PCA transformation data
│
└── hecar_framework/         # Next-Generation Multi-Modal Suite (Streamlit Web + CustomTkinter Desktop)
    ├── web_app.py               # Streamlit Cloud Application (`sirwork.streamlit.app`)
    ├── main.py                  # CustomTkinter 8-screen desktop GUI
    ├── config.py                # Global configurations and module pre-loading definitions
    └── GEMINI.md                # Full HECAR framework documentation
```

---

## Technology Stack (Baseline Application)

| Category         | Technology / Library            | Version        |
|------------------|---------------------------------|----------------|
| Language         | Python                          | 3.x            |
| GUI Framework    | Tkinter                         | Built-in       |
| Deep Learning    | Keras + TensorFlow              | 2.3.1 / 1.14.0 |
| Data Processing  | NumPy, Pandas                   | 1.19.2 / 0.25.3|
| ML Utilities     | Scikit-learn                    | 0.22.2         |
| Visualization    | Matplotlib, Seaborn             | 3.1.1 / 0.10.1 |
| Model Storage    | H5py, Pickle                    | 2.10.0         |
| Serialization    | Protobuf                        | 3.16.0         |

---

## Dataset

- **File:** `Dataset/arrhythmia.csv`
- **Source:** MIT-BIH Arrhythmia Database
- **Size:** ~4.4 MB
- **Features:** 279 ECG signal attributes (columns 0–278)
- **Target Column:** Column `279` (class label)
- **Classes (7 cardiac conditions):**

| Index | Condition |
|-------|-----------|
| 0 | Normal heart |
| 1 | Ischemic changes (Coronary Artery Disease) |
| 2 | Old Anterior Myocardial Infarction |
| 3 | Old Inferior Myocardial Infarction |
| 4 | Sinus tachycardy |
| 5 | Sinus bradycardy |
| 6 | Right bundle branch block |

---

## Application Workflow (`Main.py`)

The Tkinter GUI provides a sequential button-based pipeline:

```
[Upload Dataset] -> [Preprocess Dataset] -> [Run LSTM] -> [Run CNN] -> [Graph] -> [Performance Table]
```

### 1. Upload Dataset (`uploadDataset`)
- Opens a file dialog to select `arrhythmia.csv` from the `Dataset/` folder.
- Loads data into a Pandas DataFrame.
- Displays the first few rows and a bar chart of class distribution.

### 2. Preprocess Dataset (`preprocessDataset`)
- Fills missing values with `0`.
- Encodes labels using `LabelEncoder`.
- **Normalizes** features using L2 normalization (`sklearn.preprocessing.normalize`).
- Applies **PCA** (Principal Component Analysis) — reduces to **40 components**.
- Reshapes data for LSTM input: `(samples, 40, 1)`.
- Splits into **80% train / 20% test** using `train_test_split`.
- One-hot encodes labels with `to_categorical`.

### 3. Run LSTM (`runLSTM`)
- **Architecture:**
  - `LSTM(100)` -> `Dropout(0.2)` -> `Dense(100, relu)` -> `Dense(7, softmax)`
- Trained with `categorical_crossentropy` loss, `adam` optimizer, 100 epochs, batch size 16.
- Model persisted to `model/lstm_model.json` + `model/lstm_model_weights.h5`.
- If saved model exists, it is **loaded directly** (skipping retraining).
- Training history saved to `model/lstm_history.pckl`.

### 4. Run CNN (`runCNN`)
- Reshapes data to `(samples, 40, 1, 1)` for 2D convolutions.
- **Architecture:**
  - `Conv2D(32)` -> `MaxPool2D` -> `Conv2D(32)` -> `MaxPool2D` -> `Flatten` -> `Dense(256, relu)` -> `Dense(7, softmax)`
- Trained with same optimizer and hyperparameters as LSTM.
- Model saved/loaded similarly to LSTM.

### 5. Training Graph (`graph`)
- Loads LSTM and CNN training history from pickle files.
- Plots a combined **Accuracy & Loss graph** for both models across epochs.

### 6. Performance Table (`performanceTable`)
- Generates an HTML table (`output.html`) comparing LSTM vs CNN on:
  - Accuracy, Precision, Recall, F-Score, Sensitivity, Specificity
- Opens the HTML file in the default web browser.

---

## Evaluation Metrics

Computed by `calculateMetrics()` for each model:

| Metric      | Description |
|-------------|-------------|
| **Accuracy**    | `(TP + TN) / Total` |
| **Precision**   | Macro-averaged precision across all classes |
| **Recall**      | Macro-averaged recall across all classes |
| **F-Score**     | Macro-averaged F1-score |
| **Sensitivity** | `TP / (TP + FN)` derived from confusion matrix |
| **Specificity** | `TN / (TN + FP)` derived from confusion matrix |

A **Seaborn heatmap** confusion matrix is plotted for each algorithm.

---

## Setup & Installation

### Prerequisites
- Python 3.6–3.8 (recommended for TensorFlow 1.14 compatibility with `Main.py`)
- Python 3.12+ (recommended for modern `hecar_framework/` suite)
- Windows OS (for `run.bat`)

### Install Dependencies (Root Baseline App)

```bash
pip install numpy==1.19.2
pip install pandas==0.25.3
pip install matplotlib==3.1.1
pip install keras==2.3.1
pip install tensorflow==1.14.0
pip install h5py==2.10.0
pip install protobuf==3.16.0
pip install scikit-learn==0.22.2.post1
pip install seaborn==0.10.1
```

> **Note:** It is strongly recommended to use separate **virtual environments** for the legacy Tkinter app (`Main.py`) and the modern Streamlit web suite (`hecar_framework/web_app.py`).

```bash
python -m venv venv
venv\Scripts\activate
# Install requirements above
```

### Run the Application

**Option 1 – Batch file (Windows):**
```bat
run.bat
```

**Option 2 – Command line:**
```bash
python Main.py
```

---

## Key Design Decisions

- **Model Caching:** Both LSTM and CNN models are saved after training. On subsequent runs, models are loaded from disk instead of retraining, saving significant computation time.
- **PCA Dimensionality Reduction:** The 279 raw ECG features are reduced to 40 principal components, improving training efficiency and reducing noise.
- **Tkinter GUI:** Provides a simple desktop interface for non-technical users to run the full ML pipeline without writing code.
- **Dual Model Comparison:** Running both LSTM and CNN allows direct performance benchmarking on the same dataset split.

---

## Known Issues / Compatibility Notes

> **WARNING:** The root baseline project (`Main.py`) uses **TensorFlow 1.14** and **Keras 2.3.1**, which are outdated. The following APIs have been deprecated or removed in newer versions:

- `keras.utils.np_utils.to_categorical` -> use `tensorflow.keras.utils.to_categorical`
- `lstm._make_predict_function()` -> not needed in TF 2.x
- `Dense(output_dim=...)` -> replaced by `Dense(units=...)` in Keras 2.x+
- `Convolution2D` -> replaced by `Conv2D`
- `model_from_json` -> available in `tensorflow.keras.models`

> **TIP:** For modern TensorFlow 2.x + Streamlit + XGBoost workflows, use the updated implementation inside `hecar_framework/`.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `Main.py` | Complete application: GUI setup, data loading, preprocessing, model training/loading, evaluation, and visualization |
| `requirements.txt` | List of pip install commands for all dependencies |
| `run.bat` | Windows shortcut to launch `Main.py` |
| `output.html` | Auto-generated performance metrics HTML table |
| `Dataset/arrhythmia.csv` | MIT-BIH ECG arrhythmia dataset in CSV format |
| `model/lstm_model.json` | Serialized LSTM model architecture |
| `model/lstm_model_weights.h5` | Trained LSTM model weights |
| `model/lstm_history.pckl` | LSTM training accuracy/loss history (per epoch) |
| `model/cnn_model.json` | Serialized CNN model architecture |
| `model/cnn_model_weights.h5` | Trained CNN model weights |
| `model/cnn_history.pckl` | CNN training accuracy/loss history (per epoch) |
| `model/pca.txt` | Stored PCA transformation data |

---

## Authors & Context

This project was developed as an academic implementation of deep learning-based cardiac arrhythmia detection, comparing **LSTM** against **CNN** on ECG feature data. It has since expanded into the comprehensive **HECAR Framework** (`hecar_framework/`) combining PDF parsing, clinical risk fusion, and cloud deployment.
