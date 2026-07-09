import logging
import base64
import webbrowser
from pathlib import Path
from typing import Dict, Any

try:
    from jinja2 import Template
except ImportError:
    pass

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HECAR Clinical Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0D1117; color: #E6EDF3; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background-color: #161B22; border-radius: 8px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .header { border-bottom: 2px solid #30363D; padding-bottom: 20px; margin-bottom: 20px; display: flex; justify-content: space-between; }
        h1, h2, h3 { color: #00D4AA; margin-top: 0; }
        .patient-info { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .info-item { font-size: 14px; }
        .info-label { color: #8B949E; font-weight: bold; }
        
        .section { margin-bottom: 30px; background-color: #1C2128; padding: 20px; border-radius: 8px; border: 1px solid #30363D; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .metric-card { background-color: #21262D; padding: 15px; border-radius: 6px; text-align: center; }
        .metric-val { font-size: 24px; font-weight: bold; color: #E6EDF3; }
        .metric-label { font-size: 12px; color: #8B949E; }
        
        .result-highlight { font-size: 20px; font-weight: bold; padding: 15px; border-radius: 6px; margin-bottom: 15px; }
        .normal { background-color: rgba(63, 185, 80, 0.2); color: #3FB950; border: 1px solid #3FB950; }
        .warning { background-color: rgba(210, 153, 34, 0.2); color: #D29922; border: 1px solid #D29922; }
        .danger { background-color: rgba(248, 81, 73, 0.2); color: #F85149; border: 1px solid #F85149; }
        
        .risk-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .risk-card { background-color: #21262D; padding: 20px; border-radius: 6px; text-align: center; }
        .risk-score { font-size: 36px; font-weight: bold; margin: 10px 0; }
        
        .shap-image { width: 100%; max-width: 600px; margin: 0 auto; display: block; border-radius: 6px; }
        ul { padding-left: 20px; }
        li { margin-bottom: 8px; }
        .footer { text-align: center; margin-top: 40px; font-size: 12px; color: #8B949E; border-top: 1px solid #30363D; padding-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>HECAR Diagnostic Report</h1>
                <p style="color: #8B949E; margin: 0;">Hybrid ECG & Cardiovascular Assessment</p>
            </div>
            <div style="text-align: right;">
                <div class="info-label">Date</div>
                <div>{{ patient_info.date }}</div>
            </div>
        </div>

        <div class="section">
            <h2>Patient Information</h2>
            <div class="patient-info">
                <div class="info-item"><span class="info-label">Name:</span> {{ patient_info.name }}</div>
                <div class="info-item"><span class="info-label">Patient ID:</span> {{ patient_info.patient_id }}</div>
                <div class="info-item"><span class="info-label">Age:</span> {{ patient_info.age }}</div>
                <div class="info-item"><span class="info-label">Gender:</span> {{ patient_info.gender }}</div>
            </div>
        </div>

        <div class="section">
            <h2>Arrhythmia Classification (CNN-BiLSTM)</h2>
            <div class="result-highlight {% if arrhythmia_result.class_name == 'Normal Sinus Rhythm' %}normal{% else %}warning{% endif %}">
                {{ arrhythmia_result.class_name }} ({{ arrhythmia_result.confidence }}% Confidence)
            </div>
            <p>{{ arrhythmia_result.description }}</p>
            
            <h3 style="margin-top: 20px;">ECG Parameters</h3>
            <div class="grid-3">
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.AR_bpm }}</div><div class="metric-label">AR (bpm)</div></div>
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.VR_bpm }}</div><div class="metric-label">VR (bpm)</div></div>
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.QRSD_ms }}</div><div class="metric-label">QRSD (ms)</div></div>
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.QT_ms }}</div><div class="metric-label">QT (ms)</div></div>
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.QTcB_ms }}</div><div class="metric-label">QTcB (ms)</div></div>
                <div class="metric-card"><div class="metric-val">{{ ecg_metrics.PRI_ms }}</div><div class="metric-label">PRI (ms)</div></div>
            </div>
        </div>

        <div class="section">
            <h2>Cardiovascular Risk Prediction (XGBoost)</h2>
            <div class="risk-section">
                <div class="risk-card">
                    <div class="metric-label">10-Year Stroke Risk</div>
                    <div class="risk-score {% if stroke_risk.level == 'High' %}danger{% elif stroke_risk.level == 'Medium' %}warning{% else %}normal{% endif %}">
                        {{ stroke_risk.score }}%
                    </div>
                    <div style="font-weight: bold;">{{ stroke_risk.level }} Risk</div>
                </div>
                <div class="risk-card">
                    <div class="metric-label">10-Year CHD Risk</div>
                    <div class="risk-score {% if chd_risk.level == 'High' %}danger{% elif chd_risk.level == 'Medium' %}warning{% else %}normal{% endif %}">
                        {{ chd_risk.score }}%
                    </div>
                    <div style="font-weight: bold;">{{ chd_risk.level }} Risk</div>
                </div>
            </div>
        </div>

        {% if shap_chart_path %}
        <div class="section">
            <h2>SHAP Feature Importance</h2>
            <p>Top factors influencing the risk predictions for this patient:</p>
            <img class="shap-image" src="data:image/png;base64,{{ shap_chart_path }}" alt="SHAP Plot">
        </div>
        {% endif %}

        <div class="section">
            <h2>Clinical Recommendations</h2>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>

        <div class="footer">
            {{ disclaimer }}<br>
            Generated by HECAR Framework
        </div>
    </div>
</body>
</html>
"""

class ReportGenerator:
    """Generates styled HTML clinical reports."""
    
    def __init__(self):
        self.template = Template(HTML_TEMPLATE)

    def _encode_image(self, image_path: str) -> str:
        if not image_path or not Path(image_path).exists():
            return ""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return ""

    def generate(self, report_data: Dict[str, Any], output_path: str) -> str:
        try:
            # Encode image if provided
            if "shap_chart_path" in report_data and report_data["shap_chart_path"]:
                report_data["shap_chart_path"] = self._encode_image(report_data["shap_chart_path"])
                
            # Default disclaimer
            if "disclaimer" not in report_data:
                report_data["disclaimer"] = "Disclaimer: This report is AI-generated for investigational/diagnostic assistance purposes. It does not replace professional medical judgment."
                
            html_content = self.template.render(**report_data)
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            logger.info(f"Report generated at {output_path}")
            return output_path
        except Exception as e:
            logger.exception(f"Failed to generate report: {e}")
            return ""

    def open_in_browser(self, html_path: str):
        if Path(html_path).exists():
            webbrowser.open(f"file://{Path(html_path).resolve()}")
