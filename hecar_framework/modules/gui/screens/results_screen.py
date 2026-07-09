import customtkinter as ctk
from PIL import Image
from pathlib import Path
from modules.gui.theme import HECAR_THEME
from modules.gui.components import SectionHeader, StyledButton, MetricCard, ScrollableResultFrame
from modules.explainability.report_generator import ReportGenerator
from config import OUTPUTS_DIR
import datetime

class ResultsScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        
        SectionHeader(self, "Diagnosis Results").pack(anchor="w", pady=(0, 10))
        
        # Header (Patient Info)
        self.lbl_patient = ctk.CTkLabel(self, text="Patient: -", font=HECAR_THEME.font_subheading)
        self.lbl_patient.pack(anchor="w", pady=(0, 20))
        
        scroll = ScrollableResultFrame(self)
        scroll.pack(fill="both", expand=True)
        
        # Arrhythmia Section
        arr_frame = ctk.CTkFrame(scroll, fg_color=HECAR_THEME.bg_card)
        arr_frame.pack(fill="x", pady=10, padx=10, ipady=15)
        
        ctk.CTkLabel(arr_frame, text="Arrhythmia Classification", font=HECAR_THEME.font_subheading, text_color=HECAR_THEME.text_secondary).pack(pady=(10,5))
        self.lbl_arr_class = ctk.CTkLabel(arr_frame, text="-", font=("Inter", 28, "bold"), text_color=HECAR_THEME.accent_primary)
        self.lbl_arr_class.pack()
        self.lbl_arr_conf = ctk.CTkLabel(arr_frame, text="-", font=HECAR_THEME.font_body)
        self.lbl_arr_conf.pack()
        
        # Risk Section
        risk_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        risk_frame.pack(fill="x", pady=10, padx=10)
        
        self.mc_stroke = MetricCard(risk_frame, "10-Yr Stroke Risk", "-", "%", HECAR_THEME.danger)
        self.mc_stroke.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.mc_chd = MetricCard(risk_frame, "10-Yr CHD Risk", "-", "%", HECAR_THEME.danger)
        self.mc_chd.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # SHAP Image
        self.img_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self.img_frame.pack(fill="x", pady=20, padx=10)
        
        self.shap_label = ctk.CTkLabel(self.img_frame, text="")
        self.shap_label.pack()
        
        # Buttons
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", pady=20)
        
        StyledButton(btns, "Generate Full Report", command=self.generate_report).pack(side="left", padx=(0, 20))

    def on_show(self):
        meta = self.app.shared_state.get("ecg_metadata", {})
        clin = self.app.shared_state.get("clinical_data", {})
        arr = self.app.shared_state.get("arrhythmia_result", {})
        risk = self.app.shared_state.get("risk_results", {})
        
        name = meta.get("patient_name") or "Unknown"
        age = clin.get("age") or meta.get("age") or "--"
        
        self.lbl_patient.configure(text=f"Patient: {name} (Age: {age})")
        
        if arr:
            self.lbl_arr_class.configure(text=arr.get("class_name", "-"))
            if arr.get("class_name") == "Normal Sinus Rhythm":
                self.lbl_arr_class.configure(text_color=HECAR_THEME.success)
            else:
                self.lbl_arr_class.configure(text_color=HECAR_THEME.warning)
            self.lbl_arr_conf.configure(text=f"{arr.get('confidence', 0)}% Confidence")
            
        if risk:
            self.mc_stroke.set_value(risk.get("stroke_score", "-"))
            self.mc_chd.set_value(risk.get("chd_score", "-"))
            
            sc = risk.get("stroke_level", "")
            if sc == "Low": self.mc_stroke.set_value(risk.get("stroke_score"), HECAR_THEME.success)
            elif sc == "Medium": self.mc_stroke.set_value(risk.get("stroke_score"), HECAR_THEME.warning)
            
            cc = risk.get("chd_level", "")
            if cc == "Low": self.mc_chd.set_value(risk.get("chd_score"), HECAR_THEME.success)
            elif cc == "Medium": self.mc_chd.set_value(risk.get("chd_score"), HECAR_THEME.warning)
            
        shap_path = self.app.shared_state.get("shap_chart_path")
        if shap_path and Path(shap_path).exists():
            img = Image.open(shap_path)
            # scale down
            img.thumbnail((800, 600))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.shap_label.configure(image=ctk_img)

    def generate_report(self):
        meta = self.app.shared_state.get("ecg_metadata", {})
        clin = self.app.shared_state.get("clinical_data", {})
        
        report_data = {
            "patient_info": {
                "name": meta.get("patient_name") or "Unknown",
                "patient_id": meta.get("patient_id") or "N/A",
                "age": clin.get("age") or meta.get("age") or "N/A",
                "gender": clin.get("gender") or meta.get("gender") or "N/A",
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            },
            "ecg_metrics": meta,
            "arrhythmia_result": self.app.shared_state.get("arrhythmia_result", {}),
            "stroke_risk": self.app.shared_state.get("risk_results", {}),
            "chd_risk": self.app.shared_state.get("risk_results", {}),
            "shap_chart_path": self.app.shared_state.get("shap_chart_path"),
            "recommendations": [
                "Schedule a follow-up appointment.",
                "Review lifestyle factors."
            ]
        }
        
        path = OUTPUTS_DIR / f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        rg = ReportGenerator()
        out = rg.generate(report_data, str(path))
        
        if out:
            self.app.shared_state["report_path"] = out
            self.app.update_status("Report Generated", HECAR_THEME.success)
            self.app.show_screen("Report")
