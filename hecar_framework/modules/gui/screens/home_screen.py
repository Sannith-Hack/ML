import customtkinter as ctk
from pathlib import Path
from modules.gui.theme import HECAR_THEME
from modules.gui.components import MetricCard, StyledButton
from config import OUTPUTS_DIR, MODEL_SAVED_DIR, ECG_PDFS_DIR

class HomeScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 30))
        
        ctk.CTkLabel(header, text="HECAR Dashboard", font=("Inter", 32, "bold"), text_color=HECAR_THEME.text_primary).pack(anchor="w")
        ctk.CTkLabel(header, text="AI-Powered Cardiac Arrhythmia Classification & Cardiovascular Risk Prediction", 
                     font=HECAR_THEME.font_body, text_color=HECAR_THEME.accent_primary).pack(anchor="w", pady=(5,0))
                     
        # Stats Row
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(fill="x", pady=20)
        
        self.stat_pdfs = MetricCard(stats_frame, "ECG PDFs Found", "0")
        self.stat_pdfs.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.stat_models = MetricCard(stats_frame, "Models Available", "0")
        self.stat_models.pack(side="left", fill="x", expand=True, padx=10)
        
        self.stat_reports = MetricCard(stats_frame, "Reports Generated", "0")
        self.stat_reports.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # Pipeline Flow
        ctk.CTkLabel(self, text="Pipeline Workflow", font=HECAR_THEME.font_subheading, text_color=HECAR_THEME.text_primary).pack(anchor="w", pady=(20,10))
        flow_frame = ctk.CTkFrame(self, fg_color="transparent")
        flow_frame.pack(fill="x")
        
        steps = ["Upload", "Preprocess", "Train", "Clinical", "Predict", "Report"]
        for i, step in enumerate(steps):
            f = ctk.CTkFrame(flow_frame, fg_color=HECAR_THEME.bg_secondary, corner_radius=8, width=120, height=60)
            f.pack(side="left", padx=5, expand=True, fill="both")
            f.pack_propagate(False)
            ctk.CTkLabel(f, text=f"{i+1}. {step}", font=HECAR_THEME.font_body).place(relx=0.5, rely=0.5, anchor="center")
            
        # Quick Actions
        actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        actions_frame.pack(fill="x", pady=40)
        
        StyledButton(actions_frame, "Start New Diagnosis", command=lambda: self.app.show_screen("Upload")).pack(side="left", padx=(0, 20))
        StyledButton(actions_frame, "View Last Report", variant="secondary", command=lambda: self.app.show_screen("Report")).pack(side="left")

    def on_show(self):
        """Update stats when screen is shown."""
        try:
            pdf_count = len(list(Path(ECG_PDFS_DIR).glob("*.pdf")))
            model_count = len(list(Path(MODEL_SAVED_DIR).glob("*.keras"))) + len(list(Path(MODEL_SAVED_DIR).glob("*.pkl")))
            report_count = len(list(Path(OUTPUTS_DIR).glob("*.html")))
            
            self.stat_pdfs.set_value(str(pdf_count))
            self.stat_models.set_value(str(model_count))
            self.stat_reports.set_value(str(report_count))
        except Exception:
            pass
