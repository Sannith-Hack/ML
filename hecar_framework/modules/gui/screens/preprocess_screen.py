import customtkinter as ctk
import threading
from modules.gui.theme import HECAR_THEME
from modules.gui.components import StyledButton, SectionHeader, MetricCard
from modules.feature_engineering.dataset_builder import DatasetBuilder
from config import TRICOG_DATA_DIR, PROCESSED_DIR
from pathlib import Path

class PreprocessScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.builder = DatasetBuilder()
        self.df = None
        
        SectionHeader(self, "Dataset Preprocessing").pack(anchor="w", pady=(0, 20))
        
        # Top Stats
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="x", pady=10)
        
        self.mc_total = MetricCard(self.stats_frame, "Total Records", "0")
        self.mc_total.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.mc_classes = MetricCard(self.stats_frame, "Unique Classes", "0")
        self.mc_classes.pack(side="left", fill="x", expand=True, padx=10)
        
        # Controls
        ctrl = ctk.CTkFrame(self, fg_color=HECAR_THEME.bg_secondary)
        ctrl.pack(fill="x", pady=20, ipady=10)
        
        StyledButton(ctrl, "Build from TRICOG PDFs", command=self.build_dataset).pack(side="left", padx=20)
        StyledButton(ctrl, "Augment Dataset", variant="secondary", command=self.augment_dataset).pack(side="left", padx=10)
        StyledButton(ctrl, "Save Final CSV", variant="secondary", command=self.save_csv).pack(side="left", padx=10)
        
        # Log area
        SectionHeader(self, "Processing Log").pack(anchor="w", pady=(10, 5))
        self.log_box = ctk.CTkTextbox(self, fg_color=HECAR_THEME.bg_card, text_color=HECAR_THEME.text_secondary, font=HECAR_THEME.font_small)
        self.log_box.pack(fill="both", expand=True)

    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def build_dataset(self):
        self.log("Starting dataset build from PDFs...")
        self.app.update_status("Building dataset...", HECAR_THEME.warning)
        
        def run():
            try:
                self.df = self.builder.build_from_folder(str(TRICOG_DATA_DIR))
                self.after(0, self._update_stats)
                self.after(0, lambda: self.log(f"Successfully built dataset with {len(self.df)} records."))
                self.app.update_status("Dataset built", HECAR_THEME.success)
            except Exception as e:
                self.after(0, lambda: self.log(f"Error: {str(e)}"))
                self.app.update_status("Build failed", HECAR_THEME.danger)
        
        threading.Thread(target=run, daemon=True).start()

    def augment_dataset(self):
        if self.df is None:
            self.log("No dataset loaded. Build first.")
            return
            
        self.log("Augmenting dataset to balance classes (target=20)...")
        def run():
            try:
                self.df = self.builder.augment(self.df, target_per_class=20)
                self.after(0, self._update_stats)
                self.after(0, lambda: self.log(f"Augmentation complete. New size: {len(self.df)} records."))
            except Exception as e:
                self.after(0, lambda: self.log(f"Error: {str(e)}"))
                
        threading.Thread(target=run, daemon=True).start()

    def save_csv(self):
        if self.df is not None:
            path = Path(PROCESSED_DIR) / "hecar_dataset.csv"
            self.builder.save(self.df, str(path))
            self.log(f"Saved to {path}")

    def _update_stats(self):
        if self.df is not None:
            self.mc_total.set_value(str(len(self.df)))
            self.mc_classes.set_value(str(self.df["arrhythmia_class"].nunique()))
