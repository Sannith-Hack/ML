import customtkinter as ctk
import tkinter.filedialog as fd
import threading
from pathlib import Path

from modules.gui.theme import HECAR_THEME
from modules.gui.components import StyledButton, SectionHeader, StatusBadge, ECGMetricRow, ScrollableResultFrame
from modules.pdf_processor.pdf_loader import PDFLoader
from modules.pdf_processor.ocr_extractor import OCRExtractor
from config import TRICOG_DATA_DIR

class UploadScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        
        self.loader = PDFLoader()
        self.ocr = OCRExtractor()
        
        SectionHeader(self, "Upload ECG Report").pack(anchor="w", pady=(0, 20))
        
        # Upload Area
        upload_frame = ctk.CTkFrame(self, fg_color=HECAR_THEME.bg_secondary, border_width=2, border_color=HECAR_THEME.border)
        upload_frame.pack(fill="x", pady=10, ipady=30)
        
        self.lbl_selected = ctk.CTkLabel(upload_frame, text="No PDF Selected", font=HECAR_THEME.font_body, text_color=HECAR_THEME.text_secondary)
        self.lbl_selected.pack(pady=10)
        
        btns = ctk.CTkFrame(upload_frame, fg_color="transparent")
        btns.pack()
        StyledButton(btns, "Browse Single PDF", command=self.browse_file).pack(side="left", padx=10)
        StyledButton(btns, "Load TRICOG Folder", variant="secondary", command=self.load_tricog).pack(side="left", padx=10)
        
        # Extract Button
        self.btn_extract = StyledButton(self, "Extract Data", command=self.extract_data, state="disabled")
        self.btn_extract.pack(pady=20)
        
        # Results Area
        res_container = ctk.CTkFrame(self, fg_color="transparent")
        res_container.pack(fill="both", expand=True)
        
        # Left: PDF List
        list_frame = ctk.CTkFrame(res_container, fg_color="transparent")
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        SectionHeader(list_frame, "Loaded Files").pack(anchor="w")
        self.scroll_list = ScrollableResultFrame(list_frame)
        self.scroll_list.pack(fill="both", expand=True, pady=10)
        
        # Right: Extracted Data
        data_frame = ctk.CTkFrame(res_container, fg_color="transparent")
        data_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        SectionHeader(data_frame, "Extracted Metadata").pack(anchor="w")
        self.scroll_data = ScrollableResultFrame(data_frame)
        self.scroll_data.pack(fill="both", expand=True, pady=10)
        
        self.pdf_paths = []
        self.current_raw = None

    def browse_file(self):
        file = fd.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file:
            self.pdf_paths = [file]
            self.lbl_selected.configure(text=Path(file).name)
            self.btn_extract.configure(state="normal")
            self._update_list()

    def load_tricog(self):
        if Path(TRICOG_DATA_DIR).exists():
            files = list(Path(TRICOG_DATA_DIR).glob("*.pdf"))
            if files:
                self.pdf_paths = [str(f) for f in files]
                self.lbl_selected.configure(text=f"Loaded {len(files)} files from TRICOG folder.")
                self.btn_extract.configure(state="normal")
                self._update_list()

    def _update_list(self):
        for widget in self.scroll_list.winfo_children():
            widget.destroy()
            
        for path in self.pdf_paths:
            f = ctk.CTkFrame(self.scroll_list, fg_color=HECAR_THEME.bg_card, corner_radius=6)
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=Path(path).name).pack(side="left", padx=10, pady=5)
            
    def extract_data(self):
        if not self.pdf_paths: return
        self.app.update_status("Extracting PDF...", HECAR_THEME.warning)
        
        # For GUI demo, we just extract the first one selected for the preview
        path = self.pdf_paths[0]
        self.app.shared_state["current_pdf_path"] = path
        
        def run():
            data = self.loader.load(path)
            if data:
                meta = self.ocr.extract_metadata(data["raw_text"])
                self.app.shared_state["ecg_metadata"] = meta
                self.after(0, self._show_extracted, meta)
                self.app.update_status("Extraction complete", HECAR_THEME.success)
            else:
                self.app.update_status("Extraction failed", HECAR_THEME.danger)
                
        threading.Thread(target=run, daemon=True).start()
        
    def _show_extracted(self, meta):
        for widget in self.scroll_data.winfo_children():
            widget.destroy()
            
        for k, v in meta.items():
            ECGMetricRow(self.scroll_data, k.replace("_", " ").title(), str(v)).pack(fill="x", pady=2)
            
        StyledButton(self.scroll_data, "Proceed to Clinical Data", command=lambda: self.app.show_screen("Clinical")).pack(pady=20)
