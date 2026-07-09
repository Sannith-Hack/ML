import customtkinter as ctk
import webbrowser
from pathlib import Path

from modules.gui.theme import HECAR_THEME
from modules.gui.components import StyledButton, SectionHeader

class ReportScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        
        SectionHeader(self, "Clinical Report").pack(anchor="w", pady=(0, 20))
        
        self.lbl_path = ctk.CTkLabel(self, text="No report generated yet.", font=HECAR_THEME.font_body, text_color=HECAR_THEME.text_secondary)
        self.lbl_path.pack(pady=20)
        
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(pady=20)
        
        self.btn_open = StyledButton(btns, "Open in Web Browser", command=self.open_browser, state="disabled")
        self.btn_open.pack(side="left", padx=10)

    def on_show(self):
        path = self.app.shared_state.get("report_path")
        if path and Path(path).exists():
            self.lbl_path.configure(text=f"Report ready: {Path(path).name}")
            self.btn_open.configure(state="normal")
        else:
            self.lbl_path.configure(text="No report generated yet.")
            self.btn_open.configure(state="disabled")

    def open_browser(self):
        path = self.app.shared_state.get("report_path")
        if path and Path(path).exists():
            webbrowser.open(f"file://{Path(path).resolve()}")
