import customtkinter as ctk
import logging

from .theme import apply_theme, HECAR_THEME
from config import APP_TITLE, APP_WIDTH, APP_HEIGHT

# Placeholder imports for screens (will be created next)
from .screens.home_screen import HomeScreen
from .screens.upload_screen import UploadScreen
from .screens.preprocess_screen import PreprocessScreen
from .screens.train_screen import TrainScreen
from .screens.clinical_screen import ClinicalScreen
from .screens.predict_screen import PredictScreen
from .screens.results_screen import ResultsScreen
from .screens.report_screen import ReportScreen

logger = logging.getLogger(__name__)

class HECARApp(ctk.CTk):
    """Main Application Window."""
    
    def __init__(self):
        super().__init__()
        
        apply_theme()
        
        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.configure(fg_color=HECAR_THEME.bg_primary)
        
        # Shared State across screens
        self.shared_state = {
            "current_pdf_path": None,
            "ecg_metadata": {},
            "clinical_data": {},
            "arrhythmia_result": None,
            "risk_results": None,
            "report_path": None
        }
        
        # Grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Build Sidebar
        self._build_sidebar()
        
        # Main Content Area
        self.main_container = ctk.CTkFrame(self, fg_color=HECAR_THEME.bg_primary, corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Initialize Screens
        self.screens = {}
        self._init_screens()
        
        # Show default screen
        self.show_screen("Home")

    def _build_sidebarself(self):
        pass

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=HECAR_THEME.bg_sidebar)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1)
        
        # Logo / App Name
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="HECAR", 
            font=("Inter", 28, "bold"), text_color=HECAR_THEME.accent_primary
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        self.sub_label = ctk.CTkLabel(
            self.sidebar, text="AI Framework", 
            font=("Inter", 12), text_color=HECAR_THEME.text_secondary
        )
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 30))
        
        # Nav Buttons
        nav_items = [
            ("Home", "🏠 Home"),
            ("Upload", "📁 Upload ECG"),
            ("Preprocess", "⚙️ Preprocess"),
            ("Train", "🧠 Train Model"),
            ("Clinical", "🩺 Clinical Data"),
            ("Predict", "🔍 Predict"),
            ("Results", "📊 Results"),
            ("Report", "📄 Report")
        ]
        
        self.nav_buttons = {}
        for i, (name, label) in enumerate(nav_items):
            btn = ctk.CTkButton(
                self.sidebar, text=label, anchor="w",
                fg_color="transparent", text_color=HECAR_THEME.text_primary,
                hover_color=HECAR_THEME.bg_card, font=HECAR_THEME.font_body,
                command=lambda n=name: self.show_screen(n)
            )
            btn.grid(row=i+2, column=0, padx=10, pady=5, sticky="ew")
            self.nav_buttons[name] = btn
            
        # Status Label at bottom
        self.status_lbl = ctk.CTkLabel(self.sidebar, text="Ready", text_color=HECAR_THEME.success, font=HECAR_THEME.font_small)
        self.status_lbl.grid(row=10, column=0, padx=20, pady=20, sticky="sw")

    def _init_screens(self):
        # Pass self so screens can access shared_state and show_screen
        self.screens["Home"] = HomeScreen(self.main_container, self)
        self.screens["Upload"] = UploadScreen(self.main_container, self)
        self.screens["Preprocess"] = PreprocessScreen(self.main_container, self)
        self.screens["Train"] = TrainScreen(self.main_container, self)
        self.screens["Clinical"] = ClinicalScreen(self.main_container, self)
        self.screens["Predict"] = PredictScreen(self.main_container, self)
        self.screens["Results"] = ResultsScreen(self.main_container, self)
        self.screens["Report"] = ReportScreen(self.main_container, self)
        
        for screen in self.screens.values():
            screen.grid(row=0, column=0, sticky="nsew")

    def show_screen(self, screen_name: str):
        # Update sidebar highlighting
        for name, btn in self.nav_buttons.items():
            if name == screen_name:
                btn.configure(fg_color=HECAR_THEME.bg_card, text_color=HECAR_THEME.accent_primary)
            else:
                btn.configure(fg_color="transparent", text_color=HECAR_THEME.text_primary)
                
        # Raise screen
        screen = self.screens.get(screen_name)
        if screen:
            screen.tkraise()
            if hasattr(screen, 'on_show'):
                screen.on_show()

    def update_status(self, msg: str, color: str = None):
        if color is None:
            color = HECAR_THEME.text_secondary
        self.status_lbl.configure(text=msg, text_color=color)

    def run(self):
        self.mainloop()
