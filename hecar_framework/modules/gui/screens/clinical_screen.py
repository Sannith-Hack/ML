import customtkinter as ctk
from modules.gui.theme import HECAR_THEME
from modules.gui.components import StyledButton, SectionHeader, ScrollableResultFrame
from modules.fusion.clinical_features import ClinicalFeatures

class ClinicalScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        
        SectionHeader(self, "Patient Clinical Data").pack(anchor="w", pady=(0, 20))
        
        scroll = ScrollableResultFrame(self)
        scroll.pack(fill="both", expand=True)
        
        # Form grid
        form = ctk.CTkFrame(scroll, fg_color="transparent")
        form.pack(fill="x", pady=10, padx=10)
        
        self.entries = {}
        
        # Demographics
        ctk.CTkLabel(form, text="Demographics", font=HECAR_THEME.font_subheading).grid(row=0, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        self.add_field(form, 1, "age", "Age:")
        self.add_combo(form, 2, "gender", "Gender:", ["Female", "Male"])
        
        # Vitals
        ctk.CTkLabel(form, text="Vitals & Labs", font=HECAR_THEME.font_subheading).grid(row=3, column=0, columnspan=2, sticky="w", pady=(20, 5))
        self.add_field(form, 4, "bp_systolic", "Systolic BP:")
        self.add_field(form, 5, "bp_diastolic", "Diastolic BP:")
        self.add_field(form, 6, "bmi", "BMI:")
        self.add_field(form, 7, "hba1c", "HbA1c (%):")
        self.add_field(form, 8, "cholesterol", "Cholesterol (mg/dL):")
        
        # Lifestyle
        ctk.CTkLabel(form, text="Lifestyle", font=HECAR_THEME.font_subheading).grid(row=9, column=0, columnspan=2, sticky="w", pady=(20, 5))
        self.add_switch(form, 10, "smoking", "Smoker")
        self.add_switch(form, 11, "alcohol", "Alcohol Consumption")
        self.add_combo(form, 12, "physical_activity", "Physical Activity:", ["Low", "Moderate", "High"])
        
        # History
        ctk.CTkLabel(form, text="Medical History", font=HECAR_THEME.font_subheading).grid(row=13, column=0, columnspan=2, sticky="w", pady=(20, 5))
        self.add_switch(form, 14, "diabetes", "Diabetes")
        self.add_switch(form, 15, "hypertension", "Hypertension")
        self.add_switch(form, 16, "heart_disease", "Heart Disease History")
        self.add_switch(form, 17, "stroke", "Prior Stroke")
        self.add_switch(form, 18, "family_history", "Family History of CVD")
        
        # Actions
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", pady=20)
        StyledButton(actions, "Save & Proceed", command=self.save_data).pack(side="left", padx=(0, 20))
        StyledButton(actions, "Fill Default (Demo)", variant="secondary", command=self.fill_demo).pack(side="left")

    def add_field(self, parent, row, key, label):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        entry = ctk.CTkEntry(parent, width=200)
        entry.grid(row=row, column=1, sticky="w", padx=10, pady=5)
        self.entries[key] = entry
        
    def add_combo(self, parent, row, key, label, vals):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        combo = ctk.CTkComboBox(parent, values=vals, width=200)
        combo.grid(row=row, column=1, sticky="w", padx=10, pady=5)
        self.entries[key] = combo
        
    def add_switch(self, parent, row, key, label):
        switch = ctk.CTkSwitch(parent, text=label, progress_color=HECAR_THEME.accent_primary)
        switch.grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        self.entries[key] = switch

    def on_show(self):
        # Auto-fill age and gender from parsed PDF metadata if available
        meta = self.app.shared_state.get("ecg_metadata", {})
        if "age" in meta and meta["age"]:
            self.entries["age"].delete(0, 'end')
            self.entries["age"].insert(0, str(meta["age"]))
        if "gender" in meta and meta["gender"]:
            self.entries["gender"].set(str(meta["gender"]).capitalize())

    def fill_demo(self):
        demo = {
            "age": "67", "bp_systolic": "135", "bp_diastolic": "85",
            "bmi": "26.5", "hba1c": "6.1", "cholesterol": "190"
        }
        for k, v in demo.items():
            self.entries[k].delete(0, 'end')
            self.entries[k].insert(0, v)
        self.entries["physical_activity"].set("Moderate")
        self.entries["hypertension"].select()

    def save_data(self):
        data = {}
        for k, w in self.entries.items():
            if isinstance(w, ctk.CTkEntry):
                try:
                    data[k] = float(w.get())
                except:
                    data[k] = None
            elif isinstance(w, ctk.CTkComboBox):
                data[k] = w.get()
            elif isinstance(w, ctk.CTkSwitch):
                data[k] = w.get() == 1
                
        # Move history booleans into array
        hist = []
        for h in ["diabetes", "hypertension", "heart_disease", "stroke"]:
            if data.get(h):
                hist.append(h)
        data["medical_history"] = hist
        
        self.app.shared_state["clinical_data"] = data
        self.app.update_status("Clinical Data Saved", HECAR_THEME.success)
        self.app.show_screen("Predict")
