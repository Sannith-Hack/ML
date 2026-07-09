import customtkinter as ctk
from .theme import HECAR_THEME

class MetricCard(ctk.CTkFrame):
    def __init__(self, parent, label, value, unit='', color=HECAR_THEME.accent_primary, **kwargs):
        super().__init__(parent, fg_color=HECAR_THEME.bg_card, corner_radius=HECAR_THEME.corner_radius, **kwargs)
        
        self.val_label = ctk.CTkLabel(self, text=str(value), font=("Inter", 24, "bold"), text_color=color)
        self.val_label.pack(pady=(15, 5))
        
        text = f"{label} ({unit})" if unit else label
        self.title_label = ctk.CTkLabel(self, text=text, font=HECAR_THEME.font_small, text_color=HECAR_THEME.text_secondary)
        self.title_label.pack(pady=(0, 15))

    def set_value(self, value, color=None):
        self.val_label.configure(text=str(value))
        if color:
            self.val_label.configure(text_color=color)


class SectionHeader(ctk.CTkLabel):
    def __init__(self, parent, text, **kwargs):
        super().__init__(
            parent, 
            text=text, 
            font=HECAR_THEME.font_subheading, 
            text_color=HECAR_THEME.accent_primary,
            anchor="w",
            **kwargs
        )


class StyledButton(ctk.CTkButton):
    def __init__(self, parent, text, command=None, variant="primary", **kwargs):
        fg = HECAR_THEME.accent_primary if variant == "primary" else HECAR_THEME.bg_secondary
        hover = HECAR_THEME.accent_secondary if variant == "primary" else HECAR_THEME.border
        text_col = "#000000" if variant == "primary" else HECAR_THEME.text_primary
        
        if variant == "danger":
            fg = HECAR_THEME.danger
            hover = "#d73a49"
            text_col = "#ffffff"
            
        super().__init__(
            parent,
            text=text,
            command=command,
            fg_color=fg,
            hover_color=hover,
            text_color=text_col,
            font=HECAR_THEME.font_body,
            height=HECAR_THEME.button_height,
            corner_radius=HECAR_THEME.corner_radius,
            **kwargs
        )


class StatusBadge(ctk.CTkLabel):
    def __init__(self, parent, text, status="normal", **kwargs):
        colors = {
            "normal": HECAR_THEME.success,
            "warning": HECAR_THEME.warning,
            "danger": HECAR_THEME.danger,
            "info": HECAR_THEME.accent_secondary
        }
        color = colors.get(status, HECAR_THEME.text_primary)
        
        super().__init__(
            parent,
            text=f" {text} ",
            font=HECAR_THEME.font_small,
            fg_color=f"{color}33", # Add transparency for background
            text_color=color,
            corner_radius=4,
            **kwargs
        )


class ProgressCard(ctk.CTkFrame):
    def __init__(self, parent, label, **kwargs):
        super().__init__(parent, fg_color=HECAR_THEME.bg_card, corner_radius=HECAR_THEME.corner_radius, **kwargs)
        
        self.lbl = ctk.CTkLabel(self, text=label, font=HECAR_THEME.font_body, text_color=HECAR_THEME.text_primary)
        self.lbl.pack(anchor="w", padx=15, pady=(15, 5))
        
        self.progress = ctk.CTkProgressBar(self, progress_color=HECAR_THEME.accent_primary)
        self.progress.pack(fill="x", padx=15, pady=(0, 5))
        self.progress.set(0)
        
        self.pct_lbl = ctk.CTkLabel(self, text="0%", font=HECAR_THEME.font_small, text_color=HECAR_THEME.text_secondary)
        self.pct_lbl.pack(anchor="e", padx=15, pady=(0, 15))

    def set_progress(self, value):
        self.progress.set(value)
        self.pct_lbl.configure(text=f"{int(value*100)}%")


class ECGMetricRow(ctk.CTkFrame):
    def __init__(self, parent, name, val, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self, text=name, font=HECAR_THEME.font_body, text_color=HECAR_THEME.text_secondary).grid(row=0, column=0, sticky="w")
        
        self.val_lbl = ctk.CTkLabel(self, text=str(val), font=HECAR_THEME.font_body, text_color=HECAR_THEME.text_primary)
        self.val_lbl.grid(row=0, column=1, sticky="e")

    def set_value(self, val):
        self.val_lbl.configure(text=str(val))


class ScrollableResultFrame(ctk.CTkScrollableFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=HECAR_THEME.bg_secondary,
            corner_radius=HECAR_THEME.corner_radius,
            **kwargs
        )
