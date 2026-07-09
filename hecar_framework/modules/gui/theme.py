import customtkinter as ctk
from config import THEME

def apply_theme():
    """Sets up the CustomTkinter appearance and custom color theme."""
    ctk.set_appearance_mode("dark")
    
    # We can't directly override all internal colors easily without a JSON theme file,
    # but we can set default widget colors to our theme.
    
    # We will mainly rely on passing our THEME dictionary to components
    # when creating them.
    pass

class HECAR_THEME:
    """Wrapper for the theme dictionary for easy access in GUI components."""
    bg_primary = THEME["bg_primary"]
    bg_secondary = THEME["bg_secondary"]
    bg_card = THEME["bg_card"]
    bg_sidebar = THEME["bg_sidebar"]
    
    accent_primary = THEME["accent_primary"]
    accent_secondary = THEME["accent_secondary"]
    
    text_primary = THEME["text_primary"]
    text_secondary = THEME["text_secondary"]
    text_muted = THEME["text_muted"]
    
    success = THEME["success"]
    warning = THEME["warning"]
    danger = THEME["danger"]
    
    border = THEME["border"]
    
    # Fonts
    font_heading = ("Inter", 24, "bold")
    font_subheading = ("Inter", 16, "bold")
    font_body = ("Inter", 13)
    font_small = ("Inter", 11)
    
    # Styles
    corner_radius = 12
    border_width = 1
    button_height = 40
