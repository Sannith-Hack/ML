import logging
import numpy as np
from typing import Dict, Tuple, List, Any

from config import CLINICAL_FEATURES

logger = logging.getLogger(__name__)

class ClinicalFeatures:
    """Handles clinical data validation and feature encoding."""
    
    def __init__(self):
        pass

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # Age
        age = data.get("age")
        if age is None or not (18 <= age <= 120):
            errors.append("Age must be between 18 and 120.")
            
        # BP
        sys = data.get("bp_systolic")
        if sys is None or not (70 <= sys <= 250):
            errors.append("Systolic BP must be between 70 and 250.")
            
        dia = data.get("bp_diastolic")
        if dia is None or not (40 <= dia <= 150):
            errors.append("Diastolic BP must be between 40 and 150.")
            
        # Labs & Vitals
        hba1c = data.get("hba1c")
        if hba1c is None or not (3.0 <= hba1c <= 15.0):
            errors.append("HbA1c must be between 3.0 and 15.0.")
            
        chol = data.get("cholesterol")
        if chol is None or not (100 <= chol <= 400):
            errors.append("Cholesterol must be between 100 and 400.")
            
        bmi = data.get("bmi")
        if bmi is None or not (10.0 <= bmi <= 60.0):
            errors.append("BMI must be between 10.0 and 60.0.")
            
        return len(errors) == 0, errors

    def to_feature_vector(self, data: Dict[str, Any]) -> np.ndarray:
        vec = []
        
        # Numeric
        vec.append(float(data.get("age", 50)))
        
        # Gender encoded
        gender = str(data.get("gender", "")).lower()
        vec.append(1.0 if gender.startswith('m') else 0.0)
        
        vec.append(float(data.get("bp_systolic", 120)))
        vec.append(float(data.get("bp_diastolic", 80)))
        vec.append(float(data.get("hba1c", 5.5)))
        vec.append(float(data.get("cholesterol", 180)))
        vec.append(float(data.get("bmi", 24.0)))
        
        # Boolean
        vec.append(1.0 if data.get("smoking") else 0.0)
        vec.append(1.0 if data.get("alcohol") else 0.0)
        
        # Categorical
        pa = str(data.get("physical_activity", "")).lower()
        if pa == "high": pa_val = 2.0
        elif pa == "moderate": pa_val = 1.0
        else: pa_val = 0.0
        vec.append(pa_val)
        
        # Medical History Array
        history = data.get("medical_history", [])
        if isinstance(history, str):
            history = [history]
        history_lower = [str(h).lower() for h in history]
        
        vec.append(1.0 if "diabetes" in history_lower else 0.0)
        vec.append(1.0 if "hypertension" in history_lower else 0.0)
        vec.append(1.0 if "heart_disease" in history_lower else 0.0)
        vec.append(1.0 if "stroke" in history_lower else 0.0)
        
        # Family history
        vec.append(1.0 if data.get("family_history") else 0.0)
        
        return np.array(vec)

    def get_feature_names(self) -> List[str]:
        return CLINICAL_FEATURES
