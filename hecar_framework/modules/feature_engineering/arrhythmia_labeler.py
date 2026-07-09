import logging
import pandas as pd
from typing import Dict, Tuple, Any

from config import ARRHYTHMIA_CLASSES, ARRHYTHMIA_RULES

logger = logging.getLogger(__name__)

class ArrhythmiaLabeler:
    """Labels ECG records using clinical rules based on extracted parameters."""
    
    def __init__(self):
        self.rules = ARRHYTHMIA_RULES
        self.classes = ARRHYTHMIA_CLASSES

    def label(self, parsed_record: Dict[str, Any]) -> int:
        """Returns the class index (0-6) based on clinical rules priority."""
        idx, _, _ = self.label_with_confidence(parsed_record)
        return idx

    def label_with_confidence(self, parsed_record: Dict[str, Any]) -> Tuple[int, float, str]:
        """Returns class index, confidence score, and reason string."""
        
        qrs_wide = parsed_record.get("qrs_wide", False)
        pr_long = parsed_record.get("pr_long", False)
        qtc_prolonged = parsed_record.get("qtc_prolonged", False)
        
        ar = parsed_record.get("AR_bpm")
        
        p_axis = parsed_record.get("P_axis")
        r_axis = parsed_record.get("R_axis")
        
        # 1. Bundle Branch Block (highest priority for structural issues)
        if qrs_wide:
            return 3, 0.95, "QRS duration > 100ms indicates Bundle Branch Block."
            
        # 2. 1st Degree AV Block
        if pr_long:
            return 5, 0.95, "PR interval > 200ms indicates 1st Degree AV Block."
            
        # 3. Long QT Syndrome
        if qtc_prolonged:
            return 4, 0.90, "Prolonged QTc interval for gender indicates Long QT Syndrome."
            
        # 4. Sinus Tachycardia
        if ar is not None and ar > self.rules["AR_HIGH_THRESHOLD"]:
            return 1, 0.95, f"Atrial rate {ar} > {self.rules['AR_HIGH_THRESHOLD']} bpm indicates Sinus Tachycardia."
            
        # 5. Sinus Bradycardia
        if ar is not None and ar < self.rules["AR_LOW_THRESHOLD"]:
            return 2, 0.95, f"Atrial rate {ar} < {self.rules['AR_LOW_THRESHOLD']} bpm indicates Sinus Bradycardia."
            
        # 6. Abnormal Axis
        axis_abnormal = False
        if p_axis is not None and (p_axis < self.rules["P_AXIS_MIN"] or p_axis > self.rules["P_AXIS_MAX"]):
            axis_abnormal = True
        if r_axis is not None and (r_axis < self.rules["R_AXIS_MIN"] or r_axis > self.rules["R_AXIS_MAX"]):
            axis_abnormal = True
            
        if axis_abnormal:
            return 6, 0.85, "Electrical axis falls outside normal reference range."
            
        # 7. Normal Sinus Rhythm
        return 0, 0.99, "All ECG parameters fall within normal reference ranges."

    def label_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies labeling rules to a DataFrame of parsed records."""
        labels = []
        confidences = []
        reasons = []
        
        for _, row in df.iterrows():
            idx, conf, reason = self.label_with_confidence(row.to_dict())
            labels.append(idx)
            confidences.append(conf)
            reasons.append(reason)
            
        df["arrhythmia_class"] = labels
        df["class_name"] = df["arrhythmia_class"].map(self.classes)
        df["confidence"] = confidences
        df["diagnosis_reason"] = reasons
        
        return df
