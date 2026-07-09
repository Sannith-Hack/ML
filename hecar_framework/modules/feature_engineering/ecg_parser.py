import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ECGParser:
    """Parses, validates, and engineers features from extracted ECG text metadata."""
    
    def __init__(self):
        pass

    def parse(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and engineers new features for a single record."""
        parsed = metadata.copy()
        
        # 1. Ensure numeric types (handle None)
        numeric_fields = ["AR_bpm", "VR_bpm", "QRSD_ms", "QT_ms", "QTcB_ms", "PRI_ms", "P_axis", "R_axis", "T_axis", "age"]
        for field in numeric_fields:
            if parsed.get(field) is None:
                # Default to NaN/None for now, imputer handles later
                parsed[field] = None
        
        # 2. Gender encoding (0=Female, 1=Male)
        gender = str(parsed.get("gender", "")).strip().lower()
        if gender.startswith('m'):
            parsed["gender_encoded"] = 1
        elif gender.startswith('f'):
            parsed["gender_encoded"] = 0
        else:
            parsed["gender_encoded"] = None
            
        # 3. Derived Boolean Features
        # Wide QRS (> 100ms)
        qrsd = parsed.get("QRSD_ms")
        parsed["qrs_wide"] = bool(qrsd and qrsd > 100)
        
        # Long PR (> 200ms)
        pri = parsed.get("PRI_ms")
        parsed["pr_long"] = bool(pri and pri > 200)
        
        # Prolonged QTcB (gender specific)
        qtcb = parsed.get("QTcB_ms")
        gen_enc = parsed.get("gender_encoded")
        
        if qtcb is not None and gen_enc is not None:
            if gen_enc == 0 and qtcb > 460: # Female
                parsed["qtc_prolonged"] = True
            elif gen_enc == 1 and qtcb > 440: # Male
                parsed["qtc_prolonged"] = True
            else:
                parsed["qtc_prolonged"] = False
        else:
            parsed["qtc_prolonged"] = False
            
        # 4. Categorical derivations
        ar = parsed.get("AR_bpm")
        if ar:
            if ar > 100:
                parsed["heart_rate_category"] = "tachycardia"
            elif ar < 60:
                parsed["heart_rate_category"] = "bradycardia"
            else:
                parsed["heart_rate_category"] = "normal"
        else:
            parsed["heart_rate_category"] = "unknown"
            
        age = parsed.get("age")
        if age:
            if age < 40:
                parsed["age_group"] = "young"
            elif age <= 60:
                parsed["age_group"] = "middle"
            else:
                parsed["age_group"] = "senior"
        else:
            parsed["age_group"] = "unknown"
            
        return parsed

    def parse_batch(self, metadata_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Parses a list of records and returns a DataFrame."""
        parsed_list = [self.parse(meta) for meta in metadata_list]
        df = pd.DataFrame(parsed_list)
        return df
