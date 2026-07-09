import re
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class OCRExtractor:
    """Parses raw text extracted from ECG PDFs to pull structured fields."""
    
    def __init__(self):
        pass

    def extract_metadata(self, raw_text: str) -> Dict[str, Any]:
        """Extracts patient metadata and ECG metrics from the raw text."""
        result: Dict[str, Any] = {
            "hospital_name": None,
            "patient_name": None,
            "age": None,
            "gender": None,
            "patient_id": None,
            "acquired_at": None,
            "AR_bpm": None,
            "VR_bpm": None,
            "QRSD_ms": None,
            "QT_ms": None,
            "QTcB_ms": None,
            "PRI_ms": None,
            "P_axis": None,
            "R_axis": None,
            "T_axis": None
        }
        
        if not raw_text:
            return result
            
        lines = raw_text.split('\n')
        if lines:
            result["hospital_name"] = lines[0].strip()
            
        for line in lines:
            line = line.strip()
            
            if 'Patient Name:' in line:
                result["patient_name"] = line.split('Patient Name:')[-1].strip()
                
            if 'Age / Gender:' in line:
                age_gen_str = line.split('Age / Gender:')[-1].split('Acquired')[0].strip()
                if '/' in age_gen_str:
                    age_str, gender_str = age_gen_str.split('/')
                    try:
                        result["age"] = int(age_str.strip())
                        result["gender"] = gender_str.strip()
                    except ValueError:
                        pass
                        
            if 'Patient ID:' in line:
                result["patient_id"] = line.split('Patient ID:')[-1].split('Reported')[0].strip()
                
            if 'Acquired At:' in line:
                result["acquired_at"] = line.split('Acquired At:')[-1].strip()
                
            # Regex for metrics
            ar_match = re.search(r'AR:\s*(\d+)bpm', line)
            if ar_match: result["AR_bpm"] = int(ar_match.group(1))
            
            vr_match = re.search(r'VR:\s*(\d+)bpm', line)
            if vr_match: result["VR_bpm"] = int(vr_match.group(1))
            
            qrsd_match = re.search(r'QRSD:\s*(\d+)ms', line)
            if qrsd_match: result["QRSD_ms"] = int(qrsd_match.group(1))
            
            qt_match = re.search(r'QT:\s*(\d+)ms', line)
            if qt_match: result["QT_ms"] = int(qt_match.group(1))
            
            qtcb_match = re.search(r'QTcB:\s*(\d+)ms', line)
            if qtcb_match: result["QTcB_ms"] = int(qtcb_match.group(1))
            
            pri_match = re.search(r'PRI:\s*(\d+)ms', line)
            if pri_match: result["PRI_ms"] = int(pri_match.group(1))
            
            prt_match = re.search(r'P-R-T:\s*(-?\d+)\??\s*(-?\d+)\??\s*(-?\d+)\??', line)
            if prt_match:
                result["P_axis"] = int(prt_match.group(1))
                result["R_axis"] = int(prt_match.group(2))
                result["T_axis"] = int(prt_match.group(3))
                
        return result
