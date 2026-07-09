import numpy as np
from typing import Dict, Any
from scipy.stats import skew, kurtosis

from config import ECG_TEXT_FEATURES

class FeatureExtractor:
    """Extracts features from 1D signals or text-based ECG parameters."""
    
    def __init__(self):
        pass

    def extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        if len(signal) == 0:
            return {}
        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "rms": float(np.sqrt(np.mean(signal**2))),
            "skewness": float(skew(signal)),
            "kurtosis": float(kurtosis(signal))
        }

    def extract_frequency_domain(self, signal: np.ndarray, fs: float = 360.0) -> Dict[str, float]:
        if len(signal) == 0:
            return {}
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        dom_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
        total_power = np.sum(fft_vals**2)
        power_ratio = (fft_vals[0]**2 / total_power) if total_power > 0 else 0
        
        # Spectral entropy
        psd = fft_vals**2
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
        
        return {
            "dominant_frequency": float(dom_freq),
            "power_ratio": float(power_ratio),
            "spectral_entropy": float(spectral_entropy)
        }

    def extract_from_ecg_params(self, ecg_dict: Dict[str, Any]) -> np.ndarray:
        """Converts structured textual ECG parameters into a feature vector."""
        vec = []
        for feat in ECG_TEXT_FEATURES:
            val = ecg_dict.get(feat, 0.0)
            if val is None or pd.isna(val) if 'pd' in globals() and hasattr(val, 'isna') else val is None:
                val = 0.0
            vec.append(float(val))
        return np.array(vec)
