import numpy as np

class Normalizer:
    """Normalizes 1D signals."""
    
    def __init__(self):
        pass

    def l2_normalize(self, signal: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(signal)
        if norm == 0:
            return signal
        return signal / norm

    def minmax_normalize(self, signal: np.ndarray) -> np.ndarray:
        s_min = np.min(signal)
        s_max = np.max(signal)
        if s_max == s_min:
            return signal - s_min
        return (signal - s_min) / (s_max - s_min)

    def zscore_normalize(self, signal: np.ndarray) -> np.ndarray:
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std

    def normalize(self, signal: np.ndarray, method: str = 'minmax') -> np.ndarray:
        if method == 'minmax':
            return self.minmax_normalize(signal)
        elif method == 'l2':
            return self.l2_normalize(signal)
        elif method == 'zscore':
            return self.zscore_normalize(signal)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
