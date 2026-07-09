import numpy as np
from scipy.signal import butter, filtfilt, medfilt

class NoiseFilter:
    """Filters noise and baseline wander from 1D ECG signals."""
    
    def __init__(self):
        pass

    def bandpass_filter(self, signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = 360.0, order: int = 4) -> np.ndarray:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered

    def remove_baseline_wander(self, signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
        # Median filter approach: filter out baseline wander using two median filters
        # Window size ~0.2s for first, ~0.6s for second
        w1 = int(0.2 * fs)
        if w1 % 2 == 0: w1 += 1
        w2 = int(0.6 * fs)
        if w2 % 2 == 0: w2 += 1
        
        baseline = medfilt(signal, kernel_size=w1)
        baseline = medfilt(baseline, kernel_size=w2)
        
        return signal - baseline

    def filter(self, signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
        filtered = self.remove_baseline_wander(signal, fs)
        filtered = self.bandpass_filter(filtered, fs=fs)
        return filtered
