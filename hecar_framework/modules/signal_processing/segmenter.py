import numpy as np
from scipy.signal import find_peaks

class Segmenter:
    """Segments continuous ECG signals into individual heartbeat windows."""
    
    def __init__(self):
        pass

    def detect_qrs_peaks(self, signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
        """Detect R-peaks using basic find_peaks (simplified Pan-Tompkins step)."""
        # Minimum distance between peaks: ~200ms (max 300 bpm)
        distance = int(0.2 * fs)
        # Height threshold: half of max signal
        height = np.max(signal) * 0.5
        
        peaks, _ = find_peaks(signal, distance=distance, height=height)
        return peaks

    def segment_beats(self, signal: np.ndarray, peaks: np.ndarray, window_ms: int = 600, fs: float = 360.0) -> np.ndarray:
        """Extract a window around each peak."""
        window_samples = int((window_ms / 1000.0) * fs)
        half_window = window_samples // 2
        
        beats = []
        for peak in peaks:
            start = peak - half_window
            end = peak + half_window
            if start >= 0 and end <= len(signal):
                beats.append(signal[start:end])
                
        if not beats:
            return np.array([])
        return np.array(beats)

    def segment(self, signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
        """Full segmentation pipeline."""
        peaks = self.detect_qrs_peaks(signal, fs)
        return self.segment_beats(signal, peaks, fs=fs)
