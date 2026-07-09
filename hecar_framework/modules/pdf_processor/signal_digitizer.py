import logging
import numpy as np
from typing import Optional
try:
    import cv2
    from scipy.interpolate import interp1d
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SignalDigitizer:
    """Digitizes an ECG waveform image into a 1D numpy array."""
    
    def __init__(self):
        pass

    def digitize(self, ecg_image: np.ndarray, target_length: int = 1000) -> Optional[np.ndarray]:
        """Convert a cropped ECG grid image to a 1D signal array."""
        try:
            gray = cv2.cvtColor(ecg_image, cv2.COLOR_BGR2GRAY)
            # Threshold to isolate the dark signal line
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            height, width = thresh.shape
            signal_raw = []
            
            # For each column, find the top-most black pixel (white in inverted thresh)
            for x in range(width):
                col = thresh[:, x]
                # Find indices of non-zero (white) pixels
                y_indices = np.where(col > 0)[0]
                if len(y_indices) > 0:
                    # Take the median or mean y position of the line thickness
                    y_pos = np.mean(y_indices)
                    # Invert Y axis so higher values go up
                    signal_raw.append(height - y_pos)
                else:
                    # If no pixel found, repeat the last value or use 0
                    if signal_raw:
                        signal_raw.append(signal_raw[-1])
                    else:
                        signal_raw.append(height / 2.0)
                        
            signal_arr = np.array(signal_raw)
            
            # Normalize to [-1, 1]
            sig_min, sig_max = np.min(signal_arr), np.max(signal_arr)
            if sig_max > sig_min:
                signal_arr = 2.0 * ((signal_arr - sig_min) / (sig_max - sig_min)) - 1.0
            else:
                signal_arr = np.zeros_like(signal_arr)
                
            # Interpolate to target length
            x_old = np.linspace(0, 1, len(signal_arr))
            f = interp1d(x_old, signal_arr, kind='cubic')
            x_new = np.linspace(0, 1, target_length)
            signal_target = f(x_new)
            
            return signal_target
        except Exception as e:
            logger.exception(f"Error digitizing signal: {e}")
            return None
