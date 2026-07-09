import logging
import numpy as np
from typing import Optional
try:
    from pdf2image import convert_from_path
    import cv2
except ImportError:
    pass

logger = logging.getLogger(__name__)

class WaveformExtractor:
    """Extracts waveform region images from PDFs using OpenCV."""
    
    def __init__(self):
        pass

    def extract_image(self, pdf_path: str) -> Optional[np.ndarray]:
        try:
            pages = convert_from_path(pdf_path, dpi=200)
            if not pages:
                return None
            page = pages[0]
            # Convert PIL image to OpenCV format
            open_cv_image = np.array(page) 
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            return open_cv_image
        except Exception as e:
            logger.exception(f"Failed to convert PDF to image: {e}")
            return None

    def detect_ecg_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the grid area containing the ECG waves."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Thresholding to find dark lines/grid
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Find the largest bounding box (assuming it's the ECG grid)
            max_area = 0
            best_rect = None
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > max_area:
                    max_area = area
                    best_rect = (x, y, w, h)
                    
            if best_rect:
                x, y, w, h = best_rect
                # Add some padding
                pad = 10
                y1 = max(0, y - pad)
                y2 = min(image.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(image.shape[1], x + w + pad)
                return image[y1:y2, x1:x2]
            return None
        except Exception as e:
            logger.exception(f"Failed to detect ECG region: {e}")
            return None

    def extract(self, pdf_path: str) -> Optional[np.ndarray]:
        """Full pipeline: pdf to image -> detect region -> return cropped image."""
        img = self.extract_image(pdf_path)
        if img is not None:
            return self.detect_ecg_region(img)
        return None
