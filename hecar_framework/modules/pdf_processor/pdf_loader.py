import pdfplumber
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class PDFLoader:
    """Handles loading and extracting raw text from ECG PDF reports."""
    
    def __init__(self):
        pass

    def validate(self, pdf_path: str) -> bool:
        """Check if the file is a valid PDF."""
        path = Path(pdf_path)
        if not path.exists():
            logger.error(f"File not found: {pdf_path}")
            return False
        if path.suffix.lower() != '.pdf':
            logger.error(f"Not a PDF file: {pdf_path}")
            return False
        return True

    def load(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract text and metadata from a PDF file."""
        if not self.validate(pdf_path):
            return None
            
        try:
            with pdfplumber.open(pdf_path) as pdf:
                raw_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
                        
                return {
                    "file_path": pdf_path,
                    "filename": Path(pdf_path).name,
                    "raw_text": raw_text,
                    "num_pages": len(pdf.pages),
                    "metadata": pdf.metadata
                }
        except Exception as e:
            logger.exception(f"Error loading PDF {pdf_path}: {e}")
            return None

    def batch_load(self, folder_path: str) -> List[Dict[str, Any]]:
        """Load all PDFs from a given directory."""
        path = Path(folder_path)
        if not path.is_dir():
            logger.error(f"Directory not found: {folder_path}")
            return []
            
        results = []
        for pdf_file in path.glob('*.pdf'):
            data = self.load(str(pdf_file))
            if data:
                results.append(data)
                
        logger.info(f"Loaded {len(results)} PDFs from {folder_path}")
        return results
