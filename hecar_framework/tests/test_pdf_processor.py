"""HECAR Framework — Test: PDF Processor"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from modules.pdf_processor.ocr_extractor import OCRExtractor


class TestOCRExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = OCRExtractor()
        self.sample_text = (
            "Bhageerath Cardiac Care Center Healing Hearts-1\n"
            "Patient Name: Shyalamadevi\n"
            "Age / Gender: 67/Female Acquired At: 9 Jun 2026 7:09 PM\n"
            "Patient ID: 1 Reported At: N/A\n"
            "AR: 76bpm VR: 76bpm QRSD: 80ms QT: 380ms QTcB: 428ms PRI: 144ms P-R-T: 25? 51? 63?\n"
        )

    def test_extract_patient_name(self):
        result = self.extractor.extract_metadata(self.sample_text)
        self.assertEqual(result["patient_name"], "Shyalamadevi")

    def test_extract_age_gender(self):
        result = self.extractor.extract_metadata(self.sample_text)
        self.assertEqual(result["age"], 67)
        self.assertEqual(result["gender"], "Female")

    def test_extract_ecg_metrics(self):
        result = self.extractor.extract_metadata(self.sample_text)
        self.assertEqual(result["AR_bpm"], 76)
        self.assertEqual(result["VR_bpm"], 76)
        self.assertEqual(result["QRSD_ms"], 80)
        self.assertEqual(result["QT_ms"], 380)
        self.assertEqual(result["QTcB_ms"], 428)
        self.assertEqual(result["PRI_ms"], 144)

    def test_extract_axes(self):
        result = self.extractor.extract_metadata(self.sample_text)
        self.assertEqual(result["P_axis"], 25)
        self.assertEqual(result["R_axis"], 51)
        self.assertEqual(result["T_axis"], 63)

    def test_missing_fields_return_none(self):
        result = self.extractor.extract_metadata("Only partial text here.")
        self.assertIsNone(result["patient_name"])
        self.assertIsNone(result["AR_bpm"])


if __name__ == "__main__":
    unittest.main()
