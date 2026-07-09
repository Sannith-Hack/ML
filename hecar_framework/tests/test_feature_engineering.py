"""HECAR Framework — Test: Feature Engineering"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
from modules.feature_engineering.arrhythmia_labeler import ArrhythmiaLabeler
from modules.feature_engineering.ecg_parser import ECGParser


class TestArrhythmiaLabeler(unittest.TestCase):

    def setUp(self):
        self.labeler = ArrhythmiaLabeler()

    def _make_record(self, **kwargs):
        base = {
            "AR_bpm": 80, "VR_bpm": 80, "QRSD_ms": 80,
            "QT_ms": 380, "QTcB_ms": 430, "PRI_ms": 150,
            "P_axis": 45, "R_axis": 45, "T_axis": 45,
            "age": 40, "gender": "Male", "gender_encoded": 1,
            "qrs_wide": False, "pr_long": False, "qtc_prolonged": False,
        }
        base.update(kwargs)
        return base

    def test_normal(self):
        record = self._make_record()
        label, _, _ = self.labeler.label_with_confidence(record)
        self.assertEqual(label, 0)

    def test_tachycardia(self):
        record = self._make_record(AR_bpm=119)
        label, _, _ = self.labeler.label_with_confidence(record)
        self.assertEqual(label, 1)

    def test_bradycardia(self):
        record = self._make_record(AR_bpm=50)
        label, _, _ = self.labeler.label_with_confidence(record)
        self.assertEqual(label, 2)

    def test_bundle_branch_block(self):
        record = self._make_record(QRSD_ms=110, qrs_wide=True)
        label, _, _ = self.labeler.label_with_confidence(record)
        self.assertEqual(label, 3)

    def test_av_block(self):
        record = self._make_record(PRI_ms=210, pr_long=True)
        label, _, _ = self.labeler.label_with_confidence(record)
        self.assertEqual(label, 5)


class TestECGParser(unittest.TestCase):

    def setUp(self):
        self.parser = ECGParser()

    def test_parse_basic(self):
        meta = {
            "AR_bpm": 76, "VR_bpm": 76, "QRSD_ms": 80,
            "QT_ms": 380, "QTcB_ms": 428, "PRI_ms": 144,
            "P_axis": 25, "R_axis": 51, "T_axis": 63,
            "age": 67, "gender": "Female",
        }
        result = self.parser.parse(meta)
        self.assertEqual(result["gender_encoded"], 0)
        self.assertFalse(result["qrs_wide"])
        self.assertFalse(result["pr_long"])
        self.assertEqual(result["age_group"], "senior")

    def test_feature_vector_shape(self):
        from modules.signal_processing.feature_extractor import FeatureExtractor
        meta = {
            "AR_bpm": 76, "VR_bpm": 76, "QRSD_ms": 80,
            "QT_ms": 380, "QTcB_ms": 428, "PRI_ms": 144,
            "P_axis": 25, "R_axis": 51, "T_axis": 63,
            "age": 67, "gender": "Female",
        }
        parsed = self.parser.parse(meta)
        fe = FeatureExtractor()
        vec = fe.extract_from_ecg_params(parsed)
        self.assertEqual(vec.shape, (14,))


if __name__ == "__main__":
    unittest.main()
