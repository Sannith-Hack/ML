"""HECAR Framework — Test: Feature Fusion"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np


class TestClinicalFeatures(unittest.TestCase):

    def setUp(self):
        from modules.fusion.clinical_features import ClinicalFeatures
        self.cf = ClinicalFeatures()
        self.valid_data = {
            "age": 45, "gender": "Male",
            "bp_systolic": 130, "bp_diastolic": 85,
            "hba1c": 6.5, "cholesterol": 210, "bmi": 27.5,
            "smoking": True, "alcohol": False,
            "physical_activity": "Moderate",
            "medical_history": ["diabetes"],
            "family_history": True,
        }

    def test_valid_data(self):
        valid, errors = self.cf.validate(self.valid_data)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_age(self):
        data = {**self.valid_data, "age": 200}
        valid, errors = self.cf.validate(data)
        self.assertFalse(valid)

    def test_feature_vector_shape(self):
        vec = self.cf.to_feature_vector(self.valid_data)
        self.assertEqual(vec.shape, (15,))

    def test_feature_names_count(self):
        names = self.cf.get_feature_names()
        self.assertEqual(len(names), 15)


class TestFeatureFusion(unittest.TestCase):

    def setUp(self):
        from modules.fusion.feature_fusion import FeatureFusion
        self.ff = FeatureFusion()

    def test_fuse_for_risk_shape(self):
        ecg_features = np.random.rand(14)
        clinical_vector = np.random.rand(15)
        result = self.ff.fuse_for_risk(
            ecg_features=ecg_features,
            arrhythmia_class=1,
            arrhythmia_confidence=0.85,
            clinical_vector=clinical_vector,
        )
        # 14 + 1 + 1 + 15 = 31
        self.assertEqual(result.shape, (31,))


if __name__ == "__main__":
    unittest.main()
