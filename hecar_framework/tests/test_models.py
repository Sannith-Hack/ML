"""HECAR Framework — Test: Models"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np


class TestCNNBiLSTM(unittest.TestCase):

    def test_model_builds(self):
        from modules.models.cnn_bilstm import CNNBiLSTMModel
        builder = CNNBiLSTMModel()
        model = builder.build()
        self.assertIsNotNone(model)

    def test_model_output_shape(self):
        from modules.models.cnn_bilstm import CNNBiLSTMModel
        builder = CNNBiLSTMModel()
        model = builder.build()
        model = builder.compile(model)
        # Input: (batch=4, timesteps=1, features=14)
        X = np.random.rand(4, 1, 14).astype(np.float32)
        output = model.predict(X, verbose=0)
        self.assertEqual(output.shape, (4, 7))

    def test_predict_returns_class_and_probs(self):
        from modules.models.cnn_bilstm import CNNBiLSTMModel
        builder = CNNBiLSTMModel()
        model = builder.build()
        model = builder.compile(model)
        X = np.random.rand(3, 1, 14).astype(np.float32)
        classes, probs = builder.predict(model, X)
        self.assertEqual(len(classes), 3)
        self.assertEqual(probs.shape, (3, 7))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))


class TestModelManager(unittest.TestCase):

    def test_model_exists_false(self):
        from modules.models.model_manager import ModelManager
        mgr = ModelManager()
        self.assertFalse(mgr.model_exists("nonexistent_xyz_model"))


if __name__ == "__main__":
    unittest.main()
