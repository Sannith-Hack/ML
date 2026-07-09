import numpy as np
from typing import List

class FeatureFusion:
    """Fuses ECG features and Clinical features for downstream risk models."""
    
    def __init__(self):
        pass

    def fuse(self, ecg_features: np.ndarray, ecg_embedding: np.ndarray, arrhythmia_probs: np.ndarray, clinical_vector: np.ndarray) -> np.ndarray:
        """Concatenate all into a single vector."""
        return np.concatenate([
            ecg_features.flatten(),
            ecg_embedding.flatten(),
            arrhythmia_probs.flatten(),
            clinical_vector.flatten()
        ])

    def fuse_for_risk(self, ecg_features: np.ndarray, arrhythmia_class: int, arrhythmia_confidence: float, clinical_vector: np.ndarray) -> np.ndarray:
        """Simplified fusion for XGBoost (excludes high-dim embeddings)."""
        # (14,) + (1,) + (1,) + (15,) = (31,)
        return np.concatenate([
            ecg_features.flatten(),
            np.array([float(arrhythmia_class)]),
            np.array([float(arrhythmia_confidence)]),
            clinical_vector.flatten()
        ])

    def get_fusion_feature_names(self, ecg_feature_names: List[str], clinical_feature_names: List[str]) -> List[str]:
        return ecg_feature_names + ["arrhythmia_class", "arrhythmia_confidence"] + clinical_feature_names
