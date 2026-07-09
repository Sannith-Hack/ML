import logging
import numpy as np
from typing import Tuple, Dict

try:
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    pass

from config import XGBOOST_CONFIG

logger = logging.getLogger(__name__)

class StrokePredictor:
    """XGBoost model for Stroke Risk Prediction."""
    
    def __init__(self):
        self.config = XGBOOST_CONFIG
        self.model = None

    def build(self) -> 'XGBClassifier':
        self.model = XGBClassifier(**self.config)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)
        logger.info("Stroke Predictor trained.")
        return self.model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        probs = self.model.predict_proba(X)
        classes = self.model.predict(X)
        return classes, probs

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        classes, probs = self.predict(X_test)
        
        acc = accuracy_score(y_test, classes)
        # Binary classification assumed
        if probs.shape[1] == 2:
            auc = roc_auc_score(y_test, probs[:, 1])
        else:
            auc = 0.0
            
        return {"accuracy": acc, "auc": auc}
