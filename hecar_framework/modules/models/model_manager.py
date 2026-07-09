import logging
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

try:
    from tensorflow import keras
except ImportError:
    pass

from config import MODEL_SAVED_DIR, MODEL_HISTORY_DIR

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages saving and loading of Keras and scikit-learn/XGBoost models."""
    
    def __init__(self):
        self.saved_dir = Path(MODEL_SAVED_DIR)
        self.history_dir = Path(MODEL_HISTORY_DIR)

    def save_keras_model(self, model, name: str):
        path = self.saved_dir / f"{name}.keras"
        model.save(str(path))
        logger.info(f"Keras model saved to {path}")

    def load_keras_model(self, name: str):
        path = self.saved_dir / f"{name}.keras"
        if not path.exists():
            return None
        return keras.models.load_model(str(path))

    def save_sklearn_model(self, model, name: str):
        path = self.saved_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Sklearn/XGBoost model saved to {path}")

    def load_sklearn_model(self, name: str):
        path = self.saved_dir / f"{name}.pkl"
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_history(self, history: Dict[str, Any], name: str):
        path = self.history_dir / f"{name}.json"
        
        # Convert NumPy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
                
        cleaned_history = {k: [convert(v) for v in val] for k, val in history.items()}
        
        with open(path, 'w') as f:
            json.dump(cleaned_history, f)
        logger.info(f"History saved to {path}")

    def load_history(self, name: str) -> Dict[str, Any]:
        path = self.history_dir / f"{name}.json"
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def model_exists(self, name: str) -> bool:
        keras_path = self.saved_dir / f"{name}.keras"
        pkl_path = self.saved_dir / f"{name}.pkl"
        return keras_path.exists() or pkl_path.exists()

    def save_scaler(self, scaler, name: str):
        self.save_sklearn_model(scaler, name)

    def load_scaler(self, name: str):
        return self.load_sklearn_model(name)
