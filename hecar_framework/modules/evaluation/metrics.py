import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
except ImportError:
    pass

class Metrics:
    """Computes evaluation metrics for models."""
    
    def __init__(self):
        pass

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        res = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        }
        
        if y_prob is not None:
            try:
                # Handle binary vs multiclass
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
                    res["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
                elif len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    res["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    res["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                res["auc_roc"] = 0.0
                
        return res

    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def compute_sensitivity_specificity(self, cm: np.ndarray) -> Tuple[float, float]:
        # For binary, or macro-average for multi-class
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return sensitivity, specificity
            
        # Multiclass macro
        sensitivities = []
        specificities = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sens)
            specificities.append(spec)
            
        return float(np.mean(sensitivities)), float(np.mean(specificities))

    def format_report(self, metrics_dict: Dict[str, float]) -> str:
        report = []
        for k, v in metrics_dict.items():
            report.append(f"{k.upper():<15}: {v:.4f}")
        return "\n".join(report)

    def compare_models(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Input: [{'model':'LSTM', 'accuracy':0.95, ...}, ...]"""
        return pd.DataFrame(results)
