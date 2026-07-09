import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

try:
    import shap
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Uses SHAP to explain XGBoost risk models."""
    
    def __init__(self):
        pass

    def explain_risk(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction."""
        try:
            explainer = shap.TreeExplainer(model)
            # Ensure 2D
            if len(X.shape) == 1:
                X_eval = X.reshape(1, -1)
            else:
                X_eval = X
                
            shap_values = explainer.shap_values(X_eval)
            expected_value = explainer.expected_value
            
            # If multi-class or similar format returned, take the positive class (index 1 or last)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
                base_val = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            else:
                if len(shap_values.shape) == 3:
                    shap_vals = shap_values[0, :, 1]
                    base_val = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                else:
                    shap_vals = shap_values[0]
                    base_val = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                    
            # Get top features
            feature_importance = list(zip(feature_names, shap_vals))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_importance[:10]
            
            return {
                "shap_values": shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals,
                "base_value": float(base_val),
                "top_features": top_features
            }
        except Exception as e:
            logger.exception(f"SHAP explanation failed: {e}")
            return {"shap_values": [], "base_value": 0.0, "top_features": []}

    def plot_shap_bar(self, shap_result: Dict[str, Any], title: str, output_path: str):
        """Generates and saves a simple matplotlib bar chart for top features."""
        try:
            top_features = shap_result.get("top_features", [])
            if not top_features:
                return
                
            # Reverse to plot highest at top
            top_features.reverse()
            
            names = [f[0] for f in top_features]
            vals = [f[1] for f in top_features]
            
            colors = ['#F85149' if v > 0 else '#1F6FEB' for v in vals]
            
            plt.figure(figsize=(8, 6), facecolor='#0D1117')
            ax = plt.gca()
            ax.set_facecolor('#0D1117')
            
            plt.barh(names, vals, color=colors)
            plt.title(title, color='white')
            plt.xlabel("SHAP Value (Impact on Risk Score)", color='white')
            
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#30363D')
                
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
            plt.close()
            logger.info(f"SHAP plot saved to {output_path}")
        except Exception as e:
            logger.exception(f"Failed to plot SHAP chart: {e}")

    def explain_batch(self, model, X_batch: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i in range(X_batch.shape[0]):
            res = self.explain_risk(model, X_batch[i], feature_names)
            results.append(res)
        return results
