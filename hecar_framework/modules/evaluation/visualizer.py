import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

logger = logging.getLogger(__name__)

class Visualizer:
    """Generates plots and visualizations for model evaluation."""
    
    def __init__(self):
        # Apply dark theme styles globally for matplotlib
        plt.style.use('dark_background')
        self.bg_color = '#0D1117'
        self.grid_color = '#30363D'
        self.text_color = '#E6EDF3'

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], title: str, output_path: str):
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.bg_color)
            ax.set_facecolor(self.bg_color)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            
            plt.title(title, color=self.text_color, pad=20)
            plt.ylabel('True Label', color=self.text_color)
            plt.xlabel('Predicted Label', color=self.text_color)
            plt.xticks(rotation=45, ha='right', color=self.text_color)
            plt.yticks(rotation=0, color=self.text_color)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor=self.bg_color)
            plt.close()
            logger.info(f"Confusion matrix saved to {output_path}")
        except Exception as e:
            logger.exception(f"Failed to plot confusion matrix: {e}")

    def plot_training_history(self, history: Dict[str, List[float]], output_path: str):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor=self.bg_color)
            
            # Accuracy
            if 'accuracy' in history:
                ax1.set_facecolor(self.bg_color)
                ax1.plot(history['accuracy'], label='Train Accuracy', color='#00D4AA')
                if 'val_accuracy' in history:
                    ax1.plot(history['val_accuracy'], label='Val Accuracy', color='#1F6FEB')
                ax1.set_title('Model Accuracy', color=self.text_color)
                ax1.set_xlabel('Epoch', color=self.text_color)
                ax1.set_ylabel('Accuracy', color=self.text_color)
                ax1.legend(facecolor=self.grid_color, edgecolor=self.grid_color, labelcolor=self.text_color)
                ax1.grid(color=self.grid_color, linestyle='--', alpha=0.5)
                ax1.tick_params(colors=self.text_color)
                
            # Loss
            if 'loss' in history:
                ax2.set_facecolor(self.bg_color)
                ax2.plot(history['loss'], label='Train Loss', color='#00D4AA')
                if 'val_loss' in history:
                    ax2.plot(history['val_loss'], label='Val Loss', color='#1F6FEB')
                ax2.set_title('Model Loss', color=self.text_color)
                ax2.set_xlabel('Epoch', color=self.text_color)
                ax2.set_ylabel('Loss', color=self.text_color)
                ax2.legend(facecolor=self.grid_color, edgecolor=self.grid_color, labelcolor=self.text_color)
                ax2.grid(color=self.grid_color, linestyle='--', alpha=0.5)
                ax2.tick_params(colors=self.text_color)
                
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor=self.bg_color)
            plt.close()
            logger.info(f"Training history plot saved to {output_path}")
        except Exception as e:
            logger.exception(f"Failed to plot training history: {e}")

    def plot_performance_comparison(self, comparison_df: pd.DataFrame, output_path: str):
        try:
            df_melted = comparison_df.melt(id_vars='model', var_name='Metric', value_name='Score')
            
            plt.figure(figsize=(10, 6), facecolor=self.bg_color)
            ax = plt.gca()
            ax.set_facecolor(self.bg_color)
            
            sns.barplot(x='Metric', y='Score', hue='model', data=df_melted, palette='viridis', ax=ax)
            
            plt.title('Model Performance Comparison', color=self.text_color)
            plt.ylim(0, 1.05)
            plt.legend(facecolor=self.grid_color, edgecolor=self.grid_color, labelcolor=self.text_color)
            ax.tick_params(colors=self.text_color)
            ax.grid(color=self.grid_color, axis='y', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor=self.bg_color)
            plt.close()
        except Exception as e:
            logger.exception(f"Failed to plot performance comparison: {e}")

    def plot_risk_distribution(self, stroke_scores: List[float], chd_scores: List[float], output_path: str):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor=self.bg_color)
            
            sns.histplot(stroke_scores, bins=20, kde=True, ax=ax1, color='#F85149')
            ax1.set_title('Stroke Risk Distribution', color=self.text_color)
            ax1.set_facecolor(self.bg_color)
            ax1.tick_params(colors=self.text_color)
            
            sns.histplot(chd_scores, bins=20, kde=True, ax=ax2, color='#1F6FEB')
            ax2.set_title('CHD Risk Distribution', color=self.text_color)
            ax2.set_facecolor(self.bg_color)
            ax2.tick_params(colors=self.text_color)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor=self.bg_color)
            plt.close()
        except Exception as e:
            logger.exception(f"Failed to plot risk distribution: {e}")

    def plot_arrhythmia_distribution(self, labels: List[int], class_names: Dict[int, str], output_path: str):
        try:
            counts = pd.Series(labels).value_counts()
            names = [class_names[idx] for idx in counts.index]
            
            plt.figure(figsize=(10, 8), facecolor=self.bg_color)
            plt.pie(counts, labels=names, autopct='%1.1f%%', 
                    textprops={'color': self.text_color}, 
                    colors=sns.color_palette("husl", len(names)))
            plt.title('Arrhythmia Class Distribution', color=self.text_color)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor=self.bg_color)
            plt.close()
        except Exception as e:
            logger.exception(f"Failed to plot arrhythmia distribution: {e}")
