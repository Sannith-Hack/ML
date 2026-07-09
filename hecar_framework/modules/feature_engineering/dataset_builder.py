import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from modules.pdf_processor.pdf_loader import PDFLoader
from modules.pdf_processor.ocr_extractor import OCRExtractor
from modules.feature_engineering.ecg_parser import ECGParser
from modules.feature_engineering.arrhythmia_labeler import ArrhythmiaLabeler
from config import ECG_TEXT_FEATURES

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Builds and augments the tabular ECG dataset from PDF reports."""
    
    def __init__(self):
        self.loader = PDFLoader()
        self.ocr = OCRExtractor()
        self.parser = ECGParser()
        self.labeler = ArrhythmiaLabeler()

    def build_from_folder(self, folder_path: str) -> pd.DataFrame:
        """Complete pipeline: PDFs -> text -> metadata -> features -> labels."""
        logger.info(f"Building dataset from folder: {folder_path}")
        
        raw_data = self.loader.batch_load(folder_path)
        if not raw_data:
            return pd.DataFrame()
            
        metadata_list = []
        for rd in raw_data:
            meta = self.ocr.extract_metadata(rd["raw_text"])
            meta["file_path"] = rd["file_path"]
            meta["filename"] = rd["filename"]
            metadata_list.append(meta)
            
        df_parsed = self.parser.parse_batch(metadata_list)
        df_labeled = self.labeler.label_batch(df_parsed)
        
        # Drop rows with entirely missing core metrics
        core_cols = ["AR_bpm", "VR_bpm", "QRSD_ms", "QT_ms"]
        df_labeled.dropna(subset=core_cols, how='all', inplace=True)
        
        # Fill remaining missing numeric features with column means
        num_cols = df_labeled.select_dtypes(include=[np.number]).columns
        df_labeled[num_cols] = df_labeled[num_cols].fillna(df_labeled[num_cols].mean())
        
        logger.info(f"Built dataset with {len(df_labeled)} records.")
        return df_labeled

    def augment(self, df: pd.DataFrame, target_per_class: int = 20) -> pd.DataFrame:
        """Augments underrepresented classes by adding Gaussian noise to numeric features."""
        logger.info("Augmenting dataset to balance classes...")
        
        augmented_dfs = [df]
        numeric_cols = [c for c in ECG_TEXT_FEATURES if c in df.columns and c not in ["qrs_wide", "pr_long", "qtc_prolonged", "gender_encoded"]]
        
        class_counts = df["arrhythmia_class"].value_counts()
        
        for cls in range(7):
            count = class_counts.get(cls, 0)
            if 0 < count < target_per_class:
                needed = target_per_class - count
                class_data = df[df["arrhythmia_class"] == cls]
                
                # Sample with replacement
                samples = class_data.sample(n=needed, replace=True).copy()
                
                # Apply noise (std = 2% of mean)
                for col in numeric_cols:
                    if col in samples.columns:
                        std = df[col].std()
                        if pd.isna(std) or std == 0:
                            std = df[col].mean() * 0.02
                        noise = np.random.normal(0, std * 0.5, size=len(samples))
                        samples[col] = samples[col] + noise
                
                # Re-parse derived boolean features based on noisy data
                # Simplified re-eval
                samples["qrs_wide"] = samples["QRSD_ms"] > 100
                samples["pr_long"] = samples["PRI_ms"] > 200
                
                augmented_dfs.append(samples)
                
        final_df = pd.concat(augmented_dfs, ignore_index=True)
        logger.info(f"Augmented dataset size: {len(final_df)} records.")
        return final_df

    def save(self, df: pd.DataFrame, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

    def load(self, input_path: str) -> pd.DataFrame:
        return pd.read_csv(input_path)

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts X (feature matrix) and y (labels) for training."""
        # Ensure all expected features are present
        X_df = pd.DataFrame(index=df.index)
        for col in ECG_TEXT_FEATURES:
            if col in df.columns:
                X_df[col] = df[col]
            else:
                X_df[col] = 0  # Fallback
                
        # Fill remaining nans
        X_df = X_df.fillna(X_df.mean().fillna(0))
        
        # Explicitly cast to float32 for Keras
        X = X_df.values.astype(np.float32)
        y = df["arrhythmia_class"].values.astype(np.int32)
        return X, y
