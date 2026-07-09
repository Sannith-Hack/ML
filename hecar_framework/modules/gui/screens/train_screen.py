import customtkinter as ctk
import threading
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from modules.gui.theme import HECAR_THEME
from modules.gui.components import StyledButton, SectionHeader, ProgressCard, ScrollableResultFrame
from modules.models.cnn_bilstm import CNNBiLSTMModel
from modules.models.stroke_predictor import StrokePredictor
from modules.models.heart_disease_predictor import HeartDiseasePredictor
from modules.models.model_manager import ModelManager
from modules.feature_engineering.dataset_builder import DatasetBuilder
from config import PROCESSED_DIR

class TrainScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.mgr = ModelManager()
        
        SectionHeader(self, "Model Training (CNN-BiLSTM)").pack(anchor="w", pady=(0, 20))
        
        # Hyperparameters
        hp_frame = ctk.CTkFrame(self, fg_color=HECAR_THEME.bg_secondary)
        hp_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(hp_frame, text="Epochs:", text_color=HECAR_THEME.text_secondary).grid(row=0, column=0, padx=10, pady=10)
        self.epochs = ctk.CTkSlider(hp_frame, from_=10, to=200, number_of_steps=19, width=150)
        self.epochs.set(100)
        self.epochs.grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(hp_frame, text="Batch Size:", text_color=HECAR_THEME.text_secondary).grid(row=0, column=2, padx=10, pady=10)
        self.batch = ctk.CTkComboBox(hp_frame, values=["8", "16", "32", "64"], width=80)
        self.batch.set("16")
        self.batch.grid(row=0, column=3, padx=10, pady=10)
        
        StyledButton(hp_frame, "Start Training", command=self.start_training).grid(row=0, column=4, padx=20, pady=10)
        
        # Progress
        self.progress = ProgressCard(self, "Training Progress")
        self.progress.pack(fill="x", pady=20)
        
        # Log
        SectionHeader(self, "Training Log").pack(anchor="w")
        self.log_box = ctk.CTkTextbox(self, fg_color=HECAR_THEME.bg_card, text_color=HECAR_THEME.text_secondary, font=HECAR_THEME.font_small)
        self.log_box.pack(fill="both", expand=True)

    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def start_training(self):
        dataset_path = Path(PROCESSED_DIR) / "hecar_dataset.csv"
        if not dataset_path.exists():
            self.log("Dataset not found. Please preprocess first.")
            return
            
        self.progress.set_progress(0.1)
        self.log("Loading dataset...")
        
        def run():
            try:
                df = pd.read_csv(dataset_path)
                builder = DatasetBuilder()
                X, y = builder.get_feature_matrix(df)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.log(f"Data split: {len(X_train)} train, {len(X_val)} val")
                
                # BiLSTM
                bilstm = CNNBiLSTMModel()
                # Override configs from UI
                bilstm.config["epochs"] = int(self.epochs.get())
                bilstm.config["batch_size"] = int(self.batch.get())
                
                model = bilstm.build()
                model = bilstm.compile(model)
                
                self.log("Starting CNN-BiLSTM training (see console for details)...")
                # For GUI, we just run fit. In real app, we'd use a callback to update progress bar.
                hist = bilstm.train(model, X_train, y_train, X_val, y_val)
                
                self.mgr.save_keras_model(model, "cnn_bilstm")
                self.mgr.save_history(hist, "cnn_bilstm_history")
                
                self.log("Training complete. Model saved.")
                self.progress.set_progress(1.0)
                self.app.update_status("Model ready", HECAR_THEME.success)
                
            except Exception as e:
                self.after(0, lambda: self.log(f"Error during training: {str(e)}"))
                self.app.update_status("Training failed", HECAR_THEME.danger)
                
        threading.Thread(target=run, daemon=True).start()
