import logging
import numpy as np
from typing import Tuple, Dict

try:
    from tensorflow import keras
    from tensorflow.keras.layers import Conv1D, BatchNormalization, Bidirectional, LSTM, Dropout, Dense, Input
    from tensorflow.keras.models import Model
except ImportError:
    pass

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class CNNBiLSTMModel:
    """CNN-BiLSTM Architecture for 1D ECG Feature Sequences."""
    
    def __init__(self):
        self.config = MODEL_CONFIG

    def build(self) -> 'keras.Model':
        input_shape = self.config["input_shape"]
        
        inputs = Input(shape=input_shape)
        
        x = Conv1D(self.config["cnn_filters_1"], self.config["cnn_kernel_size"], activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        
        x = Conv1D(self.config["cnn_filters_2"], self.config["cnn_kernel_size"], activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(self.config["lstm_units_1"], return_sequences=True))(x)
        x = Dropout(self.config["dropout_rate"])(x)
        
        x = Bidirectional(LSTM(self.config["lstm_units_2"]))(x)
        x = Dropout(self.config["dropout_rate"])(x)
        
        embedding = Dense(self.config["dense_units"], activation='relu', name='embedding_layer')(x)
        
        outputs = Dense(self.config["num_classes"], activation='softmax')(embedding)
        
        model = Model(inputs=inputs, outputs=outputs)
        logger.info("CNN-BiLSTM model built successfully.")
        return model

    def compile(self, model: 'keras.Model') -> 'keras.Model':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, model: 'keras.Model', X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        # Check shapes
        if len(X_train.shape) == 2:
            X_train = np.expand_dims(X_train, axis=1)
        if len(X_val.shape) == 2:
            X_val = np.expand_dims(X_val, axis=1)
            
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.config["patience"], restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=[early_stopping],
            verbose=1
        )
        return history.history

    def predict(self, model: 'keras.Model', X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
        probs = model.predict(X)
        classes = np.argmax(probs, axis=-1)
        return classes, probs

    def get_embedding(self, model: 'keras.Model', X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
        embed_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
        return embed_model.predict(X)
