import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import os

class FormAutoencoder:
    def __init__(self, sequence_length=30, num_features=30, model_path=None):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()
        self.threshold = 0.5  # Fallback threshold
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            
        # Add tensorflow-macos / tensorflow-metal logic
        # TF automatically uses metal layer on M1 if tensorflow-metal is installed
        
    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, self.num_features))
        
        # Encoder
        encoded = LSTM(64, activation='tanh', return_sequences=True)(inputs)
        encoded = LSTM(32, activation='tanh', return_sequences=False)(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(32, activation='tanh', return_sequences=True)(decoded)
        decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
        outputs = TimeDistributed(Dense(self.num_features))(decoded)
        
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
        
    def set_threshold_from_training(self, train_data):
        """Calculate the 95th percentile reconstruction error from perfect data."""
        reconstructions = self.model.predict(train_data)
        mse = np.mean(np.square(train_data - reconstructions), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)
        print(f"Set Autoencoder 95th Percentile Threshold: {self.threshold}")
        return self.threshold

    def predict_anomaly(self, sequence):
        """
        Run inference on a single sequence (shape: seq_len, num_features).
        Returns a tuple: (is_anomaly, reconstruction_error)
        """
        if len(sequence) < self.sequence_length:
            return False, 0.0
            
        # Just use the last `sequence_length` frames
        seq_input = np.array([sequence[-self.sequence_length:]], dtype=np.float32)
        
        reconstruction = self.model.predict(seq_input, verbose=0)
        mse = np.mean(np.square(seq_input - reconstruction))
        
        is_anomaly = float(mse) > self.threshold
        return is_anomaly, float(mse)

    def train_model(self, train_data, epochs=50, batch_size=32, save_path="autoencoder_weights.h5"):
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        from tensorflow.keras import backend as K
        
        # Save weights as .h5 file and stop early if validation loss plateaus
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Training Autoencoder...")
        self.model.fit(
            train_data, train_data, # Autoencoder target is its input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint, early_stop]
        )
        
        print("Training complete, explicitly clearing Keras session...")
        K.clear_session()
