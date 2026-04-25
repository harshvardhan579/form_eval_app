#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for exercise form anomaly detection.")
    parser.add_argument('--exercise', type=str, required=True, choices=['curl', 'squat'],
                        help="The exercise to train on ('curl' or 'squat').")
    args = parser.parse_args()

    exercise = args.exercise
    data_path = os.path.join(DATA_DIR, f"{exercise}_perfect.npy")
    model_path = os.path.join(MODELS_DIR, f"lstm_{exercise}_best.weights.h5")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    print(f"Data loaded. Shape: {data.shape}")

    if len(data.shape) != 3 or data.shape[1:] != (30, 99):
        print(f"Error: Expected data shape (X, 30, 99), but got {data.shape}")
        return

    # Train-val split (80/20)
    split_idx = int(len(data) * 0.8)
    X_train = data[:split_idx]
    X_val = data[split_idx:]
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build LSTM Autoencoder Architecture
    print("Building LSTM Autoencoder model...")
    model = Sequential([
        Input(shape=(30, 99)),
        
        # Encoder
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        
        # Bridge
        RepeatVector(30),
        
        # Decoder
        LSTM(32, return_sequences=True),
        LSTM(64, return_sequences=True),
        
        # Output
        TimeDistributed(Dense(99))
    ])

    # Compilation
    # Using clipnorm=1.0 and learning_rate=0.0001 as required
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=True,
        verbose=1
    )

    # Training
    print(f"Starting training for {exercise}...")
    history = model.fit(
        X_train, X_train,  # Autoencoder reconstructs input
        epochs=100,        # You can adjust epochs as needed, EarlyStopping will halt it
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    print(f"Training complete. Best weights saved to {model_path}")

if __name__ == "__main__":
    main()
