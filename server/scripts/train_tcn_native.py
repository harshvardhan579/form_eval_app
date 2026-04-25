#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
# --- OVERRIDE: Blind TF to the Metal GPU to prevent pointer crash ---
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def main():
    perfect_data_path = os.path.join(DATA_DIR, "squat_perfect.npy")
    bad_data_path = os.path.join(DATA_DIR, "squat_bad.npy")
    model_path = os.path.join(MODELS_DIR, "tcn_squat_v2_best.weights.h5")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(perfect_data_path) or not os.path.exists(bad_data_path):
        print("Error: Data files not found. Ensure both squat_perfect.npy and squat_bad.npy exist.")
        return

    print("Loading datasets...")
    perfect_data = np.load(perfect_data_path)
    bad_data = np.load(bad_data_path)
    
    print(f"Perfect data loaded. Shape: {perfect_data.shape}")
    print(f"Bad data loaded. Shape: {bad_data.shape}")

    if len(perfect_data.shape) != 3 or perfect_data.shape[1:] != (30, 99):
        print(f"Error: Expected data shape (X, 30, 99), but got {perfect_data.shape} for perfect data")
        return
        
    if len(bad_data.shape) != 3 or bad_data.shape[1:] != (30, 99):
        print(f"Error: Expected data shape (X, 30, 99), but got {bad_data.shape} for bad data")
        return

    print("Assigning labels and concatenating data...")
    # Labels: 1.0 for perfect form, 0.0 for bad form
    perfect_labels = np.ones((perfect_data.shape[0], 1), dtype=np.float32)
    bad_labels = np.zeros((bad_data.shape[0], 1), dtype=np.float32)

    X = np.concatenate([perfect_data, bad_data], axis=0)
    y = np.concatenate([perfect_labels, bad_labels], axis=0)
    
    # Shuffle the dataset
    print("Shuffling datasets...")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Train-val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build TCN Architecture
    print("Building Temporal Convolutional Network (TCN)...")
    model = Sequential([
        Input(shape=(30, 99)),
        
        # 1st Convolutional Block
        Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
        MaxPooling1D(),
        
        # 2nd Convolutional Block (with Dilation)
        Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu', dilation_rate=2),
        
        # Pooling & Dense Layers
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        
        # Output Layer (0.0 to 1.0 Quality Score)
        Dense(1, activation='sigmoid')
    ])

    # Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
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
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,  # EarlyStopping will halt it
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    print(f"Training complete. Best weights saved to {model_path}")

if __name__ == "__main__":
    main()
