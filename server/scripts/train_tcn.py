import os
import tensorflow as tf
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.tcn_model import TCNModel

CACHE_DIR = os.path.join(BASE_DIR, "data", "tcn_cache")
TRAIN_CACHE = os.path.join(CACHE_DIR, "train")
VAL_CACHE = os.path.join(CACHE_DIR, "val")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_npy(filepath_tensor, label_tensor):
    def _read_npy(filepath):
        import numpy as np
        return np.load(filepath.decode('utf-8')).astype(np.float32)
        
    npy_data = tf.numpy_function(_read_npy, [filepath_tensor], tf.float32)
    npy_data.set_shape((30, 30)) # sequence_length=30, num_features=30
    return npy_data, label_tensor

def build_dataset(cache_dir, batch_size=16):
    csv_path = os.path.join(cache_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: No labels.csv found in {cache_dir}")
        return None
        
    df = pd.read_csv(csv_path)
    file_paths = [os.path.join(cache_dir, f) for f in df['filename']]
    labels = df['score'].values.astype('float32')
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train():
    train_dataset = build_dataset(TRAIN_CACHE)
    val_dataset = build_dataset(VAL_CACHE)
    
    if train_dataset is None or val_dataset is None:
        print("Datasets missing. Please run extract_features.py first.")
        return
        
    tcn = TCNModel(sequence_length=30, num_features=30)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODELS_DIR, "tcn_squat_best.weights.h5"), 
                                           monitor='val_loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    print("Starting TCN Training...")
    history = tcn.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks
    )
    
    print("Training finished!")
    
    # Print final validation metrics
    val_loss = history.history['val_loss'][-1]
    print(f"Final Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()
