#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf

# --- OVERRIDE: Blind TF to the Metal GPU to prevent pointer crash ---
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
GRAPHICS_DIR = os.path.join(BASE_DIR, "data", "graphics")

def build_v2_model():
    """Builds the Native TCN V2 Architecture expecting (30, 99) inputs."""
    model = Sequential([
        Input(shape=(30, 99)),
        Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
        MaxPooling1D(),
        Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu', dilation_rate=2),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def main():
    perfect_data_path = os.path.join(DATA_DIR, "squat_perfect.npy")
    bad_data_path = os.path.join(DATA_DIR, "squat_bad.npy")
    model_path = os.path.join(BASE_DIR, "models", "tcn_squat_v2_best.weights.h5")
    
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    print("Loading datasets natively (no dimensional reduction)...")
    perfect_data = np.load(perfect_data_path)
    bad_data = np.load(bad_data_path)
    
    print(f"Perfect data shape: {perfect_data.shape}")
    print(f"Bad data shape: {bad_data.shape}")
    
    print("Building Native TCN (V2) Architecture and loading weights...")
    model = build_v2_model()
    model.load_weights(model_path)
    
    print("Running predictions...")
    # TCN output is 0.0-1.0 sigmoid
    perfect_preds_raw = model.predict(perfect_data, verbose=0).flatten()
    bad_preds_raw = model.predict(bad_data, verbose=0).flatten()
    
    # Convert sigmoid to strict binary predictions (Threshold = 0.5)
    perfect_preds = (perfect_preds_raw >= 0.5).astype(int)
    bad_preds = (bad_preds_raw >= 0.5).astype(int)
    
    # Create true labels array (1 for perfect, 0 for bad)
    perfect_true = np.ones_like(perfect_preds)
    bad_true = np.zeros_like(bad_preds)
    
    # Combine predictions and true labels
    y_true = np.concatenate([bad_true, perfect_true])
    y_pred = np.concatenate([bad_preds, perfect_preds])
    
    print("Generating Confusion Matrix graphic...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row-wise percentages for annotations
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create text annotations (Count + Percentage)
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            # Example format: "1187\n(100.0%)"
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]*100:.1f}%)"
            
    # Set up matplotlib and seaborn for poster styling
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=2, linecolor='black')
    
    # Labels and Titles
    ax.set_title('Native TCN (V2) Evaluation: Squat Form', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Form', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Actual Form', fontsize=16, fontweight='bold', labelpad=15)
    
    # Set tick labels (0 = Bad, 1 = Perfect)
    tick_labels = ['Bad Form', 'Perfect Form']
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_yticklabels(tick_labels, fontsize=14, rotation=0)
    
    plt.tight_layout()
    
    output_path = os.path.join(GRAPHICS_DIR, "tcn_squat_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved Confusion Matrix graphic to {output_path}")

if __name__ == "__main__":
    main()
