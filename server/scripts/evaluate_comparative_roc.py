#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf

# --- OVERRIDE: Blind TF to the Metal GPU to prevent pointer crash ---
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
GRAPHICS_DIR = os.path.join(BASE_DIR, "data", "graphics")

def build_lstm_model():
    """Builds the exact LSTM Autoencoder architecture."""
    model = Sequential([
        Input(shape=(30, 99)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        RepeatVector(30),
        LSTM(32, return_sequences=True),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(99))
    ])
    return model

def build_tcn_v2_model():
    """Builds the exact Native TCN V2 architecture."""
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
    lstm_weights_path = os.path.join(BASE_DIR, "models", "lstm_squat_best.weights.h5")
    tcn_weights_path = os.path.join(BASE_DIR, "models", "tcn_squat_v2_best.weights.h5")
    
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    
    if not os.path.exists(lstm_weights_path) or not os.path.exists(tcn_weights_path):
        print("Error: Model weights not found. Ensure both LSTM and TCN weights exist in server/models/.")
        return
        
    print("Loading datasets natively (X, 30, 99)...")
    perfect_data = np.load(perfect_data_path)
    bad_data = np.load(bad_data_path)
    
    # Create true labels array: 1 for perfect, 0 for bad
    perfect_true = np.ones((perfect_data.shape[0],))
    bad_true = np.zeros((bad_data.shape[0],))
    
    # Combine datasets
    X_all = np.concatenate([perfect_data, bad_data], axis=0)
    y_true = np.concatenate([perfect_true, bad_true], axis=0)
    
    # ==========================
    # 1. LSTM Evaluation
    # ==========================
    print("Building LSTM Autoencoder and loading weights...")
    lstm_model = build_lstm_model()
    lstm_model.load_weights(lstm_weights_path)
    
    print("Running LSTM inference...")
    lstm_reconstructions = lstm_model.predict(X_all, verbose=0)
    # Calculate MSE for every sequence
    lstm_mse = np.mean(np.square(X_all - lstm_reconstructions), axis=(1, 2))
    
    # Math Trick: Multiply MSE by -1 so high MSE (Bad Form) becomes a lower score,
    # and low MSE (Perfect Form) becomes a higher score (acting as a "Confidence in Perfect Form")
    lstm_scores = -1.0 * lstm_mse
    
    # ==========================
    # 2. Native TCN (V2) Evaluation
    # ==========================
    print("Building Native TCN (V2) and loading weights...")
    tcn_model = build_tcn_v2_model()
    tcn_model.load_weights(tcn_weights_path)
    
    print("Running TCN inference...")
    # Raw sigmoid probabilities (0.0 to 1.0) directly represent confidence in "Perfect Form"
    tcn_scores = tcn_model.predict(X_all, verbose=0).flatten()
    
    # ==========================
    # ROC Curve Calculations
    # ==========================
    print("Calculating ROC Curves and AUC metrics...")
    lstm_fpr, lstm_tpr, _ = roc_curve(y_true, lstm_scores)
    lstm_auc = auc(lstm_fpr, lstm_tpr)
    
    tcn_fpr, tcn_tpr, _ = roc_curve(y_true, tcn_scores)
    tcn_auc = auc(tcn_fpr, tcn_tpr)
    
    # ==========================
    # Generate Academic Graphic
    # ==========================
    print("Generating IEEE styled Comparative ROC Graphic...")
    
    # IEEE Academic Styling: White background with clean grid lines
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Line 1: TCN (Bold, solid blue line)
    ax.plot(tcn_fpr, tcn_tpr, color='#1f77b4', linestyle='-', linewidth=3.5, 
            label=f'Temporal Convolutional Network (AUC = {tcn_auc:.3f})')
    
    # Line 2: LSTM (Dashed, red line)
    ax.plot(lstm_fpr, lstm_tpr, color='#d62728', linestyle='--', linewidth=2.5, 
            label=f'LSTM Autoencoder (AUC = {lstm_auc:.3f})')
            
    # Random guessing (Dotted diagonal grey line)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=2.0)
    
    # Titles and Labels
    ax.set_title('Model Architecture Comparison: Squat Anomaly Detection', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set rigid bounds for academic standard
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    
    # Configure the legend
    ax.legend(loc="lower right", fontsize=12, frameon=True, shadow=False, edgecolor='black', facecolor='white')
    
    plt.tight_layout()
    
    output_path = os.path.join(GRAPHICS_DIR, "squat_architecture_roc.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved comparative ROC graphic to {output_path}")

if __name__ == "__main__":
    main()
