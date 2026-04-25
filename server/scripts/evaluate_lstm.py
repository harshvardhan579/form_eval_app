#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHICS_DIR = os.path.join(BASE_DIR, "data", "graphics")

def build_model():
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

def calculate_reconstruction_error(model, data):
    # Predict to get reconstructed data
    predictions = model.predict(data, verbose=0)
    # Mean Squared Error for each sequence (average across time steps and features)
    mse = np.mean(np.square(data - predictions), axis=(1, 2))
    return mse

def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM Autoencoder and generate academic graphics.")
    parser.add_argument('--exercise', type=str, required=True, choices=['curl', 'squat'],
                        help="The exercise to evaluate ('curl' or 'squat').")
    args = parser.parse_args()

    exercise = args.exercise
    perfect_data_path = os.path.join(DATA_DIR, f"{exercise}_perfect.npy")
    bad_data_path = os.path.join(DATA_DIR, f"{exercise}_bad.npy")
    model_path = os.path.join(MODELS_DIR, f"lstm_{exercise}_best.weights.h5")

    # Ensure output directory exists
    os.makedirs(GRAPHICS_DIR, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    if not os.path.exists(perfect_data_path) or not os.path.exists(bad_data_path):
        print(f"Error: Data files not found for '{exercise}'. Ensure both perfect and bad data exist.")
        return

    print(f"Loading data for '{exercise}'...")
    perfect_data = np.load(perfect_data_path)
    bad_data = np.load(bad_data_path)
    
    print("Building model and loading weights...")
    model = build_model()
    model.load_weights(model_path)
    
    print("Running inference...")
    perfect_errors = calculate_reconstruction_error(model, perfect_data)
    bad_errors = calculate_reconstruction_error(model, bad_data)
    
    print("Calculating threshold...")
    # Define threshold as 95th percentile of the perfect data's error
    threshold = np.percentile(perfect_errors, 95)
    print(f"Calculated Threshold (95th percentile of Perfect Form): {threshold:.6f}")

    print("Generating academic-quality graphics...")
    # Set styling for academic quality (clean, minimalist, white background)
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LSTM Autoencoder Reconstruction Error: {exercise.capitalize()}", fontsize=16, fontweight='bold', y=1.02)
    
    # Define muted academic colors
    color_perfect = "#4C72B0" # Muted Blue
    color_bad = "#C44E52"     # Muted Red
    
    # ==========================
    # 1. Violin Plot
    # ==========================
    ax1 = axes[0]
    data_list = [perfect_errors, bad_errors]
    
    parts = ax1.violinplot(data_list, showmeans=False, showmedians=True, showextrema=True)
    
    # Style the violin plot components
    for pc, color in zip(parts['bodies'], [color_perfect, color_bad]):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.5)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Perfect Form', 'Bad Form'])
    ax1.set_ylabel('Reconstruction Error (MSE)')
    ax1.set_title('Error Distribution Comparison', fontweight='bold')
    
    # Add dynamic threshold line
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (95th% = {threshold:.4f})')
    ax1.legend(loc="upper left")

    # ==========================
    # 2. Histogram / KDE Plot
    # ==========================
    ax2 = axes[1]
    sns.histplot(perfect_errors, bins=50, color=color_perfect, kde=True, label="Perfect Form", stat="density", ax=ax2, alpha=0.6)
    sns.histplot(bad_errors, bins=50, color=color_bad, kde=True, label="Bad Form", stat="density", ax=ax2, alpha=0.6)
    
    # Add dynamic threshold line
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (95th% = {threshold:.4f})')
    
    ax2.set_xlabel('Reconstruction Error (MSE)')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Density & Overlap', fontweight='bold')
    ax2.legend(loc="upper right")
    
    # Clean up layout
    sns.despine()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(GRAPHICS_DIR, f"{exercise}_evaluation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved graphics to {output_path}")

if __name__ == "__main__":
    main()
