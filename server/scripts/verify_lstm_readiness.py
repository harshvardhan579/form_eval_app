#!/usr/bin/env python3

import os
import sys
import glob
import re

def print_status(status, message):
    colors = {
        "PASS": "\033[92m",    # Green
        "FAIL": "\033[91m",    # Red
        "WARNING": "\033[93m", # Yellow
        "INFO": "\033[94m",    # Blue
        "ENDC": "\033[0m"      # Reset
    }
    color = colors.get(status, "")
    print(f"{color}[{status}]{colors['ENDC']} {message}")

def check_environment():
    print("\n--- Environment Integrity ---")
    # TensorFlow
    try:
        # Suppress TF logging for cleaner output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print_status("PASS", f"TensorFlow recognizes GPU device(s): {[device.name for device in gpu_devices]}")
        else:
            print_status("FAIL", "TensorFlow does not recognize any GPU devices (Apple Metal GPU expected).")
    except ImportError:
        print_status("FAIL", "TensorFlow is not installed.")
    except Exception as e:
        print_status("FAIL", f"Error checking TensorFlow: {e}")

    # Numpy
    try:
        import numpy as np
        np_version = np.__version__
        
        # Safely parse major version
        major_version_match = re.match(r"^(\d+)", np_version)
        major_version = int(major_version_match.group(1)) if major_version_match else 0
        
        if major_version >= 2:
            print_status("WARNING", f"numpy version {np_version} is >= 2.0.0. This may break MediaPipe.")
        else:
            print_status("PASS", f"numpy version is {np_version} (Safe for MediaPipe).")
    except ImportError:
        print_status("FAIL", "numpy is not installed.")

def check_directories(base_dir):
    print("\n--- Directory Architecture ---")
    directories_to_check = [
        "server/models",
        "server/data/raw_videos",
        "server/data/extracted_features"
    ]
    for directory in directories_to_check:
        full_path = os.path.join(base_dir, directory)
        if os.path.isdir(full_path):
            print_status("PASS", f"Directory exists: {directory}")
        else:
            print_status("FAIL", f"Directory missing: {directory}")

def check_data(base_dir):
    print("\n--- Data Availability & Shape Auditing ---")
    try:
        import numpy as np
    except ImportError:
        print_status("FAIL", "numpy is not installed. Cannot audit data.")
        return

    features_dir = os.path.join(base_dir, "server/data/extracted_features")
    if not os.path.isdir(features_dir):
        print_status("FAIL", f"Cannot check data. Directory missing: {features_dir}")
        return

    npy_files = glob.glob(os.path.join(features_dir, "*.npy"))
    if not npy_files:
        print_status("WARNING", f"No .npy files found in {features_dir}.")
        return
    
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        try:
            data = np.load(npy_file, allow_pickle=True)
            shape = data.shape
            
            # Check shape: Expected (X, Y, 99)
            if len(shape) == 3 and shape[2] == 99:
                print_status("PASS", f"File: {filename} | Shape: {shape} | Status: Valid")
            else:
                print_status("FAIL", f"File: {filename} | Shape: {shape} | Status: Corrupt (Expected (X, Y, 99))")
        except Exception as e:
            print_status("FAIL", f"File: {filename} | Could not load: {e}")

def check_script(base_dir):
    print("\n--- Script Check ---")
    script_path = os.path.join(base_dir, "server/scripts/extract_features.py")
    if os.path.isfile(script_path):
        print_status("PASS", "Script exists: server/scripts/extract_features.py")
    else:
        print_status("FAIL", "Script missing: server/scripts/extract_features.py")

def main():
    print("Starting System and Data Audit for LSTM Autoencoder Training...")
    
    # Resolve the base directory of the project.
    # Assuming this script is located in base_dir/server/scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    
    check_environment()
    check_directories(base_dir)
    check_data(base_dir)
    check_script(base_dir)
    
    print("\nAudit Complete.\n")

if __name__ == "__main__":
    main()
