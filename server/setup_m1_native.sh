#!/bin/bash

# Move into the server directory
cd "$(dirname "$0")" || exit 1

echo "Removing old native venv if it exists..."
rm -rf venv_native

echo "Creating new arm64 virtual environment..."
# Force the system Python to execute under arm64 architecture
arch -arm64 /usr/bin/python3 -m venv venv_native

echo "Activating native venv..."
source venv_native/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing ML dependencies native to Apple Silicon..."
# Install tensorflow-macos and tensorflow-metal to hook into the M1 GPU
pip install tensorflow-macos tensorflow-metal opencv-python mediapipe numpy

echo "Installing backend web dependencies..."
pip install fastapi uvicorn websockets

echo "Testing native architecture and GPU presence..."
# Run a quick verification check without dumping extra Python files
python3 -c "
import tensorflow as tf
print('\n================================')
print('TensorFlow Version:', tf.__version__)
print('Physical Devices (All):', tf.config.list_physical_devices())
gpu_devices = tf.config.list_physical_devices('GPU')
print('GPU Devices:', gpu_devices)
if len(gpu_devices) > 0:
    print('✅ SUCCESS - Metal GPU detected!')
else:
    print('❌ FAILURE - No GPU detected!')
print('================================\n')
"
