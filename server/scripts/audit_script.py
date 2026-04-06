import os
import tensorflow as tf

print("\n--- Hardware & Environment Check ---")
print("TensorFlow Version:", tf.__version__)
print("Physical Devices (All):", tf.config.list_physical_devices())
gpu_devices = tf.config.list_physical_devices('GPU')
print("GPU Devices:", gpu_devices)
if len(gpu_devices) > 0:
    print("✅ SUCCESS - Metal GPU detected!")
else:
    print("❌ FAILURE - No GPU detected!")

print("\n--- Data Inventory ---")
for d in [
    '/Users/hvsingh/Desktop/form_eval_app/server/data/training_data/perfect',
    '/Users/hvsingh/Desktop/form_eval_app/server/data/testing_data/flawed',
    '/Users/hvsingh/Desktop/form_eval_app/server/data/training_data/perfect/Bicep Curls',
    '/Users/hvsingh/Desktop/form_eval_app/server/data/training_data/perfect/Squats',
    '/Users/hvsingh/Desktop/form_eval_app/server/data/testing_data/flawed/Bicep Curls',
    '/Users/hvsingh/Desktop/form_eval_app/server/data/testing_data/flawed/Squats'
]:
    print(f"Directory: {d}")
    if os.path.exists(d):
        npy_files = []
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))
        print(f"Found {len(npy_files)} .npy files.")
        if len(npy_files) > 0:
            import numpy as np
            shapes = set()
            for nf in npy_files[:10]: # Check shapes of first 10
                arr = np.load(nf)
                shapes.add(arr.shape)
            print(f"Shapes of sample arrays: {shapes}")
    else:
        print("Directory DOES NOT EXIST.")

print("\n--- Fitness-AQA Mapping ---")
labels_path = '/Users/hvsingh/Desktop/form_eval_app/labels'
# Let's count some json files or whatever is there
if os.path.exists(labels_path):
    files = os.listdir(labels_path)
    print(f"Found {len(files)} items in {labels_path}.")
else:
    print(f"{labels_path} DOES NOT EXIST.")
