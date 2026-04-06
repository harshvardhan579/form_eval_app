import os
import sys

def verify_gpu():
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("[FAIL] No Metal GPUs detected by TensorFlow.")
            return False
            
        print("[SUCCESS] Metal GPU detected!")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Could not import TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
