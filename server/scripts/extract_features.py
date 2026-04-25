import os
import cv2
import numpy as np
import glob
import sys

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted_features")

sys.path.append(BASE_DIR)
from core.pose_module import PoseDetector

def extract_features():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = PoseDetector()

    video_files = glob.glob(os.path.join(VIDEOS_DIR, "*.mov")) + glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))

    for video_path in video_files:
        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing {vid_id}...")
        
        cap = cv2.VideoCapture(video_path)
        sequence = []

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            
            img = detector.findPose(img, draw=False)
            lm_list = detector.findPosition(img, draw=False)
            
            if len(lm_list) >= 33:
                frame_features = []
                for idx in range(33):
                    # lm_list[idx] is [id, x, y, z, visibility]
                    # We need x, y, z
                    if len(lm_list[idx]) >= 4:
                        frame_features.extend([lm_list[idx][1], lm_list[idx][2], lm_list[idx][3]])
                    else:
                        frame_features.extend([lm_list[idx][1], lm_list[idx][2], 0.0])
                sequence.append(frame_features)
                
        cap.release()
        
        if len(sequence) == 0:
            print(f"No features extracted for {vid_id}")
            continue
            
        seq_len = 30
        chunks = []
        # Create sliding window chunks
        for i in range(max(1, len(sequence) - seq_len + 1)):
            chunk = sequence[i:i+seq_len]
            # Pad if shorter than seq_len
            if len(chunk) < seq_len:
                padding = [chunk[-1]] * (seq_len - len(chunk))
                chunk.extend(padding)
            chunks.append(chunk)
            
        chunks_array = np.array(chunks, dtype=np.float32)
        
        chunk_path = os.path.join(OUTPUT_DIR, f"{vid_id}.npy")
        np.save(chunk_path, chunks_array)
        print(f"Saved {chunk_path} with shape {chunks_array.shape}")

if __name__ == "__main__":
    extract_features()
