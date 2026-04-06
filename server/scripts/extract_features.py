import os
import json
import cv2
import numpy as np
import csv
from tqdm import tqdm
import sys

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FITNESS_AQA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Fitness-AQA")
SPLITS_DIR = os.path.join(FITNESS_AQA_DIR, "Squat 2", "Labeled_Dataset", "Splits")
LABELS_DIR1 = os.path.join(FITNESS_AQA_DIR, "Squat 2", "Labeled_Dataset", "Labels")
LABELS_DIR2 = os.path.join(FITNESS_AQA_DIR, "Squat 2", "Labeled_Dataset", "Shallow_Squat_Error_Dataset")
VIDEOS_DIR = os.path.join(FITNESS_AQA_DIR, "Squat", "Labeled_Dataset", "videos")

CACHE_DIR = os.path.join(BASE_DIR, "data", "tcn_cache")
TRAIN_CACHE = os.path.join(CACHE_DIR, "train")
VAL_CACHE = os.path.join(CACHE_DIR, "val")

sys.path.append(BASE_DIR)
from core.pose_module import PoseDetector
from services.data_recorder import DataRecorder

def parse_all_errors():
    label_map = {}
    json_paths = [
        os.path.join(LABELS_DIR1, "error_knees_forward.json"),
        os.path.join(LABELS_DIR1, "error_knees_inward.json"),
        os.path.join(LABELS_DIR2, "labels_shallow_depth.json")
    ]
    
    for path in json_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for vid_id, errors in data.items():
                if vid_id not in label_map:
                    label_map[vid_id] = []
                if isinstance(errors, list):
                    label_map[vid_id].extend(errors)
        except Exception as e:
            print(f"Failed parsing {path}: {e}")
            
    return label_map

def extract_features():
    os.makedirs(TRAIN_CACHE, exist_ok=True)
    os.makedirs(VAL_CACHE, exist_ok=True)
    
    # Load splits
    try:
        with open(os.path.join(SPLITS_DIR, "train_keys.json"), "r") as f:
            train_keys = json.load(f)
        with open(os.path.join(SPLITS_DIR, "val_keys.json"), "r") as f:
            val_keys = json.load(f)
    except FileNotFoundError:
        print("Split JSONs not found, please check path.")
        return

    error_map = parse_all_errors()
    detector = PoseDetector()
    recorder = DataRecorder()

    def process_split(keys, output_dir, split_name):
        csv_path = os.path.join(output_dir, "labels.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "score"])
            
            for vid_id in tqdm(keys, desc=f"Processing {split_name}"):
                video_path = os.path.join(VIDEOS_DIR, f"{vid_id}.mp4")
                if not os.path.exists(video_path):
                    continue
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                sequence = []
                timestamps = []
                frame_idx = 0
                
                while cap.isOpened():
                    success, img = cap.read()
                    if not success:
                        break
                        
                    lm_list = detector.findPosition(detector.findPose(img, draw=False), draw=False)
                    if len(lm_list) >= 33:
                        landmarks = []
                        for idx in range(33):
                            landmarks.append({'x': lm_list[idx][1], 'y': lm_list[idx][2]})
                        normalized = recorder.normalize_landmarks(landmarks)
                        sequence.append(normalized)
                        timestamps.append(frame_idx / fps)
                    frame_idx += 1
                cap.release()
                
                seq_len = 30
                error_intervals = error_map.get(vid_id, [])
                
                if len(sequence) >= seq_len:
                    for i in range(len(sequence) - seq_len + 1):
                        seq_chunk = sequence[i:i+seq_len]
                        chunk_start = timestamps[i]
                        chunk_end = timestamps[i+seq_len-1]
                        
                        is_error = False
                        for interval in error_intervals:
                            if len(interval) >= 2:
                                e_start, e_end = interval[0], interval[1]
                                if chunk_start <= e_end and chunk_end >= e_start:
                                    is_error = True
                                    break
                        
                        target_score = 0.0 if is_error else 1.0
                        chunk_filename = f"{vid_id}_{i}.npy"
                        chunk_path = os.path.join(output_dir, chunk_filename)
                        np.save(chunk_path, np.array(seq_chunk, dtype=np.float32))
                        writer.writerow([chunk_filename, target_score])

    process_split(train_keys, TRAIN_CACHE, "train")
    process_split(val_keys, VAL_CACHE, "val")

if __name__ == "__main__":
    extract_features()
