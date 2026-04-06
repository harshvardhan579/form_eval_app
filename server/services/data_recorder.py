import os
import json
import numpy as np
import time
from pathlib import Path

# MediaPipe Landmark Indices (15 key points for Squat/Curl form)
# 0: Nose
# 11, 12: Shoulders
# 13, 14: Elbows
# 15, 16: Wrists
# 23, 24: Hips
# 25, 26: Knees
# 27, 28: Ankles
# 31, 32: Feet (toes)
TARGET_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

class DataRecorder:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.perfect_dir = os.path.join(base_dir, "training_data", "perfect")
        self.flawed_dir = os.path.join(base_dir, "testing_data", "flawed")
        
        os.makedirs(self.perfect_dir, exist_ok=True)
        os.makedirs(self.flawed_dir, exist_ok=True)
        
        # Buffer to hold current rep frames
        self.current_sequence = []
        
    def normalize_landmarks(self, landmarks):
        """
        Translates all landmarks to the hip center (23, 24) 
        and scales by torso length (11 to 23).
        """
        try:
            # Hip center
            l_hip = landmarks[23]
            r_hip = landmarks[24]
            hip_center_x = (l_hip['x'] + r_hip['x']) / 2.0
            hip_center_y = (l_hip['y'] + r_hip['y']) / 2.0
            
            # Torso length
            l_shoulder = landmarks[11]
            torso_length = np.hypot(
                l_shoulder['x'] - l_hip['x'], 
                l_shoulder['y'] - l_hip['y']
            )
            # Avoid division by zero
            if torso_length < 0.001:
                torso_length = 1.0
                
            normalized_frame = []
            for idx in TARGET_LANDMARKS:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    norm_x = (lm['x'] - hip_center_x) / torso_length
                    norm_y = (lm['y'] - hip_center_y) / torso_length
                    normalized_frame.extend([norm_x, norm_y])
                else:
                    normalized_frame.extend([0.0, 0.0])
            
            return normalized_frame
        except Exception as e:
            # Return zero vector if error
            return [0.0] * (len(TARGET_LANDMARKS) * 2)

    def append_frame(self, landmarks):
        """Append a normalized frame to the current sequence."""
        norm_frame = self.normalize_landmarks(landmarks)
        self.current_sequence.append(norm_frame)

    def save_sequence(self, label, exercise_type):
        """
        Save the current buffered sequence to .npy with a timestamp.
        label: 'Good', 'Knee Cave', 'Shallow', 'Sway', etc.
        """
        if not self.current_sequence:
            return None
            
        seq_array = np.array(self.current_sequence, dtype=np.float32)
        timestamp = int(time.time() * 1000)
        
        filename = f"{exercise_type.replace(' ', '_')}_{label.replace(' ', '_')}_{timestamp}.npy"
        
        if label.lower() == 'good':
            save_path = os.path.join(self.perfect_dir, filename)
        else:
            save_path = os.path.join(self.flawed_dir, filename)
            
        np.save(save_path, seq_array)
        self.current_sequence = []  # Clear buffer after saving
        return save_path

