import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, BatchNormalization
import sys

from services.data_recorder import DataRecorder

class TCNModel:
    def __init__(self, sequence_length=30, num_features=30, model_path=None):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            
    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, self.num_features))
        
        # Dilated Convolutions
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal')(x)
        x = BatchNormalization()(x)
        
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x) # Binary quality score 0-1
        
        tcn = Model(inputs, outputs)
        adam_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        tcn.compile(optimizer=adam_opt, loss='binary_crossentropy')
        return tcn
        
    def predict_quality(self, sequence):
        if len(sequence) < self.sequence_length:
            return 100.0 # Default high score if sequence is too short
            
        seq_input = np.array([sequence[-self.sequence_length:]], dtype=np.float32)
        score = self.model.predict(seq_input, verbose=0)[0][0]
        # Scale back to 0-100 score for display
        return max(0.0, min(100.0, float(score) * 100.0))

    def train_model(self, X_train, Y_train, epochs=50, batch_size=32, save_path="tcn_weights.h5"):
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        from tensorflow.keras import backend as K
        
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Training TCN...")
        self.model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint, early_stop]
        )
        
        print("Training complete, explicitly clearing Keras session...")
        K.clear_session()


class TCNDataLoader:
    def __init__(self, labels_path, videos_dir):
        import cv2
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from core.pose_module import PoseDetector
        self.labels_path = labels_path
        self.videos_dir = videos_dir
        self.detector = PoseDetector()
        self.recorder = DataRecorder()

    def parse_labels(self):
        import glob
        label_map = {}
        
        # Determine if labels_path is a directory or file
        if os.path.exists(self.labels_path) and os.path.isdir(self.labels_path):
            json_files = []
            for root, dirs, files in os.walk(self.labels_path):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))
        else:
            json_files = [self.labels_path]
            
        # Aggregate the individual error JSONs into a single temporal label mapping (vid_id -> intervals)
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for vid_id, errors in data.items():
                    if vid_id not in label_map:
                        label_map[vid_id] = []
                    
                    if isinstance(errors, list):
                        label_map[vid_id].extend(errors)
            except Exception as e:
                print(f"Error parsing {json_file}: {e}")
                
        return label_map

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback to 30 FPS if unable to determine
        
        sequence = []
        timestamps = []
        frame_idx = 0
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
                
            # Extract landmarks
            lm_list = self.detector.findPosition(self.detector.findPose(img, draw=False), draw=False)
            
            # Format lm_list for normalizer
            if len(lm_list) >= 33:
                landmarks = []
                for idx in range(33):
                    landmarks.append({'x': lm_list[idx][1], 'y': lm_list[idx][2]})
                
                normalized = self.recorder.normalize_landmarks(landmarks)
                sequence.append(normalized)
                timestamps.append(frame_idx / fps)
                
            frame_idx += 1
            
        cap.release()
        return sequence, timestamps
        
    def generate_training_data(self):
        label_map = self.parse_labels()
        X = []
        Y = []
        
        for vid_id, error_intervals in label_map.items():
            video_file = os.path.join(self.videos_dir, f"{vid_id}.mp4")
            if os.path.exists(video_file):
                print(f"Processing {video_file}...")
                sequence, timestamps = self.process_video(video_file)
                # Create chunks of sequence_length
                seq_len = 30
                if len(sequence) >= seq_len:
                    for i in range(len(sequence) - seq_len + 1):
                        seq_chunk = sequence[i:i+seq_len]
                        
                        # Determine if this sequence overlaps with any error interval
                        chunk_start = timestamps[i]
                        chunk_end = timestamps[i+seq_len-1]
                        
                        is_error = False
                        for interval in error_intervals:
                            if len(interval) >= 2:
                                e_start, e_end = interval[0], interval[1]
                                # Check for overlap: chunk_start <= e_end and chunk_end >= e_start
                                if chunk_start <= e_end and chunk_end >= e_start:
                                    is_error = True
                                    break
                        
                        # Temporal Label Stream: 0.0 for Bad/Error, 1.0 for Good (No Error)
                        target_score = 0.0 if is_error else 1.0
                        
                        X.append(seq_chunk)
                        Y.append(target_score)
                        
        return np.array(X), np.array(Y)
