"""
RecorderService — Stateful, memory-safe recording buffer for labeled landmark sequences.

Manages a single recording session at a time. Landmarks are normalized on ingest
and flushed to disk as .npy files organized by label/exercise.

Directory structure:
    server/data/training_data/{label}/{exercise}/{timestamp}.npy
"""

import os
import time
import numpy as np
from services.data_recorder import TARGET_LANDMARKS

# ---------------------------------------------------------------------------
# Normalization (mirrors DataRecorder.normalize_landmarks but is standalone)
# ---------------------------------------------------------------------------

def normalize_landmarks(landmarks):
    """Translate all landmarks to hip-center and scale by torso length."""
    try:
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        hip_cx = (l_hip['x'] + r_hip['x']) / 2.0
        hip_cy = (l_hip['y'] + r_hip['y']) / 2.0

        l_shoulder = landmarks[11]
        torso_len = np.hypot(
            l_shoulder['x'] - l_hip['x'],
            l_shoulder['y'] - l_hip['y'],
        )
        if torso_len < 0.001:
            torso_len = 1.0

        frame = []
        for idx in TARGET_LANDMARKS:
            if idx < len(landmarks):
                lm = landmarks[idx]
                frame.extend([
                    (lm['x'] - hip_cx) / torso_len,
                    (lm['y'] - hip_cy) / torso_len,
                ])
            else:
                frame.extend([0.0, 0.0])
        return frame
    except Exception:
        return [0.0] * (len(TARGET_LANDMARKS) * 2)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RecorderService:
    """One instance per WebSocket connection."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.is_recording: bool = False
        self._buffer: list = []
        self._exercise: str = ""
        self._label: str = ""

    # -- public API ----------------------------------------------------------

    def start_recording(self, exercise: str, label: str) -> None:
        """Begin a new recording session, discarding any prior buffer."""
        self._buffer = []          # always start clean
        self._exercise = exercise
        self._label = label
        self.is_recording = True

    def add_frame(self, landmarks: list) -> None:
        """Normalize and buffer a single frame (no-op when not recording)."""
        if not self.is_recording:
            return
        self._buffer.append(normalize_landmarks(landmarks))

    def stop_and_save(self) -> dict:
        """
        Stop recording, flush the buffer to an .npy file, and
        **immediately clear the buffer** (M1 memory safety).

        Returns:
            dict with keys: frames_saved (int), filepath (str)
        """
        self.is_recording = False
        frames = len(self._buffer)

        if frames == 0:
            self._buffer = []
            return {"frames_saved": 0, "filepath": ""}

        # Build directory: data/training_data/{label}/{exercise}/
        safe_label = self._label.replace(" ", "_")
        safe_exercise = self._exercise.replace(" ", "_")
        save_dir = os.path.join(
            self.base_dir, "training_data", safe_label, safe_exercise
        )
        os.makedirs(save_dir, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.npy"
        filepath = os.path.join(save_dir, filename)

        seq_array = np.array(self._buffer, dtype=np.float32)
        np.save(filepath, seq_array)

        # M1 memory safety: release immediately
        self._buffer = []

        return {"frames_saved": frames, "filepath": filepath}

    def cancel(self) -> None:
        """Discard the current recording without saving."""
        self.is_recording = False
        self._buffer = []

    @property
    def frame_count(self) -> int:
        return len(self._buffer)
