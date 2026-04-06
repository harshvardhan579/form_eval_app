"""Unit tests for RecorderService."""

import os
import tempfile
import numpy as np
import pytest

from services.recorder_service import RecorderService


# -- helpers -----------------------------------------------------------------

def _make_landmarks(n=33):
    """Return a list of 33 fake landmarks (dicts with x, y, z, visibility)."""
    return [{"x": 0.5 + i * 0.001, "y": 0.5 + i * 0.001, "z": 0.0, "visibility": 1.0}
            for i in range(n)]


# -- tests -------------------------------------------------------------------

class TestRecorderService:

    def test_start_sets_recording_flag(self):
        svc = RecorderService(base_dir=tempfile.mkdtemp())
        assert svc.is_recording is False
        svc.start_recording("Squat", "Perfect")
        assert svc.is_recording is True

    def test_add_frame_only_when_recording(self):
        svc = RecorderService(base_dir=tempfile.mkdtemp())
        lm = _make_landmarks()

        # Not recording — frame discarded
        svc.add_frame(lm)
        assert svc.frame_count == 0

        # Recording — frame stored
        svc.start_recording("Squat", "Perfect")
        svc.add_frame(lm)
        assert svc.frame_count == 1

    def test_stop_and_save_creates_npy(self):
        tmp = tempfile.mkdtemp()
        svc = RecorderService(base_dir=tmp)
        svc.start_recording("Squat", "Perfect")
        for _ in range(10):
            svc.add_frame(_make_landmarks())

        result = svc.stop_and_save()
        assert result["frames_saved"] == 10
        assert os.path.isfile(result["filepath"])

        # Validate array shape (10 frames x 30 features)
        arr = np.load(result["filepath"])
        assert arr.shape == (10, 30)
        assert arr.dtype == np.float32

    def test_buffer_cleared_after_save(self):
        tmp = tempfile.mkdtemp()
        svc = RecorderService(base_dir=tmp)
        svc.start_recording("Squat", "Perfect")
        for _ in range(5):
            svc.add_frame(_make_landmarks())

        svc.stop_and_save()
        assert svc.frame_count == 0
        assert svc.is_recording is False

    def test_cancel_clears_without_saving(self):
        tmp = tempfile.mkdtemp()
        svc = RecorderService(base_dir=tmp)
        svc.start_recording("Bicep Curl", "Flawed")
        for _ in range(5):
            svc.add_frame(_make_landmarks())

        svc.cancel()
        assert svc.frame_count == 0
        assert svc.is_recording is False

        # No file should exist
        target_dir = os.path.join(tmp, "training_data", "Flawed", "Bicep_Curl")
        assert not os.path.exists(target_dir)

    def test_directory_structure(self):
        tmp = tempfile.mkdtemp()
        svc = RecorderService(base_dir=tmp)
        svc.start_recording("Bicep Curl", "Perfect")
        svc.add_frame(_make_landmarks())
        result = svc.stop_and_save()

        # Check path structure
        expected_dir = os.path.join(tmp, "training_data", "Perfect", "Bicep_Curl")
        assert os.path.isdir(expected_dir)
        assert result["filepath"].startswith(expected_dir)

    def test_stop_empty_buffer(self):
        svc = RecorderService(base_dir=tempfile.mkdtemp())
        svc.start_recording("Squat", "Perfect")
        result = svc.stop_and_save()
        assert result["frames_saved"] == 0
        assert result["filepath"] == ""
