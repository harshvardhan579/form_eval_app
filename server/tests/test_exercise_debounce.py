"""Tests for incident debouncing in BicepCurl and Squat."""

import pytest
from core.exercise_logic import BicepCurl, Squat


# -- helpers -----------------------------------------------------------------

def _base_landmarks(n=33):
    """Return 33 neutral landmarks at center (0.5, 0.5)."""
    return [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(n)]


def _curl_landmarks(angle_hint="down", sway=False):
    """
    Build landmarks that drive BicepCurl state machine.
    angle_hint: 'down' (~170°), 'up' (~20°)
    sway: shift shoulder x to trigger sway detection
    """
    lm = _base_landmarks()
    # Shoulder (12) and Hip (23/24) for torso length
    lm[11] = {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 1.0}  # L Shoulder
    lm[12] = {"x": 0.6, "y": 0.3, "z": 0.0, "visibility": 1.0}  # R Shoulder
    lm[23] = {"x": 0.4, "y": 0.6, "z": 0.0, "visibility": 1.0}  # L Hip
    lm[24] = {"x": 0.6, "y": 0.6, "z": 0.0, "visibility": 1.0}  # R Hip

    if angle_hint == "down":
        # Arm straight: 12(shoulder) -> 14(elbow) -> 16(wrist) ~170°
        lm[14] = {"x": 0.6, "y": 0.45, "z": 0.0, "visibility": 1.0}
        lm[16] = {"x": 0.6, "y": 0.58, "z": 0.0, "visibility": 1.0}
    else:  # up
        # Arm curled: angle ~20°
        lm[14] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 1.0}
        lm[16] = {"x": 0.6, "y": 0.28, "z": 0.0, "visibility": 1.0}

    if sway:
        # Shift shoulder far from initial position to trigger sway
        lm[12] = {"x": 0.9, "y": 0.3, "z": 0.0, "visibility": 1.0}

    return lm


def _squat_landmarks(angle_hint="standing", cave=False):
    """
    Build landmarks that drive Squat state machine.
    angle_hint: 'standing' (~170°), 'deep' (~80°)
    cave: move knees inward to trigger knee cave
    """
    lm = _base_landmarks()
    lm[11] = {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 1.0}  # L Shoulder
    lm[23] = {"x": 0.45, "y": 0.6, "z": 0.0, "visibility": 1.0}  # L Hip
    lm[24] = {"x": 0.55, "y": 0.6, "z": 0.0, "visibility": 1.0}  # R Hip

    if angle_hint == "standing":
        # Right leg straight: 24(hip)->26(knee)->28(ankle) ~170°
        lm[26] = {"x": 0.55, "y": 0.75, "z": 0.0, "visibility": 1.0}
        lm[28] = {"x": 0.55, "y": 0.9, "z": 0.0, "visibility": 1.0}
    else:  # deep squat
        lm[26] = {"x": 0.55, "y": 0.7, "z": 0.0, "visibility": 1.0}
        lm[28] = {"x": 0.45, "y": 0.65, "z": 0.0, "visibility": 1.0}

    if cave:
        # Knees very close together
        lm[25] = {"x": 0.49, "y": 0.75, "z": 0.0, "visibility": 1.0}
        lm[26] = {"x": 0.51, "y": 0.75, "z": 0.0, "visibility": 1.0}
    else:
        lm[25] = {"x": 0.45, "y": 0.75, "z": 0.0, "visibility": 1.0}
        lm[26] = {"x": 0.55, "y": 0.75, "z": 0.0, "visibility": 1.0}

    return lm


# -- tests -------------------------------------------------------------------

class TestBicepCurlDebounce:

    def test_sway_incident_counts_once_per_occurrence(self):
        """Sustained sway over many frames should only count as 1 incident."""
        curl = BicepCurl()
        # Prime with a normal frame so initial_shoulder_x is set to non-swayed position
        curl.process(_curl_landmarks(angle_hint="up", sway=False))
        curl.is_moving = True

        # Sway for 15 frames — should only count as 1 incident
        for _ in range(15):
            curl.process(_curl_landmarks(angle_hint="up", sway=True))

        assert curl.shoulder_sway_incidents == 1, (
            f"Expected 1 incident, got {curl.shoulder_sway_incidents}"
        )

    def test_sway_two_occurrences(self):
        """Sway → clear → sway should count as 2 incidents."""
        curl = BicepCurl()
        # Prime with a normal frame so initial_shoulder_x is set
        curl.process(_curl_landmarks(angle_hint="up", sway=False))
        curl.is_moving = True

        # First sway episode (10 frames)
        for _ in range(10):
            curl.process(_curl_landmarks(angle_hint="up", sway=True))

        # Clear the sway (10 frames, no sway)
        for _ in range(10):
            curl.process(_curl_landmarks(angle_hint="up", sway=False))

        # Second sway episode (10 frames)
        for _ in range(10):
            curl.process(_curl_landmarks(angle_hint="up", sway=True))

        assert curl.shoulder_sway_incidents == 2, (
            f"Expected 2 incidents, got {curl.shoulder_sway_incidents}"
        )

    def test_reset_clears_incidents(self):
        """reset_state must zero out incident counters and flags."""
        curl = BicepCurl()
        curl.shoulder_sway_incidents = 5
        curl.knee_cave_incidents = 3
        curl._is_swaying = True
        curl._is_caving = True

        curl.reset_state()

        assert curl.shoulder_sway_incidents == 0
        assert curl.knee_cave_incidents == 0
        assert curl._is_swaying is False
        assert curl._is_caving is False


class TestSquatDebounce:

    def test_knee_cave_incident_counts_once(self):
        """Sustained knee cave should only count as 1 incident."""
        squat = Squat()
        squat.is_moving = True

        for _ in range(15):
            squat.process(_squat_landmarks(angle_hint="deep", cave=True))

        assert squat.knee_cave_incidents == 1, (
            f"Expected 1 incident, got {squat.knee_cave_incidents}"
        )

    def test_knee_cave_two_occurrences(self):
        """Cave → clear → cave should count as 2 incidents."""
        squat = Squat()
        squat.is_moving = True

        for _ in range(10):
            squat.process(_squat_landmarks(angle_hint="deep", cave=True))
        for _ in range(10):
            squat.process(_squat_landmarks(angle_hint="deep", cave=False))
        for _ in range(10):
            squat.process(_squat_landmarks(angle_hint="deep", cave=True))

        assert squat.knee_cave_incidents == 2, (
            f"Expected 2 incidents, got {squat.knee_cave_incidents}"
        )

    def test_reset_clears_squat_incidents(self):
        """reset_state must zero out knee cave incident counters."""
        squat = Squat()
        squat.knee_cave_incidents = 4
        squat._is_caving = True

        squat.reset_state()

        assert squat.knee_cave_incidents == 0
        assert squat._is_caving is False
