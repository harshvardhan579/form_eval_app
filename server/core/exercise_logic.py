import math
import time

class Exercise:
    def __init__(self):
        self.count = 0
        self.dir = 0
        self.p_count = 0 
        self.feedback = "Start the rep"
        self.rep_feedback = "Waiting for first rep..."
        self.rep_start_time = 0
        self.is_moving = False
        self.calibrated_height = 0
        self.calibrated_hip_width = 0
        self.is_calibrated = False
        self.torso_length = 1.0 # default to avoid div/0
        self.rep_percent = 0
        self.previous_angle = None
        self.active_angle = None
        self.active_joint_coords = None
        
        # Phase 2 Hysteresis Buffers (Consecutive frames)
        self.knee_cave_frames = 0
        self.shoulder_sway_frames = 0
        # Debounced incident counters (increment once per occurrence, not per frame)
        self.shoulder_sway_incidents = 0
        self.knee_cave_incidents = 0
        self._is_swaying = False
        self._is_caving = False
        self.shoulder_normal_frames = 0
        self.knee_normal_frames = 0

    def reset_state(self):
        self.count = 0
        self.dir = 0
        self.feedback = "Start the rep"
        self.rep_feedback = "Waiting for first rep..."
        self.rep_start_time = 0
        self.is_moving = False
        self.rep_percent = 0
        self.previous_angle = None
        self.active_angle = None
        self.active_joint_coords = None
        # Reset incident tracking
        self.knee_cave_frames = 0
        self.shoulder_sway_frames = 0
        self.shoulder_sway_incidents = 0
        self.knee_cave_incidents = 0
        self._is_swaying = False
        self._is_caving = False
        self.shoulder_normal_frames = 0
        self.knee_normal_frames = 0

    def calculate_angle(self, landmarks, p1, p2, p3):
        try:
            x1, y1 = landmarks[p1]['x'], landmarks[p1]['y']
            x2, y2 = landmarks[p2]['x'], landmarks[p2]['y']
            x3, y3 = landmarks[p3]['x'], landmarks[p3]['y']
            
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360
            if angle > 180:
                angle = 360 - angle
                
            if getattr(self, 'previous_angle', None) is None:
                self.previous_angle = angle
            else:
                self.previous_angle = (0.85 * angle) + (0.15 * self.previous_angle)
                
            return self.previous_angle
        except Exception:
            return 0

    def set_calibration(self, height, hip_width):
        self.calibrated_height = height
        self.calibrated_hip_width = hip_width
        self.is_calibrated = True
        
    def calculate_torso_length(self, landmarks):
        # Using Torso Length (Distance between L-Shoulder 11 to L-Hip 23) as robust distance measurement
        try:
            x11, y11 = landmarks[11]['x'], landmarks[11]['y']
            x23, y23 = landmarks[23]['x'], landmarks[23]['y']
            return math.hypot(x11 - x23, y11 - y23)
        except Exception:
            return 1.0

class BicepCurl(Exercise):
    def __init__(self):
        super().__init__()
        
    def process(self, landmarks):
        # Frame Guard for Bicep Curls
        try:
            if any(landmarks[i].get('visibility', 0.0) < 0.5 for i in [11, 12, 15, 16]):
                self.feedback = "OUT_OF_FRAME"
                return int(self.count)
        except Exception:
            self.feedback = "OUT_OF_FRAME"
            return int(self.count)
            
        # Update torso length for normalization
        self.torso_length = self.calculate_torso_length(landmarks)
        
        # Right Arm: 12 (Shoulder), 14 (Elbow), 16 (Wrist)
        angle = self.calculate_angle(landmarks, 12, 14, 16)
        self.active_angle = round(angle, 1)
        self.active_joint_coords = {'x': landmarks[14]['x'], 'y': landmarks[14]['y']}
        
        # Track Amount of Rep Completed (0% to 100%)
        if self.dir == 0: # Phase 1: Curl Up (Angle 160 -> 30) maps to 0% -> 50%
            rep_percent = max(0, min(50, int(((160 - angle) / 130) * 50)))
        else: # Phase 2: Lower Down (Angle 30 -> 160) maps to 50% -> 100%
            rep_percent = max(50, min(100, int(50 + ((angle - 30) / 130) * 50)))
        self.rep_percent = rep_percent
        
        # Start tracking time when leaving the bottom position
        if self.dir == 0 and angle < 160 and not self.is_moving:
            self.rep_start_time = time.time()
            self.is_moving = True
            
        # Sway Detection: track shoulder 11 relative to torso
        try:
            x11_current = landmarks[11]['x']
            x12_current = landmarks[12]['x']
            shoulder_width = abs(x11_current - x12_current)
            
            if not hasattr(self, 'initial_shoulder_x'):
                self.initial_shoulder_x = x12_current
                
            # Sway tolerance is 15% of torso length
            if abs(x12_current - self.initial_shoulder_x) > (self.torso_length * 0.15):
                self.shoulder_sway_frames += 1
                self.shoulder_normal_frames = 0
            else:
                self.shoulder_sway_frames = 0
                self.shoulder_normal_frames += 1
                
            if self.is_moving and self.shoulder_sway_frames >= 5:
                 self.feedback = "Keep Shoulders Still"
                 if not self._is_swaying:
                     self.shoulder_sway_incidents += 1
                     self._is_swaying = True
                     
            if self._is_swaying and self.shoulder_normal_frames >= 5:
                 self._is_swaying = False
        except Exception:
            pass

        # State Machine
        if angle > 160:
            if getattr(self, 'shoulder_sway_frames', 0) < 5:
                self.feedback = "Curl Up"
            self.initial_shoulder_x = landmarks[12]['x'] if 12 in landmarks else getattr(self, 'initial_shoulder_x', 0)
            
            if self.dir == 1: # Was up, now down
                self.count += 0.5
                self.dir = 0 # Down
                self.is_moving = False
                
                # Rep completed, calculate quality
                if self.rep_start_time > 0:
                    rep_duration = time.time() - self.rep_start_time
                    if rep_duration < 1.5:
                        self.rep_feedback = "Rep too fast! Slow down."
                    elif rep_duration > 5.0:
                        self.rep_feedback = "A bit too slow! Maintain speed."
                    else:
                        self.rep_feedback = "Excellent rep quality!"
                
        if angle < 30:
            if getattr(self, 'shoulder_sway_frames', 0) < 5:
                 self.feedback = "Good! Lower Down"
            if self.dir == 0: # Was down, now up
                self.count += 0.5
                self.dir = 1 # Up
        
        # Intermediate/Guidance
        if 30 <= angle <= 160:
            if getattr(self, 'shoulder_sway_frames', 0) < 5:
                if self.dir == 0: # Down state, need to go Up
                    self.feedback = "Curl Up"
                elif self.dir == 1: # Up state, need to go Down
                    self.feedback = "Lower Slowly"
                
        return int(self.count)


class Squat(Exercise):
    def process(self, landmarks):
        # Frame Guard for Squats
        try:
            if any(landmarks[i].get('visibility', 0.0) < 0.5 for i in [11, 12, 27, 28]):
                self.feedback = "OUT_OF_FRAME"
                return int(self.count)
        except Exception:
            self.feedback = "OUT_OF_FRAME"
            return int(self.count)
            
        # Update torso length for normalization
        self.torso_length = self.calculate_torso_length(landmarks)
        
        # Right Leg: 24, 26, 28 | Left Leg: 23, 25, 27
        angle = self.calculate_angle(landmarks, 24, 26, 28)
        self.active_angle = round(angle, 1)
        self.active_joint_coords = {'x': landmarks[26]['x'], 'y': landmarks[26]['y']}
        
        # Track Amount of Rep Completed (0% to 100%)
        if self.dir == 0: # Phase 1: Squat Down (Angle 170 -> 90) maps to 0% -> 50%
            rep_percent = max(0, min(50, int(((170 - angle) / 80) * 50)))
        else: # Phase 2: Stand Up (Angle 90 -> 170) maps to 50% -> 100%
            rep_percent = max(50, min(100, int(50 + ((angle - 90) / 80) * 50)))
        self.rep_percent = rep_percent
        
        # Knee Cave Detection
        try:
            # Lowered visibility requirement for Knees to 0.3
            vis25 = landmarks[25].get('visibility', 1.0)
            vis26 = landmarks[26].get('visibility', 1.0)
            
            if vis25 >= 0.3 and vis26 >= 0.3:
                x23, y23 = landmarks[23]['x'], landmarks[23]['y'] # L Hip
                x24, y24 = landmarks[24]['x'], landmarks[24]['y'] # R Hip
                # Hips (23, 24) might be partially obscured, so we don't strict check their visibility
                hip_width = math.hypot(x23 - x24, y23 - y24)
                
                x25 = landmarks[25]['x'] # L Knee
                x26 = landmarks[26]['x'] # R Knee
                knee_distance = abs(x25 - x26)
                
                # If knees are significantly closer than hips (normalized by torso length to account for camera angles)
                # A tighter tolerance means knees cave inward.
                cave_threshold = hip_width - (self.torso_length * 0.1)
                
                if knee_distance < cave_threshold:
                     self.knee_cave_frames += 1
                     self.knee_normal_frames = 0
                else:
                     self.knee_cave_frames = 0
                     self.knee_normal_frames += 1
                     # Near-miss logging
                     if knee_distance < hip_width:
                         import sys
                         sys.stdout.write(f"[DIAGNOSTIC] Near-Miss! Knee Dist: {knee_distance:.3f} | Hip Dist: {hip_width:.3f} | Threshold: {cave_threshold:.3f}\n")
                         sys.stdout.flush()
                      
                if self.is_moving and self.knee_cave_frames >= 5:
                     self.feedback = "Knees Caving In!"
                     if not self._is_caving:
                         self.knee_cave_incidents += 1
                         self._is_caving = True
                         
                if self._is_caving and self.knee_normal_frames >= 5:
                     self._is_caving = False
        except Exception:
            pass
            
        # Standing ~ 170-180, Squat ~ < 90
        
        if self.dir == 0 and angle < 160 and not self.is_moving:
            self.rep_start_time = time.time()
            self.is_moving = True
        
        if angle > 160:
            if getattr(self, 'knee_cave_frames', 0) < 5:
                self.feedback = "Start Squat"
            if self.dir == 1: # Was down
               self.count += 0.5
               self.dir = 0
               self.is_moving = False
               
               if self.rep_start_time > 0:
                   rep_duration = time.time() - self.rep_start_time
                   if rep_duration < 2.0:
                       self.rep_feedback = "Too fast! Control your descent."
                   elif rep_duration > 6.0:
                       self.rep_feedback = "Too slow! Rise up steadily."
                   else:
                       self.rep_feedback = "Perfect rep timing!"
        
        elif angle < 90: # Deep squat
            if getattr(self, 'knee_cave_frames', 0) < 5:
                self.feedback = "Good Depth! Up"
            if self.dir == 0:
                self.count += 0.5
                self.dir = 1
        else: # Intermediate tracking
            if getattr(self, 'knee_cave_frames', 0) < 5:
                if self.dir == 0:
                    self.feedback = "Go Lower"
                else:
                    self.feedback = "Stand Up"
            
        return int(self.count)


