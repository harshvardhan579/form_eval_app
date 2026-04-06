
import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)
        self.lm_list = []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, lm.x, lm.y, lm.z, lm.visibility]) 
                # Keeping normalized coordinates (0.0 - 1.0) and visibility
                # We will convert to pixel coordinates only when needed or drawing
                if draw:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Retrieve landmarks based on id
        # User constraint: Strictly enforce coordinate indexing format
        try:
            # landmarks are in self.lm_list which is [id, x, y, z, visibility]
            # however, to be safer and compatible with pixel logic, let's get pixel coords here
            h, w, c = img.shape
            
            # Get the specific landmarks
            # Note: lm_list is a list of lists. The index in lm_list corresponds to the landmark ID
            # IF and ONLY IF the list is full. MediaPipe returns 33 landmarks always if detected.
            
            x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
            x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
            x3, y3 = self.lm_list[p3][1], self.lm_list[p3][2]
            
            # Convert to pixels for drawing and calculation 
            # (Standard angle calculation usually works better in consistent units, pixels are fine for 2D)
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            px3, py3 = int(x3 * w), int(y3 * h)

            # Calculate the Angle
            radians = math.atan2(py3 - py2, px3 - px2) - \
                      math.atan2(py1 - py2, px1 - px2)
            angle = math.degrees(radians)
            
            if angle < 0:
                angle += 360

            # Draw
            if draw:
                cv2.line(img, (px1, py1), (px2, py2), (255, 255, 255), 3)
                cv2.line(img, (px3, py3), (px2, py2), (255, 255, 255), 3)
                cv2.circle(img, (px1, py1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (px1, py1), 15, (0, 0, 255), 2)
                cv2.circle(img, (px2, py2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (px2, py2), 15, (0, 0, 255), 2)
                cv2.circle(img, (px3, py3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (px3, py3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (px2 - 50, py2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            return angle
            
        except IndexError:
             # Pose not detected or index out of bounds
             return 0
