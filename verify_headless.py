
import cv2
import numpy as np
import pose_module as pm
import exercise_logic as el
import db_manager as dbm
import os

def test_imports():
    print("Testing Imports...")
    try:
        import mediapipe
        import tkinter
        import PIL
        print("Imports Successful.")
    except ImportError as e:
        print(f"Import Error: {e}")
        return False
    return True

def test_logic():
    print("Testing Logic on Mock Landmarks...")
    # Initialize Detector & Exercise
    detector = pm.PoseDetector()
    curl = el.BicepCurl()
    
    # Mock Landmarks (Blank Image)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Initialize with all zero list
    # Format: [id, x, y, z, vis]
    mock_lm = [[i, 0.5, 0.5, 0.0, 1.0] for i in range(33)]
    
    # Simulate Right Arm UP (Angle ~20) - Should set dir=1
    # Shoulder (12) at (0.8, 0.2)
    # Elbow (14) at (0.8, 0.5)
    # Wrist (16) at (0.8, 0.21) -> Folded up close to shoulder
    mock_lm[12] = [12, 0.8, 0.2, 0.0, 1.0]
    mock_lm[14] = [14, 0.8, 0.5, 0.0, 1.0]
    mock_lm[16] = [16, 0.8, 0.21, 0.0, 1.0] 
    
    detector.lm_list = mock_lm
    
    print("Step 1: Initial State (R: Up)")
    curl.process(detector, img)
    print(f"Total Count: {curl.count}")
    
    # Should be 0.5 (Half Rep Up)
    if curl.count != 0.5:
        print(f"FAILED: Expected 0.5, got {curl.count}")
        return False
        
    # Step 2: Right Arm DOWN (Angle ~180) - Should set dir=0 and complete rep?
    # Actually BicepCurl Logic: 
    # > 160: dir=0
    # < 30: dir=1, count+=0.5
    # > 160 (again): dir=0, count+=0.5 (Full rep on way back if logic does that?)
    # Let's check logic: 
    # if angle > 160: if dir==1: count+=0.5, dir=0.
    
    mock_lm[16] = [16, 0.8, 0.8, 0.0, 1.0] # Straight down
    detector.lm_list = mock_lm
    
    print("Step 2: Swap State (R: Down)")
    curl.process(detector, img)
    print(f"Total Count: {curl.count}")
    
    if curl.count != 1.0:
        print(f"FAILED: Expected 1.0, got {curl.count}")
        return False

    print("Logic Verification PASSED.")
    return True

def test_db():
    print("Testing Database...")
    try:
        db = dbm.DBManager("test_fitness.db")
        with db as conn:
            conn.log_session("Test Exercise", 10, 60)
        print("Database Write Successful.")
        
        # Verify
        reps = db.get_total_reps("Test Exercise")
        print(f"Total Reps Retrieved: {reps}")
        
        os.remove("test_fitness.db")
    except Exception as e:
        print(f"Database Error: {e}")
        return False
    return True

if __name__ == "__main__":
    if test_imports() and test_logic() and test_db():
        print("ALL HEADLESS TESTS PASSED.")
    else:
        print("TESTS FAILED.")
