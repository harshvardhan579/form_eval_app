
import tkinter as tk
from tkinter import ttk, Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import time
import datetime
import pose_module as pm
import exercise_logic as el
import db_manager as dbm
import sys
import math

class TrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("M1 AI Exercise Trainer")
        self.root.geometry("1400x900")
        
        # Initialize Modules
        self.detector = pm.PoseDetector(detectionCon=0.7, trackCon=0.7)
        self.db = dbm.DBManager()
        
        # Exercises
        self.exercises = {
            "Bicep Curl": el.BicepCurl(),
            "Squat": el.Squat()
        }
        self.current_exercise_name = "Bicep Curl"
        self.current_exercise = self.exercises[self.current_exercise_name]
        
        # Session State
        self.session_start = time.time()
        self.current_session_reps = 0
        
        # Calibration State
        self.is_calibrating = False
        self.calibration_start = 0
        self.calibration_data = [] # List of (height, hip_width) tuples
        
        # Camera Setup (Safe Init)
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Camera could not be opened.")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not initialize camera: {e}\nPlease check permissions.")
            sys.exit()
        
        # UI Setup
        self.setup_ui()
        
        # FPS Control
        self.target_fps = 30
        self.frame_duration = 1.0 / self.target_fps
        
        # Start Loop
        self.video_loop()

    def setup_ui(self):
        # Control Panel
        control_frame = tk.Frame(self.root, bg="#333", width=300, bd=5, relief="ridge") # Added border/padding
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5) # Added padding to pack
        
        lbl_title = tk.Label(control_frame, text="Controls", bg="#333", fg="white", font=("Arial", 20, "bold"))
        lbl_title.pack(pady=30)
        
        # Exercise Selector
        lbl_select = tk.Label(control_frame, text="Select Exercise:", bg="#333", fg="#ccc", font=("Arial", 12))
        lbl_select.pack(pady=(10, 5))
        
        self.ex_var = tk.StringVar(value=self.current_exercise_name)
        cbox = ttk.Combobox(control_frame, textvariable=self.ex_var, values=list(self.exercises.keys()), state="readonly", font=("Arial", 12))
        cbox.bind("<<ComboboxSelected>>", self.change_exercise)
        cbox.pack(pady=10, ipady=5, padx=20)
        
        # Calibration
        btn_calib = tk.Button(control_frame, text="Calibrate (3s)", command=self.start_calibration, 
                              bg="#555", fg="white", font=("Arial", 12), highlightbackground="#333")
        btn_calib.pack(pady=20, fill=tk.X, padx=20)
        
        # Stats
        # Stats
        self.lbl_reps = tk.Label(control_frame, text="Reps: 0", bg="#333", fg="white", font=("Arial", 30, "bold")) 
        self.lbl_reps.pack(pady=10)
        
        # Feedback (Made Prominent)
        self.lbl_feedback = tk.Label(control_frame, text="Start Exercise", bg="#333", fg="#ffcc00", font=("Arial", 28, "bold"), wraplength=280)
        self.lbl_feedback.pack(pady=10)
        
        # Rep Quality Feedback
        self.lbl_rep_feedback = tk.Label(control_frame, text="Waiting for first rep...", bg="#333", fg="#00ffcc", font=("Arial", 16, "italic"), wraplength=280)
        self.lbl_rep_feedback.pack(pady=10)
        
        # Reset Button
        btn_reset = tk.Button(control_frame, text="RESET REPS", command=self.reset_exercise,
                              bg="#ff4444", fg="black", font=("Arial", 14, "bold")) # High contrast reset button
        btn_reset.pack(pady=10, fill=tk.X, padx=20)
        
        # Quit
        btn_quit = tk.Button(control_frame, text="Quit & Save", command=self.on_closing, 
                             bg="#900", fg="white", font=("Arial", 14, "bold"))
        btn_quit.pack(side=tk.BOTTOM, pady=40, fill=tk.X, padx=20)
        
        # Video Frame
        self.canvas = tk.Label(self.root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start = time.time()
        self.calibration_data = []
        self.lbl_feedback.config(text="CALIBRATING... STAND STILL")

    def change_exercise(self, event):
        new_ex = self.ex_var.get()
        if new_ex in self.exercises:
            # Log previous session
            duration = time.time() - self.session_start
            session_id = self.db.log_session(self.current_exercise_name, self.current_exercise.count, duration)
            self.db.export_session_summary(session_id)
            
            # Switch
            self.current_exercise_name = new_ex
            self.current_exercise = self.exercises[new_ex]
            
            # Reset counters
            self.current_exercise.reset_state()
            self.session_start = time.time()
            self.lbl_reps.config(text="Reps: 0")
            self.lbl_feedback.config(text="Feedback: Session Saved")

    def reset_exercise(self):
        self.current_exercise.reset_state()
        self.lbl_reps.config(text="Reps: 0")
        self.lbl_feedback.config(text="Start Exercise")
        self.lbl_rep_feedback.config(text="Waiting for first rep...")

    def video_loop(self):
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if ret:
            # 1. Process Frame
            frame = cv2.resize(frame, (1080, 720)) # Keep consistent size
            frame = self.detector.findPose(frame, draw=False)
            lm_list = self.detector.findPosition(frame, draw=False) # Get Normalized Coords
            
            if len(lm_list) != 0:
                # Calibration Logic (omitted for brevity in search, but exists)
                if self.is_calibrating:
                    elapsed = time.time() - self.calibration_start
                    if elapsed < 3:
                        self.lbl_feedback.config(text=f"Calibrating... {3-int(elapsed)}")
                        # Collect Data: Height (Nose to Ankle approx) and Hip Width
                        # Nose: 0
                        # Left Ankle: 27, Right Ankle: 28
                        # Left Hip: 23, Right Hip: 24
                        try:
                            # Hip Width
                            x23, y23 = lm_list[23][1], lm_list[23][2]
                            x24, y24 = lm_list[24][1], lm_list[24][2]
                            hip_w = math.hypot(x23-x24, y23-y24)
                            
                            # Height (Nose to Mid-Ankle)
                            x0, y0 = lm_list[0][1], lm_list[0][2]
                            x27, y27 = lm_list[27][1], lm_list[27][2]
                            x28, y28 = lm_list[28][1], lm_list[28][2]
                            mid_ankle_x = (x27 + x28) / 2
                            mid_ankle_y = (y27 + y28) / 2
                            height = math.hypot(x0-mid_ankle_x, y0-mid_ankle_y)
                            
                            self.calibration_data.append((height, hip_w))
                        except Exception:
                            pass
                    else:
                        # Finish Calibration
                        self.is_calibrating = False
                        if self.calibration_data:
                            avg_h = sum(d[0] for d in self.calibration_data) / len(self.calibration_data)
                            avg_w = sum(d[1] for d in self.calibration_data) / len(self.calibration_data)
                            
                            # Update all exercises with calibration
                            for ex in self.exercises.values():
                                if hasattr(ex, 'set_calibration'):
                                    ex.set_calibration(avg_h, avg_w)
                            
                            self.lbl_feedback.config(text="CALIBRATION COMPLETE")
                        else:
                            self.lbl_feedback.config(text="CALIBRATION FAILED - No User")

                else:
                    # Normal Exercise Logic
                    self.current_exercise.process(self.detector, frame)
                    reps = self.current_exercise.count
                    
                    self.lbl_reps.config(text=f"Reps: {int(reps)}")
                    
                    fb = getattr(self.current_exercise, 'feedback', '-')
                    self.lbl_feedback.config(text=fb)
                    
                    rep_fb = getattr(self.current_exercise, 'rep_feedback', '')
                    self.lbl_rep_feedback.config(text=rep_fb)
                    
            # Convert to Tkinter Image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.imgtk = img_tk
            self.canvas.config(image=img_tk)
        
        # 3. FPS Cap logic
        process_time = time.time() - start_time
        delay = max(10, int((self.frame_duration - process_time) * 1000))
        
        self.root.after(delay, self.video_loop)

    def on_closing(self):
        # Save current session
        duration = time.time() - self.session_start
        session_id = self.db.log_session(self.current_exercise_name, self.current_exercise.count, duration)
        self.db.export_session_summary(session_id)
        
        # Cleanup
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
