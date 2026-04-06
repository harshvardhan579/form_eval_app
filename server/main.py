from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
import json
import time
import math
import asyncio
from core.exercise_logic import BicepCurl, Squat
from db.db_manager import DBManager
from services.data_recorder import DataRecorder
from services.recorder_service import RecorderService
from models.autoencoder_model import FormAutoencoder
from models.tcn_model import TCNModel

# Initialize models globally (load weights if available)
auto_model = FormAutoencoder(model_path="data/autoencoder_weights.h5")
tcn_model = TCNModel(model_path="data/tcn_weights.h5")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/exercise")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Unique instances per connection (isolated session)
    db = DBManager(db_path="data/fitness_data.db")
    data_recorder = DataRecorder(base_dir="data")
    recorder = RecorderService(base_dir="data")
    exercises = {
        "Bicep Curl": BicepCurl(),
        "Squat": Squat()
    }
    
    current_exercise_name = "Bicep Curl"
    current_exercise = exercises[current_exercise_name]
    session_start_time = time.time()
    
    try:
        while True:
            data_text = await websocket.receive_text()
            
            # 1. Fast-drain the buffer to eliminate WebSocket queuing (Stale Drop Logic)
            batch = [data_text]
            try:
                while True:
                    batch.append(await asyncio.wait_for(websocket.receive_text(), timeout=0.001))
            except asyncio.TimeoutError:
                pass
                
            # 2. Separate routing: Execute all controls in sequence, but keep ONLY the absolute newest landmarks payload
            landmarks_payload = None
            control_payloads = []
            
            for txt in batch:
                payload = json.loads(txt)
                if payload.get("type") == "landmarks":
                    landmarks_payload = payload
                else:
                    control_payloads.append(payload)
                    
            # 3. Process Control Signals First
            for payload in control_payloads:
                msg_type = payload.get("type")
                
                if msg_type == "set_exercise":
                    new_ex = payload.get("exercise")
                    if new_ex in exercises and new_ex != current_exercise_name:
                        duration = time.time() - session_start_time
                        if current_exercise.count > 0 or duration > 10:
                            db.log_session(
                                current_exercise_name, 
                                math.floor(current_exercise.count), 
                                int(duration),
                                getattr(current_exercise, 'knee_cave_incidents', 0),
                                getattr(current_exercise, 'shoulder_sway_incidents', 0)
                            )
                            await websocket.send_json({"type": "session_saved", "message": "SAVED"})
                        
                        current_exercise_name = new_ex
                        current_exercise = exercises[new_ex]
                        current_exercise.reset_state()
                        data_recorder.current_sequence = []
                        session_start_time = time.time()
                        
                elif msg_type == "reset":
                    current_exercise.reset_state()
                    session_start_time = time.time()
                    
                elif msg_type == "recording_control":
                    action = payload.get("action", "").upper()
                    if action == "START":
                        rec_exercise = payload.get("exercise", current_exercise_name)
                        rec_label = payload.get("label", "Perfect")
                        session_start_time = time.time()
                        current_reps = 0
                        current_exercise.reset_state()
                        recorder.start_recording(rec_exercise, rec_label)
                        await websocket.send_json({
                            "type": "recording_started",
                            "exercise": rec_exercise,
                            "label": rec_label
                        })
                    elif action == "STOP":
                        result = recorder.stop_and_save()
                        duration_seconds = int(time.time() - session_start_time)
                        db.log_session(
                            current_exercise_name,
                            math.floor(current_exercise.count),
                            duration_seconds,
                            getattr(current_exercise, 'knee_cave_incidents', 0),
                            getattr(current_exercise, 'shoulder_sway_incidents', 0)
                        )
                        await websocket.send_json({
                            "type": "RECORDING_SAVED",
                            "frames_saved": result["frames_saved"]
                        })
                    elif action == "CANCEL":
                        recorder.cancel()
                        await websocket.send_json({"type": "recording_cancelled"})
            
            # 4. Process ONLY the most recent Landmarks (discarding all stale ones)
            if landmarks_payload:
                landmarks = landmarks_payload.get("landmarks", [])
                
                if landmarks:
                    current_exercise.process(landmarks)
                    data_recorder.append_frame(landmarks)
                    recorder.add_frame(landmarks)
                    
                    if len(data_recorder.current_sequence) > 60:
                        data_recorder.current_sequence = data_recorder.current_sequence[-30:]
                        
                    seq = data_recorder.current_sequence
                    
                    # Core ML execution: Continue propagating sequences to the Models, 
                    # but strip all heavy TCN/Anomaly scoring from the socket return payload for M1 Optimization
                    if len(seq) >= 30:
                        loop = asyncio.get_event_loop()
                        if current_exercise_name == "Squat":
                            future_tcn = loop.run_in_executor(None, tcn_model.predict_quality, seq)
                            future_auto = loop.run_in_executor(None, auto_model.predict_anomaly, seq)
                            await asyncio.gather(future_tcn, future_auto)
                        else:
                            future_auto = loop.run_in_executor(None, auto_model.predict_anomaly, seq)
                            await future_auto
                            
                # 5. M1 Optimization: Strip dead-weight JSON out of the Hot-Loop Response
                await websocket.send_json({
                    "reps": current_exercise.count,
                    "rep_percent": getattr(current_exercise, 'rep_percent', 0),
                    "feedback": current_exercise.feedback,
                    "active_angle": getattr(current_exercise, 'active_angle', None),
                    "active_joint_coords": getattr(current_exercise, 'active_joint_coords', None)
                })
                
    except WebSocketDisconnect:
        # WebSocket Cleanup: Robust disconnect handling to ensure data is not lost
        duration = time.time() - session_start_time
        if current_exercise.count > 0 or duration > 5:
            db.log_session(
                current_exercise_name, 
                math.floor(current_exercise.count), 
                int(duration),
                getattr(current_exercise, 'knee_cave_incidents', 0),
                getattr(current_exercise, 'shoulder_sway_incidents', 0)
            )
            print(f"Session saved on disconnect: {current_exercise_name} - {current_exercise.count} reps")


@app.get("/api/sessions")
def get_recent_sessions():
    """Retrieve the most recent sessions."""
    db = DBManager()
    try:
        rows = db.get_recent_sessions(limit=10)
        return rows
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

