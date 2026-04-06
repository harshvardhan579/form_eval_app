# AI Form Evaluator (V2 Production)

A real-time, zero-latency computer vision application that evaluates exercise biomechanics (Squats & Bicep Curls) directly in the browser. 

This is a complete architectural rewrite of my V1 project, transitioning from a static file-upload API to a live WebSocket streaming HUD.

## 🚀 Key Upgrades in V2
* **Real-Time Inference:** Replaced standard REST endpoints with a WebSocket pipeline (`wss://`), dropping stale frames to achieve 30FPS zero-latency feedback.
* **Serverless Cloud Architecture:** Migrated backend from local SQLite to a cloud Postgres database (Supabase) and containerized the Python backend via Docker for deployment on Railway.
* **Venture-Grade HUD:** Built a custom React frontend with a CSS Grid 'zero-scroll' architecture designed for 6-foot distance visibility, complete with dynamic HTML Canvas skeleton overlays.
* **ML Hysteresis/Debouncing:** Implemented rising-edge debouncing in the physics engine to filter out coordinate noise and prevent false-positive fault detection.

## 🛠️ Tech Stack
* **Frontend:** React, Vite, WebSockets, HTML5 Canvas
* **Backend:** Python, FastAPI, OpenCV, MediaPipe
* **Database:** Postgres (Supabase)

## 🏃‍♂️ Coming Next
Training a Temporal Convolutional Network (TCN) on the `Fitness-AQA` dataset to replace the current rules-based heuristic engine with a fully supervised Quality Scoring model.
