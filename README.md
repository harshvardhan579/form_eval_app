# AI Form Evaluator (V2 Production)

An enterprise-grade, real-time, zero-latency computer vision application designed to evaluate and correct exercise biomechanics (Squats and Bicep Curls) directly in the browser.

This is a complete production-ready rewrite of the V1 architecture, transitioning from high-latency static video analysis to a live WebSocket streaming telemetry HUD with advanced physics debouncing, diagnostic tracking, and real-time audio coaching.

---

## 🚀 Key Features

*   **Premium Dynamic Telemetry HUD:** Uses `react-resizable-panels` to present a beautiful, fully adjustable desktop workspace. Users can customize their layout by dragging borders between the live camera feed, ML configuration controls, and the massive rep counter/progress console.
*   **Stale-Frame Drop Pipeline:** Designed for M1 Macs and optimized client-server execution. Replaced REST endpoints with an active WebSocket pipeline that drops queued stale frames when system latency spikes, keeping response rates locked at a fluid 30 FPS.
*   **Dynamic Skeleton overlay:** HTML5 Canvas layer rendering coordinates directly onto the live user video stream, shifting segment colors (e.g., turning limbs red) immediately upon fault detection.
*   **Biomechanical Physics Engine:** Includes custom rules-based algorithms tracking skeletal angles, velocity, and torso-relative spatial variations.
*   **Voice Coach & Debounce Locks:** Real-time audio synthesis (Web Speech API) calls out rep counts and provides form corrections. Features built-in debounce locks to prevent overlapping audio queues.
*   **Interactive Set Recorder:** A built-in state-machine recorder (Warmup Countdown → Capture & Frame Counting → Disk Persistence) allowing users to easily record custom telemetry streams for future model training.
*   **Comprehensive Test Suite:** High-quality unit and service test coverage validating the core physics models and debounce tracking, maintaining high reliability.

---

## 🛠️ Biomechanical Physics & Correction Logic

The server-side biomechanics engine (`server/core/exercise_logic.py`) normalizes user distance and camera position by computing a dynamic **Torso Length** (distance between left shoulder and left hip). This acts as a spatial scaling baseline.

### 🏋️‍♂️ 1. Squat Biomechanics
*   **Angle Target:** Right Leg Hip-Knee-Ankle (landmarks `24` ➔ `26` ➔ `28`).
*   **Rep State Machine:**
    *   **Phase 1 (Descend):** Hip-Knee-Ankle flexion from $170^\circ \rightarrow 90^\circ$ maps to $0\% \rightarrow 50\%$ rep completion.
    *   **Phase 2 (Ascend):** Knee extension from $90^\circ \rightarrow 170^\circ$ maps to $50\% \rightarrow 100\%$ rep completion.
*   **Knee Caving (Valgus Collapse) Detection:**
    *   Measures the absolute horizontal distance between knees (`25` & `26`) against hip width (`23` & `24`).
    *   If knee distance drops below $\text{Hip Width} - (0.1 \times \text{Torso Length})$ for $5$ consecutive frames, a valgus collapse incident is flagged.
    *   Displays a prominent HUD warning: `"Knees Caving In!"`
*   **Speed/Tempo Validation:**
    *   Measures duration between starting descent and returning to stand.
    *   Descent-to-ascent time $< 2.0\text{s}$ flags `"Too fast! Control your descent."`
    *   Duration $> 6.0\text{s}$ flags `"Too slow! Rise up steadily."`

### 🦾 2. Bicep Curl Biomechanics
*   **Angle Target:** Right Arm Shoulder-Elbow-Wrist (landmarks `12` ➔ `14` ➔ `16`).
*   **Rep State Machine:**
    *   **Phase 1 (Curl):** Elbow flexion from $160^\circ \rightarrow 30^\circ$ maps to $0\% \rightarrow 50\%$ rep completion.
    *   **Phase 2 (Extension):** Elbow extension from $30^\circ \rightarrow 160^\circ$ maps to $50\% \rightarrow 100\%$ rep completion.
*   **Shoulder Sway (Cheating) Detection:**
    *   Tracks horizontal displacement of the right shoulder (`12`) relative to its starting rest coordinate.
    *   If displacement exceeds $15\%$ of the scaled torso length for $5$ consecutive frames, the system detects cheating via body sway.
    *   Displays a prominent HUD warning: `"Keep Shoulders Still"`.
*   **Speed/Tempo Validation:**
    *   Requires curls to be executed between $1.5\text{s}$ and $5.0\text{s}$. Curls faster than $1.5\text{s}$ trigger `"Rep too fast! Slow down."`

---

## 📂 Project Architecture

```directory
form_eval_app/
├── client/                     # Vite + React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── CameraFeed.jsx  # MediaPipe camera input & HTML5 Canvas overlay
│   │   │   ├── StatsPanel.jsx  # Telemetry display & layout slot
│   │   │   └── RecorderUI.jsx  # Recording state machine & telemetry buttons
│   │   ├── styles/
│   │   │   └── App.css         # Modern neon-dark HUD styles & glassmorphism
│   │   ├── App.jsx             # Resizable PanelGroup layout & WebSocket connection
│   │   └── main.jsx
│   ├── package.json            # npm package dependencies (react-resizable-panels)
│   └── vite.config.js
├── server/                     # FastAPI Backend
│   ├── core/
│   │   └── exercise_logic.py   # Mathematical physics models & debounced rules
│   ├── db/
│   │   └── db_manager.py       # Local SQLite session persistence logic
│   ├── services/
│   │   ├── data_recorder.py    # Standard pose sequence pipeline
│   │   └── recorder_service.py # Telemetry set recorder (warmups & cancellations)
│   ├── tests/
│   │   ├── test_exercise_debounce.py  # Unit tests for debounced form rules
│   │   └── test_recorder_service.py   # Tests for recording and session logic
│   ├── main.py                 # FastAPI uvicorn runner & WebSocket endpoints
│   ├── requirements.txt        # Python backend library requirements
│   └── setup_m1_native.sh      # Developer setup scripts for M1 Macs
├── main.py                     # Legacy Desktop Tkinter Application
└── README.md                   # Project Documentation
```

---

## ⚙️ Local Installation & Setup

Follow these steps to run both the frontend and backend servers on your local machine.

### 🐍 1. Backend Server Setup (FastAPI)

1.  Navigate into the `server` directory:
    ```bash
    cd server
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv_native
    source venv_native/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the FastAPI application server:
    ```bash
    python main.py
    ```
    *The server will start running at `http://localhost:8000/` and open WebSocket connections on `/ws/exercise`.*

### ⚛️ 2. Frontend Client Setup (Vite + React)

1.  Navigate to the `client` directory (in a new terminal tab):
    ```bash
    cd client
    ```
2.  Install the required npm packages:
    ```bash
    npm install
    ```
3.  Start the local development server:
    ```bash
    npm run dev
    ```
4.  Open your browser and navigate to the address shown: `http://localhost:5173/`
    *Make sure to grant browser camera permissions so MediaPipe can parse coordinates.*

---

## 🧪 Testing & Validation

High test coverage ensures the app remains robust. Use these commands to verify code quality.

### 🧪 Run Backend Tests
Run the `pytest` suite inside the activated virtual environment, ensuring the `PYTHONPATH` points to the `server` root directory:
```bash
cd server
source venv_native/bin/activate
PYTHONPATH=. pytest
```
*All 13 tests checking physics debouncing, joint angle calculations, and set recording logic should pass successfully.*

### 🧹 Run Frontend Linter
Ensure no syntax issues, type conflicts, or unused variables clutter the React client:
```bash
cd client
npm run lint
```
*The ESLint suite runs smoothly and will return clean output with no errors.*

### 🏗️ Test Production Build
Verify the frontend builds correctly for production distribution:
```bash
cd client
npm run build
```

---

## 🛠️ Tech Stack & Dependencies

*   **Frontend Ecosystem:**
    *   **Framework:** React 19.2 (Vite-backed SPA)
    *   **Layout engine:** `react-resizable-panels` 4.10.0 (aliased to Group/Separator for compatibility)
    *   **Sensors:** MediaPipe Pose 0.5 (Dynamic skeletal coordinate tracking)
    *   **Audio engine:** Web Speech API (Biomechanical feedback speaker)
*   **Backend Ecosystem:**
    *   **Web server:** FastAPI 0.100+ & Uvicorn (Ultra-low latency web-sockets)
    *   **Physics pipeline:** MediaPipe Python 0.10.3 & NumPy (Biomechanics trigonometry)
    *   **Storage:** SQLite (Configured under `server/data/fitness_data.db`)
    *   **Testing Framework:** pytest 8.4
