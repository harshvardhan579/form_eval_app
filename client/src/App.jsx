import { useState, useEffect, useRef, useCallback } from 'react'
import { Group as PanelGroup, Panel, Separator as PanelResizeHandle } from "react-resizable-panels";
import CameraFeed from './components/CameraFeed'
import StatsPanel from './components/StatsPanel'
import RecorderUI from './components/RecorderUI'
import './styles/App.css'

const REP_WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty'];

function App() {
  const [exercise, setExercise] = useState("Bicep Curl");
  const [formLabel, setFormLabel] = useState("Perfect");
  const [isRecording, setIsRecording] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [toast, setToast] = useState(null);
  const [audioMuted, setAudioMuted] = useState(false);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  // --- Audio Feedback Engine ---
  const audioMutedRef = useRef(audioMuted);
  const lastSpokenRepRef = useRef(0);
  const lastSpokenFaultRef = useRef("");

  useEffect(() => { audioMutedRef.current = audioMuted; }, [audioMuted]);

  const speak = useCallback((text) => {
    if (audioMutedRef.current || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }, []);

  const [stats, setStats] = useState({
    reps: 0,
    repPercent: 0,
    feedback: "-",
    repFeedback: "-",
    anomalyScore: 0.0,
    tcnScore: 100.0,
    conflictWarning: false,
    activeAngle: null,
    activeJointCoords: null
  });
  const wsRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Rep count announcer
  useEffect(() => {
    const wholeReps = Math.floor(stats.reps);
    if (wholeReps > 0 && wholeReps !== lastSpokenRepRef.current) {
      lastSpokenRepRef.current = wholeReps;
      const word = wholeReps <= 20 ? REP_WORDS[wholeReps] : String(wholeReps);
      speak(word);
    }
  }, [stats.reps, speak]);

  // Fault callout with debounce lock
  useEffect(() => {
    const msg = (stats.feedback || "").toLowerCase();
    const isFault = msg.includes("knee") || msg.includes("shoulder") || msg.includes("cave") || msg.includes("sway");
    if (isFault && stats.feedback !== lastSpokenFaultRef.current) {
      lastSpokenFaultRef.current = stats.feedback;
      speak(stats.feedback);
    }
    if (!isFault) {
      lastSpokenFaultRef.current = "";
    }
  }, [stats.feedback, speak]);

  useEffect(() => {
    // If Vercel doesn't have the environment variable set, it safely falls back to your Render backend.
    // Notice the 'wss://' prefix instead of 'https://'
    const WS_URL = import.meta.env.VITE_WS_URL || "wss://form-eval-app.onrender.com";
    const ws = new WebSocket(`${WS_URL}/ws/exercise`);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      ws.send(JSON.stringify({ type: 'set_exercise', exercise }));
    };

    ws.onmessage = (event) => {
      isProcessingRef.current = false;
      const data = JSON.parse(event.data);

      if (data.type === "RECORDING_SAVED") {
        setIsSaving(false);
        setIsRecording(false);
        showToast("Session Saved!");
        if (ws._recorderHandler) ws._recorderHandler(data);
        return;
      }

      // Route recording-related messages to RecorderUI handler
      if (data.type === 'recording_started' || data.type === 'recording_saved' || data.type === 'recording_cancelled') {
        if (ws._recorderHandler) {
          ws._recorderHandler(data);
        }
        return;
      }

      // Default: stats update
      setStats({
        reps: data.reps,
        repPercent: data.rep_percent !== undefined ? data.rep_percent : 0,
        feedback: data.feedback,
        repFeedback: data.rep_feedback,
        anomalyScore: data.anomaly_score !== undefined ? data.anomaly_score : 0.0,
        tcnScore: data.tcn_score !== undefined ? data.tcn_score : 100.0,
        conflictWarning: data.conflict_warning || false,
        activeAngle: data.active_angle,
        activeJointCoords: data.active_joint_coords
      });
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
    };

    wsRef.current = ws;

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []); // Only run once on mount

  const onLandmarks = useCallback((landmarks) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (isProcessingRef.current) return;
      isProcessingRef.current = true;
      wsRef.current.send(JSON.stringify({
        type: 'landmarks',
        landmarks: landmarks
      }));
    }
  }, []);

  const handleExerciseChange = (e) => {
    const newEx = e.target.value;
    setExercise(newEx);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_exercise', exercise: newEx }));
    }
    setStats({
      reps: 0, repPercent: 0, feedback: "Starting...", repFeedback: "Waiting for first rep...",
      anomalyScore: 0.0, tcnScore: 100.0, conflictWarning: false
    });
  };

  const handleReset = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
    setStats({
      reps: 0, repPercent: 0, feedback: "Starting...", repFeedback: "Waiting for first rep...",
      anomalyScore: 0.0, tcnScore: 100.0, conflictWarning: false
    });
  }

  return (
    <div className="app-container">
      <PanelGroup direction="horizontal" autoSaveId="main-layout">
        {/* ----- LEFT SIDEBAR (COMMAND CENTER) ----- */}
        <Panel defaultSize={40} minSize={20}>
          <PanelGroup direction="vertical" autoSaveId="left-sidebar-layout">
            <Panel defaultSize={50} minSize={30}>
              <div className="camera-module" style={{ width: '100%', height: '100%' }}>
                <CameraFeed onLandmarks={onLandmarks} isRecording={isRecording} feedback={stats.feedback} activeAngle={stats.activeAngle} activeJointCoords={stats.activeJointCoords} />
              </div>
            </Panel>

            <PanelResizeHandle className="resize-handle-horizontal" />

            <Panel defaultSize={50} minSize={20}>
              <div className="config-module" style={{ width: '100%', height: '100%', boxSizing: 'border-box' }}>
                <label className="select-label" style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '1.2rem', opacity: 0.8 }}>
                  LIVE ML MODEL:
                  <select value={exercise} onChange={handleExerciseChange} className="glass-select" style={{ fontSize: '1.5rem', padding: '12px 16px', fontWeight: 'bold' }}>
                    <option value="Bicep Curl">Bicep Curl</option>
                    <option value="Squat">Squat</option>
                  </select>
                </label>
                <label className="select-label" style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '1.2rem', opacity: 0.8 }}>
                  FORM LABEL:
                  <select value={formLabel} onChange={(e) => setFormLabel(e.target.value)} className="glass-select" style={{ fontSize: '1.5rem', padding: '12px 16px', fontWeight: 'bold' }}>
                    <option value="Perfect">Perfect</option>
                    <option value="Flawed">Flawed</option>
                  </select>
                </label>
                <button onClick={handleReset} className="reset-btn glass-btn" style={{ fontSize: '1.2rem', padding: '12px' }}>🔄 Reset Engine</button>
                <button onClick={() => setAudioMuted(m => !m)} className="glass-btn" style={{ fontSize: '1.2rem', padding: '12px', background: audioMuted ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)', color: audioMuted ? '#f87171' : '#4ade80', borderColor: audioMuted ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)' }}>{audioMuted ? '🔇 Voice Coach Off' : '🔊 Voice Coach On'}</button>
              </div>
            </Panel>
          </PanelGroup>
        </Panel>

        <PanelResizeHandle className="resize-handle-vertical" />

        {/* ----- RIGHT MAIN STAGE (TELEMETRY) ----- */}
        <Panel defaultSize={60} minSize={30}>
          <div className="right-main" style={{ width: '100%', height: '100%' }}>
            {toast && (
              <div style={{ position: 'absolute', top: 20, right: 20, padding: '12px', background: '#22c55e', color: 'white', fontWeight: 'bold', borderRadius: '8px', textAlign: 'center', zIndex: 100, fontSize: '1.5rem', boxShadow: '0 4px 15px rgba(0,0,0,0.5)' }}>
                {toast}
              </div>
            )}

            <StatsPanel
              reps={stats.reps}
              repPercent={stats.repPercent}
              feedback={stats.feedback}
              repFeedback={stats.repFeedback}
              anomalyScore={stats.anomalyScore}
              tcnScore={stats.tcnScore}
              conflictWarning={stats.conflictWarning}
              exercise={exercise}
            >
              <RecorderUI
                wsRef={wsRef}
                onRecordingStateChange={setIsRecording}
                exercise={exercise}
                label={formLabel}
              />
            </StatsPanel>
          </div>
        </Panel>
      </PanelGroup>
    </div>
  )
}

export default App
