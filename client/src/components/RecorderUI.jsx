import React, { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Recording state machine: idle → warmup → recording → saving → idle
 */
const STATES = {
    IDLE: 'idle',
    WARMUP: 'warmup',
    RECORDING: 'recording',
    SAVING: 'saving',
};

const RecorderUI = ({ wsRef, onRecordingStateChange, exercise, label }) => {
    const [recState, setRecState] = useState(STATES.IDLE);
    const [countdown, setCountdown] = useState(3);
    const [frameCount, setFrameCount] = useState(0);
    const [toast, setToast] = useState(null);

    const countdownRef = useRef(null);
    const frameIntervalRef = useRef(null);

    // Notify parent of recording state changes
    useEffect(() => {
        if (onRecordingStateChange) {
            onRecordingStateChange(recState === STATES.RECORDING);
        }
    }, [recState, onRecordingStateChange]);

    // Clean up timers on unmount
    useEffect(() => {
        return () => {
            clearInterval(countdownRef.current);
            clearInterval(frameIntervalRef.current);
        };
    }, []);

    // Dismiss toast after 4 seconds
    useEffect(() => {
        if (toast) {
            const t = setTimeout(() => setToast(null), 4000);
            return () => clearTimeout(t);
        }
    }, [toast]);

    const sendControl = useCallback((action, extra = {}) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'recording_control',
                action,
                ...extra,
            }));
        }
    }, [wsRef]);

    // -----------------------------------------------------------------------
    // Start → Warmup countdown → Recording
    // -----------------------------------------------------------------------
    const handleStart = () => {
        setRecState(STATES.WARMUP);
        setCountdown(3);
        setFrameCount(0);
        setToast(null);

        let remaining = 3;
        countdownRef.current = setInterval(() => {
            remaining -= 1;
            if (remaining <= 0) {
                clearInterval(countdownRef.current);
                // Transition to recording
                setCountdown(0);
                setRecState(STATES.RECORDING);
                sendControl('START', { exercise, label });

                // Start a frame-count poller (poll WS ack or use local tick)
                let localFrames = 0;
                frameIntervalRef.current = setInterval(() => {
                    localFrames += 1;  // approximate: ~30 fps landmark stream
                    setFrameCount(localFrames);
                }, 33); // ~30 fps
            } else {
                setCountdown(remaining);
            }
        }, 1000);
    };

    // -----------------------------------------------------------------------
    // Stop → Save
    // -----------------------------------------------------------------------
    const handleStop = () => {
        clearInterval(frameIntervalRef.current);
        setRecState(STATES.SAVING);
        sendControl('STOP');
    };

    // -----------------------------------------------------------------------
    // Cancel
    // -----------------------------------------------------------------------
    const handleCancel = () => {
        clearInterval(countdownRef.current);
        clearInterval(frameIntervalRef.current);
        setRecState(STATES.IDLE);
        sendControl('CANCEL');
        setFrameCount(0);
        setToast(null);
    };

    // -----------------------------------------------------------------------
    // Handle incoming WS messages (called from App.jsx via prop)
    // -----------------------------------------------------------------------
    const handleRecordingMessage = useCallback((data) => {
        if (data.type === 'recording_started') {
            // already handled locally
        } else if (data.type === 'recording_saved' || data.type === 'RECORDING_SAVED') {
            setRecState(STATES.IDLE);
            const parts = data.filepath ? data.filepath.split('/') : [];
            const shortPath = parts.length >= 2 ? `${parts[parts.length - 3]}/${parts[parts.length - 2]}` : 'disk';
            setToast(`✅ Saved ${data.frames_saved} frames to ${shortPath}`);
            setFrameCount(0);
        } else if (data.type === 'recording_cancelled') {
            setRecState(STATES.IDLE);
            setFrameCount(0);
        }
    }, []);

    // Expose handler to parent via ref-like pattern
    useEffect(() => {
        // Attach handler to wsRef for parent to call
        if (wsRef.current) {
            wsRef.current._recorderHandler = handleRecordingMessage;
        }
    }, [wsRef, handleRecordingMessage]);

    const isIdle = recState === STATES.IDLE;
    const isWarmup = recState === STATES.WARMUP;
    const isRecording = recState === STATES.RECORDING;
    const isSaving = recState === STATES.SAVING;

    return (
        <div className="recorder-state-display" style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
            {isIdle && (
                <button
                    id="rec-start-btn"
                    className="hud-rec-btn hud-rec-btn-start"
                    onClick={handleStart}
                >
                    RECORD SET
                </button>
            )}

            {isWarmup && (
                <div className="countdown-display" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <div className="countdown-number" style={{ fontSize: '5rem', fontWeight: '900', lineHeight: 1 }}>{countdown}</div>
                    <div className="countdown-label" style={{ fontSize: '1.5rem', opacity: 0.8, letterSpacing: '2px', marginTop: '1rem' }}>GET READY...</div>
                    <button
                        id="rec-cancel-btn"
                        className="hud-rec-btn"
                        onClick={handleCancel}
                        style={{ marginTop: '1.5rem', background: 'rgba(255,255,255,0.1)' }}
                    >
                        CANCEL
                    </button>
                </div>
            )}

            {isRecording && (
                <div className="recording-active" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: '1.5rem', padding: '0 1rem' }}>
                        <div className="rec-live-indicator" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span className="rec-dot-inline" style={{ width: '24px', height: '24px', backgroundColor: '#ef4444', borderRadius: '50%', boxShadow: '0 0 15px #ef4444', animation: 'pulse-hud-red 1s infinite' }}></span>
                            <span className="rec-live-text" style={{ fontSize: '1.8rem', fontWeight: '900', color: '#ef4444', letterSpacing: '3px' }}>RECORDING</span>
                        </div>
                        <div className="rec-frame-counter">
                            <span className="frame-count-value" style={{ fontSize: '3rem', fontWeight: '900', fontVariantNumeric: 'tabular-nums' }}>{frameCount}</span>
                            <span className="frame-count-label" style={{ fontSize: '1.2rem', marginLeft: '8px', opacity: 0.8, textTransform: 'uppercase' }}>Frames</span>
                        </div>
                    </div>
                    
                    <div className="rec-active-buttons" style={{ display: 'flex', gap: '1rem', width: '100%' }}>
                        <button
                            id="rec-stop-btn"
                            className="hud-rec-btn hud-rec-btn-stop"
                            onClick={handleStop}
                            style={{ flex: 3 }}
                        >
                            STOP & SAVE
                        </button>
                        <button
                            id="rec-cancel-active-btn"
                            className="hud-rec-btn"
                            onClick={handleCancel}
                            style={{ flex: 1, background: 'rgba(255,255,255,0.1)', color: '#f87171' }}
                        >
                            TRASH
                        </button>
                    </div>
                </div>
            )}

            {isSaving && (
                <div className="saving-display" style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.3)', padding: '2rem', borderRadius: '1rem', border: '1px solid rgba(255,255,255,0.1)' }}>
                    <div className="saving-spinner" style={{ width: '40px', height: '40px', border: '4px solid rgba(255,255,255,0.1)', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                    <span style={{ fontSize: '1.5rem', marginLeft: '1.5rem', fontWeight: 'bold', letterSpacing: '2px' }}>FINALIZING...</span>
                </div>
            )}
            
            <style jsx="true">{`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}</style>
        </div>
    );
};

export default RecorderUI;
