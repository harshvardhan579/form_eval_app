import React from 'react';
import '../styles/App.css';

const StatsPanel = React.memo(({ reps, repPercent, feedback, repFeedback, anomalyScore, tcnScore, conflictWarning, exercise, children }) => {
    const msg = (feedback || "").toLowerCase();
    const isOutOfFrame = msg === "out_of_frame";
    const isInitialState = reps === 0 && repPercent === 0;
    const isReadyMsg = msg === "-" || msg.includes("ready") || msg.includes("waiting") || msg.includes("starting") || msg === "start the rep" || msg === "curl up" || msg === "start squat" || msg === "squat down";
    const isReady = isInitialState || isReadyMsg;
    const isFault = !isOutOfFrame && !isReady && repPercent > 0 && !msg.includes("good") && !msg.includes("perfect");

    let bannerClass = "ready";
    let bannerText = "READY";
    let bannerStyle = {};
    if (isOutOfFrame) {
        bannerClass = "fault";
        bannerText = "MOVE BACK: FULL BODY NOT VISIBLE";
        bannerStyle = { color: "#eab308", borderColor: "rgba(234, 179, 8, 0.4)", backgroundColor: "rgba(234, 179, 8, 0.15)", boxShadow: "0 0 30px rgba(234, 179, 8, 0.2)", animation: "none" };
    } else if (isFault) {
        bannerClass = "fault";
        bannerText = "FAULT DETECTED: " + feedback.toUpperCase();
    } else if (!isReady) {
        bannerClass = "perfect";
        bannerText = "PERFECT FORM";
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
            {/* Top Banner */}
            <div className={`form-banner ${bannerClass}`} style={bannerStyle}>
                {bannerText}
            </div>
            
            {/* Massive Rep Counter */}
            <div className="rep-counter-container">
                <span className="rep-counter-massive">{reps}</span>
            </div>
            
            {/* Action & Progress - Progress Bar + Children (RecorderUI) */}
            <div className="action-progress-module">
                <div className="hud-progress-container">
                    <div className="hud-progress-bar" style={{ width: `${repPercent}%` }}></div>
                    <span className="hud-progress-text">{repPercent}%</span>
                </div>
                {/* HUD Recorder Buttons Slot Here via App.jsx wrapping */}
                {children}
            </div>
        </div>
    );
});

export default StatsPanel;
