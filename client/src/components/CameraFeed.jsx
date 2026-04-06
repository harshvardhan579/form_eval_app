import React, { useRef, useEffect, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';

const CameraFeed = ({ onLandmarks, isRecording, feedback, activeAngle, activeJointCoords }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const feedbackRef = useRef(feedback);
    const activeAngleRef = useRef(activeAngle);
    const activeJointCoordsRef = useRef(activeJointCoords);
    const [cameraActive, setCameraActive] = useState(false);

    useEffect(() => {
        feedbackRef.current = feedback;
        activeAngleRef.current = activeAngle;
        activeJointCoordsRef.current = activeJointCoords;
    }, [feedback, activeAngle, activeJointCoords]);
    const [telemetry, setTelemetry] = useState({
        ratio: 0,
        torso: 0,
        visibility: 0,
        caving: false
    });

    useEffect(() => {
        const videoElement = videoRef.current;
        let camera;

        const pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });

        pose.setOptions({
            modelComplexity: 1, /* Balanced for 30fps on M1 */
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: false,
            minDetectionConfidence: 0.5, /* Lowered to allow skeleton lock even if face is off-screen */
            minTrackingConfidence: 0.8  /* Increased to enforce stickiness once body is found */
        });

        pose.onResults((results) => {
            const canvasElement = canvasRef.current;
            const canvasCtx = canvasElement?.getContext('2d');
            if (!canvasCtx || !canvasElement || !videoElement) return;

            // Dynamically scale canvas resolution to exact container client boundaries for the Portrait layout
            if (canvasElement.width !== canvasElement.clientWidth || canvasElement.height !== canvasElement.clientHeight) {
                canvasElement.width = canvasElement.clientWidth;
                canvasElement.height = canvasElement.clientHeight;
            }

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            if (results.poseLandmarks) {
                const lms = results.poseLandmarks;

                // Telemetry Calculation
                let isCaving = false;
                try {
                    // Torso: 11 (L Shoulder) to 23 (L Hip)
                    const torsoLen = Math.hypot(lms[11].x - lms[23].x, lms[11].y - lms[23].y);

                    // Hips: 23 & 24
                    const hipDist = Math.hypot(lms[23].x - lms[24].x, lms[23].y - lms[24].y);

                    // Knees: 25 & 26
                    const kneeDist = Math.abs(lms[25].x - lms[26].x);

                    // Threshold check
                    const threshold = hipDist - (torsoLen * 0.1);
                    isCaving = kneeDist < threshold;

                    // Visibility Avg
                    const visAvg = ((lms[25].visibility || 0) + (lms[26].visibility || 0)) / 2;

                    setTelemetry({
                        ratio: (kneeDist / hipDist).toFixed(2),
                        torso: torsoLen.toFixed(3),
                        visibility: visAvg.toFixed(2),
                        caving: isCaving
                    });
                } catch (e) {
                    // Ignore math errors on missing landmarks
                }

                // Draw dynamic knee line unconditionally
                if (lms[25] && lms[26]) {
                    canvasCtx.beginPath();
                    // MediaPipe coords are normalized 0-1, multiply by canvas dimensions
                    canvasCtx.moveTo(lms[25].x * canvasElement.width, lms[25].y * canvasElement.height);
                    canvasCtx.lineTo(lms[26].x * canvasElement.width, lms[26].y * canvasElement.height);
                    canvasCtx.strokeStyle = isCaving ? '#FF0000' : '#FFFF00';
                    canvasCtx.lineWidth = 6;
                    canvasCtx.stroke();
                }

                // Parse current feedback to determine if we should flash red
                const msg = (feedbackRef.current || "").toLowerCase();
                const hasKneeError = msg.includes("knee");
                const hasShoulderError = msg.includes("shoulder") || msg.includes("form");

                const armColor = hasShoulderError ? '#ef4444' : '#22c55e';
                const legColor = hasKneeError ? '#ef4444' : '#22c55e';
                const torsoColor = hasShoulderError ? '#ef4444' : '#22c55e';

                const drawLine = (p1, p2, color, width) => {
                    if (!p1 || !p2 || p1.visibility < 0.5 || p2.visibility < 0.5) return;
                    canvasCtx.beginPath();
                    canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
                    canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
                    canvasCtx.strokeStyle = color;
                    canvasCtx.lineWidth = width;
                    canvasCtx.lineCap = 'round';
                    canvasCtx.lineJoin = 'round';
                    canvasCtx.stroke();
                };

                const drawPath = (indices, color) => {
                    for (let i = 0; i < indices.length - 1; i++) {
                        drawLine(lms[indices[i]], lms[indices[i+1]], color, 5);
                    }
                };

                // Draw Custom Skeleton
                // Torso
                drawPath([11, 12, 24, 23, 11], torsoColor);
                // Arms
                drawPath([11, 13, 15], armColor);
                drawPath([12, 14, 16], armColor);
                // Legs
                drawPath([23, 25, 27, 29, 31, 27], legColor);
                drawPath([24, 26, 28, 30, 32, 28], legColor);

                if (activeAngleRef.current !== undefined && activeAngleRef.current !== null && activeJointCoordsRef.current) {
                    const angleText = `${activeAngleRef.current}°`;
                    const px = (activeJointCoordsRef.current.x * canvasElement.width) + 30;
                    const py = activeJointCoordsRef.current.y * canvasElement.height;

                    canvasCtx.fillStyle = 'white';
                    canvasCtx.font = 'bold 24px system-ui';
                    canvasCtx.textAlign = 'left';
                    canvasCtx.textBaseline = 'middle';
                    canvasCtx.shadowColor = 'rgba(0,0,0,0.8)';
                    canvasCtx.shadowBlur = 4;
                    canvasCtx.shadowOffsetX = 2;
                    canvasCtx.shadowOffsetY = 2;

                    canvasCtx.fillText(angleText, px, py);

                    canvasCtx.shadowColor = 'transparent';
                    canvasCtx.shadowBlur = 0;
                    canvasCtx.shadowOffsetX = 0;
                    canvasCtx.shadowOffsetY = 0;
                }

                // Pass landmarks to parent component mapped as {x,y,z,visibility}
                onLandmarks(lms);
            }

            if (feedbackRef.current === "OUT_OF_FRAME") {
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.75)";
                canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
                canvasCtx.fillStyle = "#eab308";
                canvasCtx.font = "bold 48px system-ui";
                canvasCtx.textAlign = "center";
                canvasCtx.textBaseline = "middle";
                canvasCtx.fillText("STEP BACK", canvasElement.width / 2, canvasElement.height / 2 - 30);
                canvasCtx.font = "bold 24px system-ui";
                canvasCtx.fillText("FULL BODY NOT VISIBLE", canvasElement.width / 2, canvasElement.height / 2 + 20);
            }

            canvasCtx.restore();
        });

        if (videoElement) {
            camera = new Camera(videoElement, {
                onFrame: async () => {
                    await pose.send({ image: videoElement });
                },
                width: 1080,
                height: 720
            });
            camera.start().then(() => setCameraActive(true));
        }

        return () => {
            if (camera) {
                camera.stop();
            }
            pose.close();
        };
    }, [onLandmarks]);

    return (
        <>
            {!cameraActive && <div className="loading-camera">Loading Camera Model...</div>}
            <video
                ref={videoRef}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                autoPlay
                playsInline
                muted
            />
            <canvas
                ref={canvasRef}
                width="1080"
                height="720"
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    pointerEvents: 'none',
                    zIndex: 10
                }}
            />

            {/* Debug Telemetry Overlay */}
            <div style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: '#00FF00',
                padding: '10px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                zIndex: 20, /* Above canvas */
                border: '1px solid #00FF00'
            }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#FFF' }}>SENSOR DIAGNOSTIC</div>
                <div>Knee/Hip Ratio: {telemetry.ratio}</div>
                <div>Torso Length: {telemetry.torso}</div>
                <div>Knee Visibility: {telemetry.visibility}</div>
                <div style={{ color: telemetry.caving ? '#FF0000' : '#00FF00', fontWeight: 'bold', marginTop: '4px' }}>
                    Status: {telemetry.caving ? 'CAVE DETECTED' : 'GOOD'}
                </div>
            </div>

            {/* Recording indicator overlay */}
            {isRecording && (
                <div className="rec-overlay" style={{ zIndex: 20 }}>
                    <span className="rec-dot"></span>
                    <span className="rec-text">REC</span>
                </div>
            )}
        </>
    );
};

export default CameraFeed;
