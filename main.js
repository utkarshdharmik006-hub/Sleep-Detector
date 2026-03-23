import {
    FaceLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let results = undefined;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("start-btn");
const testBtn = document.getElementById("test-alert-btn");
const alertBanner = document.getElementById("alert-banner");
const earDisplay = document.getElementById("ear-display");
const stateDisplay = document.getElementById("state-display");
const attentionLevel = document.getElementById("attention-level");
const systemStatus = document.getElementById("system-status");
const alertSound = document.getElementById("alert-sound");

const customAudioInput = document.getElementById("custom-audio");
const customAudioLabel = document.querySelector(".custom-audio-label");

// Default Alert Sound (Arpit Bala Scream)
let currentAlertSound = "uth_jaa(256k).mp3.mpeg";
alertSound.src = currentAlertSound;

// Custom Audio Handler
customAudioInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        const fileURL = URL.createObjectURL(file);
        alertSound.src = fileURL;
        customAudioLabel.innerText = "ALARM: " + file.name;
        customAudioLabel.style.borderColor = "#00ff88";
        customAudioLabel.style.color = "#00ff88";
    }
});

// Thresholds
const EYE_AR_THRESH = 0.22;
const EYE_AR_CONSEC_FRAMES = 60; // Approx 2 seconds at 30fps
let counter = 0;
let isAlertActive = false;

// Initialize Face Landmarker
async function initializeMediaPipe() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode,
            numFaces: 1
        });
        
        // Hide loading screen, wait for user click
        document.getElementById("loading-overlay").style.opacity = "0";
        setTimeout(() => {
            document.getElementById("loading-overlay").style.display = "none";
            startBtn.disabled = false;
        }, 500);
        
    } catch (error) {
        console.error("Failed to load MediaPipe:", error);
        systemStatus.innerText = "MODEL LOAD ERROR - SEE CONSOLE";
        systemStatus.parentElement.querySelector(".dot").className = "dot red";
    }
}

startBtn.disabled = true;
initializeMediaPipe();

// EAR Calculation (Eye Aspect Ratio)
function calculateEAR(landmarks) {
    const getDist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
    const l_v1 = getDist(landmarks[160], landmarks[144]);
    const l_v2 = getDist(landmarks[158], landmarks[153]);
    const l_h = getDist(landmarks[33], landmarks[133]);
    const leftEAR = (l_v1 + l_v2) / (2.0 * l_h);
    const r_v1 = getDist(landmarks[385], landmarks[380]);
    const r_v2 = getDist(landmarks[387], landmarks[373]);
    const r_h = getDist(landmarks[362], landmarks[263]);
    const rightEAR = (r_v1 + r_v2) / (2.0 * r_h);
    return (leftEAR + rightEAR) / 2.0;
}

async function predictWebcam() {
    if (!faceLandmarker) return;

    canvasElement.style.width = video.clientWidth + "px";
    canvasElement.style.height = video.clientHeight + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, performance.now());
    }

    if (results && results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            const ear = calculateEAR(landmarks);
            if (earDisplay) earDisplay.innerText = `EAR: ${ear.toFixed(2)}`;
            
            if (ear < EYE_AR_THRESH) {
                counter++;
                if (stateDisplay) {
                    stateDisplay.innerText = "EYES: CLOSED";
                    stateDisplay.className = "state-bad";
                }
                
                const attention = Math.max(0, 100 - (counter / EYE_AR_CONSEC_FRAMES) * 100);
                if (attentionLevel) {
                    attentionLevel.style.width = `${attention}%`;
                    if (attention < 50) attentionLevel.classList.add("warning");
                }

                if (counter >= EYE_AR_CONSEC_FRAMES) {
                    triggerAlert();
                }
            } else {
                counter = 0;
                if (stateDisplay) {
                    stateDisplay.innerText = "EYES: OPEN";
                    stateDisplay.className = "state-good";
                }
                if (attentionLevel) {
                    attentionLevel.style.width = "100%";
                    attentionLevel.classList.remove("warning");
                }
                stopAlert();
            }

            // Draw futuristic facial mesh
            if (DrawingUtils && FaceLandmarker) {
                const drawingUtils = new DrawingUtils(canvasCtx);
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#ffffff20", lineWidth: 1 });
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#00ff88", lineWidth: 3 });
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#00ff88", lineWidth: 3 });
            }
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function triggerAlert() {
    if (!isAlertActive) {
        isAlertActive = true;
        if (alertBanner) alertBanner.classList.remove("hidden");
        if (systemStatus) {
            systemStatus.innerText = "CRITICAL ALERT";
            systemStatus.parentElement.querySelector(".dot").className = "dot red";
        }
        if (alertSound) alertSound.play().catch(e => console.log("Audio play failed", e));
        if ("vibrate" in navigator) navigator.vibrate([200, 100, 200]);
    }
}

function stopAlert() {
    if (isAlertActive) {
        isAlertActive = false;
        if (alertBanner) alertBanner.classList.add("hidden");
        if (systemStatus) {
            systemStatus.innerText = "MONITORING ACTIVE";
            systemStatus.parentElement.querySelector(".dot").className = "dot green";
        }
        if (alertSound) {
            alertSound.pause();
            alertSound.currentTime = 0;
        }
    }
}

// Start/Stop Logic
startBtn.addEventListener("click", async () => {
    if (!webcamRunning) {
        try {
            // Request high-definition camera for clearer UI
            const constraints = { 
                video: { 
                    width: { ideal: 1920 }, 
                    height: { ideal: 1080 },
                    facingMode: "user"
                } 
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            
            video.addEventListener("loadeddata", () => {
                video.classList.add("active");
                if (document.getElementById("cam-placeholder")) {
                    document.getElementById("cam-placeholder").style.display = "none";
                }
                predictWebcam();
            });
            
            webcamRunning = true;
            startBtn.innerText = "STOP MONITORING";
            systemStatus.innerText = "MONITORING ACTIVE";
        } catch (err) {
            console.error("Camera error:", err);
            systemStatus.innerText = "CAMERA BLOCKED - PLEASE ALLOW ACCESS";
            systemStatus.parentElement.querySelector(".dot").className = "dot red";
        }
    } else {
        webcamRunning = false;
        if (video.srcObject) {
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
            video.classList.remove("active");
            if (document.getElementById("cam-placeholder")) {
                document.getElementById("cam-placeholder").style.display = "flex";
            }
        }
        startBtn.innerText = "START MONITORING";
        systemStatus.innerText = "SYSTEM PAUSED";
        systemStatus.parentElement.querySelector(".dot").className = "dot";
        systemStatus.parentElement.querySelector(".dot").style.background = "#555";
        stopAlert();
    }
});

testBtn.addEventListener("click", () => {
    triggerAlert();
    setTimeout(stopAlert, 3000);
});
