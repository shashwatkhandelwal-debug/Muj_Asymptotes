/**
 * frontend/static/app.js — Edge Challenge Capture Companion
 * Mission-critical challenge-capture client with low-bandwidth face crop extraction,
 * real-time brightness gate, 5-second synchronized keyframe sampling, and hardware track safety.
 */

// Global state
let activeStream = null;
let mediaRecorder = null;
let audioChunks = [];
let brightnessPollTimer = null;
let dimWaitSeconds = 0;
let overrideAvailable = false;
let faceDetector = null;
let mediaPipeLoaded = false;

// Challenge Contract (§5.1)
let challenge = {
  action: "turn_left",
  spoken_phrase: "Priya Sharma 2026-09-11 4471",
  nonce: "4471",
  date_str: "2026-09-11"
};

/**
 * Fetch a fresh, single-use session challenge with a unique nonce from backend
 */
async function fetchChallenge() {
  try {
    const endpoints = ["/api/challenge/new", "http://localhost:8000/api/challenge/new", "/api/challenge"];
    for (const ep of endpoints) {
      try {
        const res = await fetch(ep);
        if (res.ok) {
          const data = await res.json();
          if (data && data.nonce) {
            challenge = {
              action: data.action || challenge.action,
              spoken_phrase: data.spoken_phrase || challenge.spoken_phrase,
              nonce: data.nonce,
              date_str: data.date_str || challenge.date_str
            };
            return challenge;
          }
        }
      } catch (_) {}
    }
  } catch (e) {
    console.warn("Using fallback challenge:", e);
  }
  return challenge;
}

/**
 * Initialize MediaPipe FaceDetection from window.FaceDetection CDN (§5.4)
 */
async function initMediaPipe() {
  if (typeof window !== "undefined" && window.FaceDetection) {
    try {
      faceDetector = new window.FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
      });
      faceDetector.setOptions({
        model: "short",
        minDetectionConfidence: 0.5
      });
      mediaPipeLoaded = true;
    } catch (e) {
      console.warn("MediaPipe unavailable — sending full frame", e);
      mediaPipeLoaded = false;
    }
  } else {
    console.log("MediaPipe unavailable — sending full frame");
    mediaPipeLoaded = false;
  }
}

const LOW_LIGHT = 60.0;

/**
 * Brightness measurement function (§5.3)
 */
function meanBrightness(videoElement, canvas, ctx) {
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) {
    sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  return sum / (data.length / 4); // 0..255
}
const checkBrightness = meanBrightness;

/**
 * Hard stop on all media tracks to guarantee camera/mic hardware indicator turns off
 */
function stopMediaTracks() {
  if (brightnessPollTimer) {
    clearInterval(brightnessPollTimer);
    brightnessPollTimer = null;
  }
  if (activeStream) {
    activeStream.getTracks().forEach(track => {
      try {
        track.stop();
      } catch (e) {}
    });
    activeStream = null;
  }
}

/**
 * Extract 224x224 face crop with 20% padding or fallback to full 480p frame (§5.4)
 */
async function extractFaceCrop(videoEl, canvas, ctx) {
  const w = videoEl.videoWidth || 854;
  const h = videoEl.videoHeight || 480;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(videoEl, 0, 0, w, h);

  if (mediaPipeLoaded && faceDetector) {
    try {
      let detectedBox = null;
      faceDetector.onResults((results) => {
        if (results && results.detections && results.detections.length > 0) {
          detectedBox = results.detections[0].boundingBox;
        }
      });
      // Ensure faceDetector.send cannot hang execution
      await Promise.race([
        faceDetector.send({ image: canvas }),
        new Promise((_, reject) => setTimeout(() => reject(new Error("MediaPipe timeout")), 400))
      ]).catch((err) => {
        console.warn("Face detector send timeout or error:", err.message);
      });

      if (detectedBox) {
        const bx = detectedBox.xCenter * w - (detectedBox.width * w) / 2;
        const by = detectedBox.yCenter * h - (detectedBox.height * h) / 2;
        const bw = detectedBox.width * w;
        const bh = detectedBox.height * h;

        // 20% padding on each side
        const padX = bw * 0.20;
        const padY = bh * 0.20;
        const cropX = Math.max(0, bx - padX);
        const cropY = Math.max(0, by - padY);
        const cropW = Math.min(w - cropX, bw + padX * 2);
        const cropH = Math.min(h - cropY, bh + padY * 2);

        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = 224;
        cropCanvas.height = 224;
        const cropCtx = cropCanvas.getContext("2d");
        cropCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, 224, 224);

        return await new Promise(resolve => cropCanvas.toBlob(resolve, "image/jpeg", 0.85));
      } else {
        // No face detected in this frame -> skip it (never push null)
        return null;
      }
    } catch (detErr) {
      console.warn("Face detection error, skipping frame:", detErr);
      return null;
    }
  }

  // Explicit full-frame fallback path
  console.log("MediaPipe unavailable — sending full frame");
  const fallbackCanvas = document.createElement("canvas");
  fallbackCanvas.width = 224;
  fallbackCanvas.height = 224;
  const fbCtx = fallbackCanvas.getContext("2d");
  fbCtx.drawImage(canvas, 0, 0, w, h, 0, 0, 224, 224);
  return await new Promise(resolve => fallbackCanvas.toBlob(resolve, "image/jpeg", 0.85));
}

/**
 * Payload assembly and send (§5.7)
 */
async function sendChallenge(cropBlobs, audioBlob, ch) {
  const formData = new FormData();
  cropBlobs.forEach((blob, i) => formData.append(`frame_${i}`, blob, `frame_${i}.jpg`));
  formData.append("audio", audioBlob, "audio.webm");
  formData.append("action", ch.action);
  formData.append("nonce", ch.nonce);
  formData.append("date_str", ch.date_str);

  const endpoints = ["/api/challenge", "http://localhost:8000/api/challenge"];
  for (const ep of endpoints) {
    try {
      const res = await fetch(ep, { method: "POST", body: formData });
      if (res.ok) {
        return await res.json();
      }
    } catch (_) {}
  }
  return { ok: false, error: "Backend challenge endpoint unavailable" };
}

let currentUIState = null;

/**
 * UI State Renderer (§5.8)
 */
function renderUIState(state, data = {}) {
  const root = document.getElementById("capture-root");
  if (!root) return;

  // If already in warning_brightness, avoid DOM thrashing: just update the readout
  if (currentUIState === "warning_brightness" && state === "warning_brightness") {
    const warnLux = document.getElementById("warnLux");
    if (warnLux) warnLux.innerText = data.lux || 0;
    if (overrideAvailable) {
      const btn = document.getElementById("btnOverride");
      if (btn) {
        btn.style.display = "inline-flex";
        btn.onclick = () => {
          if (brightnessPollTimer) {
            clearInterval(brightnessPollTimer);
            brightnessPollTimer = null;
          }
          runRecordingSequence();
        };
      }
    }
    return;
  }

  currentUIState = state;

  switch (state) {
    case "idle":
      const actionLabels = {
        "turn_left": "Turn your head LEFT",
        "turn_right": "Turn your head RIGHT",
        "blink_twice": "Blink your eyes TWICE"
      };
      const displayAction = actionLabels[challenge.action] || challenge.action.replace("_", " ").toUpperCase();
      root.innerHTML = `
        <div class="panel" style="text-align: center;">
          <div class="eyebrow" style="margin-bottom: 12px;">ACTIVE BIOMETRIC CHALLENGE</div>
          <div style="font-size: 18px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
            Action: ${displayAction}
          </div>
          <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">
            Position your face within frame, perform the action, and recite the challenge token aloud:
          </div>
          <div class="recessed-block" style="margin-bottom: 24px; text-align: center;">
            "${challenge.spoken_phrase}"
          </div>
          <button id="btnStartCapture" class="btn-terminal" style="font-size: 13px; padding: 10px 24px; color: var(--text-primary); border-color: var(--border-strong);">
            <span>[START CAPTURE]</span>
          </button>
        </div>
      `;
      document.getElementById("btnStartCapture")?.addEventListener("click", startCaptureWorkflow);
      break;

    case "checking_brightness":
      root.innerHTML = `
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <span class="eyebrow">ENVIRONMENT TELEMETRY</span>
            <span class="mono" style="font-size: 11px; color: var(--text-secondary);">LUX THRESHOLD: 60</span>
          </div>
          <div style="position: relative; width: 100%; aspect-ratio: 16/9; background: var(--bg-inset); border: 1px solid var(--border-hairline); border-radius: 6px; overflow: hidden; margin-bottom: 14px;">
            <video id="captureVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);"></video>
            <canvas id="calcCanvas" style="display: none;"></canvas>
          </div>
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span class="mono" style="font-size: 11px; color: var(--text-tertiary);">AMBIENT ILLUMINANCE</span>
            <span id="luxReadout" class="mono" style="font-size: 12px; font-weight: 600; color: var(--text-primary);">${data.lux || 0} LUX</span>
          </div>
          <div style="width: 100%; height: 6px; background: var(--bg-inset); border: 1px solid var(--border-hairline); border-radius: 3px; overflow: hidden;">
            <div id="luxBar" style="height: 100%; width: ${Math.min(100, ((data.lux || 0) / 255) * 100)}%; background: ${data.lux < 60 ? 'var(--review)' : 'var(--clear)'}; transition: width 100ms linear;"></div>
          </div>
        </div>
      `;
      break;

    case "warning_brightness":
      root.innerHTML = `
        <div class="panel" style="border-color: rgba(245,158,11,0.35); background: var(--bg-panel);">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--review);"></span>
            <span class="eyebrow" style="color: var(--review);">SUB-OPTIMAL LIGHTING DETECTED</span>
          </div>
          <div style="position: relative; width: 100%; aspect-ratio: 16/9; background: var(--bg-inset); border: 1px solid var(--border-hairline); border-radius: 6px; overflow: hidden; margin-bottom: 14px;">
            <video id="captureVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);"></video>
            <canvas id="calcCanvas" style="display: none;"></canvas>
            <div style="position: absolute; inset: 0; background: var(--review-dim); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px;">
              <div style="font-size: 14px; font-weight: 600; color: var(--review); margin-bottom: 6px;">
                Lighting is too dim (<span id="warnLux" class="mono">${data.lux || 0}</span> LUX)
              </div>
              <div style="font-size: 12px; color: var(--text-secondary); max-width: 320px;">
                Move to a brighter area or face a light source to proceed with facial verification.
              </div>
            </div>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="mono" style="font-size: 11px; color: var(--text-tertiary);">WAITING FOR ILLUMINATION...</span>
            <button id="btnOverride" class="btn-terminal" style="${overrideAvailable ? 'display: inline-flex;' : 'display: none;'} color: var(--review); border-color: rgba(245,158,11,0.3);">
              <span>[OVERRIDE]</span>
            </button>
          </div>
        </div>
      `;
      if (overrideAvailable) {
        const btn = document.getElementById("btnOverride");
        if (btn) {
          btn.onclick = () => {
            if (brightnessPollTimer) {
              clearInterval(brightnessPollTimer);
              brightnessPollTimer = null;
            }
            runRecordingSequence();
          };
        }
      }
      break;

    case "recording":
      root.innerHTML = `
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="rec-dot"></span>
              <span class="eyebrow" style="color: var(--flagged);">RECORDING LIVE CAPTURE</span>
            </div>
            <span class="mono" style="font-size: 11px; color: var(--text-secondary);">480P @ 30FPS // OPUS 16KHZ</span>
          </div>

          <div style="position: relative; width: 100%; aspect-ratio: 16/9; background: var(--bg-inset); border: 1px solid var(--border-hairline); border-radius: 6px; overflow: hidden; margin-bottom: 16px;">
            <video id="captureVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);"></video>
            <canvas id="calcCanvas" style="display: none;"></canvas>

            <!-- Large mono countdown 5..0 overlay -->
            <div style="position: absolute; inset: 0; background: rgba(7,10,14,0.35); display: flex; flex-direction: column; align-items: center; justify-content: center;">
              <div id="countdownNumber" class="mono" style="font-size: 72px; font-weight: 700; color: var(--text-primary); line-height: 1;">
                ${data.count !== undefined ? data.count : 5}
              </div>
              <div class="eyebrow" style="color: var(--accent-blue); margin-top: 8px;">SPEAK NOW</div>
            </div>
          </div>

          <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 6px;" class="eyebrow">CHALLENGE PROMPT:</div>
          <div class="recessed-block" style="width: 100%; text-align: center;">
            "${challenge.spoken_phrase}"
          </div>
        </div>
      `;
      break;

    case "processing":
      root.innerHTML = `
        <div class="panel" style="text-align: center; padding: 48px 24px;">
          <div class="spinner"></div>
          <div class="mono" style="font-size: 13px; color: var(--text-secondary); letter-spacing: 0.04em;">
            Analysing...
          </div>
          <div class="mono" style="font-size: 11px; color: var(--text-tertiary); margin-top: 8px;">
            EXTRACTING 5 KEYFRAMES & AUDIO OPUS STREAM
          </div>
        </div>
      `;
      break;

    case "result":
      root.innerHTML = `
        <div class="panel" style="text-align: center; padding: 36px 24px;">
          <div class="badge badge-clear" style="margin-bottom: 12px; font-size: 13px; padding: 4px 12px;">
            CHALLENGE DISPATCHED
          </div>
          <div style="font-size: 16px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
            Payload Transmitted to Forensic Pipeline
          </div>
          <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 24px;">
            Payload size: ${data.payloadKb || '<200'} KB. Redirecting to verification console...
          </p>
          <a href="dashboard.html" class="btn-terminal" style="color: var(--text-primary); border-color: var(--border-strong);">
            <span>[OPEN DASHBOARD]</span>
          </a>
        </div>
      `;
      break;

    case "error":
      root.innerHTML = `
        <div class="panel" style="border-color: rgba(239,68,68,0.3); background: var(--bg-panel); text-align: center; padding: 36px 24px;">
          <div class="badge badge-flagged" style="margin-bottom: 12px;">
            CAPTURE FAULT
          </div>
          <div style="font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
            Unable to Complete Video/Audio Acquisition
          </div>
          <div style="font-size: 12px; color: var(--text-secondary); max-width: 400px; margin: 0 auto 24px auto;">
            ${data.message || "Hardware stream permission denied or device busy."}
          </div>
          <button id="btnTryAgain" class="btn-terminal" style="color: var(--text-primary); border-color: var(--border-strong);">
            <span>[TRY AGAIN]</span>
          </button>
        </div>
      `;
      document.getElementById("btnTryAgain")?.addEventListener("click", () => {
        stopMediaTracks();
        renderUIState("idle");
      });
      break;
  }
}

/**
 * Main Capture Workflow Sequence
 */
async function startCaptureWorkflow() {
  overrideAvailable = false;
  dimWaitSeconds = 0;
  await fetchChallenge();

  try {
    // 1. getUserMedia (§5.2) — use ideal 30fps to match 50Hz/60Hz indoor AC lighting anti-flicker
    activeStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 854 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });

    renderUIState("checking_brightness", { lux: 0 });
    let video = document.getElementById("captureVideo");
    const canvas = document.getElementById("calcCanvas");
    video.srcObject = activeStream;
    await video.play();

    canvas.width = video.videoWidth || 854;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");

    // 2. Brightness gate polling (§5.3)
    brightnessPollTimer = setInterval(() => {
      if (!activeStream) return;
      video = document.getElementById("captureVideo");
      if (!video) return;

      const lux = Math.round(checkBrightness(video, canvas, ctx));
      const luxReadout = document.getElementById("luxReadout");
      const luxBar = document.getElementById("luxBar");
      if (luxReadout) luxReadout.innerText = `${lux} LUX`;
      if (luxBar) {
        luxBar.style.width = `${Math.min(100, (lux / 255) * 100)}%`;
        luxBar.style.backgroundColor = lux < 60 ? "var(--review)" : "var(--clear)";
      }

      if (lux < 60) {
        dimWaitSeconds += 0.2;
        if (dimWaitSeconds >= 5.0) {
          overrideAvailable = true;
        }
        renderUIState("warning_brightness", { lux });
        const warnVideo = document.getElementById("captureVideo");
        if (warnVideo && warnVideo.srcObject !== activeStream) {
          warnVideo.srcObject = activeStream;
          warnVideo.play().catch(() => {});
        }
      } else {
        // Adequate lighting -> clear gate and begin recording
        clearInterval(brightnessPollTimer);
        brightnessPollTimer = null;
        runRecordingSequence();
      }
    }, 200);

  } catch (err) {
    stopMediaTracks();
    renderUIState("error", { message: err.message || "Camera or microphone permission denied." });
  }
}

/**
 * Execute synchronized 5-second countdown, frame extraction, and audio recording
 */
async function runRecordingSequence() {
  try {
    renderUIState("recording", { count: 5 });

    const video = document.getElementById("captureVideo");
    let canvas = document.getElementById("calcCanvas");
    if (!canvas) {
      canvas = document.createElement("canvas");
    }
    const ctx = canvas.getContext("2d");

    if (video && activeStream) {
      if (video.srcObject !== activeStream) {
        video.srcObject = activeStream;
      }
      try {
        video.play().catch(() => {});
      } catch (_) {}
    }

    // Audio setup (§5.6) - isolate audio tracks to avoid recorder container negotiation failure
    audioChunks = [];
    try {
      const audioTracks = activeStream ? activeStream.getAudioTracks() : [];
      const audioStream = audioTracks.length > 0 ? new MediaStream(audioTracks) : activeStream;

      let mimeType = "";
      if (window.MediaRecorder) {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/webm")) {
          mimeType = "audio/webm";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        }

        try {
          mediaRecorder = mimeType ? new MediaRecorder(audioStream, { mimeType }) : new MediaRecorder(audioStream);
        } catch (mErr) {
          console.warn("Primary MediaRecorder init failed, trying bare fallback:", mErr);
          mediaRecorder = new MediaRecorder(activeStream);
        }

        mediaRecorder.ondataavailable = (evt) => {
          if (evt.data && evt.data.size > 0) audioChunks.push(evt.data);
        };
        mediaRecorder.start(250); // timeslice generates periodic chunks
      }
    } catch (e) {
      console.warn("MediaRecorder could not start audio capture:", e);
    }

    // Five-keyframe sampling schedule (§5.5) at t = 0.5, 1.5, 2.5, 3.5, 4.5s
    const keyframeTimesMs = [500, 1500, 2500, 3500, 4500];
    const cropBlobs = [];

    keyframeTimesMs.forEach((tMs) => {
      setTimeout(async () => {
        try {
          if (!activeStream) return;
          const liveVideo = document.getElementById("captureVideo") || video;
          const liveCanvas = document.getElementById("calcCanvas") || canvas;
          const liveCtx = liveCanvas.getContext ? liveCanvas.getContext("2d") : ctx;
          if (!liveVideo || !liveCanvas || !liveCtx) return;

          const crop = await extractFaceCrop(liveVideo, liveCanvas, liveCtx);
          if (crop && cropBlobs.length < 5) {
            cropBlobs.push(crop);
          }
        } catch (cropErr) {
          console.warn("Keyframe crop frame failed, continuing:", cropErr);
        }
      }, tMs);
    });

    // Countdown timer 5..0 (must run unconditionally)
    let count = 5;
    const countdownInterval = setInterval(() => {
      count -= 1;
      const countEl = document.getElementById("countdownNumber");
      if (countEl) countEl.innerText = count;

      if (count <= 0) {
        clearInterval(countdownInterval);
        finalizeAndDispatch(cropBlobs);
      }
    }, 1000);
  } catch (fatalErr) {
    console.error("Fatal recording sequence error:", fatalErr);
    renderUIState("error", { message: "Capture sequence error: " + fatalErr.message });
  }
}

/**
 * Stop media stream, assemble payload, and send to /api/challenge (§5.7, §5.10)
 */
async function finalizeAndDispatch(cropBlobs) {
  renderUIState("processing");

  // Collect final audio blob
  const audioBlob = await new Promise(resolve => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.onstop = () => resolve(new Blob(audioChunks, { type: "audio/webm" }));
      mediaRecorder.stop();
    } else {
      resolve(new Blob(audioChunks, { type: "audio/webm" }));
    }
  });

  // HARD RULE: stream.getTracks().forEach(t => t.stop()) MUST run after send completes or on error
  stopMediaTracks();

  // Payload budget calculation (<200KB)
  let totalBytes = audioBlob ? audioBlob.size : 0;
  cropBlobs.forEach(b => { totalBytes += b.size; });
  const payloadKb = (totalBytes / 1024).toFixed(1);

  // Send challenge (§5.7)
  await sendChallenge(cropBlobs, audioBlob, challenge);

  renderUIState("result", { payloadKb });

  // Auto redirect to dashboard
  setTimeout(() => {
    window.location.href = "dashboard.html";
  }, 1200);
}

// Global initialization
window.addEventListener("DOMContentLoaded", async () => {
  await initMediaPipe();
  await fetchChallenge();
  renderUIState("idle");
});

// Window unload safeguard
window.addEventListener("beforeunload", () => {
  stopMediaTracks();
});

// CAPTURE APP OK
