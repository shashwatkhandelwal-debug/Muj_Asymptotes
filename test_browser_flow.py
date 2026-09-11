"""
test_browser_flow.py — End-to-End Real Browser Automation Test with Playwright Chromium.
Runs the complete 4-stage user journey with fake media devices, verifies each stage transition,
and pulls the resulting flow_debug_log from audit.db.
"""
import os
import sys
import time
import json
import sqlite3
import subprocess
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

def run_browser_test():
    import uvicorn
    import threading
    from playwright.sync_api import sync_playwright

    print("=" * 70)
    print("STARTING REAL BROWSER FLOW DIAGNOSTIC TEST")
    print("=" * 70)

    # 1. Clear previous debug logs for clean trace
    db = sqlite3.connect("audit.db")
    c = db.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS flow_debug_log (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    c.execute("DELETE FROM flow_debug_log")
    db.commit()
    db.close()

    # 2. Check if FastAPI server is reachable
    import requests
    server_running = False
    try:
        r = requests.get("http://127.0.0.1:8000/api/challenge/new", timeout=1.0)
        if r.status_code == 200:
            server_running = True
            print("[server] Found already running server on port 8000")
    except Exception:
        server_running = False

    server_thread = None
    if not server_running:
        print("[server] Starting uvicorn on port 8000...")
        def start_server():
            from api.orchestrator import app
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:8000/api/challenge/new", timeout=1.0)
                if r.status_code == 200:
                    print("[server] Server is up and accepting requests!")
                    break
            except Exception:
                time.sleep(0.5)

    # 3. Create a test sample image for Stage 1 if not exists
    sample_img_path = os.path.join(os.getcwd(), "test_sample_aadhaar.jpg")
    try:
        import cv2
        import numpy as np
        card = np.full((380, 600, 3), (240, 240, 240), dtype=np.uint8)
        cv2.putText(card, "GOVERNMENT OF INDIA", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 0), 2)
        cv2.putText(card, "AADHAAR", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(card, "Name: Priya Sharma", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(card, "DOB: 12/04/1994", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
        cv2.putText(card, "1234 5678 9010", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.imwrite(sample_img_path, card)
    except Exception as e:
        print("[sample] Error creating sample card:", e)

    stage_timings = {}
    stage_statuses = {}
    console_logs = []
    network_errors = []

    # 4. Launch Playwright with fake media devices
    with sync_playwright() as p:
        print("[playwright] Launching Chromium with fake video and audio streams...")
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--allow-file-access-from-files",
                "--disable-web-security",
            ]
        )
        context = browser.new_context(
            permissions=["camera", "microphone"],
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_logs.append(f"[PAGE ERROR] {err}"))
        page.on("requestfailed", lambda req: network_errors.append(f"[NET FAIL] {req.method} {req.url}: {req.failure}"))

        print("[browser] Navigating to http://127.0.0.1:8000/index.html...")
        page.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle")
        time.sleep(1.0)

        # -------------------------------------------------------------
        # STAGE 1: DOCUMENT UPLOAD & VERIFY
        # -------------------------------------------------------------
        print("\n--- [STAGE 1] Testing Document Upload & Verification ---")
        t1 = time.perf_counter()
        try:
            # Switch to Upload Mode
            btn_upload_mode = page.locator("#btnModeUpload")
            if btn_upload_mode.is_visible():
                btn_upload_mode.click()
                time.sleep(0.5)

            # Set file input
            file_input = page.locator("#fileInputFull")
            file_input.set_input_files(sample_img_path)
            time.sleep(0.5)

            # Load QR sample
            qr_sample_path = os.path.join(os.getcwd(), "frontend", "static", "aadhaar_qr_sample.png")
            if os.path.exists(qr_sample_path):
                page.locator("#fileInputQR").set_input_files(qr_sample_path)
                time.sleep(0.5)

            # Click Analyze & Verify Document
            print("[Stage 1] Submitting Document Verification request...")
            page.locator("#btnVerifyUpload").click()

            # Wait for inline result with adequate timeout for CPU ML inference
            page.wait_for_selector("#docResultInline", state="visible", timeout=45000)
            stage_timings["stage1"] = (time.perf_counter() - t1) * 1000.0
            stage_statuses["stage1"] = "SUCCESS - Document verified and results rendered"
            print(f"[Stage 1] Passed in {stage_timings['stage1']:.1f}ms")

            # Click Continue to Facial Liveness
            time.sleep(1.0)
            page.locator("button:has-text('CONTINUE TO FACIAL LIVENESS')").click()
        except Exception as e:
            stage_timings["stage1"] = (time.perf_counter() - t1) * 1000.0
            stage_statuses["stage1"] = f"FAILED: {str(e)}"
            print(f"[Stage 1] Failed: {e}")

        # -------------------------------------------------------------
        # STAGE 2: FACIAL LIVENESS
        # -------------------------------------------------------------
        print("\n--- [STAGE 2] Testing Facial Liveness Video Capture ---")
        t2 = time.perf_counter()
        try:
            page.wait_for_selector("#stage2.active", timeout=10000)
            print("[Stage 2] Stage 2 view active. Waiting for camera stream...")
            time.sleep(1.5)

            # Click Start Facial Capture
            btn_start_face = page.locator("#btnStartFaceCapture")
            btn_start_face.click()
            print("[Stage 2] Started facial capture. Waiting for 3-second keyframe countdown...")

            # Wait for auto-advance to Stage 3
            page.wait_for_selector("#stage3.active", timeout=20000)
            stage_timings["stage2"] = (time.perf_counter() - t2) * 1000.0
            stage_statuses["stage2"] = "SUCCESS - 5 Face keyframes captured and advanced to Stage 3"
            print(f"[Stage 2] Passed in {stage_timings['stage2']:.1f}ms")
        except Exception as e:
            stage_timings["stage2"] = (time.perf_counter() - t2) * 1000.0
            stage_statuses["stage2"] = f"FAILED: {str(e)}"
            print(f"[Stage 2] Failed: {e}")

        # -------------------------------------------------------------
        # STAGE 3: VOCAL CHALLENGE & SUBMISSION
        # -------------------------------------------------------------
        print("\n--- [STAGE 3] Testing Vocal Challenge Audio Recording & Submission ---")
        t3 = time.perf_counter()
        try:
            page.wait_for_selector("#stage3.active", timeout=10000)
            phrase_text = page.locator("#spokenPhraseDisplay").inner_text()
            print(f"[Stage 3] Active challenge phrase: {phrase_text}")

            # Click Record & Complete Challenge
            btn_record = page.locator("#btnStartVoiceRecord")
            btn_record.click()
            print("[Stage 3] Audio recording started (10-second capture timer running)...")

            # Wait for Stage 4 to become active (after 10s recording + forensic synthesis API call)
            print("[Stage 3] Waiting for challenge submission and multi-signal fusion...")
            page.wait_for_selector("#stage4.active", timeout=45000)
            stage_timings["stage3"] = (time.perf_counter() - t3) * 1000.0
            stage_statuses["stage3"] = "SUCCESS - Voice challenge dispatched and merged"
            print(f"[Stage 3] Passed in {stage_timings['stage3']:.1f}ms")
        except Exception as e:
            stage_timings["stage3"] = (time.perf_counter() - t3) * 1000.0
            stage_statuses["stage3"] = f"FAILED: {str(e)}"
            print(f"[Stage 3] Failed: {e}")

        # -------------------------------------------------------------
        # STAGE 4: FORENSIC AUDIT DASHBOARD
        # -------------------------------------------------------------
        print("\n--- [STAGE 4] Inspecting Final Forensic Audit Report ---")
        t4 = time.perf_counter()
        try:
            page.wait_for_selector("#stage4.active", timeout=10000)
            score_text = page.locator("#scoreValue").inner_text()
            tier_text = page.locator("#tierBadge").inner_text()
            summary_text = page.locator("#dossierSummary").inner_text()
            channels_text = page.locator("#channelCount").inner_text()

            stage_timings["stage4"] = (time.perf_counter() - t4) * 1000.0
            stage_statuses["stage4"] = f"SUCCESS - Final Score: {score_text}, Tier: {tier_text}, Channels: {channels_text}"
            print(f"[Stage 4] Fused Score: {score_text}")
            print(f"[Stage 4] Fused Tier: {tier_text}")
            print(f"[Stage 4] Summary: {summary_text[:120]}...")
        except Exception as e:
            stage_timings["stage4"] = (time.perf_counter() - t4) * 1000.0
            stage_statuses["stage4"] = f"FAILED: {str(e)}"
            print(f"[Stage 4] Failed: {e}")

        time.sleep(1.0)
        browser.close()

    # 5. Flush flow logger and read database rows
    from shared.flow_logger import flush_logs, get_recent_flow_logs
    flush_logs(timeout=3.0)
    logs = get_recent_flow_logs(limit=100)

    print("\n" + "=" * 70)
    print("FLOW STAGE EXECUTION SUMMARY (BROWSER PERSPECTIVE):")
    print("=" * 70)
    for stage, status in stage_statuses.items():
        timing = stage_timings.get(stage, 0.0)
        print(f"  • {stage.upper()}: {status} ({timing:.1f}ms)")

    print("\n" + "=" * 70)
    print(f"DATABASE DIAGNOSTIC LOGS FROM audit.db ({len(logs)} rows logged):")
    print("=" * 70)
    for idx, row in enumerate(logs, start=1):
        print(f"\n--- LOG ROW #{idx} [ID={row['id']}] ---")
        print(f"  Timestamp:     {row['timestamp']}")
        print(f"  Endpoint:      {row['method']} {row['endpoint']}")
        print(f"  Status Code:   {row['status_code']}")
        print(f"  Stage Name:    {row['stage_name']}")
        print(f"  Duration:      {row['duration_ms']:.2f} ms")
        print(f"  Files Summary: {json.dumps(row['files_summary'], indent=4)}")
        print(f"  Form Fields:   {json.dumps(row['form_fields'], indent=4)}")
        print(f"  Final Score:   {row['final_score']} (Tier: {row['final_tier']})")
        print(f"  Server Errors: {json.dumps(row['server_errors'], indent=4)}")
        print(f"  Evaluated Signals ({len(row.get('evaluated_signals', []))}):")
        for sig in row.get("evaluated_signals", []):
            print(f"    - {sig['signal']}: raw_score={sig.get('raw_score')}, conf={sig.get('confidence')}, triggered={sig.get('triggered')}, sev={sig.get('severity')}, ok={sig.get('ok')}")
            if sig.get("evidence", {}).get("error"):
                print(f"      [EVIDENCE ERROR]: {sig['evidence']['error']}")

    if console_logs:
        print("\n" + "=" * 70)
        print(f"BROWSER CONSOLE LOGS ({len(console_logs)} entries):")
        print("=" * 70)
        for c in console_logs[:20]:
            print(f"  {c}")

    if network_errors:
        print("\n" + "=" * 70)
        print(f"NETWORK / HTTP ERRORS ({len(network_errors)} entries):")
        print("=" * 70)
        for ne in network_errors:
            print(f"  {ne}")

    return {
        "stage_statuses": stage_statuses,
        "stage_timings": stage_timings,
        "logs": logs,
        "console_logs": console_logs,
        "network_errors": network_errors,
    }

if __name__ == "__main__":
    run_browser_test()
