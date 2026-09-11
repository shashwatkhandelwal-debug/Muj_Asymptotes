"""
scripts/test_verify_live.py
Live end-to-end verification of /api/verify endpoint using known-fake inputs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.orchestrator import app

def run_test():
    client = TestClient(app)
    doc_path = "data/calibration/genai_doc/fake/fake_genai_000.jpg"
    face_path = "data/calibration/face_deepfake/fake/fake_000.jpg" # Celeb-DF deepfake face (>99% fake)
    audio_path = "data/calibration/voice_spoof/fake/fake_000_USA_female_1_s1.mp3"

    with open(doc_path, "rb") as f_doc, open(face_path, "rb") as f_face, open(audio_path, "rb") as f_aud:
        response = client.post(
            "/api/verify",
            files={
                "document_image": ("fake_doc.jpg", f_doc, "image/jpeg"),
                "video": ("fake_face.jpg", f_face, "image/jpeg"),
                "audio": ("fake_audio.mp3", f_aud, "audio/mpeg")
            }
        )

    print(f"HTTP Status: {response.status_code}")
    data = response.json()
    print(f"Final Score: {data['final']['score']}, Tier: {data['final']['tier']}")
    print(f"Floors applied: {data['final']['floors_applied']}")
    print(f"Summary: {data['summary']}\n")
    print("Signal Lines Breakdown:")
    for line in data.get("lines", []):
        trig = "TRIGGERED" if line.get("triggered") else "CLEAR"
        print(f"  [{trig}] {line['signal']:<16} penalty={line.get('penalty'):<5} conf={line.get('confidence', 0):.2f} - {line.get('label')}")

if __name__ == "__main__":
    run_test()
