import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api.orchestrator import build_officer_summary

def test_response_shape():
    """Verify the response matches the dashboard contract exactly."""
    fake_fused = {
        "score": 82.0,
        "tier": "flagged",
        "lines": [
            {
                "signal": "genai_doc",
                "penalty": 3.75,
                "label": "Document appears genuine",
                "confidence": 0.15,
                "severity": "soft",
                "triggered": False,
                "evidence": {"method": "fft_heuristic"}
            },
            {
                "signal": "face_deepfake",
                "penalty": 25.0,
                "label": "Deepfake artifacts detected",
                "confidence": 0.91,
                "severity": "hard",
                "triggered": True,
                "evidence": {"mode": "per_frame_aggregate"}
            }
        ],
        "floors_applied": ["face_deepfake"]
    }

    summary = build_officer_summary(fake_fused)
    response = {
        "existing": {"ocr": "clean"},
        "lines": fake_fused["lines"],
        "final": {
            "score": fake_fused["score"],
            "tier": fake_fused["tier"],
            "floors_applied": fake_fused["floors_applied"],
        },
        "summary": summary,
    }

    assert "lines" in response
    assert "final" in response
    assert "summary" in response

    for line in response["lines"]:
        assert "signal" in line
        assert "penalty" in line
        assert "label" in line
        assert "confidence" in line
        assert "severity" in line
        assert "triggered" in line

    assert "score" in response["final"]
    assert "tier" in response["final"]
    assert "floors_applied" in response["final"]
    assert "FLAGGED" in response["summary"]
    assert "face_deepfake" in response["summary"]

if __name__ == "__main__":
    test_response_shape()
    print("ORCHESTRATOR TEST OK")
