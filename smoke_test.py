from shared.contracts import (SignalResult, SignalId, Severity,
                              SIGNAL_WEIGHTS, HARD_SIGNALS, TIERS)

def main():
    r = SignalResult(signal=SignalId.FACE_DEEPFAKE, raw_score=0.9,
                     confidence=0.9, severity=Severity.HARD, triggered=True,
                     label="test")
    d = r.to_dict()
    assert d["signal"] == "face_deepfake" and d["severity"] == "hard"
    assert SignalId.FACE_DEEPFAKE in HARD_SIGNALS
    assert set(SIGNAL_WEIGHTS) >= {SignalId.GENAI_DOC, SignalId.VOICE_SPOOF}
    import modules.forensics.genai_detector, modules.face.deepfake_detector
    import modules.face.challenge, modules.face.active_liveness
    import modules.audio.asr, modules.audio.antispoof
    import shared.audit_chain, shared.pepper_sss
    import modules.decision.fusion, modules.decision.calibration
    print("SMOKE OK — contracts + all stubs import clean")

if __name__ == "__main__":
    main()
