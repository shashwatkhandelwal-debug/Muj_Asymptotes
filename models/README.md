# Model weights — NOT committed to git

These files are exported from Colab and copied here manually.
`.gitignore` excludes this folder except for this README.

| File | Used by | Export notebook |
|---|---|---|
| `clip_genai_probe.pt` | genai_detector.py | notebooks/01_clip_probe.ipynb |
| `deepfake.onnx` | deepfake_detector.py | notebooks/02_deepfake_export.ipynb |
| `antispoof.onnx` | antispoof.py | notebooks/03_antispoof_export.ipynb |

If a file is absent, its detector falls back to a heuristic automatically.
The system never breaks on a missing model, it just gets weaker.
