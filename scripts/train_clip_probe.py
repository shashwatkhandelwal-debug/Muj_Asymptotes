"""
scripts/train_clip_probe.py
Extracts CLIP ViT-B/32 image embeddings from real and GenAI document samples,
fits a logistic regression probe, and exports models/clip_genai_probe.pt.
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from transformers import CLIPProcessor, CLIPModel

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_probe():
    print("Loading pre-trained CLIP model from local cache...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(model_id, local_files_only=True)
    clip_model.eval()

    def embed_image(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            inputs = processor(images=rgb, return_tensors="pt")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    feat = feat.pooler_output
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy()[0]

    # Collect data paths
    real_paths = list((REPO_ROOT / "data" / "calibration" / "genai_doc" / "real").glob("*.*"))
    fake_paths = list((REPO_ROOT / "data" / "calibration" / "genai_doc" / "fake").glob("*.*"))

    if not real_paths or not fake_paths:
        print("Dataset paths missing! Checking fallback directories...")
        real_paths = list((REPO_ROOT / "data" / "real_docs").glob("*.*"))
        fake_paths = list((REPO_ROOT / "data" / "genai_docs").glob("*.*"))

    print(f"Found {len(real_paths)} real and {len(fake_paths)} fake document images.")
    X, y = [], []

    for p in real_paths:
        try:
            X.append(embed_image(p))
            y.append(0)
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

    for p in fake_paths:
        try:
            X.append(embed_image(p))
            y.append(1)
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    print(f"Feature matrix shape: {X.shape}, label balance: {np.bincount(y)}")

    # Fit Logistic Regression probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    acc = float(clf.score(X, y))
    print(f"Probe Training Accuracy: {acc * 100:.2f}%")

    # Export probe head
    head = nn.Linear(512, 1)
    head.weight.data = torch.tensor(clf.coef_, dtype=torch.float32)
    head.bias.data = torch.tensor(clf.intercept_, dtype=torch.float32)

    out_path = MODELS_DIR / "clip_genai_probe.pt"
    torch.save({"state_dict": head.state_dict(), "accuracy": acc}, str(out_path))
    print(f"[OK] Successfully saved probe to {out_path}")

if __name__ == "__main__":
    train_probe()
