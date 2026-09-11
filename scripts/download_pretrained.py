"""
scripts/download_pretrained.py
Helper utility to pre-download and cache top open-source pre-trained models
(CLIP vision model and ViT deepfake detector) for 100% offline hackathon execution.

Usage:
    python scripts/download_pretrained.py
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_models():
    logger.info("=== 1/2 Pre-downloading HuggingFace Deepfake Classifier ===")
    try:
        from transformers import pipeline
        logger.info("Downloading dima806/deepfake_vs_real_image_detection...")
        pipe = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection"
        )
        logger.info("✓ Deepfake classifier downloaded and cached successfully.")
    except Exception as e:
        logger.warning("Could not pre-download deepfake model: %s", e)

    logger.info("=== 2/2 Pre-downloading CLIP Foundation Model for GenAI Docs ===")
    try:
        from transformers import CLIPProcessor, CLIPModel
        model_id = "openai/clip-vit-base-patch32"
        logger.info("Downloading %s...", model_id)
        CLIPProcessor.from_pretrained(model_id)
        CLIPModel.from_pretrained(model_id)
        logger.info("✓ CLIP processor and model downloaded and cached successfully.")
    except Exception as e:
        logger.warning("Could not pre-download CLIP model: %s", e)

    logger.info("=== Ready for offline demo execution! ===")

if __name__ == "__main__":
    download_models()
