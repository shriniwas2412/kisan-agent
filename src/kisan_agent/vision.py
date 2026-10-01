"""Leaf-disease classifier: MobileNetV3-Small fine-tuned on PlantDoc, run with onnxruntime.

Train it with training/train_leaf.py (writes models/leaf_classifier.onnx + labels.json).
Without a model file this returns an explicit "not available" result, so the agent
tells the farmer honestly instead of guessing a disease.
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(os.getenv("KISAN_MODEL_DIR", "models"))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# Below this, the top guess is reported as uncertain. PlantDoc test top-1 is far from
# perfect, so the agent must never present a low-confidence guess as a diagnosis.
CONFIDENT = 0.6


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_DIR / "leaf_classifier.onnx"), providers=["CPUExecutionProvider"])
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    return session, labels


def preprocess(image_bytes: bytes):
    """Same as eval in training: resize 256 → center crop 224 → ImageNet normalisation, NCHW float32."""
    import numpy as np
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    scale = 256 / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def classify(image_bytes: bytes, top_k: int = 3) -> dict:
    if not (MODEL_DIR / "leaf_classifier.onnx").exists():
        return {
            "available": False,
            "message": "Leaf classifier is not installed. Ask the farmer to describe the symptoms "
            "(spots, colour, which leaves) and to show the plant at the nearest KVK.",
        }
    import numpy as np

    try:
        x = preprocess(image_bytes)
    except Exception:
        return {"available": True, "error": "Could not read the image. Ask the farmer to send a clear photo."}

    session, labels = _load()
    logits = session.run(None, {"image": x})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(probs)[::-1][:top_k]
    guesses = [{"label": labels[i], "confidence": round(float(probs[i]), 3)} for i in top]
    return {
        "available": True,
        "top_guesses": guesses,
        "confident": guesses[0]["confidence"] >= CONFIDENT,
        "note": "Model trained on PlantDoc field photos; treat as a hint, not a diagnosis. Confirm at the KVK.",
    }
