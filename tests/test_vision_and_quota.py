import io
import json
from datetime import date

import numpy as np
import pytest
from PIL import Image

from kisan_agent import vision
from kisan_agent.ratelimit import DailyQuota


def jpeg(w=640, h=480, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (40, 140, 60)).save(buf, fmt)
    return buf.getvalue()


def test_preprocess_shape_and_normalisation():
    x = vision.preprocess(jpeg(800, 300, fmt="PNG"))  # lossless, so pixel values are exact
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert abs(x[0, 1].mean() - ((140 / 255 - 0.456) / 0.224)) < 0.01  # green channel


def test_no_model_is_reported_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "MODEL_DIR", tmp_path)
    assert vision.classify(jpeg())["available"] is False


def test_garbage_bytes_give_error(tmp_path, monkeypatch):
    (tmp_path / "leaf_classifier.onnx").write_bytes(b"x")
    monkeypatch.setattr(vision, "MODEL_DIR", tmp_path)
    assert "error" in vision.classify(b"not an image")


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    """A 3-class ONNX model built in-memory, so the inference path is tested without training."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 3))
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), tmp_path / "leaf_classifier.onnx",
                      input_names=["image"], output_names=["logits"], dynamo=False)
    (tmp_path / "labels.json").write_text(json.dumps(["Tomato leaf", "Tomato Early blight leaf", "Corn rust leaf"]))
    monkeypatch.setattr(vision, "MODEL_DIR", tmp_path)
    vision._load.cache_clear()
    yield
    vision._load.cache_clear()


def test_classify_returns_ranked_probabilities(tiny_model):
    out = vision.classify(jpeg(), top_k=3)
    confs = [g["confidence"] for g in out["top_guesses"]]
    assert out["available"] and len(confs) == 3
    assert confs == sorted(confs, reverse=True) and abs(sum(confs) - 1) < 0.01
    assert isinstance(out["confident"], bool)


def test_quota_blocks_after_limit():
    q = DailyQuota(limit=2)
    assert [q.allow(1), q.allow(1), q.allow(1)] == [True, True, False]
    assert q.allow(2)  # other users unaffected


def test_quota_resets_next_day():
    day = [date(2026, 10, 2)]
    q = DailyQuota(limit=1, today=lambda: day[0])
    assert q.allow(1) and not q.allow(1)
    day[0] = date(2026, 10, 3)
    assert q.allow(1)
