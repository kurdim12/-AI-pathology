"""
Tests for the REST API (app/api.py).

Skipped automatically if FastAPI / its TestClient isn't installed, so the core
suite still runs in a minimal environment. Uses random-init weights and
synthetic PNGs — no dataset, no downloads.

Run::

    python -m pytest tests/test_api.py -q
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi", reason="FastAPI not installed; API tests skipped.")
pytest.importorskip("httpx", reason="httpx not installed; TestClient unavailable.")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def _png_bytes(seed: int = 0) -> io.BytesIO:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "thresholds" in body and "urgent" in body["thresholds"]


def test_predict_single():
    import config

    resp = client.post("/predict", files={"file": ("slide.png", _png_bytes(1), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    # label is a class name (binary or, with a multi-class checkpoint, a subtype).
    assert body["label"] in config.CLASS_NAMES
    assert 0.0 <= body["prob_malignant"] <= 1.0
    assert body["priority"] in ("URGENT", "REVIEW", "ROUTINE")
    assert body["filename"] == "slide.png"


def test_triage_batch_is_ranked():
    files = [("files", (f"s{i}.png", _png_bytes(i), "image/png")) for i in range(6)]
    resp = client.post("/triage", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 6
    worklist = body["worklist"]
    # Ranks are contiguous 1..n.
    assert [row["rank"] for row in worklist] == list(range(1, 7))
    # Sorted by priority then descending P(malignant).
    rank = {"URGENT": 0, "REVIEW": 1, "ROUTINE": 2}
    keys = [(rank[row["priority"]], -row["prob_malignant"]) for row in worklist]
    assert keys == sorted(keys)
