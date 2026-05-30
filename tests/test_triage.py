"""
Tests for the triage queue and Grad-CAM target autodetect.

These run on CPU with randomly initialised weights and synthetic images — no
dataset, no downloads — and verify the worklist is built and sorted correctly
and that an arbitrary backbone's Grad-CAM target layer can be found by probing.

Run::

    python -m pytest tests/ -q
    # or
    python -m tests.test_triage
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

import config
from src.gradcam import GradCAM
from src.model import autodetect_target_layer, build_model
from src.triage import (
    WorklistItem,
    list_images,
    worklist_from_paths,
    write_worklist_csv,
)

_PRIORITY_RANK = {"URGENT": 0, "REVIEW": 1, "ROUTINE": 2}


def _make_images(folder: str, n: int = 6) -> None:
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, f"slide_{i:02d}.png"))


def test_list_images_finds_files():
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=4)
        paths = list_images(d)
        assert len(paths) == 4
        assert all(p.endswith(".png") for p in paths)


def test_worklist_is_sorted_and_ranked():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=6)
        items = worklist_from_paths(
            list_images(d), model=model, backbone="resnet18", trained=False
        )

    assert len(items) == 6
    assert all(isinstance(it, WorklistItem) for it in items)
    # Ranks are 1..n and contiguous.
    assert [it.rank for it in items] == list(range(1, len(items) + 1))
    # Sorted by (priority rank, descending P(malignant)).
    keys = [(_PRIORITY_RANK[it.priority], -it.prob_malignant) for it in items]
    assert keys == sorted(keys)


def test_worklist_csv_roundtrip():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=3)
        items = worklist_from_paths(
            list_images(d), model=model, backbone="resnet18", trained=False
        )
        csv_path = os.path.join(d, "worklist.csv")
        write_worklist_csv(items, csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as fh:
            lines = fh.read().strip().splitlines()
        # header + one row per item
        assert len(lines) == len(items) + 1
        assert lines[0].startswith("rank,filename")


def test_autodetect_target_layer_enables_gradcam():
    # Use an EfficientNet but discover the layer by probing rather than by name.
    model = build_model(backbone="efficientnet_b0", pretrained=False).eval()
    target = autodetect_target_layer(model)
    cam_engine = GradCAM(model, target)
    import torch

    cam, logits = cam_engine(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    cam_engine.remove()
    assert cam.shape == (1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert float(cam.min()) >= -1e-6 and float(cam.max()) <= 1 + 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\nAll {len(fns)} triage tests passed.")


if __name__ == "__main__":
    _run_all()
