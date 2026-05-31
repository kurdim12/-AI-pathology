"""
Tests for the synthetic data generator.

Verify the difficulty presets behave as intended: 'easy' is clean and separable,
'realistic' introduces label noise and overlapping distributions. CPU-only, tiny
sample, writes to a temp dir.

Run::

    python -m pytest tests/test_demo_data.py -q
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import config
import scripts.make_demo_data as mdd


def _count_images(root: str) -> dict:
    counts = {}
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            counts[cls] = len([f for f in os.listdir(d) if f.endswith(".png")])
    return counts


def test_easy_binary_is_clean_and_balanced():
    with tempfile.TemporaryDirectory() as d:
        saved = config.TRAIN_DIR
        config.TRAIN_DIR = d
        try:
            mdd.generate(per_class=20, size=48, difficulty="easy")
            counts = _count_images(d)
        finally:
            config.TRAIN_DIR = saved
    # No label noise -> exactly per_class images per class folder.
    assert counts == {"Benign": 20, "Malignant": 20}


def test_realistic_introduces_label_noise():
    # With label noise, the per-folder counts deviate from the clean balance
    # (some images get filed under a different class), but the total is conserved.
    with tempfile.TemporaryDirectory() as d:
        saved = config.TRAIN_DIR
        config.TRAIN_DIR = d
        try:
            mdd.generate(per_class=60, size=48, difficulty="realistic", seed=1)
            counts = _count_images(d)
        finally:
            config.TRAIN_DIR = saved
    assert sum(counts.values()) == 120                      # total conserved
    assert set(counts) == {"Benign", "Malignant"}
    # At least one folder is off the clean 60 (label noise happened).
    assert any(v != 60 for v in counts.values())


def test_realistic_classes_overlap_in_color():
    # The whole point of 'realistic': benign/malignant are NOT cleanly separable
    # by mean colour the way 'easy' is. Mean-colour gap should shrink markedly.
    rng = np.random.default_rng(0)

    def mean_gap(difficulty):
        hardness, _ = mdd._DIFFICULTY[difficulty]
        sigs = mdd._BINARY_SIGNATURES
        means = {}
        for name, sig in sigs.items():
            arr = np.stack([
                np.asarray(mdd._make_image(rng, sig, 48, hardness)).mean(axis=(0, 1))
                for _ in range(15)
            ])
            means[name] = arr.mean(axis=0)
        return float(np.linalg.norm(means["Benign"] - means["Malignant"]))

    # Realistic colour separation should be much smaller than easy's.
    assert mean_gap("realistic") < mean_gap("easy")


def test_multiclass_four_folders():
    with tempfile.TemporaryDirectory() as d:
        saved = config.TRAIN_DIR
        config.TRAIN_DIR = d
        try:
            mdd.generate(per_class=10, size=48, difficulty="easy", multiclass=True)
            counts = _count_images(d)
        finally:
            config.TRAIN_DIR = saved
    assert len(counts) == 4
    assert "ductal_carcinoma" in counts and "adenosis_benign" in counts


def test_invalid_difficulty_raises():
    try:
        mdd.generate(per_class=2, difficulty="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
