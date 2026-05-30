"""
Shared pytest fixtures.

The triage taxonomy lives in module-level ``config`` state (CLASS_NAMES,
MALIGNANT_CLASSES), and multi-class mode mutates it globally. To keep tests
isolated and order-independent, reset to the binary default before every test.
Tests that exercise multi-class set their own taxonomy and this fixture restores
it afterwards.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def _reset_taxonomy():
    saved = (list(config.CLASS_NAMES), list(config.MALIGNANT_CLASSES))
    config.use_multiclass(["Benign", "Malignant"], ["Malignant"])
    yield
    config.use_multiclass(*saved)
