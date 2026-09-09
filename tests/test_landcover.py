"""Unit tests: land-cover segmentation + quality layer."""
import os

import numpy as np

from core.features import compute_features, load_scene
from core.landcover import (CLASSES, class_mask, quality_indicators, segment,
                            suggest_k)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HARBOUR = os.path.join(SAMPLES, "harbour_port.png")


def _seg():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    return s, fs, segment(s, fs, k=7)


def test_fractions_sum_to_one():
    _, _, lc = _seg()
    assert abs(sum(lc.fractions.values()) - 1.0) < 1e-6
    assert set(lc.fractions) == set(CLASSES)


def test_cluster_report_schema():
    _, _, lc = _seg()
    assert 1 <= len(lc.cluster_report) <= 7
    for r in lc.cluster_report:
        assert r["assigned"] in CLASSES
        assert 0.3 <= r["confidence"] <= 1.0
        assert "signature" in r and "scores" in r


def test_suggest_k_bounded_and_deterministic():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    k1, k2 = suggest_k(fs), suggest_k(fs)
    assert k1 == k2 and 4 <= k1 <= 10


def test_uncertainty_map_shape_and_range():
    _, _, lc = _seg()
    assert lc.uncertainty is not None
    assert lc.uncertainty.shape == lc.label_map.shape
    assert float(lc.uncertainty.min()) >= 0.0 and float(lc.uncertainty.max()) <= 1.0


def test_quality_label_has_rules():
    _, _, lc = _seg()
    q = lc.quality
    assert q["label"] in ("HIGH", "MODERATE", "LOW", "INSUFFICIENT")
    assert isinstance(q["rules"], list)
    # RGB-only input must fire the NIR rule (transparent downgrade).
    assert any("NIR" in r for r in q["rules"])
    assert q["label"] in ("MODERATE", "LOW", "INSUFFICIENT")


def test_quality_indicators_function():
    _, _, lc = _seg()
    q = quality_indicators(lc, has_nir=True)
    assert not any("NIR" in r for r in q["rules"])


def test_class_mask_union():
    _, _, lc = _seg()
    m = class_mask(lc, ["water", "built_up"])
    assert m.dtype == bool and m.shape == lc.label_map.shape
    assert m.sum() > 0
