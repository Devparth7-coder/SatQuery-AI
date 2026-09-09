"""Unit tests: bi-temporal change detection."""
import json
import os

from core.change import detect, severity_reason
from core.features import compute_features, load_scene
from core.landcover import segment

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _pair():
    s1 = load_scene(os.path.join(SAMPLES, "delta_t1.png"), gsd=10.0)
    s2 = load_scene(os.path.join(SAMPLES, "delta_t2.png"), gsd=10.0)
    lc1 = segment(s1, compute_features(s1), k=7)
    return s1, s2, lc1


def test_changed_fraction_matches_ground_truth():
    with open(os.path.join(SAMPLES, "delta_truth.json")) as f:
        truth = json.load(f)
    s1, s2, lc1 = _pair()
    cr = detect(s1, s2, k=7, lc1=lc1)
    # Truth 4.47%, detected ~3.67%: assert close, not exact (algorithmic).
    assert abs(cr.changed_fraction - truth["total_changed_fraction"]) < 0.015
    assert 0.02 < cr.changed_fraction < 0.06


def test_registration_recovers_shift():
    import re
    s1, s2, lc1 = _pair()
    cr = detect(s1, s2, k=7, lc1=lc1)
    assert "ORB+RANSAC" in cr.registration
    m = re.search(r"([\d.]+)\s*px shift", cr.registration)
    assert m and abs(float(m.group(1)) - 7.2) < 2.5  # applied shift |(6,-4)|≈7.2


def test_severity_and_quality_present():
    s1, s2, lc1 = _pair()
    cr = detect(s1, s2, k=7, lc1=lc1)
    assert cr.severity in ("LOW", "MEDIUM", "HIGH")
    assert cr.quality["registration"]["quality"] in ("HIGH", "MODERATE", "LOW", "FAILED")
    assert cr.quality["histogram_matched"] is True
    assert "LOW<" in severity_reason(cr.changed_fraction, cr.quality["registration"])


def test_hotspots_schema():
    s1, s2, lc1 = _pair()
    cr = detect(s1, s2, k=7, lc1=lc1)
    assert cr.hotspots, "expected change hotspots"
    for h in cr.hotspots:
        assert len(h["bbox"]) == 4 and h["area_km2"] > 0
        assert h["severity"] in ("LOW", "MEDIUM", "HIGH")


def test_dominant_transition_is_vegetation_to_bare():
    s1, s2, lc1 = _pair()
    cr = detect(s1, s2, k=7, lc1=lc1)
    assert cr.transitions, "expected transitions"
    top = cr.transitions[0]
    assert top["from"] == "dense_vegetation" and top["to"] == "bare_soil"
    assert top["area_km2"] > 0.5
