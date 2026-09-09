"""Unit tests: object detectors + unified schema."""
import os

from core.features import compute_features, load_scene
from core.landcover import segment
from core.objects import (CONFIDENCE_NOTE, detection_to_dict, detections_table,
                          size_statistics, summarize)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HARBOUR = os.path.join(SAMPLES, "harbour_port.png")


def _objs():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    lc = segment(s, fs, k=7)
    return s, summarize(s, fs, lc)


def test_summarize_keys():
    _, objs = _objs()
    for k in ("water_regions", "veg_regions", "built_regions", "vessels",
              "bright_targets", "linear", "urban_edge_density"):
        assert k in objs


def test_harbour_has_vessels_and_water():
    _, objs = _objs()
    assert len(objs["water_regions"]) > 0
    assert len(objs["vessels"]) > 0  # harbour scene has ships


def test_unified_schema_keys():
    _, objs = _objs()
    rows = detections_table(objs)
    assert rows, "expected some detections"
    for r in rows[:5]:
        for k in ("id", "type", "bbox", "centroid", "area_px", "area_m2",
                  "confidence", "source_method", "evidence"):
            assert k in r, k
        assert len(r["bbox"]) == 4 and len(r["centroid"]) == 2
        assert 0.0 <= r["confidence"] <= 1.0


def test_detection_to_dict_single():
    _, objs = _objs()
    d = objs["vessels"][0]
    row = detection_to_dict(d, "D001")
    assert row["id"] == "D001" and row["type"] == "vessel"
    assert "length_m" in row["evidence"]


def test_size_statistics():
    _, objs = _objs()
    st = size_statistics(objs["vessels"])
    assert st["n"] == len(objs["vessels"])
    assert st["length_m"]["min"] <= st["length_m"]["median"] <= st["length_m"]["max"]
    assert size_statistics([])["n"] == 0
    assert "calibrated" in CONFIDENCE_NOTE  # honesty note present
