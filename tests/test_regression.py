"""Regression tests: known scenes / known pair / determinism / edge cases."""
import json
import os

import numpy as np

from core import objects as obj_mod
from core.change import detect
from core.features import compute_features, load_scene
from core.ingestion import ingest_file, UNAVAILABLE
from core.landcover import segment

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_harbour_water_fraction_stable():
    s = load_scene(os.path.join(SAMPLES, "harbour_port.png"), gsd=10.0)
    lc = segment(s, compute_features(s), k=7)
    assert abs(lc.fractions["water"] - 0.588) < 0.02
    assert abs(lc.fractions["built_up"] - 0.412) < 0.02


def test_delta_pair_matches_truth_document():
    with open(os.path.join(SAMPLES, "delta_truth.json")) as f:
        truth = json.load(f)
    assert truth["total_changed_fraction"] == 0.04472
    s1 = load_scene(os.path.join(SAMPLES, "delta_t1.png"), gsd=10.0)
    s2 = load_scene(os.path.join(SAMPLES, "delta_t2.png"), gsd=10.0)
    lc1 = segment(s1, compute_features(s1), k=7)
    cr = detect(s1, s2, k=7, lc1=lc1)
    assert abs(cr.changed_fraction - 0.0367) < 0.008


def test_segmentation_is_deterministic():
    s = load_scene(os.path.join(SAMPLES, "urban_mixed.png"), gsd=10.0)
    fs = compute_features(s)
    a = segment(s, fs, k=7)
    b = segment(s, fs, k=7)
    assert np.array_equal(a.label_map, b.label_map)


def test_ingestion_never_invents_geo_metadata():
    _, rep = ingest_file(os.path.join(SAMPLES, "river_wetland.png"), 10.0,
                         "river_wetland.png")
    assert rep.crs == UNAVAILABLE
    assert rep.geotransform == UNAVAILABLE
    assert "Image-space" in rep.coverage_note
    assert len(rep.checksum_sha256) == 64


def test_tiny_image_edge_case():
    import io as _io
    from PIL import Image as _IM
    p = "/tmp/satquery_tiny.png"
    _IM.new("RGB", (16, 16), (200, 30, 30)).save(p)
    s = load_scene(p, gsd=1.0)
    lc = segment(s, compute_features(s), k=4)
    assert abs(sum(lc.fractions.values()) - 1.0) < 1e-6
    objs = obj_mod.summarize(s, compute_features(s), lc)
    assert isinstance(objs["linear"], list)


def test_grayscale_input_edge_case():
    import io as _io
    from PIL import Image as _IM
    p = "/tmp/satquery_gray.png"
    _IM.new("L", (64, 64), 128).save(p)
    s = load_scene(p, gsd=5.0)
    assert s.rgb.shape == (64, 64, 3)
    lc = segment(s, compute_features(s), k=4)
    assert lc.label_map.shape == (64, 64)
