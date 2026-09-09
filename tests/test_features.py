"""Unit tests: spectral feature extraction + multi-index engine."""
import os

import numpy as np

from core.features import (compute_all_indices, compute_features, index_stats,
                           load_scene, spatial_statistics)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HARBOUR = os.path.join(SAMPLES, "harbour_port.png")


def test_load_scene_caps_resolution():
    s = load_scene(HARBOUR, gsd=10.0)
    assert max(s.width, s.height) <= 1024
    assert s.orig_shape == (764, 1400)
    assert s.rgb.shape == (s.height, s.width, 3)
    assert not s.has_nir
    assert s.px_area_m2() > 100.0  # 10 m GSD at reduced scale


def test_compute_features_ranges():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    assert fs.veg_index.shape == (s.height, s.width)
    assert fs.water_index.shape == (s.height, s.width)
    assert np.isfinite(fs.veg_index).all()
    assert fs.veg_index.min() >= -1.0 and fs.veg_index.max() <= 1.0
    assert "RGB proxy" in fs.veg_index_name  # no NIR in PNG samples


def test_indices_rgb_availability_is_honest():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    idx = compute_all_indices(s, fs)
    assert idx["VARI"].available and idx["ExG"].available
    assert idx["water_proxy"].available
    for name in ("NDVI", "NDWI", "SAVI", "EVI", "GNDVI", "MNDWI", "NDBI"):
        assert not idx[name].available, name
        assert "missing band" in idx[name].reason
        assert idx[name].formula  # formula always documented


def test_index_stats_and_histogram():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    stats, hist = index_stats(fs.veg_index)
    for k in ("min", "max", "mean", "median", "std", "p5", "p95"):
        assert k in stats and np.isfinite(stats[k])
    assert len(hist["counts"]) == 40 and len(hist["edges"]) == 41
    assert sum(hist["counts"]) == fs.veg_index.size


def test_spatial_statistics_keys():
    s = load_scene(HARBOUR, gsd=10.0)
    fs = compute_features(s)
    st = spatial_statistics(fs)
    for k in ("brightness", "saturation", "texture", "edge_density"):
        assert "mean" in st[k] and "std" in st[k]
    assert st["interpretation"]["contrast"] in ("low", "moderate", "high")
