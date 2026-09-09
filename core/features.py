"""
SatQuery AI - Spectral feature extraction layer.

Computes per-pixel radiometric / spectral / textural features from an uploaded
remote-sensing scene. Works on plain 3-band RGB (true-colour tiles, the common
case) and transparently upgrades to true NIR-based indices when a 4-band
product is supplied.

All indices are computed on float reflectance-proxy values in [0, 1].
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

EPS = 1e-6
MAX_DIM = 1024  # analysis resolution cap (keeps latency < ~1s on 2 vCPU)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """A loaded, resampled remote-sensing scene."""

    rgb: np.ndarray                      # uint8 HxWx3, analysis resolution
    nir: Optional[np.ndarray] = None     # float32 HxW in [0,1] or None
    gsd: float = 10.0                    # ground sample distance, metres/pixel
    source_name: str = "scene"
    orig_shape: tuple = (0, 0)
    scale: float = 1.0                   # analysis_px / original_px

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def n_pixels(self) -> int:
        return self.rgb.shape[0] * self.rgb.shape[1]

    @property
    def has_nir(self) -> bool:
        return self.nir is not None

    def px_area_m2(self) -> float:
        """Ground area represented by one analysis pixel, in m^2."""
        # gsd refers to the ORIGINAL raster; analysis raster is coarser by 1/scale
        return (self.gsd / max(self.scale, EPS)) ** 2


def load_scene(path: str, gsd: float = 10.0, assume_nir: bool = False) -> Scene:
    """Load an image file into a Scene, downsampling to MAX_DIM for analysis."""
    img = Image.open(path)
    ext = os.path.splitext(path)[1].lower()

    # Multi-page / multi-band TIFF: try to pull a 4th band as NIR.
    bands = None
    if img.mode in ("CMYK", "RGBA") or (img.mode not in ("RGB", "L") and ext in (".tif", ".tiff")):
        try:
            bands = np.array(img)
        except Exception:
            bands = None

    nir_full = None
    if bands is not None and bands.ndim == 3 and bands.shape[2] >= 4 and assume_nir:
        arr = bands[:, :, :3]
        nir_full = bands[:, :, 3]
    else:
        arr = np.array(img.convert("RGB"))

    if arr.dtype != np.uint8:  # 16-bit products -> percentile stretch to 8-bit
        arr = _stretch_to_u8(arr)
    if nir_full is not None and nir_full.dtype != np.uint8:
        nir_full = _stretch_to_u8(nir_full)

    h, w = arr.shape[:2]
    orig_shape = (h, w)
    scale = 1.0
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / float(max(h, w))
        arr = cv2.resize(arr, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        if nir_full is not None:
            nir_full = cv2.resize(nir_full, (arr.shape[1], arr.shape[0]),
                                  interpolation=cv2.INTER_AREA)

    nir = (nir_full.astype(np.float32) / 255.0) if nir_full is not None else None
    return Scene(rgb=arr, nir=nir, gsd=float(gsd),
                 source_name=os.path.basename(path), orig_shape=orig_shape, scale=scale)


def _stretch_to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    out = np.zeros(a.shape, np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
        out = out[:, :, None]
    for c in range(a.shape[2]):
        lo, hi = np.percentile(a[:, :, c], (2, 98))
        if hi - lo < EPS:
            hi = lo + 1
        out[:, :, c] = np.clip((a[:, :, c] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out.squeeze()


# --------------------------------------------------------------------------- #
# Feature stack
# --------------------------------------------------------------------------- #
@dataclass
class FeatureStack:
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    brightness: np.ndarray
    saturation: np.ndarray
    texture: np.ndarray          # local std-dev of luminance (5x5)
    veg_index: np.ndarray        # NDVI if NIR present, else VARI
    veg_index_name: str
    water_index: np.ndarray      # NDWI if NIR present, else RGB water proxy
    water_index_name: str
    exg: np.ndarray              # Excess-Green (RGB vegetation cue)
    ndbi_like: np.ndarray        # built-up / bare cue
    edges: np.ndarray            # Canny edge density map
    nir: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def compute_features(scene: Scene) -> FeatureStack:
    rgb = scene.rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    hsv = cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    brightness = hsv[:, :, 2] / 255.0

    gray = cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray, -1, (5, 5), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, -1, (5, 5), normalize=True)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    exg = np.clip(2.0 * g - r - b, -1.0, 1.0)

    if scene.nir is not None:
        n = scene.nir
        veg = (n - r) / (n + r + EPS)          # true NDVI
        veg_name = "NDVI"
        water = (g - n) / (g + n + EPS)        # true NDWI (McFeeters)
        water_name = "NDWI"
        ndbi_like = (n - g) / (n + g + EPS)
    else:
        # VARI: atmospherically-resistant visible vegetation index
        veg = np.clip((g - r) / (g + r - b + EPS), -1.0, 1.0)
        veg_name = "VARI (RGB proxy)"
        # RGB water proxy: water is blue/green dominant, dark and smooth
        water = np.clip((b - r) / (b + r + EPS), -1.0, 1.0)
        water_name = "Blue-Red water index (RGB proxy)"
        ndbi_like = np.clip((r - g) / (r + g + EPS), -1.0, 1.0)

    edges = cv2.Canny(scene.rgb, 60, 160).astype(np.float32) / 255.0
    edges = cv2.boxFilter(edges, -1, (9, 9), normalize=True)

    return FeatureStack(
        r=r, g=g, b=b, brightness=brightness, saturation=saturation,
        texture=texture, veg_index=veg.astype(np.float32), veg_index_name=veg_name,
        water_index=water.astype(np.float32), water_index_name=water_name,
        exg=exg.astype(np.float32), ndbi_like=ndbi_like.astype(np.float32),
        edges=edges, nir=scene.nir,
        meta={"has_nir": scene.nir is not None},
    )


def cluster_matrix(fs: FeatureStack) -> np.ndarray:
    """Feature matrix (N x D) used for unsupervised land-cover clustering."""
    layers = [fs.r, fs.g, fs.b, fs.saturation, fs.brightness,
              fs.exg, fs.water_index, fs.texture]
    if fs.nir is not None:
        layers.append(fs.nir)
        layers.append(fs.veg_index)
    m = np.stack([l.ravel() for l in layers], axis=1).astype(np.float32)
    mu, sd = m.mean(0), m.std(0) + EPS
    return (m - mu) / sd


# --------------------------------------------------------------------------- #
# v2: multi-index spectral engine (additive — existing API unchanged)
# --------------------------------------------------------------------------- #
from dataclasses import dataclass as _dataclass2  # noqa: E402
from typing import Any as _Any, Dict as _Dict, List as _List  # noqa: E402


@_dataclass2
class IndexResult:
    """One spectral index: formula, band requirements and measured stats.

    When required bands are missing the result is returned with
    ``available=False`` and a human-readable ``reason`` — the pipeline
    never substitutes an unrelated band.
    """

    name: str
    formula: str
    required_bands: _List[str]
    available: bool
    reason: str = ""
    value_range: tuple = (-1.0, 1.0)
    description: str = ""
    stats: dict = None  # type: ignore
    histogram: dict = None  # type: ignore

    def to_dict(self) -> _Dict[str, _Any]:
        return {
            "name": self.name, "formula": self.formula,
            "required_bands": self.required_bands,
            "available": self.available, "reason": self.reason,
            "value_range": list(self.value_range),
            "description": self.description,
            "stats": self.stats or {}, "histogram": self.histogram or {},
        }


INDEX_DEFS: _Dict[str, _Dict[str, _Any]] = {
    "NDVI": {"formula": "(NIR - RED) / (NIR + RED)",
             "required_bands": ["nir", "red"],
             "description": "Normalised Difference Vegetation Index; "
                            "vigorous canopy is strongly positive."},
    "NDWI": {"formula": "(GREEN - NIR) / (GREEN + NIR)",
             "required_bands": ["green", "nir"],
             "description": "McFeeters NDWI; open water is positive."},
    "MNDWI": {"formula": "(GREEN - SWIR1) / (GREEN + SWIR1)",
              "required_bands": ["green", "swir1"],
              "description": "Modified NDWI; suppresses built-up noise. "
                             "Needs SWIR."},
    "NDBI": {"formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
             "required_bands": ["swir1", "nir"],
             "description": "Normalised Difference Built-up Index. Needs SWIR."},
    "SAVI": {"formula": "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
             "required_bands": ["nir", "red"],
             "description": "Soil-Adjusted Vegetation Index (L=0.5); "
                            "stable over sparse canopy."},
    "EVI": {"formula": "2.5*(NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
            "required_bands": ["nir", "red", "blue"],
            "description": "Enhanced Vegetation Index; resists atmospheric "
                           "and canopy-background noise."},
    "GNDVI": {"formula": "(NIR - GREEN) / (NIR + GREEN)",
              "required_bands": ["nir", "green"],
              "description": "Green NDVI; sensitive to chlorophyll content."},
    "VARI": {"formula": "(GREEN - RED) / (GREEN + RED - BLUE)",
             "required_bands": ["green", "red", "blue"],
             "description": "Visible Atmospherically Resistant Index; "
                            "RGB-only vegetation proxy."},
    "ExG": {"formula": "2*GREEN - RED - BLUE",
            "required_bands": ["green", "red", "blue"],
            "description": "Excess Green; RGB-only greenness cue."},
    "water_proxy": {"formula": "(BLUE - RED) / (BLUE + RED)",
                    "required_bands": ["blue", "red"],
                    "description": "Blue-Red water proxy for RGB-only input; "
                                   "turbid water can be missed."},
}


def index_stats(arr: np.ndarray, bins: int = 40) -> tuple:
    """Robust stats + histogram for one index array (NaN-safe)."""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}, {"counts": [], "edges": []}
    lo, hi = float(a.min()), float(a.max())
    stats = {"min": round(lo, 4), "max": round(hi, 4),
             "mean": round(float(a.mean()), 4),
             "median": round(float(np.median(a)), 4),
             "std": round(float(a.std()), 4),
             "p5": round(float(np.percentile(a, 5)), 4),
             "p95": round(float(np.percentile(a, 95)), 4)}
    try:
        counts, edges = np.histogram(a, bins=bins,
                                     range=(min(lo, -1.0), max(hi, 1.0)))
        hist = {"counts": [int(c) for c in counts],
                "edges": [round(float(e), 4) for e in edges]}
    except Exception:
        hist = {"counts": [], "edges": []}
    return stats, hist


def _unavailable(name: str, have: _List[str]) -> IndexResult:
    d = INDEX_DEFS[name]
    missing = [b for b in d["required_bands"] if b not in have]
    return IndexResult(
        name=name, formula=d["formula"], required_bands=d["required_bands"],
        available=False,
        reason=f"Unavailable: missing band(s) {missing} "
               f"(have: {sorted(have)}). No substitution was made.",
        description=d["description"], stats={}, histogram={})


def compute_all_indices(scene: Scene, fs: FeatureStack) -> _Dict[str, IndexResult]:
    """Compute every supported index honestly (NIR-gated where required)."""
    have = ["red", "green", "blue"] + (["nir"] if scene.nir is not None else [])
    out: _Dict[str, IndexResult] = {}

    def _pack(name: str, arr: np.ndarray) -> IndexResult:
        d = INDEX_DEFS[name]
        stats, hist = index_stats(arr)
        return IndexResult(name=name, formula=d["formula"],
                           required_bands=d["required_bands"], available=True,
                           reason="", description=d["description"],
                           stats=stats, histogram=hist)

    # RGB-always indices (computed from the same arrays the v1 pipeline uses).
    out["VARI"] = _pack("VARI", fs.veg_index if fs.veg_index_name.startswith("VARI")
                        else np.clip((fs.g - fs.r) / (fs.g + fs.r - fs.b + EPS), -1, 1))
    out["ExG"] = _pack("ExG", fs.exg)
    out["water_proxy"] = _pack("water_proxy", fs.water_index
                               if "proxy" in fs.water_index_name.lower()
                               else np.clip((fs.b - fs.r) / (fs.b + fs.r + EPS), -1, 1))

    if scene.nir is not None:
        n = scene.nir.astype(np.float32)
        out["NDVI"] = _pack("NDVI", fs.veg_index if fs.veg_index_name == "NDVI"
                            else (n - fs.r) / (n + fs.r + EPS))
        out["NDWI"] = _pack("NDWI", fs.water_index if fs.water_index_name == "NDWI"
                            else (fs.g - n) / (fs.g + n + EPS))
        out["SAVI"] = _pack("SAVI", ((n - fs.r) / (n + fs.r + 0.5)) * 1.5)
        out["EVI"] = _pack("EVI", np.clip(
            2.5 * (n - fs.r) / (n + 6 * fs.r - 7.5 * fs.b + 1.0), -1, 1))
        out["GNDVI"] = _pack("GNDVI", (n - fs.g) / (n + fs.g + EPS))
    else:
        for name in ("NDVI", "NDWI", "SAVI", "EVI", "GNDVI"):
            out[name] = _unavailable(name, have)

    # SWIR is never present in current ingestion (honest unavailable states).
    for name in ("MNDWI", "NDBI"):
        out[name] = _unavailable(name, have)
    return out


def spatial_statistics(fs: FeatureStack) -> _Dict[str, _Any]:
    """Scene-level spatial/statistical descriptors (all measured)."""
    out: _Dict[str, _Any] = {}
    for key, arr in (("brightness", fs.brightness),
                     ("saturation", fs.saturation),
                     ("texture", fs.texture),
                     ("edge_density", fs.edges)):
        stats, _ = index_stats(arr, bins=24)
        out[key] = stats
    tex = out.get("texture", {})
    out["interpretation"] = {
        "contrast": ("high" if tex.get("std", 0) > 0.09 else
                     "moderate" if tex.get("std", 0) > 0.05 else "low"),
        "note": "Contrast judged from texture std-dev; thresholds are "
                "documented in core/config.py quality rules.",
    }
    return out
