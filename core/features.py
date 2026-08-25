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
