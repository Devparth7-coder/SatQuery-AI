"""
SatQuery AI - Unsupervised land-cover segmentation with physics-based
class assignment.

Strategy: over-segment the scene with k-means in spectral+textural feature
space, then *label* each cluster by scoring its mean spectral signature
against hand-derived rules for water / vegetation / built-up / bare soil /
cloud. This gives real, image-derived segmentation without pretrained weights,
and every label carries the evidence that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .features import FeatureStack, Scene

CLASSES = ["water", "dense_vegetation", "sparse_vegetation",
           "built_up", "bare_soil", "cloud_snow", "shadow"]

CLASS_COLORS: Dict[str, tuple] = {
    "water":            (33, 113, 214),
    "dense_vegetation": (26, 128, 52),
    "sparse_vegetation": (124, 194, 79),
    "built_up":         (222, 78, 62),
    "bare_soil":        (196, 156, 96),
    "cloud_snow":       (240, 240, 245),
    "shadow":           (58, 56, 74),
}

CLASS_LABELS: Dict[str, str] = {
    "water": "Water body",
    "dense_vegetation": "Dense vegetation",
    "sparse_vegetation": "Sparse vegetation / cropland",
    "built_up": "Built-up / impervious",
    "bare_soil": "Bare soil / barren",
    "cloud_snow": "Cloud / snow",
    "shadow": "Shadow",
}


@dataclass
class LandCover:
    label_map: np.ndarray            # HxW int, index into CLASSES
    class_names: List[str]
    fractions: Dict[str, float]      # class -> fraction of scene [0,1]
    areas_km2: Dict[str, float]
    cluster_report: List[dict]       # per-cluster evidence, for explainability
    k: int
    kmeans: object = None            # fitted classifier (reused for T2)
    ctx: dict = None                 # scene context stats used for scoring
    uncertainty: Optional[np.ndarray] = None  # v2: HxW float [0,1]
    quality: Optional[dict] = None             # v2: quality indicators
    adaptive_k_used: bool = False              # v2: k was auto-selected


def _score_signature(s: dict, has_nir: bool, ctx: Dict[str, float]) -> Dict[str, float]:
    """
    Score a cluster's mean signature against each land-cover class.

    `ctx` carries scene-level statistics so that texture/edge cues are judged
    RELATIVE to the scene. Absolute thresholds fail badly: an agricultural
    scene has high edge density everywhere (field boundaries), which would
    otherwise hand every cluster a large built-up bonus.
    """
    veg, wat, bright, sat = s["veg"], s["water"], s["bright"], s["sat"]
    r, g, b, exg = s["r"], s["g"], s["b"], s["exg"]
    blueness = b - (r + g) / 2.0

    # Scene-relative roughness: >0 means rougher/edgier than the scene median.
    tex = (s["tex"] - ctx["tex_med"]) / (ctx["tex_iqr"] + 1e-6)
    edge = (s["edge"] - ctx["edge_med"]) / (ctx["edge_iqr"] + 1e-6)
    tex_abs, edge_abs = s["tex"], s["edge"]

    sc: Dict[str, float] = {}

    # --- Water: high water index, low texture, dark, blue/green dominant
    w = 0.0
    w += 3.0 * max(wat, 0) * (2.5 if has_nir else 2.0)
    w += 1.6 * max(0.30 - bright, 0) / 0.30
    w += 1.4 * max(-tex, 0)                      # smoother than scene median
    w += 2.0 * max(blueness, 0) * 3.0
    w -= 3.0 * max(veg, 0.0) * 2.0
    w -= 2.5 * max(bright - 0.62, 0) / 0.38
    w -= 2.0 * max(exg, 0) * 3.0                 # green-dominant is not water
    sc["water"] = w

    # --- Vegetation: high veg index / excess-green
    v = 0.0
    if has_nir:
        v += 4.0 * max(veg, 0) * 2.2
        v -= 2.0 * max(wat, 0)
    else:
        # VARI is unstable on dark blue-green pixels: its denominator
        # (g + r - b) collapses toward zero over shallow/teal water, so the
        # ratio explodes and open water scores as lush canopy. Trust it only
        # where green genuinely dominates BOTH other bands, and veto whenever
        # the water index is strongly positive.
        green_dom = min(g - r, g - b)          # >0 only for truly green pixels
        vari_trust = 1.0 if green_dom > 0.012 else 0.0
        v += 3.0 * max(veg, 0) * 3.2 * vari_trust
        v += 2.6 * max(exg, 0) * 4.0
        v += 2.2 * max(green_dom, 0) * 8.0
        v -= 6.0 * max(wat, 0) * 2.0           # hard veto: water beats VARI
        v -= 3.0 * max(-green_dom, 0) * 8.0    # not green => not vegetation
    v += 0.8 * max(g - r, 0) * 6.0
    v -= 2.0 * max(bright - 0.72, 0) / 0.28
    dense = v + 1.1 * max(0.55 - bright, 0) / 0.55 + 0.7 * max(sat - 0.22, 0) * 3.0
    sparse = v * 0.82 + 0.9 * max(bright - 0.34, 0) / 0.66
    sc["dense_vegetation"] = dense
    sc["sparse_vegetation"] = sparse

    # --- Built-up: grey/neutral, bright-ish, HIGH texture and edge density
    # Greyness is the primary discriminator: built surfaces are spectrally flat
    # (R≈G≈B). Vegetation and soil are strongly chromatic, so this separates
    # them far more reliably than texture alone.
    chroma = max(r, g, b) - min(r, g, b)
    bu = 0.0
    bu += 3.4 * max(0.13 - chroma, 0) / 0.13        # flat spectrum = concrete
    bu += 2.2 * max(0.22 - sat, 0) / 0.22           # desaturated
    bu += 1.3 * max(tex, 0)                         # rougher than scene median
    bu += 1.1 * max(edge, 0)                        # edgier than scene median
    bu += 0.8 * max(bright - 0.32, 0) / 0.68
    bu -= 3.4 * max(veg, 0) * 3.0
    bu -= 3.0 * max(exg, 0) * 3.0                   # any greenness kills it
    bu -= 2.6 * max(wat, 0) * 2.0
    bu -= 2.4 * max(chroma - 0.22, 0) / 0.20        # strongly coloured => not built
    bu -= 1.4 * max(bright - 0.86, 0) / 0.14        # not blown-out white
    sc["built_up"] = bu

    # --- Bare soil: warm (R>G>B), moderate brightness, smooth-ish
    bs = 0.0
    bs += 3.0 * max(r - b, 0) * 4.0
    bs += 1.2 * max(r - g, 0) * 5.0
    bs += 1.0 * max(-tex, 0)
    bs += 0.8 * max(sat - 0.12, 0) * 2.5
    bs -= 3.0 * max(veg, 0) * 3.0
    bs -= 2.5 * max(exg, 0) * 3.0
    bs -= 3.0 * max(wat, 0) * 2.0
    sc["bare_soil"] = bs

    # --- Cloud / snow: very bright, very low saturation
    cl = 0.0
    cl += 4.2 * max(bright - 0.76, 0) / 0.24
    cl += 2.2 * max(0.14 - sat, 0) / 0.14
    cl -= 2.0 * max(tex, 0)
    cl -= 3.0 * max(veg, 0) * 2.0
    sc["cloud_snow"] = cl

    # --- Shadow: very dark, low saturation, but not water-like
    sh = 0.0
    sh += 4.0 * max(0.16 - bright, 0) / 0.16
    sh += 1.0 * max(0.24 - sat, 0) / 0.24
    sh -= 2.2 * max(wat, 0) * 2.0
    sh -= 1.5 * max(veg, 0) * 2.0
    sc["shadow"] = sh

    return sc


def scene_context(fs: FeatureStack) -> Dict[str, float]:
    """Scene-level robust statistics used to judge texture/edge cues relatively."""
    tq = np.percentile(fs.texture, (25, 50, 75))
    eq = np.percentile(fs.edges, (25, 50, 75))
    return {"tex_med": float(tq[1]), "tex_iqr": float(tq[2] - tq[0]),
            "edge_med": float(eq[1]), "edge_iqr": float(eq[2] - eq[0])}


def segment(scene: Scene, fs: FeatureStack, k: int = 7, seed: int = 42,
            km=None, ctx: Optional[Dict[str, float]] = None) -> LandCover:
    """
    Segment a scene into land-cover classes.

    `km` / `ctx` let a caller reuse a classifier fitted on another date so that
    bi-temporal pairs are labelled in one shared feature space - essential for
    change detection, otherwise independent k-means runs produce arbitrarily
    permuted clusters and fabricate huge spurious class deltas.
    """
    from .features import cluster_matrix

    X = cluster_matrix(fs)
    n = X.shape[0]
    k = int(max(4, min(k, 10)))

    if km is None:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=6,
                             batch_size=4096, max_iter=120)
        idx = np.random.RandomState(seed).choice(n, size=min(n, 60000), replace=False)
        km.fit(X[idx])
    raw = km.predict(X).reshape(fs.r.shape)

    # Spatial smoothing: majority filter removes salt-and-pepper noise
    raw = cv2.medianBlur(raw.astype(np.uint8), 5).astype(np.int32)

    has_nir = fs.nir is not None
    if ctx is None:
        ctx = scene_context(fs)
    label_map = np.zeros(raw.shape, np.int32)
    report: List[dict] = []

    for c in range(k):
        m = raw == c
        cnt = int(m.sum())
        if cnt == 0:
            continue
        sig = {
            "r": float(fs.r[m].mean()), "g": float(fs.g[m].mean()), "b": float(fs.b[m].mean()),
            "veg": float(fs.veg_index[m].mean()), "water": float(fs.water_index[m].mean()),
            "bright": float(fs.brightness[m].mean()), "sat": float(fs.saturation[m].mean()),
            "tex": float(fs.texture[m].mean()), "exg": float(fs.exg[m].mean()),
            "edge": float(fs.edges[m].mean()),
        }
        scores = _score_signature(sig, has_nir, ctx)
        best = max(scores, key=scores.get)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = ordered[0][1] - ordered[1][1]
        conf = float(np.clip(0.5 + margin / 4.0, 0.35, 0.98))
        label_map[m] = CLASSES.index(best)
        report.append({
            "cluster": c,
            "pixels": cnt,
            "fraction": cnt / float(n),
            "assigned": best,
            "assigned_label": CLASS_LABELS[best],
            "confidence": round(conf, 3),
            "signature": {kk: round(vv, 4) for kk, vv in sig.items()},
            "runner_up": ordered[1][0],
            "scores": {kk: round(vv, 3) for kk, vv in ordered[:4]},
        })

    px_area = scene.px_area_m2()
    fractions, areas = {}, {}
    for i, name in enumerate(CLASSES):
        f = float((label_map == i).sum()) / n
        fractions[name] = f
        areas[name] = f * n * px_area / 1e6

    report.sort(key=lambda d: -d["fraction"])
    lc = LandCover(label_map=label_map, class_names=CLASSES, fractions=fractions,
                   areas_km2=areas, cluster_report=report, k=k)
    lc.kmeans = km
    lc.ctx = ctx
    try:
        lc.uncertainty = _uncertainty_map(X, km, raw)
        lc.quality = quality_indicators(lc, has_nir)
    except Exception:
        lc.uncertainty = None
        lc.quality = None
    return lc


def colorize(lc: LandCover) -> np.ndarray:
    """Render the label map as an RGB thematic image."""
    out = np.zeros((*lc.label_map.shape, 3), np.uint8)
    for i, name in enumerate(CLASSES):
        out[lc.label_map == i] = CLASS_COLORS[name]
    return out


def class_mask(lc: LandCover, names: List[str]) -> np.ndarray:
    m = np.zeros(lc.label_map.shape, bool)
    for nme in names:
        if nme in CLASSES:
            m |= lc.label_map == CLASSES.index(nme)
    return m


# --------------------------------------------------------------------------- #
# v2: adaptive k, uncertainty layer, quality indicators (additive)
# --------------------------------------------------------------------------- #
def suggest_k(fs, k_min: int = 4, k_max: int = 10) -> int:
    """Deterministic adaptive cluster count from scene complexity.

    Heuristic (documented, no hidden magic): quantize colour to 3 bits per
    channel over a fixed-stride sample and count distinct signatures, then
    combine with texture variability. Colourful, textured scenes get more
    clusters; flat scenes get fewer. Always within [k_min, k_max].
    """
    h, w = fs.r.shape
    stride = max(1, int((h * w / 40000) ** 0.5))
    r = (fs.r[::stride, ::stride] * 7.0).astype(np.int32)
    g = (fs.g[::stride, ::stride] * 7.0).astype(np.int32)
    b = (fs.b[::stride, ::stride] * 7.0).astype(np.int32)
    codes = (r * 64 + g * 8 + b).ravel()
    uniq = float(len(np.unique(codes)))
    tex = float(fs.texture.std())
    color_term = min(uniq, 4000.0) / 4000.0 * 3.0
    tex_term = min(tex, 0.15) / 0.15 * 3.0
    k = int(round(4 + color_term + tex_term))
    return int(max(k_min, min(k_max, k)))


def _uncertainty_map(X: np.ndarray, km, raw: np.ndarray) -> np.ndarray:
    """Per-pixel classification uncertainty in [0,1].

    Normalised distance to the assigned cluster centroid in the
    standardised feature space (p99 normalisation). Far-from-centroid
    pixels are the ambiguous ones — e.g. mixed pixels at class borders.
    """
    labels = km.predict(X)
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    d = np.sqrt(((X - centers[labels]) ** 2).sum(axis=1))
    p99 = float(np.percentile(d, 99)) or 1.0
    return np.clip(d / p99, 0, 1).reshape(raw.shape).astype(np.float32)


def quality_indicators(lc: LandCover, has_nir: bool) -> dict:
    """Transparent classification-quality assessment.

    Starts at HIGH and downgrades one level per fired rule
    (HIGH -> MODERATE -> LOW). INSUFFICIENT is reserved for scenes that
    are mostly obscured. Every fired rule is reported — the label is
    never assigned without its justification.
    """
    rep = lc.cluster_report or []
    tot = sum(r.get("fraction", 0) for r in rep) or 1e-6
    mean_conf = sum(r.get("confidence", 0.5) * r.get("fraction", 0)
                    for r in rep) / tot
    min_frac = min((r.get("fraction", 0) for r in rep), default=0.0)
    obscured = lc.fractions.get("cloud_snow", 0.0) + lc.fractions.get("shadow", 0.0)

    label = "HIGH"
    rules: List[str] = []

    def _down(reason: str) -> None:
        nonlocal label
        rules.append(reason)
        label = {"HIGH": "MODERATE", "MODERATE": "LOW", "LOW": "LOW"}[label]

    if not has_nir:
        _down("no NIR band: vegetation/water rest on RGB proxies")
    if obscured > 0.25:
        _down(f"{obscured*100:.1f}% of pixels are cloud/shadow-obscured")
    if mean_conf < 0.55:
        _down(f"low mean cluster confidence ({mean_conf:.2f})")
    if obscured > 0.60:
        label = "INSUFFICIENT"
        rules.append("majority of scene obscured — classification unreliable")
    return {"label": label, "mean_confidence": round(float(mean_conf), 3),
            "min_cluster_fraction": round(float(min_frac), 4),
            "obscured_fraction": round(float(obscured), 4),
            "rules": rules,
            "note": "Rule-derived classes are algorithmically inferred, "
                    "not ground-truth validated."}
