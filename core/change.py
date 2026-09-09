"""
SatQuery AI - Bi-temporal change detection.

Given two scenes of the same area, co-registers them (ECC / feature-based),
then computes:
  * spectral change magnitude (CVA - change vector analysis)
  * per-class transition matrix (what became what)
  * headline gains/losses in km^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .features import FeatureStack, Scene, compute_features
from .landcover import CLASSES, CLASS_LABELS, LandCover, segment


@dataclass
class ChangeResult:
    magnitude: np.ndarray             # HxW float [0,1]
    change_mask: np.ndarray           # HxW bool
    changed_fraction: float
    transitions: List[dict]           # [{from, to, px, area_km2}]
    deltas: Dict[str, float]          # class -> delta km^2 (t2 - t1)
    registration: str
    lc1: LandCover
    lc2: LandCover
    severity: str = "UNKNOWN"          # v2: LOW / MEDIUM / HIGH
    hotspots: Optional[list] = None    # v2: top change regions
    quality: Optional[dict] = None     # v2: registration/radiometry quality


def _register(ref: np.ndarray, mov: np.ndarray) -> tuple:
    """Align `mov` to `ref`. Returns (warped, method_description)."""
    if ref.shape[:2] != mov.shape[:2]:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    g1 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)

    # Try ORB feature matching first (robust to large shifts)
    try:
        orb = cv2.ORB_create(2500)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is not None and d2 is not None and len(k1) > 25 and len(k2) > 25:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
            if len(matches) >= 12:
                p1 = np.float32([k1[m.queryIdx].pt for m in matches])
                p2 = np.float32([k2[m.trainIdx].pt for m in matches])
                M, inl = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=4.0)
                if M is not None and inl is not None and int(inl.sum()) >= 10:
                    shift = float(np.hypot(M[0, 2], M[1, 2]))
                    if shift < 0.35 * max(ref.shape[:2]):
                        warped = cv2.warpAffine(mov, M, (ref.shape[1], ref.shape[0]),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                        return warped, (f"ORB+RANSAC affine ({int(inl.sum())} inliers, "
                                        f"{shift:.1f} px shift corrected)")
    except Exception:
        pass

    # Fallback: ECC translation refinement
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        cv2.findTransformECC(g1.astype(np.float32) / 255, g2.astype(np.float32) / 255,
                             warp, cv2.MOTION_TRANSLATION, crit, None, 5)
        warped = cv2.warpAffine(mov, warp, (ref.shape[1], ref.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped, f"ECC translation ({warp[0,2]:+.1f}, {warp[1,2]:+.1f}) px"
    except Exception:
        return mov, "no co-registration applied (resampled only)"


def detect(s1: Scene, s2: Scene, k: int = 7,
           lc1: Optional[LandCover] = None) -> ChangeResult:
    warped, method = _register(s1.rgb, s2.rgb)

    # Relative radiometric normalisation BEFORE segmentation. The two dates
    # differ in sun angle / atmosphere / gain; without this, a global
    # brightness shift pushes pixels across cluster boundaries everywhere and
    # the illusory drift swamps the genuine localised change.
    try:
        from skimage.exposure import match_histograms
        warped = match_histograms(warped, s1.rgb, channel_axis=-1).astype(np.uint8)
        method += " + histogram-matched radiometry"
    except Exception:
        pass

    s2r = Scene(rgb=warped, nir=s2.nir, gsd=s1.gsd,
                source_name=s2.source_name, orig_shape=s2.orig_shape, scale=s1.scale)

    f1 = compute_features(s1)
    f2 = compute_features(s2r)

    if lc1 is None:
        lc1 = segment(s1, f1, k=k)
    # Label T2 with the SAME fitted classifier and the SAME scene context as T1.
    # Independent k-means runs would permute clusters arbitrarily and fabricate
    # enormous spurious per-class deltas even when almost nothing changed.
    lc2 = segment(s2r, f2, k=k, km=getattr(lc1, "kmeans", None),
                  ctx=getattr(lc1, "ctx", None))

    # Change Vector Analysis over normalised feature bands
    bands1 = np.stack([f1.r, f1.g, f1.b, f1.veg_index, f1.water_index, f1.brightness], -1)
    bands2 = np.stack([f2.r, f2.g, f2.b, f2.veg_index, f2.water_index, f2.brightness], -1)
    # Relative radiometric normalisation (mean/std matching) kills illumination bias
    for c in range(bands2.shape[2]):
        m1, s1_ = bands1[..., c].mean(), bands1[..., c].std() + 1e-6
        m2, s2_ = bands2[..., c].mean(), bands2[..., c].std() + 1e-6
        bands2[..., c] = (bands2[..., c] - m2) / s2_ * s1_ + m1

    mag = np.sqrt(((bands2 - bands1) ** 2).sum(-1))
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    hi = float(np.percentile(mag, 99.5)) or 1.0
    mag_n = np.clip(mag / hi, 0, 1)

    # Otsu on the magnitude histogram = data-driven change threshold
    u8 = (mag_n * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(t) / 255.0, 0.18)
    cm = mag_n > thr
    cm = cv2.morphologyEx(cm.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

    px_km2 = s1.px_area_m2() / 1e6
    trans: List[dict] = []
    for a in range(len(CLASSES)):
        for b in range(len(CLASSES)):
            if a == b:
                continue
            n = int(((lc1.label_map == a) & (lc2.label_map == b) & cm).sum())
            if n * px_km2 > 1e-4 and n > 40:
                trans.append({"from": CLASSES[a], "to": CLASSES[b],
                              "from_label": CLASS_LABELS[CLASSES[a]],
                              "to_label": CLASS_LABELS[CLASSES[b]],
                              "pixels": n, "area_km2": round(n * px_km2, 4)})
    trans.sort(key=lambda d: -d["pixels"])

    deltas = {c: round(lc2.areas_km2.get(c, 0) - lc1.areas_km2.get(c, 0), 4)
              for c in CLASSES}

    changed_fraction = float(cm.mean())
    reg_quality = _registration_quality(method)
    severity = _severity_for(changed_fraction, reg_quality)
    hotspots = _hotspots(cm, s1, mag_n, top=5)
    quality = {"registration": reg_quality,
               "histogram_matched": "histogram-matched" in method,
               "threshold_note": "Otsu on CVA magnitude (floor 0.18) + "
                                 "morphological open/close cleanup"}

    return ChangeResult(magnitude=mag_n, change_mask=cm,
                        changed_fraction=changed_fraction, transitions=trans[:14],
                        deltas=deltas, registration=method, lc1=lc1, lc2=lc2,
                        severity=severity, hotspots=hotspots, quality=quality)


# --------------------------------------------------------------------------- #
# v2: severity, hotspots, registration quality (additive)
# --------------------------------------------------------------------------- #
def _registration_quality(method: str) -> dict:
    """Transparent registration assessment parsed from the method record."""
    import re as _re
    m = _re.search(r"(\d+)\s+inliers?,\s*([\d.]+)\s*px", method or "")
    inliers = int(m.group(1)) if m else 0
    shift = float(m.group(2)) if m else 0.0
    if "ORB+RANSAC" in (method or ""):
        quality = "HIGH" if inliers >= 50 else ("MODERATE" if inliers >= 10 else "LOW")
        kind = "feature-based (ORB+RANSAC affine)"
    elif "ECC" in (method or ""):
        quality, kind = "LOW", "intensity-based (ECC translation fallback)"
    else:
        quality, kind = "FAILED", "none (resampled only)"
    return {"kind": kind, "inliers": inliers, "shift_px_corrected": round(shift, 1),
            "quality": quality,
            "note": "HIGH>=50 inliers, MODERATE>=10, LOW=ECC fallback, "
                    "FAILED=no alignment found."}


def _severity_for(changed_fraction: float, reg: dict) -> str:
    """LOW / MEDIUM / HIGH from measured change + registration gate.

    Thresholds come from core/config.py. A HIGH label is gated to MEDIUM
    when registration failed, because unaligned pixels fabricate change.
    """
    from . import config as _cfg
    f = float(changed_fraction)
    sev = ("LOW" if f < _cfg.CHANGE_SEVERITY_LOW
           else "HIGH" if f >= _cfg.CHANGE_SEVERITY_HIGH else "MEDIUM")
    if sev == "HIGH" and reg.get("quality") in ("LOW", "FAILED"):
        sev = "MEDIUM"
    return sev


def severity_reason(changed_fraction: float, reg: dict) -> str:
    from . import config as _cfg
    base = (f"{changed_fraction*100:.2f}% changed vs thresholds "
            f"LOW<{_cfg.CHANGE_SEVERITY_LOW*100:g}% / "
            f"HIGH>={_cfg.CHANGE_SEVERITY_HIGH*100:g}%")
    if reg.get("quality") in ("LOW", "FAILED"):
        base += f"; capped because registration quality is {reg.get('quality')}"
    else:
        base += f"; registration quality {reg.get('quality')} ({reg.get('inliers',0)} inliers)"
    return base + "."


def _hotspots(cm: np.ndarray, s1: Scene, mag: np.ndarray, top: int = 5) -> list:
    """Largest connected change regions with honest per-hotspot severity."""
    num, lab, stats, cent = cv2.connectedComponentsWithStats(cm.astype(np.uint8), 8)
    px_km2 = s1.px_area_m2() / 1e6
    p50 = float(np.percentile(mag[cm], 50)) if cm.any() else 0.0
    p80 = float(np.percentile(mag[cm], 80)) if cm.any() else 0.0
    out = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 40:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        mean_mag = float(mag[lab == i].mean()) if (lab == i).any() else 0.0
        sev = "HIGH" if mean_mag >= p80 else ("MEDIUM" if mean_mag >= p50 else "LOW")
        out.append({"bbox": [x, y, w, h],
                    "centroid": [round(float(cent[i][0]), 1), round(float(cent[i][1]), 1)],
                    "area_px": a, "area_km2": round(a * px_km2, 4),
                    "mean_magnitude": round(mean_mag, 3),
                    "severity": sev})
    out.sort(key=lambda d: -d["area_px"])
    return out[:top]
