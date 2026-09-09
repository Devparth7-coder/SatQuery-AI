"""
SatQuery AI - Object & structure extraction.

Classical CV detectors that operate on the land-cover context:
  * region extraction  : connected components per thematic class (lakes, patches)
  * bright-target       : ships / vehicles / metal roofs on dark or uniform bg
  * linear structures   : roads / runways / canals via Hough + morphology
Everything returns georeferenced-in-pixels geometry + real ground measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from .features import FeatureStack, Scene
from .landcover import CLASSES, CLASS_LABELS, LandCover, class_mask


@dataclass
class Detection:
    kind: str
    label: str
    bbox: tuple           # x, y, w, h in analysis pixels
    centroid: tuple
    area_px: int
    area_m2: float
    score: float
    extra: dict


def _clean(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return m.astype(bool)


def regions_for_class(scene: Scene, lc: LandCover, cls: str,
                      min_px: int = 60, top: int = 40) -> List[Detection]:
    """Connected-component regions of one land-cover class (e.g. every lake)."""
    m = _clean(class_mask(lc, [cls]))
    num, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    px_area = scene.px_area_m2()
    out: List[Detection] = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_px:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = float(cv2.arcLength(cnts[0], True)) if cnts else 0.0
        compact = float(4 * np.pi * a / (perim ** 2 + 1e-6)) if perim > 0 else 0.0
        out.append(Detection(
            kind=f"region:{cls}", label=CLASS_LABELS.get(cls, cls),
            bbox=(x, y, w, h), centroid=(float(cent[i][0]), float(cent[i][1])),
            area_px=a, area_m2=a * px_area,
            score=float(min(1.0, a / max(min_px * 8, 1))),
            extra={"compactness": round(compact, 3),
                   "perimeter_m": round(perim * np.sqrt(px_area), 1),
                   "elongation": round(max(w, h) / max(1.0, min(w, h)), 2)},
        ))
    out.sort(key=lambda d: -d.area_px)
    return out[:top]


def bright_targets(scene: Scene, fs: FeatureStack, lc: LandCover,
                   top: int = 60) -> List[Detection]:
    """
    Small bright/high-contrast blobs standing out from a locally uniform
    background - the classic signature of ships at sea, vehicles on tarmac,
    aircraft on aprons or metal rooftops.
    """
    gray = cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 31)
    diff = cv2.subtract(gray, bg)
    thr = float(np.percentile(diff, 99.3))
    thr = max(thr, 14.0)
    m = _clean(diff > thr, open_k=3, close_k=3)

    water = class_mask(lc, ["water"])
    water_ctx = cv2.dilate(water.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)

    num, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    px_area = scene.px_area_m2()
    out: List[Detection] = []
    max_px = max(400, int(0.004 * scene.n_pixels))
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 6 or a > max_px:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        elong = max(w, h) / max(1.0, min(w, h))
        cy, cx = int(cent[i][1]), int(cent[i][0])
        cy = min(max(cy, 0), scene.height - 1)
        cx = min(max(cx, 0), scene.width - 1)
        on_water = bool(water_ctx[cy, cx])
        contrast = float(diff[lab == i].mean())

        if on_water:
            kind, label = "vessel", "Probable vessel / floating target"
            score = min(1.0, 0.45 + contrast / 90.0 + (0.18 if 1.6 < elong < 7 else 0))
        else:
            kind, label = "bright_target", "Bright compact target (vehicle / structure)"
            score = min(1.0, 0.30 + contrast / 130.0)
        out.append(Detection(
            kind=kind, label=label, bbox=(x, y, w, h),
            centroid=(float(cent[i][0]), float(cent[i][1])),
            area_px=a, area_m2=a * px_area, score=round(score, 3),
            extra={"contrast": round(contrast, 1), "elongation": round(elong, 2),
                   "on_water": on_water,
                   "length_m": round(max(w, h) * np.sqrt(px_area), 1)},
        ))
    out.sort(key=lambda d: -d.score)
    return out[:top]


def linear_structures(scene: Scene, fs: FeatureStack, top: int = 40) -> List[Detection]:
    """Roads / runways / canals: long straight segments from Hough transform."""
    gray = cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(30, int(0.06 * max(scene.width, scene.height)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                            minLineLength=min_len, maxLineGap=8)
    px_m = np.sqrt(scene.px_area_m2())
    out: List[Detection] = []
    if lines is None:
        return out
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1) or 1, abs(y2 - y1) or 1
        out.append(Detection(
            kind="linear", label="Linear structure (road / runway / canal)",
            bbox=(x, y, w, h), centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            area_px=int(length), area_m2=0.0,
            score=round(min(1.0, length / (min_len * 3.0)), 3),
            extra={"length_m": round(length * px_m, 1), "orientation_deg": round(ang, 1),
                   "p1": [x1, y1], "p2": [x2, y2]},
        ))
    out.sort(key=lambda d: -d.extra["length_m"])
    return out[:top]


def summarize(scene: Scene, fs: FeatureStack, lc: LandCover) -> Dict:
    """Run the full detector suite once; the query engine reads from this."""
    water_regions = regions_for_class(scene, lc, "water", min_px=80)
    veg_regions = regions_for_class(scene, lc, "dense_vegetation", min_px=150, top=25)
    built_regions = regions_for_class(scene, lc, "built_up", min_px=150, top=25)
    targets = bright_targets(scene, fs, lc)
    lines = linear_structures(scene, fs)

    vessels = [d for d in targets if d.kind == "vessel" and d.score > 0.5]
    others = [d for d in targets if d.kind != "vessel"]

    # Urban density proxy: edge energy inside built-up areas
    bm = class_mask(lc, ["built_up"])
    urban_density = float(fs.edges[bm].mean()) if bm.any() else 0.0

    return {
        "water_regions": water_regions,
        "veg_regions": veg_regions,
        "built_regions": built_regions,
        "vessels": vessels,
        "bright_targets": others,
        "linear": lines,
        "urban_edge_density": urban_density,
    }


# --------------------------------------------------------------------------- #
# v2: unified object schema (additive — Detection dataclass unchanged)
# --------------------------------------------------------------------------- #
# Every detection is exposed to the API/UI/report layers through this schema:
#   {id, type, bbox[x,y,w,h], centroid[x,y], area_px, area_m2,
#    confidence, source_method, evidence{...}}
# `confidence` for classical detectors is a *heuristic score* (contrast /
# size cues), NOT a calibrated model probability — the note is surfaced
# wherever scores are shown.

SOURCE_METHOD = {
    "vessel": "median-background contrast blob ∩ dilated water mask",
    "bright_target": "median-background contrast blob (99.3rd percentile)",
    "linear": "probabilistic Hough transform on Canny edges",
    "region:water": "connected components of the water class mask",
    "region:dense_vegetation": "connected components of the vegetation mask",
    "region:built_up": "connected components of the built-up mask",
}

CONFIDENCE_NOTE = ("Heuristic score from contrast/size cues, not a calibrated "
                   "model probability.")


def detection_to_dict(d: Detection, id: str) -> dict:
    return {
        "id": id,
        "type": d.kind,
        "label": d.label,
        "bbox": [int(v) for v in d.bbox],
        "centroid": [round(float(d.centroid[0]), 1), round(float(d.centroid[1]), 1)],
        "area_px": int(d.area_px),
        "area_m2": round(float(d.area_m2), 1),
        "confidence": round(float(d.score), 3),
        "confidence_note": CONFIDENCE_NOTE,
        "source_method": SOURCE_METHOD.get(d.kind, "classical detector"),
        "evidence": d.extra,
    }


def detections_table(objs: dict) -> list:
    """Flatten all detector outputs into unified-schema rows (stable ids)."""
    rows: list = []
    n = 0
    for key in ("vessels", "bright_targets", "linear", "water_regions",
                "veg_regions", "built_regions"):
        for d in objs.get(key, []) or []:
            n += 1
            d.kind = d.kind  # keep kind verbatim
            rows.append(detection_to_dict(d, f"D{n:03d}"))
    return rows


def size_statistics(dets: list) -> dict:
    """Measured size distribution for one detector family (honest n)."""
    import numpy as _np
    if not dets:
        return {"n": 0, "note": "No detections in this family."}
    lengths = [float(d.extra.get("length_m", 0) or 0) for d in dets]
    lengths = [L for L in lengths if L > 0]
    areas = [float(d.area_m2) for d in dets]
    out = {"n": len(dets),
           "area_m2": {"min": round(min(areas), 1), "max": round(max(areas), 1),
                       "median": round(float(_np.median(areas)), 1)}}
    if lengths:
        out["length_m"] = {"min": round(min(lengths), 1),
                           "max": round(max(lengths), 1),
                           "median": round(float(_np.median(lengths)), 1)}
    return out
