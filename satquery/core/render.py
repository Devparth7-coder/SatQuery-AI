"""SatQuery AI - visual grounding: overlays, heatmaps, detection boxes."""
from __future__ import annotations

import base64
import io
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from .landcover import CLASS_COLORS, CLASSES, LandCover, class_mask, colorize


def to_data_uri(arr: np.ndarray, fmt: str = "PNG", quality: int = 88) -> str:
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if fmt.upper() in ("JPG", "JPEG"):
        im.convert("RGB").save(buf, "JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def landcover_overlay(rgb: np.ndarray, lc: LandCover, alpha: float = 0.55) -> np.ndarray:
    return cv2.addWeighted(rgb, 1 - alpha, colorize(lc), alpha, 0)


def class_overlay(rgb: np.ndarray, lc: LandCover, classes: List[str],
                  alpha: float = 0.60) -> np.ndarray:
    """Dim everything, light up the queried class(es), draw their outlines."""
    m = class_mask(lc, classes)
    base = (rgb.astype(np.float32) * 0.42).astype(np.uint8)   # dim background
    out = base.copy()
    tint = np.zeros_like(rgb)
    for c in classes:
        cm = class_mask(lc, [c])
        tint[cm] = CLASS_COLORS.get(c, (255, 215, 0))
    out[m] = cv2.addWeighted(rgb, 1 - alpha, tint, alpha, 0)[m]

    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap(rgb: np.ndarray, field: np.ndarray, alpha: float = 0.62,
            cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    f = field.astype(np.float32)
    lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    n = (f - lo) / (hi - lo + 1e-6)
    cm = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, cm, alpha, 0)


def change_overlay(rgb2: np.ndarray, mag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = (rgb2.astype(np.float32) * 0.5).astype(np.uint8)
    cm = cv2.applyColorMap((np.clip(mag, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    out = base.copy()
    out[mask] = cv2.addWeighted(rgb2, 0.32, cm, 0.68, 0)[mask]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, [c for c in cnts if cv2.contourArea(c) > 30], -1,
                     (255, 40, 40), 1, cv2.LINE_AA)
    return out


def detections_overlay(rgb: np.ndarray, dets, kinds: Optional[List[str]] = None) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.62).astype(np.uint8)
    palette = {"vessel": (255, 92, 92), "bright_target": (255, 214, 64),
               "linear": (80, 220, 255)}
    for d in dets:
        if kinds and d.kind not in kinds:
            continue
        col = palette.get(d.kind, (0, 255, 180))
        if d.kind == "linear":
            p1, p2 = d.extra["p1"], d.extra["p2"]
            cv2.line(out, tuple(p1), tuple(p2), col, 2, cv2.LINE_AA)
        else:
            x, y, w, h = d.bbox
            pad = 3
            cv2.rectangle(out, (max(x - pad, 0), max(y - pad, 0)),
                          (x + w + pad, y + h + pad), col, 1, cv2.LINE_AA)
            r = max(w, h) // 2 + 8
            cx, cy = int(d.centroid[0]), int(d.centroid[1])
            cv2.circle(out, (cx, cy), r, col, 1, cv2.LINE_AA)
    return out


def index_map(rgb: np.ndarray, idx: np.ndarray, kind: str = "veg") -> np.ndarray:
    cmap = cv2.COLORMAP_SUMMER if kind == "veg" else cv2.COLORMAP_OCEAN
    n = np.clip((idx + 1) / 2.0, 0, 1)
    cm = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def legend_payload(lc: LandCover) -> List[dict]:
    out = []
    from .landcover import CLASS_LABELS
    for c in CLASSES:
        f = lc.fractions.get(c, 0.0)
        if f <= 0.0005:
            continue
        r, g, b = CLASS_COLORS[c]
        out.append({"cls": c, "label": CLASS_LABELS[c],
                    "color": f"rgb({r},{g},{b})", "fraction": round(f, 4),
                    "area_km2": round(lc.areas_km2.get(c, 0.0), 4)})
    out.sort(key=lambda d: -d["fraction"])
    return out
