"""SatQuery AI v2 — satellite image ingestion 2.0.

Builds an honest :class:`IngestionReport` for every input:

* format / dimensions / bit depth / band count / band names (when known)
* CRS / geotransform / nodata — or the explicit string "Metadata unavailable"
* SHA-256 checksum (reproducibility), file size, warnings
* sensor/band mapping resolution (see :mod:`core.sensors`)

GeoTIFF tags are read with Pillow's TIFF tags when present; rasterio /
tifffile are used opportunistically if installed (fully offline) but are
NOT required. Metadata is never invented: anything unknown is reported as
unavailable and every area figure derived from user-supplied GSD is labelled
as image-space analysis.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from . import sensors
from .features import Scene, load_scene

UNAVAILABLE = "Metadata unavailable"


@dataclass
class IngestionReport:
    filename: str
    safe_name: str
    format: str
    width: int
    height: int
    bands: int
    band_names: List[str]
    bit_depth: str
    mode: str
    crs: str
    geotransform: str
    nodata: str
    resolution_note: str
    coverage_note: str
    sensor_mapping: str
    sensor_verified: bool
    checksum_sha256: str
    size_bytes: int
    gsd_m: float
    gsd_source: str
    analysis_width: int = 0
    analysis_height: int = 0
    has_nir: bool = False
    warnings: List[str] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def checksum_sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tiff_tags(path: str) -> Dict[str, Any]:
    """Best-effort GeoTIFF tag extraction (Pillow only, offline)."""
    out: Dict[str, Any] = {}
    try:
        with Image.open(path) as im:
            if im.format != "TIFF":
                return out
            tags = getattr(im, "tag_v2", {}) or {}
            # Common GeoTIFF tags: 33550 scale, 33922 tiepoint, 34735 geo keys
            if 33550 in tags:
                out["pixel_scale"] = list(tags[33550])
            if 33922 in tags:
                out["tiepoint"] = list(tags[33922])
            if 34735 in tags:
                out["geo_keys"] = list(tags[34735])[:12]
            if 34737 in tags:
                out["geo_ascii"] = str(tags[34737])[:160]
            if 42113 in tags:  # GDAL_NODATA
                out["nodata_tag"] = str(tags[42113])
    except Exception:
        pass
    return out


def _optional_raster_meta(path: str) -> Dict[str, Any]:
    """Use rasterio/tifffile when installed; otherwise return {}.

    Both are optional, offline-capable dependencies. Absence only means
    less metadata — never a failure.
    """
    meta: Dict[str, Any] = {}
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as ds:
            if ds.crs:
                meta["crs"] = str(ds.crs)
            if ds.transform:
                meta["geotransform"] = list(ds.transform)[:6]
            if ds.nodata is not None:
                meta["nodata"] = str(ds.nodata)
            if ds.descriptions and any(ds.descriptions):
                meta["band_names"] = [d or f"B{i+1}" for i, d in enumerate(ds.descriptions)]
            if ds.res and all(ds.res):
                meta["resolution"] = [float(ds.res[0]), float(ds.res[1])]
        return meta
    except Exception:
        pass
    try:
        import tifffile  # type: ignore
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags.values()}
            if "ModelPixelScaleTag" in tags:
                meta["pixel_scale"] = list(tags["ModelPixelScaleTag"])
    except Exception:
        pass
    return meta


def ingest_file(path: str, gsd: float, safe_name: str = "",
                sensor_hint: Optional[str] = None) -> Tuple[Scene, IngestionReport]:
    """Load a raster into a :class:`Scene` plus a full ingestion report."""
    warnings: List[str] = []
    size_bytes = os.path.getsize(path)
    digest = checksum_sha256_file(path)

    with Image.open(path) as im:
        im.load()
        fmt = (im.format or os.path.splitext(path)[1].lstrip(".") or "raw").upper()
        w, h = im.size
        mode = im.mode
        n_bands = len(im.getbands())

    # Bit depth (honest: what PIL reports; 16-bit gets stretched at load).
    bit_depth = {"1": "1-bit", "L": "8-bit", "P": "8-bit palette",
                 "RGB": "8-bit/channel", "RGBA": "8-bit/channel + alpha",
                 "I;16": "16-bit", "I": "32-bit"}.get(mode, mode or UNAVAILABLE)

    band_names = [f"B{i+1}" for i in range(n_bands)]
    if n_bands == 3:
        band_names = ["red", "green", "blue"]
    elif n_bands == 4 and mode == "RGBA":
        band_names = ["red", "green", "blue", "alpha"]
        warnings.append("4th band is an alpha channel; ignored (not NIR).")
    elif n_bands >= 4:
        band_names = ["red", "green", "blue", "nir*"] + [f"B{i+1}" for i in range(4, n_bands)]
        warnings.append("4th band assumed to be NIR (generic 4-band layout).")

    mapping = sensors.resolve_mapping(n_bands, mode, sensor_hint)
    if sensor_hint and sensor_hint in sensors.SENSOR_REGISTRY and \
            not sensors.SENSOR_REGISTRY[sensor_hint].verified:
        warnings.append(f"Sensor '{sensor_hint}' is planned, not verified; "
                        f"using {mapping.sensor} mapping instead.")

    # Geo metadata: real tags only, else explicit "unavailable".
    crs = UNAVAILABLE
    geotransform = UNAVAILABLE
    nodata = UNAVAILABLE
    rmeta = _optional_raster_meta(path)
    ttags = _tiff_tags(path)
    if rmeta.get("crs"):
        crs = str(rmeta["crs"])
    elif ttags.get("geo_ascii"):
        crs = str(ttags["geo_ascii"])
    if rmeta.get("geotransform"):
        geotransform = str(rmeta["geotransform"])
    elif ttags.get("tiepoint") and ttags.get("pixel_scale"):
        geotransform = f"tiepoint={ttags['tiepoint'][:6]} scale={ttags['pixel_scale']}"
    if rmeta.get("nodata") is not None:
        nodata = str(rmeta["nodata"])
    elif ttags.get("nodata_tag"):
        nodata = str(ttags["nodata_tag"])
    if rmeta.get("band_names"):
        band_names = list(rmeta["band_names"])[:n_bands]

    if crs == UNAVAILABLE:
        coverage_note = ("Image-space analysis: no georeferencing found. "
                         "Positions are reported in pixel coordinates; areas use "
                         "the user-supplied GSD.")
    else:
        coverage_note = f"Georeferenced ({crs}). Pixel coordinates retained for audit."

    if rmeta.get("resolution"):
        rx = rmeta["resolution"][0]
        resolution_note = f"{rx:g} m/px from raster metadata."
    else:
        resolution_note = (f"{float(gsd):g} m/px user-supplied GSD "
                           f"(no resolution metadata in file).")

    # Load analysis scene (4-band TIFF -> NIR; RGBA alpha is NOT NIR).
    assume_nir = n_bands >= 4 and mode != "RGBA"
    scene = load_scene(path, gsd=gsd, assume_nir=assume_nir)

    report = IngestionReport(
        filename=os.path.basename(path), safe_name=safe_name or os.path.basename(path),
        format=fmt, width=w, height=h, bands=n_bands, band_names=band_names,
        bit_depth=bit_depth, mode=mode, crs=crs, geotransform=geotransform,
        nodata=nodata, resolution_note=resolution_note, coverage_note=coverage_note,
        sensor_mapping=mapping.sensor, sensor_verified=mapping.verified,
        checksum_sha256=digest, size_bytes=size_bytes, gsd_m=float(gsd),
        gsd_source="user input (GSD form field)",
        analysis_width=scene.width, analysis_height=scene.height,
        has_nir=scene.has_nir, warnings=warnings, status="ok",
    )
    return scene, report
