"""SatQuery AI v2 — sensor / band abstraction.

Instead of assuming ``RGB means fixed wavelengths`` everywhere, the pipeline
resolves a lightweight :class:`BandMapping` at ingestion time. Spectral-index
code then asks the mapping which logical bands exist instead of guessing.

Only mappings flagged ``verified=True`` are claimed as supported. Planned
sensors ship their band tables for future use but are explicitly marked
experimental — the system never claims to support a sensor it cannot
actually ingest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Logical band names used across the pipeline.
BLUE = "blue"
GREEN = "green"
RED = "red"
NIR = "nir"
SWIR1 = "swir1"
SWIR2 = "swir2"
ALPHA = "alpha"


@dataclass
class BandMapping:
    sensor: str
    bands: Dict[str, int]          # logical name -> raster band index (0-based)
    band_count: int
    verified: bool                 # True only if actually tested end-to-end
    notes: str = ""

    def has(self, *names: str) -> bool:
        return all(n in self.bands for n in names)

    def available_bands(self) -> List[str]:
        return sorted(self.bands)


SENSOR_REGISTRY: Dict[str, BandMapping] = {
    # --- Verified: exercised by the test-suite and sample scenes. ---
    "generic_rgb": BandMapping(
        sensor="generic_rgb", bands={RED: 0, GREEN: 1, BLUE: 2},
        band_count=3, verified=True,
        notes="Plain 3-band true-colour imagery. RGB proxies are used for "
              "vegetation/water cues; NIR indices are reported unavailable."),
    "generic_rgb_nir": BandMapping(
        sensor="generic_rgb_nir", bands={RED: 0, GREEN: 1, BLUE: 2, NIR: 3},
        band_count=4, verified=True,
        notes="4-band product with NIR as the 4th band (e.g. stacked GeoTIFF). "
              "True NDVI/NDWI family indices are computed."),
    # --- Planned: band tables reserved for future ingestion work. ---
    "sentinel2": BandMapping(
        sensor="sentinel2",
        bands={BLUE: 1, GREEN: 2, RED: 3, NIR: 7, SWIR1: 10, SWIR2: 11},
        band_count=12, verified=False,
        notes="Planned: Sentinel-2 L1C/L2A band layout (0-based indices into "
              "B01..B12). Not yet wired to a reader."),
    "landsat89": BandMapping(
        sensor="landsat89",
        bands={BLUE: 1, GREEN: 2, RED: 3, NIR: 4, SWIR1: 5, SWIR2: 6},
        band_count=7, verified=False,
        notes="Planned: Landsat 8/9 OLI band layout."),
    "planet": BandMapping(
        sensor="planet", bands={BLUE: 0, GREEN: 1, RED: 2, NIR: 3},
        band_count=4, verified=False,
        notes="Planned: PlanetScope 4-band layout."),
    "cartosat": BandMapping(
        sensor="cartosat", bands={RED: 0, GREEN: 0, BLUE: 0},
        band_count=1, verified=False,
        notes="Planned: Cartosat panchromatic — single band; multispectral "
              "indices do not apply."),
    "resourcesat": BandMapping(
        sensor="resourcesat", bands={GREEN: 0, RED: 1, NIR: 2},
        band_count=3, verified=False,
        notes="Planned: Resourcesat LISS-III/IV green/red/NIR layout."),
}


def resolve_mapping(band_count: int, mode: str = "",
                    hint: Optional[str] = None) -> BandMapping:
    """Pick the honest band mapping for an ingested raster.

    * ``hint`` may name a registry sensor, but unverified sensors are never
      silently used — the generic mapping is returned with a note instead.
    * RGBA input maps to ``generic_rgb`` (alpha is ignored, never mistaken
      for NIR).
    """
    if hint and hint in SENSOR_REGISTRY:
        m = SENSOR_REGISTRY[hint]
        if m.verified:
            return m
        # Fall through to the generic mapping; the caller surfaces the note.
    if band_count >= 4 and mode not in ("RGBA", "RGBa"):
        return SENSOR_REGISTRY["generic_rgb_nir"]
    return SENSOR_REGISTRY["generic_rgb"]


def supported_sensors() -> List[Dict[str, object]]:
    return [{"sensor": m.sensor, "verified": m.verified,
             "bands": m.available_bands(), "notes": m.notes}
            for m in SENSOR_REGISTRY.values()]
