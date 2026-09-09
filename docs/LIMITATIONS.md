# SatQuery AI v2 — Limitations (surfaced, not hidden)

These are stated in-app (quality panels, evidence limitations, reports)
wherever they apply.

- **RGB-only input**: without NIR, vegetation/water rest on VARI/ExG/Blue-Red
  proxies. Turbid or sediment-laden water is red-dominant and can be confused
  with bare soil. MNDWI/NDBI are unavailable without SWIR.
- **Rule-derived classes** are algorithmically inferred, not survey-grade or
  ground-truth validated. The quality label + fired rules quantify this.
- **Classical detectors** (contrast blobs, Hough lines) are not trained object
  detectors. Low-contrast/mored craft can be missed; wave crests can
  false-positive. Scores are heuristic, not calibrated probabilities.
- **Areas assume your GSD**. Set it per sensor (Sentinel-2 ≈ 10 m,
  LISS-IV ≈ 5.8 m, Cartosat-2 ≈ 0.65 m). Non-georeferenced inputs are
  image-space analyses; positions are pixel coordinates.
- **Change detection** needs the same area at two dates; failed registration
  caps severity and is reported. Illumination/atmosphere differences are
  mitigated (histogram match + band normalisation), not eliminated.
- **AI reasoning** only rephrases supplied evidence (offline default). It
  cannot see the image and must refuse when evidence is insufficient.
- **Sessions expire** (in-memory, LRU). Persist via history; reproduce from
  stored inputs while they exist.
