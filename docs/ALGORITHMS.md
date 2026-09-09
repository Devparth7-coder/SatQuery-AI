# SatQuery AI v2 — Algorithms

Versions recorded with every analysis (see `core/config.py` `ALGO_VERSIONS`).

## Spectral (`spectral 2.0`)

- Float reflectance proxies in [0,1]; 16-bit inputs percentile-stretched.
- NIR present → true **NDVI**, **NDWI** (McFeeters), **SAVI** (L=0.5),
  **EVI**, **GNDVI**. RGB-only → **VARI**, **ExG**, Blue-Red water proxy.
- **MNDWI/NDBI** need SWIR: reported `available:false` with reason (no
  substitution). Each index ships formula, band requirements, stats
  (min/max/mean/median/std/p5/p95) and a 40-bin histogram.

## Land cover (`landcover 1.1`)

- MiniBatch k-means (seed 42) over standardised spectral+textural features,
  median-blur smoothing, then rule-based class assignment from cluster mean
  signatures (water / dense+sparse vegetation / built-up / bare / cloud / shadow).
- Scene-relative texture/edge cues (median/IQR context) — absolute thresholds
  are not used where they fail.
- v2 additions: deterministic adaptive-k heuristic (colour diversity + texture),
  per-pixel uncertainty (normalised centroid distance), transparent quality
  label (HIGH→MODERATE→LOW→INSUFFICIENT with fired rules).
- Labels are **algorithmically inferred**, never ground truth.

## Objects (`objects 1.1`)

- Connected components per class mask (lakes, patches).
- Median-background top-hat blobs → **vessels** (∩ dilated water mask) or
  **bright targets**; probabilistic Hough → **linear structures**.
- Scores are heuristic (contrast/size), **not calibrated probabilities** —
  stated wherever shown. Unified schema:
  `{id, type, bbox, centroid, area, confidence, source_method, evidence}`.

## Change (`change 1.1`)

- ORB(2500)/RANSAC affine registration (ECC translation fallback),
  histogram matching, shared-classifier labelling of both dates, CVA over
  6 normalised bands + Otsu (floor 0.18) + morphological cleanup.
- Transition matrix masked by change mask; per-class km² deltas.
- v2: severity LOW<2% / HIGH≥8% (config) gated by registration quality;
  hotspot regions; registration quality record (inliers, shift, kind).

## Query (`query 2.0`)

- Lexicon intent/entity parsing → v2 taxonomy (`scene_summary`,
  `land_cover`, `spectral_index`, `object_count`, `object_location`,
  `temporal_change`, `comparison`, `area_estimation`,
  `spatial_statistics`, `evidence_explanation`, `presence`) with target /
  operation / required-evidence / tools.
- Deterministic realisation: every sentence backed by a computed number;
  insufficient evidence → explicit refusal, never invention.
