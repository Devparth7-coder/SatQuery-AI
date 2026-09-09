"""SatQuery AI v2 — centralized configuration.

Single source of truth for product versioning, limits, algorithm versions
(used for reproducibility records) and tunable thresholds. No magic constants
scattered through the pipeline: everything user- or ops-tunable lives here
or in environment variables.
"""
from __future__ import annotations

import os

PRODUCT = "SatQuery AI"
VERSION = "2.0.0"
PIPELINE_VERSION = "2.0"
BUILD = os.environ.get("SATQUERY_BUILD", "dev")

# --------------------------------------------------------------------------- #
# Runtime limits (overridable via environment)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_MB = int(os.environ.get("SATQUERY_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_DIM = int(os.environ.get("SATQUERY_MAX_IMAGE_DIM", "4096"))
MAX_IMAGE_PIXELS = int(os.environ.get("SATQUERY_MAX_IMAGE_PIXELS", "33554432"))  # 32 MP
ANALYSIS_MAX_DIM = 1024  # analysis raster cap (features.py MAX_DIM, kept in sync)
MAX_SESSIONS = int(os.environ.get("SATQUERY_MAX_SESSIONS", "12"))
MAX_HISTORY_RECORDS = int(os.environ.get("SATQUERY_MAX_HISTORY", "100"))
QUERY_TIMEOUT_S = int(os.environ.get("SATQUERY_QUERY_TIMEOUT_S", "120"))

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
ALLOWED_MIMES = {
    "image/png", "image/jpeg", "image/tiff", "image/webp", "image/bmp",
    "application/octet-stream",  # browsers often send this for .tif
}

# --------------------------------------------------------------------------- #
# Algorithm versions — recorded with every analysis for reproducibility
# --------------------------------------------------------------------------- #
ALGO_VERSIONS = {
    "ingestion": "2.0",
    "spectral": "2.0",      # indices + stats engine
    "landcover": "1.1",     # v1 physics rules + adaptive-k + uncertainty layer
    "objects": "1.1",       # v1 detectors + unified schema
    "change": "1.1",        # v1 CVA/Otsu + severity + hotspots + quality
    "query": "2.0",         # structured planner, grounded realisation
    "render": "1.1",
    "evidence": "2.0",
}

# --------------------------------------------------------------------------- #
# Interpretable thresholds (documented; surfaced in reports, never hidden)
# --------------------------------------------------------------------------- #
CHANGE_SEVERITY_LOW = 0.02    # changed_fraction < 2%  -> LOW
CHANGE_SEVERITY_HIGH = 0.08   # changed_fraction >= 8% -> HIGH, else MEDIUM

QUALITY_RULES = {
    "no_nir_penalty": 0.90,       # RGB proxies weaker than true NIR indices
    "min_registration_inliers": 10,
    "large_shadow_fraction": 0.15,
    "large_cloud_fraction": 0.12,
    "min_cluster_margin_conf": 0.35,
}

CORS_ORIGINS = os.environ.get("SATQUERY_CORS", "*").split(",")
