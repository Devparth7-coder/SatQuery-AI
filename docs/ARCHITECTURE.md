# SatQuery AI v2 — Architecture

## Pipeline

```
INGESTION → VALIDATION → PREPROCESSING → GEO/SENSOR METADATA
        → ANALYSIS ORCHESTRATOR
              ├── Spectral Engine      (core/features.py)
              ├── Land-Cover Engine    (core/landcover.py)
              ├── Object Engine        (core/objects.py)
              ├── Change Engine        (core/change.py)
              ├── Spatial Statistics   (core/features.spatial_statistics)
              ├── Evidence Engine      (core/evidence.py)
              └── AI Reasoning Layer   (core/ai_reasoning.py, presentational only)
                      → Evidence-Grounded Answer → Visualization + Report
```

The query planner (`core/query.py`) is decoupled from the image-processing
implementations: it emits a structured plan (intent / target / operation /
required evidence / tools) and the deterministic tools run against the
already-computed scene graph. The planner never touches pixels directly.

## Module map

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI routes, sessions, middleware, history wiring |
| `core/config.py` | Versions, limits, thresholds (single source of truth) |
| `core/schemas.py` | Standard API envelopes |
| `core/validation.py` | Upload hardening (MIME, magic bytes, size, dims) |
| `core/ingestion.py` | Ingestion 2.0: format/band/geo-metadata report |
| `core/sensors.py` | Band-mapping abstraction + sensor registry |
| `core/features.py` | Scene loading, features, multi-index engine |
| `core/landcover.py` | k-means + physics rules, adaptive k, uncertainty, quality |
| `core/objects.py` | Classical detectors + unified object schema |
| `core/change.py` | Registration, CVA+Otsu, severity, hotspots, quality |
| `core/query.py` | Structured planner + grounded realisation |
| `core/evidence.py` | Evidence items + per-session evidence graph |
| `core/ai_reasoning.py` | Provider-agnostic grounded explanation (offline default) |
| `core/history.py` | Persistent records + reproduce |
| `core/report.py` | Self-contained HTML report generator |
| `core/render.py` | Overlays, heatmaps, legend payloads |
| `core/logging_utils.py` | Structured request logging |

## Data flow (single scene)

1. Bytes validated (`validation`) → stored under `data/<sid>/` with a safe name.
2. `ingest_file` builds the `Scene` + `IngestionReport` (checksum, dims, bands,
   CRS-or-unavailable, sensor mapping).
3. `compute_features` → `segment` (fixed or adaptive k) → `summarize` →
   `compute_all_indices` + `spatial_statistics`.
4. Evidence graph seeded; history record persisted; overlays rendered.
5. Queries parse → plan → answer against the scene graph → overlay + evidence.

## Bi-temporal path

T2 is ingested identically, co-registered to T1 (ORB/RANSAC, ECC fallback),
histogram-matched, then labelled with T1's fitted classifier + context so
both dates share one feature space. Change magnitude (CVA), Otsu mask,
transition matrix, severity, hotspots and quality are derived from there.

## Upgrade boundary (v1 → v2)

v1 response shapes are preserved on all original routes; v2 adds fields and
new endpoints. The v1 algorithms (physics rules, detectors, CVA) are intact —
v2 wraps them with ingestion reports, quality indicators, evidence IDs,
planner taxonomy, history and reports.
