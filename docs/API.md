# SatQuery AI v2 — API

Base URL: `http://localhost:8000`. All v1 routes keep their historical shapes
and gain additive `meta` / evidence fields. New routes use the envelope:

```json
{"success": true, "data": {}, "evidence": [], "warnings": [], "errors": [], "meta": {}}
```

Every response carries `X-Request-ID`. Errors on new routes are structured;
`HTTPException` routes return `{"detail": ...}` (v1-compatible).

## v1 routes (preserved)

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | multipart `image`, optional `image2`, `gsd`, `k`, `adaptive_k`, `sensor` → session + overlays + stats + ingestion + quality |
| `POST /api/query` | `{session, query}` → answer, evidence, overlay, confidence, **plan**, **evidence_v2** |
| `POST /api/analyze_sample` | same as analyze, by bundled sample name(s) |
| `GET /api/samples` | bundled scenes (+ dimensions/size) |
| `GET /api/health` | liveness + version + sessions + uptime |

## v2 routes

| Endpoint | Purpose |
|---|---|
| `GET /api/version` | product / version / build / pipeline_version / algorithms |
| `GET /api/sensors` | sensor registry with verified/planned status |
| `GET /api/providers` | reasoning providers + availability |
| `GET /api/indices?session=` | all spectral indices w/ formula, stats, histogram, availability |
| `GET /api/objects?session=` | unified-schema detections per family + size stats |
| `GET /api/evidence?session=` | evidence graph + query log |
| `GET /api/quality?session=` | classification + change quality + limitations |
| `GET /api/inspect?session=&x=&y=&date=t1` | pixel readout (RGB, indices, class, uncertainty, change) |
| `POST /api/explain` | `{session, query, provider?}` → answer + plan + pipeline steps + grounded reasoning + overlay |
| `GET /api/report?session=` | self-contained HTML report (prints to PDF) |
| `GET /api/history` | list records |
| `GET /api/history/{id}` | full record |
| `DELETE /api/history/{id}` | delete record |
| `POST /api/history/{id}/duplicate` | copy record |
| `POST /api/history/{id}/reproduce` | re-run deterministic pipeline from stored inputs |
| `POST /api/benchmark` | `{name, name2?, gsd?, k?}` → measured stage timings + query latency + memory |

## Notes

- Sessions are in-memory (cap 12, LRU-evicted); history is on disk.
- Uploads: ≤50 MB, ≤4096 px, PNG/JPG/TIFF/WebP/BMP, magic-byte verified.
- `POST /api/analyze` and `/api/benchmark` are rate-limited (30/min/IP).
- Coordinates are image-space pixels unless the ingestion report shows real CRS.
