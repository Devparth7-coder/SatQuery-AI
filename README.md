# SatQuery AI v2

**Ask Questions. Explore Earth.**

An evidence-first intelligence platform for understanding satellite imagery.
Upload a scene, ask in plain English, and get answers measured from the
pixels — each with visual proof, methodology, uncertainty and an auditable
evidence trail.

```
"How much of the area is water?"      → 58.8% · 62.9 km² + highlighted water mask
"How many ships are there?"           → 49 vessels + ringed detections + lengths
"What is the NDVI here?"              → honest unavailable state on RGB-only input
"What changed between the two dates?" → 3.67% changed · severity MEDIUM + hotspots
"Show the evidence behind this."      → pipeline trace + evidence IDs + methods
```

## Why it exists

Earth-observation answers must be **measured, not generated**. SatQuery keeps
deterministic remote-sensing code as the sole source of numbers; natural
language only plans, interprets and explains — and every claim cites evidence.

## Capabilities

- **Ingestion 2.0** — PNG/JPG/TIFF/WebP/BMP + multi-band; format, dims, bands,
  bit depth, CRS-or-`Metadata unavailable`, checksum, sensor mapping report.
- **Sensor-aware processing** — band-mapping abstraction; verified generic
  RGB/RGB+NIR today, Sentinel-2/Landsat/Planet/Cartosat/Resourcesat tables
  reserved (honestly marked planned).
- **Spectral engine** — NDVI/NDWI/SAVI/EVI/GNDVI with NIR; VARI/ExG/water-proxy
  on RGB; MNDWI/NDBI unavailable-without-SWIR states; formula + stats +
  histogram per index.
- **Land-cover intelligence** — k-means + physics rules, adaptive k, class
  proportions/areas, cluster signatures, uncertainty layer, transparent
  quality label (HIGH→INSUFFICIENT with fired rules).
- **Object intelligence** — water/vegetation/built regions, vessels,
  bright targets, linear structures in a unified schema with heuristic-score
  honesty notes.
- **Temporal intelligence** — ORB/RANSAC registration, histogram matching,
  CVA+Otsu, transition matrix, severity (LOW/MEDIUM/HIGH), hotspots, quality.
- **Map workspace** — pan/zoom viewer, layer toggles + opacity, legend,
  image-space coordinates, pixel inspector, measure/box/marker tools,
  before/after slider. Non-georeferenced inputs are labelled image-space.
- **Query planner 2.0** — structured intents (scene summary, land cover,
  spectral index, object count/location, temporal change, comparison, area,
  spatial statistics, evidence explanation) with visible
  QUERY → INTERPRETATION → TOOLS → EVIDENCE → ANSWER traces.
- **Grounded AI layer** — offline deterministic composer by default; optional
  provider interface (Ollama stub). Never invents measurements.
- **Evidence graph** — every answer traces to `E#` items (type/source/method/
  value/confidence/visualization).
- **Reports / history / reproducibility** — self-contained HTML reports,
  persistent versioned records (open/duplicate/export/delete) and one-click
  re-execution from stored inputs + checksums + seed 42.
- **Security + tests** — hardened uploads, no stack-trace leaks, rate limits,
  8 test suites (unit/integration/regression), live benchmarks.

## Quick start

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Click **Explore Demo** for the ground-truthed
bi-temporal pair, or upload your own scene. Set GSD correctly
(Sentinel-2 ≈ 10 m, LISS-IV ≈ 5.8 m, Cartosat-2 ≈ 0.65 m).

## Demo workflow

1. **Explore Demo** → loads `delta_t1/t2` (known edits: clearings, urban
   block, reservoir, 6/−4 px shift).
2. Ask *"What changed between the two dates?"* → 3.67% changed, severity,
   transitions, hotspots.
3. Open **Change Detection** → inspect the change map + hotspot table.
4. Ask *"Show the evidence behind this result"* (or toggle Explain mode).
5. **Reports → Generate Report** → download/print the audit-ready HTML.
6. **History → Reproduce** to re-run deterministically.

## Architecture

```
Text query → Planner (intent/target/operation/evidence/tools)
Scene → Features → Land cover → Scene graph → Evidence → Grounded answer + overlay
```

No pretrained weights, no API keys, fully offline. See `docs/ARCHITECTURE.md`.

| Layer | File |
|---|---|
| Ingestion/validation/sensors | `core/ingestion.py`, `core/validation.py`, `core/sensors.py` |
| Spectral + stats | `core/features.py` |
| Land cover + quality | `core/landcover.py` |
| Objects | `core/objects.py` |
| Change + severity | `core/change.py` |
| Query planner | `core/query.py` |
| Evidence | `core/evidence.py` |
| AI reasoning | `core/ai_reasoning.py` |
| History/reproduce | `core/history.py` |
| Reports | `core/report.py` |
| Rendering | `core/render.py` |

## API

v1 routes preserved (`/api/analyze`, `/api/query`, `/api/analyze_sample`,
`/api/samples`, `/api/health`); v2 adds `/api/version`, `/api/explain`,
`/api/indices`, `/api/objects`, `/api/evidence`, `/api/quality`,
`/api/inspect`, `/api/report`, `/api/history/*`, `/api/benchmark`,
`/api/sensors`, `/api/providers`. Details: `docs/API.md`.

## Validation

Ground-truthed demo pair (`samples/delta_truth.json`):

| Metric | Truth | Detected |
|---|---|---|
| Changed area | 4.47% | 3.67% |
| Applied shift | +6, −4 px (‖·‖≈7.2) | 5.2 px via ORB+RANSAC |
| Dominant transition | vegetation → bare | vegetation → bare, 1.17 km² |

Run `python -m pytest tests/ -q` and `POST /api/benchmark` for live numbers.

## Known limits

RGB-only proxies, inferred (not survey-grade) classes, classical detectors,
GSD-dependent areas, image-space coordinates without CRS. Full list:
`docs/LIMITATIONS.md`.

## Docs

`docs/ARCHITECTURE.md` · `docs/API.md` · `docs/ALGORITHMS.md` ·
`docs/EVIDENCE.md` · `docs/DEVELOPMENT.md` · `docs/SECURITY.md` ·
`docs/LIMITATIONS.md`

## Roadmap (extension points ready)

STAC/Sentinel-2/Landsat readers (`sensors.py` registry), GeoJSON/GeoTIFF
export, cloud masking, MapLibre/Leaflet GIS layer, VLM adapters
(`VisionModelProvider`), RAG over RS docs, agentic orchestration.
