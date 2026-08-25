# SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

ISRO · Space Technology · Software

---

## What it is

Upload a satellite scene, ask questions in plain English, get answers that are
**measured from the pixels** — each with a visual overlay showing exactly where
the answer came from, and an auditable evidence trail showing how each number
was computed.

```
"How much of the area is water?"      → 58.8% · 62.9 km² + highlighted water mask
"How many ships are there?"           → 27 vessels, 55–465 m + ringed detections
"Where is the built-up area?"         → centroid, extent, largest patch + overlay
"What changed between the two dates?" → per-class deltas + transition matrix
```

## Run

```bash
pip install fastapi "uvicorn[standard]" python-multipart numpy opencv-python pillow scikit-learn scikit-image scipy
cd satquery
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Click any sample scene, or drop your own image.
For change detection, click **Load bi-temporal demo pair**.

## Architecture

```
Text query ──► NLU (intent + entity parsing)
                     │
Scene ──► Features ──► Land-cover segmentation ──► Scene graph ──► Evidence retrieval
          (indices)    (k-means + physics rules)    (objects)            │
                                                                Grounded answer
                                                                + overlay + citations
```

**No pretrained weights, no API keys, fully offline.** Every figure is derived
from the image on device.

| Layer | File | What it does |
|---|---|---|
| Spectral features | `core/features.py` | NDVI/NDWI with NIR; VARI + water proxies for RGB. Texture, edge density, excess-green |
| Land cover | `core/landcover.py` | MiniBatch k-means over spectral+textural space, then rule-based class assignment from cluster signatures |
| Objects | `core/objects.py` | Connected-component regions, top-hat bright-target detection (vessels), Hough linear structures |
| Change | `core/change.py` | ORB/RANSAC co-registration, histogram matching, change vector analysis + Otsu, class transition matrix |
| Query engine | `core/query.py` | Intent/entity parsing → evidence retrieval → deterministic language realisation |
| Grounding | `core/render.py` | Class overlays, heatmaps, detection boxes |

### Why clustering + rules, not a black box

k-means finds what is *actually* in this image; the rule layer names it using
physical spectral behaviour. This means **no training data is required**, the
system generalises to any sensor, and — critically for operational use — every
label is explainable. The `clusters` field in `/api/analyze` returns each
cluster's mean signature and the scores that produced its label.

## Validation

The bi-temporal demo pair is **synthetically edited from a real Sentinel-2 scene**,
so change detection is checked against known ground truth (`samples/delta_truth.json`):

| Metric | Truth | Detected |
|---|---|---|
| Changed area | 4.47% | 3.67% |
| Applied shift | +6, −4 px | recovered 5.2 px via ORB+RANSAC |
| Dominant transition | vegetation → bare | vegetation → bare, 1.17 km² |

All five injected edits (3 clearings, 1 urban block, 1 reservoir) are localised
with no false positives elsewhere in the scene.

## Known limits (deliberately surfaced, not hidden)

- **RGB-only input**: turbid/sediment-laden water is red-dominant and can be
  confused with bare soil. The assistant states this in its own answers. Supply
  a 4-band product for true NDVI/NDWI separation.
- Vessel detection is contrast-based: low-contrast or moored craft may be missed,
  and bright wave crests can false-positive.
- Areas assume the GSD you enter — set it correctly (Sentinel-2 ≈ 10 m,
  LISS-IV ≈ 5.8 m, Cartosat-2 ≈ 0.65 m).
- Class labels are physics-derived, not survey-grade cadastral classes.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | multipart: `image`, optional `image2`, `gsd`, `k` → session + overlays + stats |
| `POST /api/query` | `{session, query}` → answer, evidence[], overlay image, confidence |
| `POST /api/analyze_sample` | analyse a bundled sample by name |
| `GET /api/samples` | list bundled scenes |
| `GET /api/health` | liveness |

## Extending to a true VLM

`core/query.py` is a drop-in seam. The scene graph (land-cover fractions,
detections, indices) is already a structured text-serialisable context — pass it
as grounding context to a Qwen-VL / LLaVA / Gemini call for free-form phrasing
while keeping the numbers authoritative and hallucination-free.
