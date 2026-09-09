# SatQuery AI v2 — Evidence model

```
ANSWER → EVIDENCE E17 → LAND-COVER MAP → PIXEL MASK
       → FEATURE EXTRACTION → SOURCE IMAGE
```

## Item schema

```json
{"evidence_id": "E17", "type": "measurement", "source": "change.detect",
 "method": "CVA over 6 normalised bands + Otsu", "value": 0.0367,
 "unit": "fraction", "confidence": null, "quality": "",
 "visualization": "change", "note": "changed_area_fraction"}
```

Types: `measurement`, `inference`, `detection`, `classification`,
`registration`, `metadata`, `limitation`. Limitations are first-class
evidence: missing bands, missing georeferencing and weak registration are
recorded, not hidden.

## Lifecycle

1. Analysis seeds the graph (input metadata, checksum, footprint, top class
   fractions, detector counts, quality, change severity).
2. Each query/explain appends its evidence rows with `query:<intent>` source
   and the overlay layer as `visualization`.
3. The UI Evidence panel, the chat replies, the report generator and the AI
   reasoning layer all read the same graph — numbers cannot diverge between
   surfaces.

## Grounding policy (AI reasoning layer)

- Only the evidence bundle is visible to the reasoning provider.
- No unsupported numerical claims, objects, dates, sensors, coordinates or
  certainty; every figure cites its evidence ID.
- Empty bundle → `"I don't have enough evidence to answer this reliably."`
- Default provider is offline + deterministic (template composer). External
  VLM adapters implement `VisionModelProvider`; their output must be checked
  against the bundle before display.
