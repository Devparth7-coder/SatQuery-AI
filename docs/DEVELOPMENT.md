# SatQuery AI v2 — Development

## Setup

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Fully offline; no API keys.

## Tests

```bash
python -m pytest tests/ -q
```

Suites: `test_features`, `test_landcover`, `test_objects`, `test_change`,
`test_query`, `test_validation`, `test_api` (integration via TestClient),
`test_regression` (known scenes, delta truth, determinism, edge cases).

Every major bug fix must add a regression test.

## Benchmarks

```bash
curl -X POST localhost:8000/api/benchmark \
  -H 'Content-Type: application/json' -d '{"name":"delta_t1.png","name2":"delta_t2.png"}'
```

Benchmarks are measured live on the host — never hard-code numbers.

## Conventions

- Typed dataclasses with `to_dict()` for API-facing structures.
- No magic constants: put them in `core/config.py`.
- Structured logging via `core/logging_utils.log_event`.
- Additive changes: keep v1 route shapes; extend, don't break.
- Never invent metadata/coordinates/confidence — surface `unavailable` states.
- Docstrings on all public functions; honest comments on heuristics.

## Environment knobs

`SATQUERY_BUILD`, `SATQUERY_MAX_UPLOAD_MB`, `SATQUERY_MAX_IMAGE_DIM`,
`SATQUERY_MAX_SESSIONS`, `SATQUERY_CORS`, `SATQUERY_OLLAMA_URL`,
`SATQUERY_OLLAMA_MODEL`.
