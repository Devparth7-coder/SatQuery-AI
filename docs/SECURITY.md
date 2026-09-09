# SatQuery AI v2 — Security

## Upload hardening (`core/validation.py`)

- Extension + MIME allow-lists (PNG/JPG/TIFF/WebP/BMP).
- Magic-byte sniffing — the filename is never trusted.
- Size cap (50 MB), dimension cap (4096 px), pixel cap (32 MP,
  `Image.MAX_IMAGE_PIXELS` tightened against decompression bombs).
- Full PIL decode (`load()`) before acceptance; truncated/malicious
  payloads rejected.
- Server-side safe filenames (`<sanitised>_<rand>.<ext>`); user names are
  never used as paths; `os.path.basename` on sample names (no traversal).

## Runtime

- Uploaded files are never executed and never served as code.
- No stack traces in responses (global handler logs server-side, returns a
  generic message + request ID).
- Security headers: `X-Content-Type-Options: nosniff`, `Referrer-Policy`,
  `Permissions-Policy`. (Framing headers intentionally omitted for preview
  proxies.)
- Rate limiting on expensive endpoints (30/min/IP for analyze/benchmark).
- Request IDs on every response (`X-Request-ID`) for audit trails.
- CORS configurable via `SATQUERY_CORS` (default `*` for local use).

## Data

- Sessions in memory (cap 12, LRU eviction); history JSON on disk under
  `data/history/` (cap 100, oldest pruned).
- History stores parameters/checksums/results — pixels only in the
  per-session upload directory, never duplicated per record.
