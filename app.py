"""SatQuery AI v2 — FastAPI application server.

An evidence-first AI Earth Observation intelligence platform.

Backward compatibility: all v1 routes keep their response shapes
(``/api/analyze``, ``/api/query``, ``/api/analyze_sample``, ``/api/samples``,
``/api/health``) and gain additive ``meta`` / evidence fields. New
capabilities ship as additive endpoints using the standard envelope from
:mod:`core.schemas`.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from core import change as change_mod
from core import history as history_mod
from core import objects as obj_mod
from core import query as query_mod
from core import render
from core import ai_reasoning, config, schemas
from core.evidence import EvidenceGraph, INFERENCE, LIMITATION, MEASUREMENT, METADATA
from core.features import (compute_all_indices, compute_features,
                           spatial_statistics)
from core.ingestion import ingest_file
from core.landcover import CLASS_LABELS, segment, suggest_k
from core.logging_utils import get_logger, log_event, set_request_id
from core.report import generate_html_report
from core.validation import validate_upload

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
STATIC = os.path.join(BASE, "static")
SAMPLES = os.path.join(BASE, "samples")
os.makedirs(DATA, exist_ok=True)

log = get_logger("satquery")
START_TS = time.time()

app = FastAPI(title="SatQuery AI", version=config.VERSION,
              description="Evidence-first AI Earth Observation intelligence platform")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Middleware: request IDs + security headers (framing headers deliberately
# omitted so the app keeps working behind preview proxies).
# --------------------------------------------------------------------------- #
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        set_request_id(rid)
        request.state.request_id = rid
        t0 = time.time()
        try:
            resp = await call_next(request)
        except Exception as e:  # never leak stack traces to clients
            log_event(log, "unhandled_error", level=40, error=f"{type(e).__name__}: {e}",
                      path=request.url.path)
            return JSONResponse({"detail": "Internal error while handling request.",
                                 "request_id": rid}, status_code=500)
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        log_event(log, "request", path=request.url.path, method=request.method,
                  status=resp.status_code, ms=int((time.time() - t0) * 1000))
        return resp


app.add_middleware(RequestContextMiddleware)

# --------------------------------------------------------------------------- #
# Lightweight rate limiting for expensive endpoints (in-memory, per-IP).
# --------------------------------------------------------------------------- #
_RATE: Dict[str, list] = {}
_RATE_LIMIT = 30
_RATE_WINDOW = 60.0


def _rate_check(request: Request, key: str) -> None:
    ip = (request.client.host if request.client else "?") + ":" + key
    now = time.time()
    hits = [t for t in _RATE.get(ip, []) if now - t < _RATE_WINDOW]
    if len(hits) >= _RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded — please wait a minute and retry.")
    hits.append(now)
    _RATE[ip] = hits


# --------------------------------------------------------------------------- #
# Sessions (in-memory) + persistent history (disk)
# --------------------------------------------------------------------------- #
SESSIONS: Dict[str, dict] = {}


def _evict() -> None:
    while len(SESSIONS) > config.MAX_SESSIONS:
        oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1]["ts"])[0][0]
        SESSIONS.pop(oldest, None)


def _rid(request: Optional[Request] = None) -> str:
    return getattr(getattr(request, "state", None), "request_id", "-")


# --------------------------------------------------------------------------- #
# Analysis pipeline with stage timings + evidence seeding
# --------------------------------------------------------------------------- #
def _analyse2(path: str, gsd: float, k: int, adaptive_k: bool = False,
              safe_name: str = "", sensor_hint: Optional[str] = None) -> dict:
    stages: Dict[str, int] = {}
    t = time.time()
    scene, ingestion = ingest_file(path, gsd, safe_name or os.path.basename(path),
                                   sensor_hint)
    stages["ingest_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    fs = compute_features(scene)
    stages["features_ms"] = int((time.time() - t) * 1000)

    kk = suggest_k(fs) if adaptive_k else int(max(4, min(k, 10)))
    t = time.time()
    lc = segment(scene, fs, k=kk)
    lc.adaptive_k_used = bool(adaptive_k)
    stages["landcover_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    objs = obj_mod.summarize(scene, fs, lc)
    stages["objects_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    indices = compute_all_indices(scene, fs)
    stats = spatial_statistics(fs)
    stages["indices_ms"] = int((time.time() - t) * 1000)

    ms = sum(stages.values())
    return {"scene": scene, "features": fs, "landcover": lc, "objects": objs,
            "indices": indices, "spatial_stats": stats, "ingestion": ingestion,
            "k_used": kk, "path": path, "ms": ms, "stages": stages,
            "ts": time.time()}


def _seed_evidence(sess: dict) -> EvidenceGraph:
    g = EvidenceGraph()
    a = sess["a"]
    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    ing = a.get("ingestion")
    if ing is not None:
        g.add(type=METADATA, source=ing.filename,
              method="ingestion report (measured headers + file bytes)",
              value=f"{ing.width}x{ing.height} {ing.format}, {ing.bands} bands",
              note="input_metadata")
        g.add(type=METADATA, source=ing.filename, method="SHA-256 of input bytes",
              value=ing.checksum_sha256[:16] + "…", note="input_checksum")
        if ing.crs == "Metadata unavailable":
            g.add_limitation(source="ingestion",
                             note="No georeferencing found — image-space analysis; "
                                  "areas use the user-supplied GSD.")
    total_km2 = scene.n_pixels * scene.px_area_m2() / 1e6
    g.add(type=MEASUREMENT, source="scene", method="pixel count × (GSD)²",
          value=round(total_km2, 4), unit="km²", note="scene_footprint")
    for name, frac in sorted(lc.fractions.items(), key=lambda kv: -kv[1])[:4]:
        if frac > 0.0005:
            g.add(type=INFERENCE, source="landcover.segment",
                  method="MiniBatch k-means + physics rule labelling",
                  value=round(frac, 4), unit="fraction",
                  note=f"fraction_{name}", visualization="landcover")
    g.add(type=MEASUREMENT, source="objects.summarize",
          method="contrast blobs ∩ water mask",
          value=len(objs["vessels"]), unit="count",
          note="vessel_count", visualization="detections")
    g.add(type=MEASUREMENT, source="objects.summarize",
          method="probabilistic Hough on Canny edges",
          value=len(objs["linear"]), unit="count",
          note="linear_count", visualization="detections")
    if not scene.has_nir:
        g.add_limitation(source="features",
                         note="No NIR band — vegetation/water rest on RGB proxies "
                              "(VARI, Blue-Red); turbid water may be confused "
                              "with bare soil.")
    q = getattr(lc, "quality", None) or {}
    if q.get("label"):
        g.add(type=INFERENCE, source="landcover.quality",
              method="transparent downgrade rules: " + "; ".join(q.get("rules", [])) or "none fired",
              value=q.get("label"), note="classification_quality",
              confidence=q.get("mean_confidence"))
    if sess.get("change") is not None:
        cr = sess["change"]["result"]
        g.add(type=MEASUREMENT, source="change.detect",
              method="CVA over 6 normalised bands + Otsu",
              value=round(cr.changed_fraction, 4), unit="fraction",
              note="changed_area_fraction", visualization="change")
        g.add(type=INFERENCE, source="change.severity",
              method="thresholds LOW<2% / HIGH>=8% + registration gate",
              value=getattr(cr, "severity", "UNKNOWN"), note="change_severity",
              visualization="change")
    return g


def _all_dets(objs: dict):
    return objs["vessels"] + objs["bright_targets"] + objs["linear"]


def _overlay_image(sess: dict, spec: dict):
    a = sess["a"]
    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    t = (spec or {}).get("type")
    if t == "landcover":
        return render.to_data_uri(render.landcover_overlay(scene.rgb, lc), "JPEG")
    if t == "class":
        return render.to_data_uri(render.class_overlay(scene.rgb, lc, spec["classes"]), "JPEG")
    if t == "detections":
        return render.to_data_uri(
            render.detections_overlay(scene.rgb, _all_dets(objs), spec.get("kinds")), "JPEG")
    if t == "change" and sess.get("change"):
        cr = sess["change"]["result"]
        return render.to_data_uri(
            render.change_overlay(sess["b"]["scene"].rgb, cr.magnitude, cr.change_mask), "JPEG")
    return None


def _base_images(a: dict) -> dict:
    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    return {
        "original": render.to_data_uri(scene.rgb, "JPEG"),
        "landcover": render.to_data_uri(render.landcover_overlay(scene.rgb, lc), "JPEG"),
        "veg": render.to_data_uri(render.index_map(scene.rgb, fs.veg_index, "veg"), "JPEG"),
        "detections": render.to_data_uri(
            render.detections_overlay(scene.rgb, _all_dets(objs)), "JPEG"),
    }


def _session_response(sid: str, sess: dict, request_id: str) -> dict:
    a = sess["a"]
    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    ing = a.get("ingestion")
    resp = {
        "session": sid,
        "analysis_id": sess.get("analysis_id", sid),
        "ms": a["ms"],
        "stages": a.get("stages", {}),
        "scene": {
            "width": scene.orig_shape[1], "height": scene.orig_shape[0],
            "analysis_width": scene.width, "analysis_height": scene.height,
            "gsd": scene.gsd, "has_nir": scene.has_nir,
            "area_km2": round(scene.n_pixels * scene.px_area_m2() / 1e6, 4),
            "veg_index": fs.veg_index_name, "water_index": fs.water_index_name,
            "name": scene.source_name,
        },
        "legend": render.legend_payload(lc),
        "clusters": lc.cluster_report,
        "quality": getattr(lc, "quality", None),
        "k_used": a.get("k_used"),
        "adaptive_k": sess.get("adaptive_k", False),
        "counts": {
            "water_regions": len(objs["water_regions"]),
            "vessels": len(objs["vessels"]),
            "bright_targets": len(objs["bright_targets"]),
            "linear": len(objs["linear"]),
        },
        "images": _base_images(a),
        "has_second": sess.get("b") is not None,
        "ingestion": ing.to_dict() if ing is not None else None,
        "warnings": (ing.warnings if ing is not None else []),
        "georeferenced": bool(ing is not None and ing.crs != "Metadata unavailable"),
        "meta": schemas.base_meta(request_id),
    }
    if sess.get("change"):
        cr = sess["change"]["result"]
        resp["images"]["t2"] = render.to_data_uri(sess["b"]["scene"].rgb, "JPEG")
        resp["images"]["change"] = render.to_data_uri(
            render.change_overlay(sess["b"]["scene"].rgb, cr.magnitude, cr.change_mask), "JPEG")
        if lc.uncertainty is not None:
            try:
                resp["images"]["uncertainty"] = render.to_data_uri(
                    render.uncertainty_overlay(scene.rgb, lc.uncertainty), "JPEG")
            except Exception:
                pass
        resp["change"] = {
            "changed_fraction": round(cr.changed_fraction, 4),
            "registration": cr.registration,
            "transitions": cr.transitions,
            "deltas": cr.deltas,
            "severity": getattr(cr, "severity", "UNKNOWN"),
            "hotspots": getattr(cr, "hotspots", None) or [],
            "quality": getattr(cr, "quality", None),
        }
    elif lc.uncertainty is not None:
        try:
            resp["images"]["uncertainty"] = render.to_data_uri(
                render.uncertainty_overlay(scene.rgb, lc.uncertainty), "JPEG")
        except Exception:
            pass
    if sess.get("change_error"):
        resp["change_error"] = sess["change_error"]
    ev = sess.get("evidence")
    resp["evidence_count"] = len(ev) if ev is not None else 0
    return resp


def _persist_history(sid: str, sess: dict, label: str) -> None:
    try:
        rec = history_mod.build_record(sid, label, sess, sess["a"]["ms"])
        sess["analysis_id"] = rec.analysis_id
        history_mod.save_record(DATA, rec)
    except Exception as e:
        log_event(log, "history_persist_failed", error=str(e))


# --------------------------------------------------------------------------- #
# v1 routes (shapes preserved)
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "sessions": len(SESSIONS),
            "version": config.VERSION, "pipeline_version": config.PIPELINE_VERSION,
            "uptime_s": int(time.time() - START_TS),
            "request_id": _rid(request)}


@app.get("/api/version")
def version(request: Request):
    return JSONResponse(schemas.success(
        {"product": config.PRODUCT, "version": config.VERSION,
         "build": config.BUILD, "pipeline_version": config.PIPELINE_VERSION,
         "algorithms": config.ALGO_VERSIONS},
        meta=schemas.base_meta(_rid(request))))


@app.post("/api/analyze")
async def analyze(request: Request,
                  image: UploadFile = File(...),
                  image2: Optional[UploadFile] = File(None),
                  gsd: float = Form(10.0),
                  k: int = Form(7),
                  adaptive_k: bool = Form(False),
                  sensor: Optional[str] = Form(None)):
    _rate_check(request, "analyze")
    if gsd <= 0 or gsd > 10000:
        raise HTTPException(400, "gsd must be within (0, 10000] m/px")
    content = await image.read()
    v = validate_upload(image.filename or "scene", content,
                        image.content_type or "")
    if not v.ok:
        raise HTTPException(400, "; ".join(v.errors))

    sid = uuid.uuid4().hex[:12]
    sdir = os.path.join(DATA, sid)
    os.makedirs(sdir, exist_ok=True)
    p1 = os.path.join(sdir, "t1_" + v.safe_name)
    with open(p1, "wb") as f:
        f.write(content)

    try:
        a = _analyse2(p1, gsd, k, adaptive_k, v.safe_name, sensor)
    except Exception as e:
        log_event(log, "analyze_failed", error=f"{type(e).__name__}: {e}")
        raise HTTPException(400, f"could not analyse image: {type(e).__name__}")

    sess = {"a": a, "b": None, "change": None, "ts": time.time(),
            "gsd": gsd, "k": k, "adaptive_k": adaptive_k,
            "queries": [], "analysis_id": sid}

    if image2 is not None and image2.filename:
        c2 = await image2.read()
        v2 = validate_upload(image2.filename, c2, image2.content_type or "")
        if not v2.ok:
            sess["change_error"] = "; ".join(v2.errors)
        else:
            p2 = os.path.join(sdir, "t2_" + v2.safe_name)
            with open(p2, "wb") as f:
                f.write(c2)
            try:
                t = time.time()
                b = _analyse2(p2, gsd, k, adaptive_k, v2.safe_name, sensor)
                cr = change_mod.detect(a["scene"], b["scene"], k=a["k_used"],
                                       lc1=a["landcover"])
                b["stages"]["change_ms"] = int((time.time() - t) * 1000) - b["ms"]
                sess["b"] = b
                sess["change"] = {"result": cr}
            except Exception as e:
                log_event(log, "change_failed", error=f"{type(e).__name__}: {e}")
                sess["change_error"] = f"change detection failed: {type(e).__name__}"

    sess["evidence"] = _seed_evidence(sess)
    SESSIONS[sid] = sess
    _evict()
    _persist_history(sid, sess, v.safe_name)
    return JSONResponse(_session_response(sid, sess, _rid(request)))


class Ask(BaseModel):
    session: str
    query: str


@app.post("/api/query")
def ask(body: Ask, request: Request):
    sess = SESSIONS.get(body.session)
    if not sess:
        raise HTTPException(404, "session expired — please re-upload the image")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(400, "empty query")
    if len(q) > 2000:
        raise HTTPException(400, "query too long (max 2000 characters)")

    t0 = time.time()
    plan = query_mod.parse_query(q)
    plan_v2 = query_mod.plan_to_dict(plan)  # snapshot: answer() accepts v1 aliases
    result = query_mod.answer(plan, sess["a"], sess.get("change"))
    img = _overlay_image(sess, result.get("overlay", {}))
    result["overlay_image"] = img
    result["ms"] = int((time.time() - t0) * 1000)
    result["intent"] = plan_v2["intent"]
    result["parsed"] = {"intent": plan_v2["intent"], "entities": plan.entities,
                        "objects": plan.objects, "superlative": plan.superlative}
    result["plan"] = plan_v2
    # Upgrade legacy evidence rows into the session evidence graph.
    try:
        graph: EvidenceGraph = sess.get("evidence") or EvidenceGraph()
        new_items = graph.extend_legacy(result.get("evidence", []),
                                        source=f"query:{plan_v2['intent']}",
                                        visualization=(result.get("overlay", {}) or {}).get("type", ""))
        result["evidence_v2"] = [i.to_dict() for i in new_items]
        sess["evidence"] = graph
    except Exception:
        result["evidence_v2"] = []
    result["meta"] = schemas.base_meta(_rid(request))
    sess["queries"].append({"query": q, "intent": plan_v2["intent"],
                            "confidence": result.get("confidence"),
                            "ms": result["ms"], "ts": time.time()})
    try:
        _persist_history(body.session, sess, sess.get("analysis_id", body.session))
    except Exception:
        pass
    return JSONResponse(result)


@app.get("/api/samples")
def samples(request: Request):
    out = []
    if os.path.isdir(SAMPLES):
        for fn in sorted(os.listdir(SAMPLES)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                item = {"name": fn, "url": f"/samples/{fn}",
                        "title": fn.replace("_", " ").rsplit(".", 1)[0].title()}
                try:
                    from PIL import Image as _IM
                    with _IM.open(os.path.join(SAMPLES, fn)) as im:
                        item["width"], item["height"] = im.size
                    item["size_bytes"] = os.path.getsize(os.path.join(SAMPLES, fn))
                except Exception:
                    pass
                out.append(item)
    return {"samples": out, "meta": schemas.base_meta(_rid(request))}


@app.post("/api/analyze_sample")
async def analyze_sample(request: Request,
                         name: str = Form(...), name2: Optional[str] = Form(None),
                         gsd: float = Form(10.0), k: int = Form(7),
                         adaptive_k: bool = Form(False),
                         sensor: Optional[str] = Form(None)):
    _rate_check(request, "analyze")
    if gsd <= 0 or gsd > 10000:
        raise HTTPException(400, "gsd must be within (0, 10000] m/px")
    p1 = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.exists(p1):
        raise HTTPException(404, "sample not found")
    sid = uuid.uuid4().hex[:12]
    try:
        a = _analyse2(p1, gsd, k, adaptive_k, os.path.basename(name), sensor)
    except Exception as e:
        log_event(log, "analyze_failed", error=f"{type(e).__name__}: {e}")
        raise HTTPException(400, f"could not analyse sample: {type(e).__name__}")
    sess = {"a": a, "b": None, "change": None, "ts": time.time(),
            "gsd": gsd, "k": k, "adaptive_k": adaptive_k,
            "queries": [], "analysis_id": sid}
    if name2:
        p2 = os.path.join(SAMPLES, os.path.basename(name2))
        if os.path.exists(p2):
            try:
                b = _analyse2(p2, gsd, k, adaptive_k, os.path.basename(name2), sensor)
                cr = change_mod.detect(a["scene"], b["scene"], k=a["k_used"],
                                       lc1=a["landcover"])
                sess["b"] = b
                sess["change"] = {"result": cr}
            except Exception as e:
                sess["change_error"] = f"change detection failed: {type(e).__name__}"
    sess["evidence"] = _seed_evidence(sess)
    SESSIONS[sid] = sess
    _evict()
    _persist_history(sid, sess, os.path.basename(name))
    return JSONResponse(_session_response(sid, sess, _rid(request)))


# --------------------------------------------------------------------------- #
# v2 endpoints (standard envelope)
# --------------------------------------------------------------------------- #
def _need_session(sid: str) -> dict:
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "session expired — please re-run the analysis")
    return sess


@app.get("/api/sensors")
def sensors(request: Request):
    from core.sensors import supported_sensors
    return JSONResponse(schemas.success(
        {"sensors": supported_sensors()},
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/indices")
def indices(request: Request, session: str = Query(...)):
    sess = _need_session(session)
    idx = sess["a"].get("indices") or {}
    return JSONResponse(schemas.success(
        {"indices": [v.to_dict() for v in idx.values()]},
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/objects")
def objects_api(request: Request, session: str = Query(...)):
    sess = _need_session(session)
    objs = sess["a"]["objects"]
    fams = {}
    for key in ("vessels", "bright_targets", "linear", "water_regions",
                "veg_regions", "built_regions"):
        fams[key] = {"detections": [obj_mod.detection_to_dict(d, f"{key[:2].upper()}{i+1:03d}")
                                    for i, d in enumerate(objs.get(key, []) or [])]}
        try:
            if key in ("vessels", "bright_targets", "linear"):
                fams[key]["stats"] = obj_mod.size_statistics(objs.get(key, []) or [])
        except Exception:
            pass
    return JSONResponse(schemas.success(
        {"families": fams,
         "confidence_note": obj_mod.CONFIDENCE_NOTE},
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/evidence")
def evidence_api(request: Request, session: str = Query(...)):
    sess = _need_session(session)
    g: Optional[EvidenceGraph] = sess.get("evidence")
    return JSONResponse(schemas.success(
        {"evidence": g.to_list() if g else [], "queries": sess.get("queries", [])},
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/quality")
def quality_api(request: Request, session: str = Query(...)):
    sess = _need_session(session)
    lc = sess["a"]["landcover"]
    out = {"classification": getattr(lc, "quality", None),
           "has_nir": sess["a"]["scene"].has_nir}
    if sess.get("change"):
        cr = sess["change"]["result"]
        out["change"] = {"severity": getattr(cr, "severity", "UNKNOWN"),
                         "quality": getattr(cr, "quality", None)}
    lims = []
    g: Optional[EvidenceGraph] = sess.get("evidence")
    if g:
        lims = [e for e in g.to_list() if e.get("type") == "limitation"]
    out["limitations"] = lims
    return JSONResponse(schemas.success(out, meta=schemas.base_meta(_rid(request))))


@app.get("/api/inspect")
def inspect_api(request: Request, session: str = Query(...),
                x: int = Query(...), y: int = Query(...),
                date: str = Query("t1")):
    sess = _need_session(session)
    key = "b" if date == "t2" and sess.get("b") else "a"
    a = sess[key]
    scene, lc, fs = a["scene"], a["landcover"], a["features"]
    if not (0 <= x < scene.width and 0 <= y < scene.height):
        raise HTTPException(400, f"coordinates out of range (0..{scene.width-1}, 0..{scene.height-1})")
    cls_idx = int(lc.label_map[y, x])
    from core.landcover import CLASSES
    payload = {
        "x": x, "y": y, "date": key,
        "rgb": [int(v) for v in scene.rgb[y, x].tolist()],
        "veg_index": {"name": fs.veg_index_name, "value": round(float(fs.veg_index[y, x]), 4)},
        "water_index": {"name": fs.water_index_name, "value": round(float(fs.water_index[y, x]), 4)},
        "class": CLASSES[cls_idx], "class_label": CLASS_LABELS[CLASSES[cls_idx]],
        "uncertainty": (round(float(lc.uncertainty[y, x]), 3)
                        if getattr(lc, "uncertainty", None) is not None else None),
        "pixel_area_m2": round(scene.px_area_m2(), 2),
        "coordinate_space": "image-space (pixels)" if not sess.get("b") or True else "image-space",
    }
    if sess.get("change") and key == "a":
        cr = sess["change"]["result"]
        try:
            payload["change"] = {"changed": bool(cr.change_mask[y, x]),
                                 "magnitude": round(float(cr.magnitude[y, x]), 3)}
        except Exception:
            payload["change"] = None
    return JSONResponse(schemas.success(payload, meta=schemas.base_meta(_rid(request))))


class Explain(BaseModel):
    session: str
    query: str
    provider: Optional[str] = None


@app.post("/api/explain")
def explain_api(body: Explain, request: Request):
    sess = _need_session(body.session)
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(400, "empty query")
    t0 = time.time()
    plan = query_mod.parse_query(q)
    v2_intent = plan.intent
    v2_plan = query_mod.plan_to_dict(plan)
    result = query_mod.answer(plan, sess["a"], sess.get("change"))
    img = _overlay_image(sess, result.get("overlay", {}))
    a = sess["a"]
    ing = a.get("ingestion")
    steps = [
        f"Loaded image '{ing.filename if ing else a['scene'].source_name}' "
        f"({a['scene'].orig_shape[1]}x{a['scene'].orig_shape[0]} px → "
        f"analysis {a['scene'].width}x{a['scene'].height} px).",
        f"Detected {ing.bands if ing else 3} band(s); "
        f"{'NIR present — true NDVI/NDWI family.' if a['scene'].has_nir else 'no NIR — RGB proxies (VARI, Blue-Red).'}",
        f"Segmented into {a.get('k_used', '?')} clusters "
        f"({'adaptive k' if a['landcover'].adaptive_k_used else 'fixed k'}) → "
        f"{len([f for f in a['landcover'].fractions.values() if f > 0.0005])} classes present.",
        f"Ran detectors: {len(a['objects']['vessels'])} vessels, "
        f"{len(a['objects']['bright_targets'])} bright targets, "
        f"{len(a['objects']['linear'])} linear structures.",
    ]
    if sess.get("change"):
        cr = sess["change"]["result"]
        steps.append(f"Co-registered T2 ({cr.registration}); changed fraction "
                     f"{cr.changed_fraction:.4f}; severity "
                     f"{getattr(cr, 'severity', 'UNKNOWN')}.")
    steps.append(f"Parsed query → intent '{v2_intent}', target '{plan.target}', "
                 f"operation '{plan.operation}'.")
    steps.append(f"Executed tools: {', '.join(plan.tools) or 'none'}; "
                 f"produced {len(result.get('evidence', []))} evidence rows.")
    graph: EvidenceGraph = sess.get("evidence") or EvidenceGraph()
    new_items = graph.extend_legacy(result.get("evidence", []),
                                    source=f"explain:{v2_intent}")
    sess["evidence"] = graph
    bundle = ai_reasoning.build_bundle(
        intent=v2_intent, measurements={},
        detections=[], changes=[],
        limitations=[e["note"] for e in graph.to_list()
                     if e.get("type") == "limitation"],
        evidence=[i.to_dict() for i in new_items],
        scene={"name": a["scene"].source_name,
               "veg_index": a["features"].veg_index_name})
    provider = ai_reasoning.get_provider(body.provider)
    try:
        reasoning = provider.explain(bundle, q)
    except Exception as e:
        reasoning = {"text": f"Reasoning provider unavailable: {type(e).__name__}.",
                     "citations": [], "refused": True}
    data = {"answer": result.get("answer"), "intent": v2_intent,
            "plan": v2_plan, "steps": steps,
            "evidence": result.get("evidence", []),
            "evidence_v2": [i.to_dict() for i in new_items],
            "reasoning": reasoning,
            "overlay_image": img,
            "confidence": result.get("confidence"),
            "ms": int((time.time() - t0) * 1000)}
    return JSONResponse(schemas.success(data, meta=schemas.base_meta(_rid(request))))


@app.get("/api/report")
def report_api(request: Request, session: str = Query(...)):
    sess = _need_session(session)
    a = sess["a"]
    scene, lc = a["scene"], a["landcover"]
    ing = a.get("ingestion")
    summary = _session_response(session, sess, _rid(request))
    g: Optional[EvidenceGraph] = sess.get("evidence")
    evidence = g.to_list() if g else []
    limitations = [e.get("note", "") for e in evidence if e.get("type") == "limitation"]
    limitations += [
        "Rule-derived classes are algorithmically inferred, not ground-truth validated.",
        "Classical detectors are contrast/morphology-based, not trained object detectors.",
        "Areas use the user-supplied GSD; set it correctly for the sensor.",
    ]
    repro = {"product": config.PRODUCT, "version": config.VERSION,
             "pipeline_version": config.PIPELINE_VERSION, "build": config.BUILD,
             "input_checksum_sha256": ing.checksum_sha256 if ing else "n/a",
             "gsd_m": sess.get("gsd"), "k": sess.get("k"),
             "adaptive_k": sess.get("adaptive_k"), "k_used": a.get("k_used"),
             "seed": 42, "analysis_id": sess.get("analysis_id", session),
             "duration_ms": a["ms"], "stages_ms": a.get("stages", {})}
    for k2, v2 in config.ALGO_VERSIONS.items():
        repro[f"algo_{k2}"] = v2
    html_doc = generate_html_report(
        title=f"Scene analysis — {scene.source_name}",
        session_summary=summary, queries=sess.get("queries", []),
        evidence=evidence, images=summary.get("images", {}),
        reproducibility=repro, limitations=limitations)
    return HTMLResponse(html_doc)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
@app.get("/api/history")
def history_list(request: Request):
    return JSONResponse(schemas.success(
        {"records": history_mod.list_records(DATA)},
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/history/{analysis_id}")
def history_get(analysis_id: str, request: Request):
    doc = history_mod.get_record(DATA, analysis_id)
    if not doc:
        raise HTTPException(404, "history record not found")
    return JSONResponse(schemas.success({"record": doc},
                                        meta=schemas.base_meta(_rid(request))))


@app.delete("/api/history/{analysis_id}")
def history_delete(analysis_id: str, request: Request):
    ok = history_mod.delete_record(DATA, analysis_id)
    if not ok:
        raise HTTPException(404, "history record not found")
    return JSONResponse(schemas.success({"deleted": analysis_id},
                                        meta=schemas.base_meta(_rid(request))))


@app.post("/api/history/{analysis_id}/duplicate")
def history_duplicate(analysis_id: str, request: Request):
    doc = history_mod.duplicate_record(DATA, analysis_id)
    if not doc:
        raise HTTPException(404, "history record not found")
    return JSONResponse(schemas.success({"record": doc},
                                        meta=schemas.base_meta(_rid(request))))


@app.post("/api/history/{analysis_id}/reproduce")
def history_reproduce(analysis_id: str, request: Request):
    _rate_check(request, "analyze")
    doc = history_mod.get_record(DATA, analysis_id)
    if not doc:
        raise HTTPException(404, "history record not found")
    inputs = doc.get("inputs", []) or []
    params = doc.get("params", {}) or {}
    if not inputs:
        raise HTTPException(400, "record has no stored inputs to reproduce from")
    # Resolve stored input bytes: uploads live under data/<sid>/, samples under samples/.
    sid = uuid.uuid4().hex[:12]
    try:
        p1 = _resolve_history_input(inputs[0])
        a = _analyse2(p1, float(params.get("gsd", 10.0)), int(params.get("k", 7)),
                      bool(params.get("adaptive_k", False)),
                      os.path.basename(p1))
    except FileNotFoundError:
        raise HTTPException(404, "original input pixels are no longer stored — cannot reproduce")
    except Exception as e:
        raise HTTPException(400, f"reproduction failed: {type(e).__name__}")
    sess = {"a": a, "b": None, "change": None, "ts": time.time(),
            "gsd": params.get("gsd", 10.0), "k": params.get("k", 7),
            "adaptive_k": bool(params.get("adaptive_k", False)),
            "queries": [], "analysis_id": sid}
    if len(inputs) > 1:
        try:
            p2 = _resolve_history_input(inputs[1])
            b = _analyse2(p2, float(params.get("gsd", 10.0)), int(params.get("k", 7)),
                          bool(params.get("adaptive_k", False)), os.path.basename(p2))
            cr = change_mod.detect(a["scene"], b["scene"], k=a["k_used"], lc1=a["landcover"])
            sess["b"] = b
            sess["change"] = {"result": cr}
        except Exception as e:
            sess["change_error"] = f"change reproduction failed: {type(e).__name__}"
    sess["evidence"] = _seed_evidence(sess)
    SESSIONS[sid] = sess
    _evict()
    _persist_history(sid, sess, (doc.get("label", "") + " (reproduced)").strip())
    out = _session_response(sid, sess, _rid(request))
    out["reproduced_from"] = analysis_id
    return JSONResponse(out)


def _resolve_history_input(inp: dict) -> str:
    sp = inp.get("stored_path") or ""
    if sp and os.path.exists(sp):
        return sp
    fname = os.path.basename(inp.get("safe_name") or inp.get("filename") or "")
    # Search data/<sid>/t?_fname then samples/.
    for root, _, files in os.walk(DATA):
        for f in files:
            if f.endswith(fname) or fname.endswith(f):
                return os.path.join(root, f)
    cand = os.path.join(SAMPLES, fname)
    if os.path.exists(cand):
        return cand
    # Try matching by checksum prefix is overkill; fail honestly.
    raise FileNotFoundError(fname)


# --------------------------------------------------------------------------- #
# Benchmark (real measurements only)
# --------------------------------------------------------------------------- #
class BenchReq(BaseModel):
    name: str = "delta_t1.png"
    name2: Optional[str] = "delta_t2.png"
    gsd: float = 10.0
    k: int = 7


@app.post("/api/benchmark")
def benchmark(body: BenchReq, request: Request):
    _rate_check(request, "benchmark")
    p1 = os.path.join(SAMPLES, os.path.basename(body.name))
    if not os.path.exists(p1):
        raise HTTPException(404, "sample not found")
    t0 = time.time()
    a = _analyse2(p1, body.gsd, body.k)
    stages = dict(a.get("stages", {}))
    change_ms = None
    if body.name2:
        p2 = os.path.join(SAMPLES, os.path.basename(body.name2))
        if os.path.exists(p2):
            t = time.time()
            b = _analyse2(p2, body.gsd, body.k)
            cr = change_mod.detect(a["scene"], b["scene"], k=a["k_used"], lc1=a["landcover"])
            change_ms = int((time.time() - t) * 1000)
            stages["change_ms"] = change_ms
    # Query latency (real, measured on fixed questions).
    q_times = {}
    for qq in ("Describe this scene",
               "How much of the area is water?",
               "What changed between the two dates?"):
        tq = time.time()
        plan = query_mod.parse_query(qq)
        query_mod.answer(plan, a, ({"result": cr} if body.name2 and "cr" in dir() else None))
        q_times[qq] = int((time.time() - tq) * 1000)
    mem = None
    try:
        import resource
        mem = {"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    except Exception:
        mem = {"peak_rss_mb": "unavailable (resource module missing)"}
    total = int((time.time() - t0) * 1000)
    data = {"sample": body.name, "sample2": body.name2, "gsd": body.gsd, "k": body.k,
            "image": {"width": a["scene"].orig_shape[1], "height": a["scene"].orig_shape[0],
                      "analysis_width": a["scene"].width,
                      "analysis_height": a["scene"].height,
                      "pixels": a["scene"].n_pixels},
            "stages_ms": stages, "total_ms": total,
            "query_latency_ms": q_times, "memory": mem,
            "versions": {"version": config.VERSION, "pipeline": config.PIPELINE_VERSION}}
    return JSONResponse(schemas.success(
        data, warnings=["Benchmarks are measured on this host at this moment; "
                        "results vary with hardware and load."],
        meta=schemas.base_meta(_rid(request))))


@app.get("/api/providers")
def providers(request: Request):
    return JSONResponse(schemas.success(
        {"providers": ai_reasoning.list_providers()},
        meta=schemas.base_meta(_rid(request))))


app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
