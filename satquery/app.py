"""
SatQuery AI - FastAPI application server.

An Interactive Vision-Language Assistant for Multimodal Remote Sensing
Image Analysis through Text Queries.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import change as change_mod
from core import objects as obj_mod
from core import query as query_mod
from core import render
from core.features import compute_features, load_scene
from core.landcover import CLASS_LABELS, segment

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
STATIC = os.path.join(BASE, "static")
SAMPLES = os.path.join(BASE, "samples")
os.makedirs(DATA, exist_ok=True)

app = FastAPI(title="SatQuery AI", version="1.0.0",
              description="Interactive Vision-Language Assistant for Remote Sensing")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# In-memory session store: session_id -> analysis bundle
SESSIONS: Dict[str, dict] = {}
MAX_SESSIONS = 12


def _evict():
    if len(SESSIONS) > MAX_SESSIONS:
        oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1]["ts"])[0][0]
        SESSIONS.pop(oldest, None)


def _analyse(path: str, gsd: float, k: int) -> dict:
    t0 = time.time()
    scene = load_scene(path, gsd=gsd)
    fs = compute_features(scene)
    lc = segment(scene, fs, k=k)
    objs = obj_mod.summarize(scene, fs, lc)
    return {"scene": scene, "features": fs, "landcover": lc, "objects": objs,
            "path": path, "ms": int((time.time() - t0) * 1000), "ts": time.time()}


def _all_dets(objs: dict):
    return (objs["vessels"] + objs["bright_targets"] + objs["linear"])


def _overlay_image(sess: dict, spec: dict) -> Optional[str]:
    a = sess["a"]
    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    t = spec.get("type")
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


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS), "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...),
                  image2: Optional[UploadFile] = File(None),
                  gsd: float = Form(10.0),
                  k: int = Form(7)):
    if not image.filename:
        raise HTTPException(400, "no file provided")
    sid = uuid.uuid4().hex[:12]
    sdir = os.path.join(DATA, sid)
    os.makedirs(sdir, exist_ok=True)

    p1 = os.path.join(sdir, "t1_" + os.path.basename(image.filename))
    with open(p1, "wb") as f:
        f.write(await image.read())

    try:
        a = _analyse(p1, gsd, k)
    except Exception as e:
        raise HTTPException(400, f"could not read image: {e}")

    sess = {"a": a, "b": None, "change": None, "ts": time.time(), "gsd": gsd, "k": k}

    if image2 is not None and image2.filename:
        p2 = os.path.join(sdir, "t2_" + os.path.basename(image2.filename))
        with open(p2, "wb") as f:
            f.write(await image2.read())
        try:
            b = _analyse(p2, gsd, k)
            cr = change_mod.detect(a["scene"], b["scene"], k=k, lc1=a["landcover"])
            sess["b"] = b
            sess["change"] = {"result": cr}
        except Exception as e:
            sess["change_error"] = str(e)

    SESSIONS[sid] = sess
    _evict()

    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    resp = {
        "session": sid,
        "ms": a["ms"],
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
        "counts": {
            "water_regions": len(objs["water_regions"]),
            "vessels": len(objs["vessels"]),
            "bright_targets": len(objs["bright_targets"]),
            "linear": len(objs["linear"]),
        },
        "images": {
            "original": render.to_data_uri(scene.rgb, "JPEG"),
            "landcover": render.to_data_uri(render.landcover_overlay(scene.rgb, lc), "JPEG"),
            "veg": render.to_data_uri(render.index_map(scene.rgb, fs.veg_index, "veg"), "JPEG"),
            "detections": render.to_data_uri(
                render.detections_overlay(scene.rgb, _all_dets(objs)), "JPEG"),
        },
        "has_second": sess["b"] is not None,
    }
    if sess.get("change"):
        cr = sess["change"]["result"]
        resp["images"]["t2"] = render.to_data_uri(sess["b"]["scene"].rgb, "JPEG")
        resp["images"]["change"] = render.to_data_uri(
            render.change_overlay(sess["b"]["scene"].rgb, cr.magnitude, cr.change_mask), "JPEG")
        resp["change"] = {
            "changed_fraction": round(cr.changed_fraction, 4),
            "registration": cr.registration,
            "transitions": cr.transitions,
            "deltas": cr.deltas,
        }
    return JSONResponse(resp)


class Ask(BaseModel):
    session: str
    query: str


@app.post("/api/query")
def ask(body: Ask):
    sess = SESSIONS.get(body.session)
    if not sess:
        raise HTTPException(404, "session expired — please re-upload the image")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(400, "empty query")

    t0 = time.time()
    plan = query_mod.parse_query(q)
    result = query_mod.answer(plan, sess["a"], sess.get("change"))
    img = _overlay_image(sess, result.get("overlay", {}))
    result["overlay_image"] = img
    result["ms"] = int((time.time() - t0) * 1000)
    result["parsed"] = {"intent": plan.intent, "entities": plan.entities,
                        "objects": plan.objects, "superlative": plan.superlative}
    return JSONResponse(result)


@app.get("/api/samples")
def samples():
    out = []
    if os.path.isdir(SAMPLES):
        for fn in sorted(os.listdir(SAMPLES)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append({"name": fn, "url": f"/samples/{fn}",
                            "title": fn.replace("_", " ").rsplit(".", 1)[0].title()})
    return {"samples": out}


@app.post("/api/analyze_sample")
async def analyze_sample(name: str = Form(...), name2: Optional[str] = Form(None),
                         gsd: float = Form(10.0), k: int = Form(7)):
    p1 = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.exists(p1):
        raise HTTPException(404, "sample not found")
    sid = uuid.uuid4().hex[:12]
    a = _analyse(p1, gsd, k)
    sess = {"a": a, "b": None, "change": None, "ts": time.time(), "gsd": gsd, "k": k}
    if name2:
        p2 = os.path.join(SAMPLES, os.path.basename(name2))
        if os.path.exists(p2):
            b = _analyse(p2, gsd, k)
            cr = change_mod.detect(a["scene"], b["scene"], k=k, lc1=a["landcover"])
            sess["b"] = b
            sess["change"] = {"result": cr}
    SESSIONS[sid] = sess
    _evict()

    scene, lc, fs, objs = a["scene"], a["landcover"], a["features"], a["objects"]
    resp = {
        "session": sid, "ms": a["ms"],
        "scene": {"width": scene.orig_shape[1], "height": scene.orig_shape[0],
                  "analysis_width": scene.width, "analysis_height": scene.height,
                  "gsd": scene.gsd, "has_nir": scene.has_nir,
                  "area_km2": round(scene.n_pixels * scene.px_area_m2() / 1e6, 4),
                  "veg_index": fs.veg_index_name, "water_index": fs.water_index_name,
                  "name": scene.source_name},
        "legend": render.legend_payload(lc),
        "clusters": lc.cluster_report,
        "counts": {"water_regions": len(objs["water_regions"]),
                   "vessels": len(objs["vessels"]),
                   "bright_targets": len(objs["bright_targets"]),
                   "linear": len(objs["linear"])},
        "images": {
            "original": render.to_data_uri(scene.rgb, "JPEG"),
            "landcover": render.to_data_uri(render.landcover_overlay(scene.rgb, lc), "JPEG"),
            "veg": render.to_data_uri(render.index_map(scene.rgb, fs.veg_index, "veg"), "JPEG"),
            "detections": render.to_data_uri(
                render.detections_overlay(scene.rgb, _all_dets(objs)), "JPEG"),
        },
        "has_second": sess["b"] is not None,
    }
    if sess.get("change"):
        cr = sess["change"]["result"]
        resp["images"]["t2"] = render.to_data_uri(sess["b"]["scene"].rgb, "JPEG")
        resp["images"]["change"] = render.to_data_uri(
            render.change_overlay(sess["b"]["scene"].rgb, cr.magnitude, cr.change_mask), "JPEG")
        resp["change"] = {"changed_fraction": round(cr.changed_fraction, 4),
                          "registration": cr.registration,
                          "transitions": cr.transitions, "deltas": cr.deltas}
    return JSONResponse(resp)


app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
