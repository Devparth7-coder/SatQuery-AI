"""Integration tests: API v1 compatibility + v2 endpoints."""
import io

from fastapi.testclient import TestClient
from PIL import Image

import app as app_mod

client = TestClient(app_mod.app)


def _analyze(name="harbour_port.png", name2=None):
    data = {"name": name, "gsd": "10", "k": "7"}
    if name2:
        data["name2"] = name2
    r = client.post("/api/analyze_sample", data=data)
    assert r.status_code == 200, r.text[:300]
    return r.json()


def test_health_and_version():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["version"] == "2.0.0"
    v = client.get("/api/version").json()
    assert v["success"] is True
    assert v["data"]["pipeline_version"] == "2.0"
    assert "landcover" in v["data"]["algorithms"]


def test_samples_list():
    s = client.get("/api/samples").json()
    names = [x["name"] for x in s["samples"]]
    assert "harbour_port.png" in names and "delta_t1.png" in names


def test_analyze_sample_v1_shape():
    d = _analyze()
    for k in ("session", "ms", "scene", "legend", "clusters", "counts",
              "images", "has_second"):
        assert k in d, k  # v1 keys preserved
    assert d["images"]["original"].startswith("data:image/jpeg;base64,")
    assert d["scene"]["has_nir"] is False
    assert d["quality"]["label"] in ("HIGH", "MODERATE", "LOW", "INSUFFICIENT")


def test_query_v2_plan():
    d = _analyze()
    r = client.post("/api/query", json={"session": d["session"],
                                        "query": "How much of this scene is water?"})
    assert r.status_code == 200
    j = r.json()
    assert j["intent"] == "area_estimation"
    assert j["plan"]["operation"] == "measure_area"
    assert j["plan"]["tools"] and j["plan"]["required_evidence"]
    assert j["overlay_image"].startswith("data:image")
    assert "58.8%" in j["answer"]  # measured harbour water fraction


def test_query_unknown_session_404():
    r = client.post("/api/query", json={"session": "nope", "query": "hi"})
    assert r.status_code == 404


def test_indices_objects_evidence_quality():
    d = _analyze()
    sid = d["session"]
    idx = client.get("/api/indices", params={"session": sid}).json()["data"]["indices"]
    by = {i["name"]: i for i in idx}
    assert by["VARI"]["available"] and not by["NDVI"]["available"]
    objs = client.get("/api/objects", params={"session": sid}).json()["data"]
    assert "vessels" in objs["families"] and "confidence_note" in objs
    ev = client.get("/api/evidence", params={"session": sid}).json()["data"]
    assert len(ev["evidence"]) > 5
    q = client.get("/api/quality", params={"session": sid}).json()["data"]
    assert q["classification"]["label"]


def test_inspect_and_bounds():
    d = _analyze()
    sid = d["session"]
    p = client.get("/api/inspect", params={"session": sid, "x": 100, "y": 100}).json()["data"]
    assert p["rgb"] and p["class"] and "veg_index" in p
    bad = client.get("/api/inspect", params={"session": sid, "x": 99999, "y": 0})
    assert bad.status_code == 400


def test_explain_and_report():
    d = _analyze()
    sid = d["session"]
    e = client.post("/api/explain", json={"session": sid,
                                          "query": "How much water?"}).json()["data"]
    assert e["steps"] and e["plan"]["intent"] == "area_estimation"
    assert "reasoning" in e
    rep = client.get("/api/report", params={"session": sid})
    assert rep.status_code == 200 and "SatQuery AI Report" in rep.text
    assert "Reproducibility" in rep.text


def test_change_pair_end_to_end():
    d = _analyze("delta_t1.png", "delta_t2.png")
    assert d["has_second"] is True
    assert d["change"]["severity"] in ("LOW", "MEDIUM", "HIGH")
    assert d["change"]["hotspots"]
    r = client.post("/api/query", json={"session": d["session"],
                                        "query": "What changed between the two dates?"})
    assert r.json()["intent"] == "temporal_change"


def test_history_crud_and_reproduce():
    before = client.get("/api/history").json()["data"]["records"]
    d = _analyze("urban_mixed.png")
    recs = client.get("/api/history").json()["data"]["records"]
    assert len(recs) >= len(before) + 1
    aid = d["analysis_id"]
    got = client.get(f"/api/history/{aid}").json()["data"]["record"]
    assert got["analysis_id"] == aid
    dup = client.post(f"/api/history/{aid}/duplicate").json()["data"]["record"]
    assert dup["analysis_id"] != aid
    rep = client.post(f"/api/history/{aid}/reproduce")
    assert rep.status_code == 200, rep.text[:300]
    assert rep.json()["reproduced_from"] == aid
    assert client.delete(f"/api/history/{dup['analysis_id']}").status_code == 200


def test_upload_validation_rejects_garbage():
    r = client.post("/api/analyze", files={"image": ("evil.png", b"junk-bytes",
                                                     "image/png")})
    assert r.status_code == 400


def test_upload_roundtrip():
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (60, 140, 60)).save(buf, "PNG")
    r = client.post("/api/analyze", files={"image": ("green.png", buf.getvalue(),
                                                     "image/png")},
                    data={"gsd": "10", "k": "5"})
    assert r.status_code == 200
    assert r.json()["session"]


def test_benchmark_measures_real_numbers():
    b = client.post("/api/benchmark", json={"name": "delta_t1.png",
                                            "name2": "delta_t2.png"}).json()["data"]
    assert b["total_ms"] > 0
    assert b["stages_ms"]["landcover_ms"] > 0
    assert b["query_latency_ms"] and b["memory"]
