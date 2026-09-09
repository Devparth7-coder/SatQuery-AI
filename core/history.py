"""SatQuery AI v2 — persistent analysis history + reproducibility.

Each analysis lands in ``data/history/{analysis_id}.json`` with:

* input metadata (checksums, dimensions — never the pixels themselves)
* parameters, pipeline + algorithm versions, timestamp, random seed
* results summary, evidence snapshot, query log, processing duration

Large rasters are NOT duplicated: the record points at the stored inputs so
"Reproduce Analysis" can re-run the deterministic pipeline from the same
bytes + configuration and verify the checksum first.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import config


@dataclass
class AnalysisRecord:
    analysis_id: str
    timestamp: float
    inputs: List[Dict[str, Any]]
    params: Dict[str, Any]
    versions: Dict[str, str]
    seed: int
    results: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    duration_ms: int
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _history_dir(base: str) -> str:
    d = os.path.join(base, "history")
    os.makedirs(d, exist_ok=True)
    return d


def _path(base: str, analysis_id: str) -> str:
    safe = "".join(c for c in analysis_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(_history_dir(base), f"{safe}.json")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def save_record(base: str, record: AnalysisRecord) -> str:
    with open(_path(base, record.analysis_id), "w") as f:
        json.dump(record.to_dict(), f, indent=1, default=str)
    _enforce_cap(base)
    return record.analysis_id


def _enforce_cap(base: str) -> None:
    try:
        recs = list_records(base)
        if len(recs) > config.MAX_HISTORY_RECORDS:
            for r in recs[config.MAX_HISTORY_RECORDS:]:
                delete_record(base, r["analysis_id"])
    except Exception:
        pass


def list_records(base: str) -> List[Dict[str, Any]]:
    d = _history_dir(base)
    out: List[Dict[str, Any]] = []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                doc = json.load(f)
            out.append({
                "analysis_id": doc.get("analysis_id"),
                "timestamp": doc.get("timestamp", 0),
                "label": doc.get("label", ""),
                "duration_ms": doc.get("duration_ms", 0),
                "versions": doc.get("versions", {}),
                "params": doc.get("params", {}),
                "results": doc.get("results", {}),
                "n_queries": len(doc.get("queries", [])),
                "n_evidence": len(doc.get("evidence", [])),
            })
        except Exception:
            continue
    out.sort(key=lambda r: -float(r.get("timestamp") or 0))
    return out


def get_record(base: str, analysis_id: str) -> Optional[Dict[str, Any]]:
    p = _path(base, analysis_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def delete_record(base: str, analysis_id: str) -> bool:
    p = _path(base, analysis_id)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def duplicate_record(base: str, analysis_id: str) -> Optional[Dict[str, Any]]:
    doc = get_record(base, analysis_id)
    if not doc:
        return None
    doc["analysis_id"] = new_id()
    doc["timestamp"] = time.time()
    doc["label"] = (doc.get("label", "") + " (copy)").strip()
    save_record(base, AnalysisRecord(
        analysis_id=doc["analysis_id"], timestamp=doc["timestamp"],
        inputs=doc.get("inputs", []), params=doc.get("params", {}),
        versions=doc.get("versions", {}), seed=doc.get("seed", 42),
        results=doc.get("results", {}), evidence=doc.get("evidence", []),
        queries=doc.get("queries", []), duration_ms=doc.get("duration_ms", 0),
        label=doc.get("label", "")))
    return doc


def build_record(session_id: str, label: str, sess: dict,
                 duration_ms: int) -> AnalysisRecord:
    """Snapshot an in-memory session into a persistable record."""
    a = sess.get("a", {})
    scene = a.get("scene")
    lc = a.get("landcover")
    inputs: List[Dict[str, Any]] = []
    for key, role in (("a", "T1"), ("b", "T2")):
        bundle = sess.get(key) or {}
        rep = bundle.get("ingestion")
        if rep is not None:
            d = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
            d["role"] = role
            d["stored_path"] = bundle.get("path", "")
            inputs.append(d)
    results: Dict[str, Any] = {}
    if scene is not None and lc is not None:
        results = {
            "fractions": {k: round(v, 4) for k, v in lc.fractions.items()},
            "areas_km2": {k: round(v, 4) for k, v in lc.areas_km2.items()},
            "area_km2": round(scene.n_pixels * scene.px_area_m2() / 1e6, 4),
            "has_nir": scene.has_nir,
        }
        ch = sess.get("change")
        if ch is not None:
            cr = ch["result"]
            results["change"] = {
                "changed_fraction": round(cr.changed_fraction, 4),
                "registration": cr.registration,
                "severity": getattr(cr, "severity", "UNKNOWN"),
                "deltas": cr.deltas,
            }
    ev = sess.get("evidence")
    return AnalysisRecord(
        analysis_id=sess.get("analysis_id") or session_id,
        timestamp=time.time(),
        inputs=inputs,
        params={"gsd": sess.get("gsd"), "k": sess.get("k"),
                "adaptive_k": sess.get("adaptive_k", False),
                "seed": 42},
        versions={"product": config.PRODUCT, "version": config.VERSION,
                  "pipeline": config.PIPELINE_VERSION, "build": config.BUILD,
                  **{f"algo_{k}": v for k, v in config.ALGO_VERSIONS.items()}},
        seed=42,
        results=results,
        evidence=ev.to_list() if ev is not None else [],
        queries=sess.get("queries", [])[-50:],
        duration_ms=duration_ms,
        label=label or (inputs[0].get("filename", "scene") if inputs else "scene"),
    )
