"""SatQuery AI v2 — consistent API response envelopes.

v1 endpoints keep their historical response shapes (backward compatible) and
gain a lightweight ``meta`` block. All NEW endpoints use the full envelope::

    {"success": True, "data": {...}, "evidence": [...],
     "warnings": [...], "errors": [], "meta": {...}}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import config


def base_meta(request_id: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "product": config.PRODUCT,
        "version": config.VERSION,
        "pipeline_version": config.PIPELINE_VERSION,
        "build": config.BUILD,
    }
    if request_id:
        meta["request_id"] = request_id
    meta.update(extra)
    return meta


def success(data: Any = None, evidence: Optional[List[dict]] = None,
            warnings: Optional[List[str]] = None,
            meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "success": True,
        "data": data if data is not None else {},
        "evidence": evidence or [],
        "warnings": warnings or [],
        "errors": [],
        "meta": meta or base_meta(),
    }


def error(message: str, code: str = "error",
          status: int = 400, request_id: Optional[str] = None,
          details: Optional[Any] = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {
        "success": False,
        "data": {},
        "evidence": [],
        "warnings": [],
        "errors": [err],
        "meta": base_meta(request_id, http_status=status),
    }
