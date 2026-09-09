"""SatQuery AI v2 — optional AI reasoning layer (grounded, never authoritative).

Architecture::

    User Query -> Query Planner -> Deterministic Tools -> Evidence Object
        -> AI Reasoning / Explanation -> Grounded Answer

The reasoning layer receives ONLY the structured evidence bundle. It may
rephrase and organise, but must never invent measurements, objects, dates,
sensors, coordinates or certainty. Every numeric claim is rendered from the
bundle with its evidence ID cited.

The default provider is fully offline and deterministic (template-based),
so the product works with zero API keys. External VLM adapters implement
:class:`VisionModelProvider` but their output is post-checked against the
evidence bundle before display.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VisionModelProvider(ABC):
    """Provider-agnostic interface for presentational reasoning."""

    name = "base"

    @abstractmethod
    def explain(self, bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Return ``{"text": ..., "citations": [...], "refused": bool}``."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "available": True}


def _num(bundle: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    m = bundle.get("measurements", {})
    for k in keys:
        if k in m:
            return m[k]
    return default


class OfflineTemplateProvider(VisionModelProvider):
    """Deterministic, offline explanation composer. No model calls."""

    name = "offline-template"

    def explain(self, bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
        measurements = bundle.get("measurements", {}) or {}
        evidence = bundle.get("evidence", []) or []
        limitations = bundle.get("limitations", []) or []
        intent = bundle.get("intent", "scene_summary")

        if not measurements and not evidence:
            return {
                "text": ("I don't have enough evidence to answer this reliably. "
                         "Run an analysis first, then ask a question grounded in "
                         "the computed measurements."),
                "citations": [], "refused": True}

        lines: List[str] = []
        citations: List[str] = []
        lines.append(f"**Grounded interpretation** (intent: `{intent}`).")
        lines.append("")
        lines.append(f"Question: _{query}_")
        lines.append("")
        # Render each measurement with its evidence citation.
        for ev in evidence[:14]:
            eid = ev.get("evidence_id", "?")
            metric = ev.get("note") or ev.get("type", "measurement")
            val = ev.get("value")
            method = ev.get("method", "")
            if isinstance(val, (dict, list)):
                import json as _j
                val = _j.dumps(val)[:220]
            conf = ev.get("confidence")
            conf_s = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            lines.append(f"- **{metric}**: `{val}` [{eid}]{conf_s}")
            lines.append(f"  _Method: {method}_")
            citations.append(eid)
        if limitations:
            lines.append("")
            lines.append("**Limitations (from the evidence bundle):**")
            for lim in limitations[:6]:
                lines.append(f"- {lim}")
        lines.append("")
        lines.append("_Every figure above comes from the deterministic analysis "
                     "pipeline; no values were generated or estimated by a "
                     "language model._")
        return {"text": "\n".join(lines), "citations": citations, "refused": False}


class OllamaProviderStub(VisionModelProvider):
    """Extension point for a local Ollama/Qwen-VL/LLaVA adapter.

    Ships unconfigured on purpose: wiring a real endpoint without user
    consent or credentials would be dishonest. Configure via environment
    (SATQUERY_OLLAMA_URL + SATQUERY_OLLAMA_MODEL) and implement the call.
    """

    name = "ollama"

    def describe(self) -> Dict[str, Any]:
        import os
        url = os.environ.get("SATQUERY_OLLAMA_URL", "")
        model = os.environ.get("SATQUERY_OLLAMA_MODEL", "")
        return {"name": self.name, "available": bool(url and model),
                "url_set": bool(url), "model": model or "unconfigured"}

    def explain(self, bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
        d = self.describe()
        if not d["available"]:
            return {"text": ("No VLM is configured (set SATQUERY_OLLAMA_URL and "
                             "SATQUERY_OLLAMA_MODEL to enable the local-model "
                             "adapter). Falling back to the offline grounded "
                             "composer is recommended."),
                    "citations": [], "refused": True}
        raise NotImplementedError("Ollama adapter call not implemented in v2.0")


_PROVIDERS: Dict[str, VisionModelProvider] = {
    "offline-template": OfflineTemplateProvider(),
    "ollama": OllamaProviderStub(),
}


def get_provider(name: Optional[str] = None) -> VisionModelProvider:
    if name and name in _PROVIDERS:
        return _PROVIDERS[name]
    return _PROVIDERS["offline-template"]


def list_providers() -> List[Dict[str, Any]]:
    return [p.describe() for p in _PROVIDERS.values()]


def build_bundle(intent: str, measurements: Dict[str, Any],
                 detections: List[dict], changes: List[dict],
                 limitations: List[str], evidence: List[dict],
                 scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the ONLY context the reasoning layer is allowed to see."""
    return {"intent": intent, "scene": scene or {},
            "measurements": measurements, "detections": detections,
            "changes": changes, "limitations": limitations,
            "evidence": evidence}
