"""SatQuery AI v2 — evidence model.

Every answer must be traceable to evidence::

    ANSWER -> EVIDENCE E17 -> LAND-COVER MAP -> PIXEL MASK
           -> FEATURE EXTRACTION -> SOURCE IMAGE

Each :class:`EvidenceItem` carries a stable ID, a type, the source that
produced it, the method used, the value, an optional confidence/quality
note and an optional visualization reference. The UI renders these in an
Evidence Drawer; the report generator and the AI reasoning layer consume
the same objects, so numbers can never silently diverge between surfaces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Evidence kinds. "limitation" items are first-class: the absence of data
# is itself evidence the user must see.
MEASUREMENT = "measurement"
INFERENCE = "inference"
DETECTION = "detection"
CLASSIFICATION = "classification"
REGISTRATION = "registration"
METADATA = "metadata"
LIMITATION = "limitation"


@dataclass
class EvidenceItem:
    evidence_id: str
    type: str
    source: str
    method: str
    value: Any
    unit: str = ""
    confidence: Optional[float] = None   # calibrated/heuristic score or None
    quality: str = ""                    # HIGH/MODERATE/LOW or "" when n/a
    visualization: str = ""              # layer key, e.g. "landcover"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGraph:
    """Ordered, de-duplicated collection of evidence for one analysis."""

    def __init__(self, prefix: str = "E"):
        self._items: List[EvidenceItem] = []
        self._counter = 0
        self._prefix = prefix

    def add(self, type: str, source: str, method: str, value: Any,
            unit: str = "", confidence: Optional[float] = None,
            quality: str = "", visualization: str = "",
            note: str = "") -> EvidenceItem:
        self._counter += 1
        item = EvidenceItem(
            evidence_id=f"{self._prefix}{self._counter}",
            type=type, source=source, method=method, value=value, unit=unit,
            confidence=confidence, quality=quality,
            visualization=visualization, note=note)
        self._items.append(item)
        return item

    def extend_legacy(self, legacy: List[dict], source: str,
                      visualization: str = "") -> List[EvidenceItem]:
        """Upgrade v1 ``{metric, value, method}`` rows into typed items."""
        out = []
        for row in legacy or []:
            out.append(self.add(
                type=MEASUREMENT, source=source,
                method=str(row.get("method", "computed")),
                value=row.get("value"),
                note=str(row.get("metric", "")),
                visualization=visualization))
        return out

    def add_limitation(self, source: str, note: str) -> EvidenceItem:
        return self.add(type=LIMITATION, source=source,
                        method="data-availability check",
                        value="unavailable", note=note)

    def to_list(self) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in self._items]

    def find(self, evidence_id: str) -> Optional[EvidenceItem]:
        for i in self._items:
            if i.evidence_id == evidence_id:
                return i
        return None

    def __len__(self) -> int:
        return len(self._items)
