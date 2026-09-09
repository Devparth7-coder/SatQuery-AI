"""SatQuery AI v2 — HTML report generator.

Produces a professional, self-contained analysis report (inline styles, no
external assets so it renders offline and prints to PDF cleanly). Every
section tags its content as MEASURED / INFERRED / UNAVAILABLE / LIMITATION.
"""
from __future__ import annotations

import datetime
import html
from typing import Any, Dict, List, Optional

from . import config


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _tag(kind: str) -> str:
    colors = {"MEASURED": "#22d39a", "INFERRED": "#4da3ff",
              "UNAVAILABLE": "#8ea0bd", "LIMITATION": "#ffb020"}
    c = colors.get(kind, "#8ea0bd")
    return (f'<span style="display:inline-block;font-size:10px;font-weight:700;'
            f'letter-spacing:.6px;color:{c};border:1px solid {c}55;'
            f'border-radius:99px;padding:2px 9px;margin-left:8px">{kind}</span>')


def generate_html_report(title: str, session_summary: Dict[str, Any],
                         queries: List[Dict[str, Any]],
                         evidence: List[Dict[str, Any]],
                         images: Dict[str, str],
                         reproducibility: Dict[str, Any],
                         limitations: List[str],
                         methodology: Optional[List[str]] = None) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scene = session_summary.get("scene", {})
    legend = session_summary.get("legend", [])
    counts = session_summary.get("counts", {})
    change = session_summary.get("change")

    def leg_rows() -> str:
        return "".join(
            f"<tr><td><span style='display:inline-block;width:11px;height:11px;"
            f"border-radius:3px;background:{_esc(l.get('color',''))}'></span> "
            f"{_esc(l.get('label',''))}</td>"
            f"<td style='text-align:right'>{float(l.get('fraction',0))*100:.1f}%</td>"
            f"<td style='text-align:right'>{float(l.get('area_km2',0)):.3f} km²</td></tr>"
            for l in legend)

    def ev_rows() -> str:
        rows = []
        for e in evidence[:80]:
            v = e.get("value")
            if isinstance(v, (dict, list)):
                import json as _j
                v = _j.dumps(v)[:300]
            rows.append(
                f"<tr><td><code>{_esc(e.get('evidence_id',''))}</code></td>"
                f"<td>{_esc(e.get('type',''))}</td>"
                f"<td>{_esc(e.get('note') or e.get('source',''))}</td>"
                f"<td><code>{_esc(v)}</code></td>"
                f"<td style='font-size:11px'>{_esc(e.get('method',''))}</td></tr>")
        return "".join(rows) or "<tr><td colspan=5>No evidence recorded.</td></tr>"

    def query_rows() -> str:
        rows = []
        for q in queries[-20:]:
            rows.append(
                f"<div style='margin:10px 0;padding:10px 12px;border:1px solid #1e2a44;"
                f"border-radius:9px'><div style='color:#8ea0bd;font-size:12px'>Q: "
                f"{_esc(q.get('query',''))}</div><div style='font-size:12px;margin-top:4px'>"
                f"intent <code>{_esc(q.get('intent',''))}</code> · "
                f"confidence {q.get('confidence','—')} · {q.get('ms','—')} ms</div></div>")
        return "".join(rows) or "<p>No queries asked in this session.</p>"

    def img_block(key: str, caption: str, tag: str) -> str:
        uri = images.get(key, "")
        if not uri:
            return ""
        return (f"<figure style='margin:14px 0'><img src='{uri}' "
                f"style='max-width:100%;border-radius:10px;border:1px solid #1e2a44'/>"
                f"<figcaption style='color:#8ea0bd;font-size:12px;margin-top:6px'>"
                f"{caption}{_tag(tag)}</figcaption></figure>")

    meth = "".join(f"<li>{_esc(s)}</li>" for s in (methodology or [
        "Ingest + validate raster; record checksum, dimensions, band layout.",
        "Resample to analysis resolution (≤1024 px); stretch 16-bit inputs.",
        "Compute spectral / textural features (NDVI-family or RGB proxies).",
        "MiniBatch k-means segmentation + physics rule labelling.",
        "Classical detectors: connected regions, contrast blobs, Hough lines.",
        "Optional bi-temporal path: ORB/RANSAC registration, histogram match, CVA+Otsu.",
        "Natural-language plan → deterministic tools → grounded answer + overlay.",
    ]))
    lim = "".join(f"<li>{_esc(s)}</li>" for s in limitations) or "<li>None recorded.</li>"
    repro = "".join(f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>"
                    for k, v in reproducibility.items())

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{_esc(title)} — SatQuery AI Report</title>
<style>body{{background:#070b14;color:#e6edf7;font:14px/1.6 system-ui,sans-serif;
margin:0;padding:32px}}main{{max-width:920px;margin:0 auto}}h1{{font-size:26px;margin:0}}
h2{{font-size:16px;letter-spacing:1px;text-transform:uppercase;color:#8ea0bd;
border-bottom:1px solid #1e2a44;padding-bottom:8px;margin-top:34px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:7px 9px;
border-bottom:1px solid #16213a;text-align:left}}code{{background:#0a1120;
padding:1px 6px;border-radius:5px;font-size:12px}}.hdr{{color:#8ea0bd;font-size:12px}}
@media print{{body{{background:#fff;color:#111;padding:0}}h2{{color:#333}}td{{border-color:#ddd}}}}
</style></head><body><main>
<p class="hdr">SATQUERY AI · EARTH OBSERVATION INTELLIGENCE · v{config.VERSION}
(pipeline {config.PIPELINE_VERSION}) · generated {now}</p>
<h1>{_esc(title)}</h1>
<p class="hdr">Scene: {_esc(scene.get('name','—'))} · {scene.get('width','—')}×{scene.get('height','—')} px
· GSD {scene.get('gsd','—')} m · footprint {scene.get('area_km2','—')} km²
· spectral basis: {_esc(scene.get('veg_index','—'))} / {_esc(scene.get('water_index','—'))}</p>

<h2>1 · Methodology {_tag("INFERRED")}</h2><ol>{meth}</ol>
<h2>2 · Land-cover statistics {_tag("MEASURED")}</h2>
<table><tr><th>Class</th><th style="text-align:right">Fraction</th>
<th style="text-align:right">Area</th></tr>{leg_rows()}</table>
<h2>3 · Objects {_tag("MEASURED")}</h2>
<table><tr><th>Detector</th><th style="text-align:right">Count</th></tr>
<tr><td>Water regions</td><td style="text-align:right">{counts.get('water_regions','—')}</td></tr>
<tr><td>Probable vessels</td><td style="text-align:right">{counts.get('vessels','—')}</td></tr>
<tr><td>Bright targets</td><td style="text-align:right">{counts.get('bright_targets','—')}</td></tr>
<tr><td>Linear structures</td><td style="text-align:right">{counts.get('linear','—')}</td></tr></table>
<p class="hdr">Classical contrast/morphology detectors — not trained object detectors.</p>
<h2>4 · Change results {_tag("MEASURED") if change else _tag("UNAVAILABLE")}</h2>
{"<p>Changed fraction: <b>" + f"{float(change.get('changed_fraction',0))*100:.2f}%</b> · "
 + f"severity <b>{_esc(change.get('severity','—'))}</b><br/>Registration: {_esc(change.get('registration','—'))}</p>"
 if change else "<p>No second date was analysed in this session.</p>"}
<h2>5 · Visualizations {_tag("MEASURED")}</h2>
{img_block('original','Original scene (analysis resolution).','MEASURED')}
{img_block('landcover','Land-cover overlay (algorithmically inferred classes).','INFERRED')}
{img_block('veg','Vegetation-index map.','MEASURED')}
{img_block('change','Change map (before/after comparison).','INFERRED')}
<h2>6 · Queries {_tag("MEASURED")}</h2>{query_rows()}
<h2>7 · Evidence trail {_tag("MEASURED")}</h2>
<table><tr><th>ID</th><th>Type</th><th>Item</th><th>Value</th><th>Method</th></tr>{ev_rows()}</table>
<h2>8 · Limitations {_tag("LIMITATION")}</h2><ul>{lim}</ul>
<h2>9 · Reproducibility {_tag("MEASURED")}</h2>
<table>{repro}</table>
<p class="hdr">Reproduce: re-run with the same input checksum + parameters + pipeline
version. Deterministic stages use seed 42.</p>
</main></body></html>"""
