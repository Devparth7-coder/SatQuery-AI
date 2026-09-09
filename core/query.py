"""
SatQuery AI - Grounded vision-language query engine.

A retrieval-and-reasoning layer, not a text generator hallucinating over an
image. Pipeline:

  text query -> intent + entity parsing (rule/lexicon based NLU)
             -> evidence retrieval from the analysed scene graph
             -> deterministic natural-language realisation
             -> visual grounding (mask / boxes) + numeric citations

Every sentence returned is backed by a number computed from the pixels, and the
`evidence` block exposes exactly which measurement produced it. This is what
makes the assistant auditable - critical for an operational ISRO context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .landcover import CLASSES, CLASS_LABELS, LandCover, class_mask

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
ENTITY_LEXICON: Dict[str, List[str]] = {
    "water": ["water", "lake", "lakes", "river", "rivers", "sea", "ocean", "pond",
              "ponds", "reservoir", "reservoirs", "waterbody", "water body",
              "water bodies", "coast", "coastline", "canal", "flood", "flooded",
              "flooding", "inundation", "wetland", "estuary", "harbour", "harbor",
              "lagoon", "dam", "creek", "stream"],
    "dense_vegetation": ["forest", "forests", "tree", "trees", "jungle", "canopy",
                         "dense vegetation", "woodland", "green cover", "greenery",
                         "mangrove", "mangroves", "plantation", "foliage",
                         "vegetation", "vegetated", "vegetation cover"],
    "sparse_vegetation": ["crop", "crops", "cropland", "farm", "farmland", "field",
                          "fields", "agriculture", "agricultural", "grass",
                          "grassland", "pasture", "sparse vegetation", "shrub",
                          "harvest", "paddy"],
    "built_up": ["building", "buildings", "urban", "city", "cities", "town",
                 "settlement", "settlements", "built up", "built-up", "builtup",
                 "houses", "housing", "rooftop", "rooftops", "infrastructure",
                 "construction", "industrial", "impervious", "concrete",
                 "development", "sprawl", "slum", "airport", "port"],
    "bare_soil": ["bare", "barren", "soil", "sand", "desert", "dune", "dunes",
                  "rock", "rocky", "quarry", "mine", "mining", "exposed", "dry land",
                  "wasteland", "fallow"],
    "cloud_snow": ["cloud", "clouds", "cloudy", "cloud cover", "snow", "ice",
                   "glacier", "snowcap", "overcast", "haze"],
    "shadow": ["shadow", "shadows", "shaded", "dark area"],
}

OBJECT_LEXICON: Dict[str, List[str]] = {
    "vessel": ["ship", "ships", "boat", "boats", "vessel", "vessels", "barge",
               "tanker", "tankers", "fleet", "trawler", "maritime traffic"],
    "linear": ["road", "roads", "highway", "highways", "runway", "runways",
               "street", "streets", "railway", "rail", "track", "tracks", "canal",
               "route", "corridor", "bridge", "linear"],
    "bright_target": ["vehicle", "vehicles", "car", "cars", "truck", "trucks",
                      "aircraft", "plane", "planes", "object", "objects", "target",
                      "targets", "anomaly", "anomalies", "bright spot", "tank",
                      "tanks", "silo"],
}

INTENTS = [
    ("change",   ["change", "changed", "changes", "difference", "differences",
                  "compare", "comparison", "before and after", "bi-temporal",
                  "bitemporal", "since", "grown", "growth", "shrunk", "shrink",
                  "loss", "lost", "gain", "gained", "expansion", "expanded",
                  "deforestation", "encroachment", "temporal", "over time",
                  "increase", "decrease", "reduced", "converted"]),
    ("count",    ["how many", "count", "number of", "quantity", "tally",
                  "enumerate", "detect all", "no. of"]),
    ("area",     ["area", "how much", "coverage", "extent", "km2", "km²",
                  "square", "hectare", "hectares", "percentage", "percent",
                  "fraction", "proportion", "how large", "size of", "acreage"]),
    ("locate",   ["where", "locate", "location", "find", "show me", "highlight",
                  "point out", "mark", "identify where", "which part"]),
    ("presence", ["is there", "are there", "does it contain", "any ", "presence",
                  "can you see", "do you see", "contains", "visible"]),
    ("describe", ["describe", "description", "what is", "what's", "summary",
                  "summarise", "summarize", "overview", "tell me about",
                  "explain", "caption", "what do you see", "analyse", "analyze",
                  "report", "assess", "characterise"]),
]

SUPERLATIVE = ["largest", "biggest", "greatest", "widest", "longest", "maximum",
               "dominant", "most", "primary", "main", "smallest", "least"]


@dataclass
class QueryPlan:
    intent: str
    entities: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    superlative: bool = False
    raw: str = ""
    # v2 structured-planner fields (populated by parse_query)
    target: str = "scene"
    operation: str = "describe"
    required_evidence: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)


def parse_query(q: str) -> QueryPlan:
    ql = " " + q.lower().strip() + " "
    ql = re.sub(r"[^\w\s%²-]", " ", ql)
    ql = re.sub(r"\s+", " ", ql)

    entities, objects = [], []
    for cls, words in ENTITY_LEXICON.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", ql) for w in words):
            entities.append(cls)
    for obj, words in OBJECT_LEXICON.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", ql) for w in words):
            objects.append(obj)

    # Intent salience: a change/count cue anywhere in the sentence outranks a
    # generic 'area'/'describe' cue that merely happens to appear earlier.
    # ("Has the built-up AREA EXPANDED?" is a change question, not an area one.)
    PRIORITY = {"change": 100, "count": 90, "locate": 60, "area": 50,
                "presence": 40, "describe": 10}
    intent, best = None, (-1, 10 ** 9)
    for name, keys in INTENTS:
        for kw in keys:
            p = ql.find(kw)
            if p >= 0:
                cand = (PRIORITY[name], -p)          # higher priority, then earlier
                if cand > best:
                    best, intent = cand, name
    if intent is None:
        intent = "describe" if not (entities or objects) else "presence"

    # Hard overrides for unambiguous surface forms
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", ql):
        intent = "count"
    if re.search(r"\bwhere\b|\bshow me\b|\bhighlight\b", ql) and intent != "change":
        intent = "locate"

    # ---- v2 taxonomy mapping + structured plan enrichment ----
    intent = _to_v2_intent(intent, ql, entities, objects)
    target = entities[0] if entities else (objects[0] if objects else "scene")
    operation = _OPERATION_FOR.get(intent, "describe")
    required_evidence = list(_EVIDENCE_FOR.get(intent, ["land_cover"]))
    tools = list(_TOOLS_FOR.get(intent, ["landcover.segment"]))

    return QueryPlan(intent=intent, entities=entities, objects=objects,
                     superlative=any(s in ql for s in SUPERLATIVE), raw=q,
                     target=target, operation=operation,
                     required_evidence=required_evidence, tools=tools)


# --------------------------------------------------------------------------- #
# Realisation helpers
# --------------------------------------------------------------------------- #
def _fmt_area(km2: float) -> str:
    if km2 >= 1.0:
        return f"{km2:.2f} km²"
    if km2 >= 0.01:
        return f"{km2*100:.1f} ha"
    return f"{km2*1e6:,.0f} m²"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _rank_classes(lc: LandCover) -> List[tuple]:
    return sorted(lc.fractions.items(), key=lambda kv: -kv[1])


def _confidence_for(analysis, classes: List[str]) -> float:
    lc = analysis["landcover"]
    rep = [r for r in lc.cluster_report if r["assigned"] in classes]
    if not rep:
        return 0.55
    tot = sum(r["fraction"] for r in rep) or 1e-6
    conf = sum(r["confidence"] * r["fraction"] for r in rep) / tot
    if not analysis["scene"].has_nir:
        conf *= 0.90   # RGB proxies are weaker than true NIR indices
    return float(np.clip(conf, 0.3, 0.97))


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #
def answer(plan: QueryPlan, analysis: dict, change: Optional[dict] = None) -> dict:
    scene = analysis["scene"]
    lc: LandCover = analysis["landcover"]
    fs = analysis["features"]
    objs = analysis["objects"]

    total_km2 = scene.n_pixels * scene.px_area_m2() / 1e6
    ev: List[dict] = []
    overlay = {"type": "none"}
    text = ""
    conf = 0.6

    def ev_add(name, value, method):
        ev.append({"metric": name, "value": value, "method": method})

    # v2: the planner emits the v2 intent taxonomy; accept v1 names too.
    plan.intent = _ALIASES.get(plan.intent, plan.intent)

    # ---------------- CHANGE ----------------
    if plan.intent == "change":
        if not change:
            return {"answer": "I need a **second image** of the same area to answer a "
                              "change-detection question. Upload a 'Time 2' scene in the "
                              "left panel and re-ask — I'll co-register the pair and run "
                              "change vector analysis.",
                    "intent": "change", "evidence": [], "overlay": {"type": "none"},
                    "confidence": 0.0, "needs": "second_image"}
        cr = change["result"]
        lines = [f"**Change detected across {_pct(cr.changed_fraction)} of the scene** "
                 f"(~{_fmt_area(cr.changed_fraction * total_km2)} of {_fmt_area(total_km2)})."]
        ev_add("changed_area_fraction", round(cr.changed_fraction, 4),
               "Change Vector Analysis over 6 normalised bands + Otsu threshold")
        ev_add("co-registration", cr.registration, "ORB/RANSAC affine or ECC alignment")

        focus = plan.entities
        if focus:
            for c in focus:
                d = cr.deltas.get(c, 0.0)
                verb = "increased" if d > 0 else "decreased"
                arrow = "▲" if d > 0 else "▼"
                lines.append(f"{arrow} **{CLASS_LABELS[c]}** {verb} by "
                             f"**{_fmt_area(abs(d))}** "
                             f"({cr.lc1.areas_km2.get(c,0):.2f} → {cr.lc2.areas_km2.get(c,0):.2f} km²).")
                ev_add(f"delta_{c}_km2", d, "per-class area difference T2 − T1")
        else:
            top = sorted(cr.deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
            for c, d in top:
                if abs(d) < 1e-4:
                    continue
                arrow = "▲" if d > 0 else "▼"
                lines.append(f"{arrow} **{CLASS_LABELS[c]}**: {d:+.2f} km²")
                ev_add(f"delta_{c}_km2", d, "per-class area difference T2 − T1")

        if cr.transitions:
            lines.append("\n**Dominant transitions:**")
            for t in cr.transitions[:4]:
                if focus and t["from"] not in focus and t["to"] not in focus:
                    continue
                lines.append(f"- {t['from_label']} → {t['to_label']}: "
                             f"{_fmt_area(t['area_km2'])}")
            ev_add("top_transitions", cr.transitions[:4],
                   "class-transition matrix masked by the change mask")

        # Domain flags
        d_veg = cr.deltas.get("dense_vegetation", 0) + cr.deltas.get("sparse_vegetation", 0)
        d_built = cr.deltas.get("built_up", 0)
        d_water = cr.deltas.get("water", 0)
        flags = []
        if d_veg < -0.02 * total_km2:
            flags.append("⚠️ Significant **vegetation loss** — consistent with deforestation, "
                         "harvest or land clearing.")
        if d_built > 0.02 * total_km2:
            flags.append("⚠️ Notable **built-up expansion** — urban growth or new construction.")
        if d_water > 0.03 * total_km2:
            flags.append("⚠️ **Water extent increase** — possible inundation / flooding or "
                         "reservoir filling.")
        if d_water < -0.03 * total_km2:
            flags.append("⚠️ **Water extent decrease** — possible drought or reservoir drawdown.")
        if flags:
            lines.append("\n" + "\n".join(flags))

        try:
            from .change import severity_reason as _sev_reason
            _reg = ((getattr(cr, "quality", None) or {}).get("registration", {})
                    if hasattr(cr, "quality") else {})
            lines.append(f"\n**Change severity: {getattr(cr, 'severity', 'UNKNOWN')}**"
                         f" — {_sev_reason(cr.changed_fraction, _reg)}")
        except Exception:
            pass
        text = "\n".join(lines)
        overlay = {"type": "change"}
        conf = 0.78 if scene.has_nir else 0.71

    # ---------------- COUNT ----------------
    elif plan.intent == "count":
        if "vessel" in plan.objects:
            v = objs["vessels"]
            text = (f"**{len(v)} probable vessel(s)** detected on water.\n\n"
                    if v else
                    "**No vessels detected.** No small high-contrast targets were found "
                    "within water-classified regions.\n\n")
            if v:
                lens = [d.extra["length_m"] for d in v]
                text += (f"Estimated lengths span {min(lens):.0f}–{max(lens):.0f} m "
                         f"(median {np.median(lens):.0f} m) at {scene.gsd:g} m GSD. "
                         f"Detection is by local-contrast blob extraction constrained to "
                         f"the water mask, so low-contrast or moored craft may be missed.")
                ev_add("vessel_count", len(v), "top-hat contrast blobs ∩ dilated water mask")
                ev_add("length_range_m", [round(min(lens), 1), round(max(lens), 1)],
                       "bbox major axis × GSD")
            overlay = {"type": "detections", "kinds": ["vessel"]}
            conf = 0.62

        elif "linear" in plan.objects:
            L = objs["linear"]
            tot = sum(d.extra["length_m"] for d in L) / 1000.0
            text = (f"**{len(L)} linear structures** extracted (roads / runways / canals), "
                    f"cumulative traced length ≈ **{tot:.2f} km**.\n\n")
            if L:
                angs = [d.extra["orientation_deg"] for d in L]
                hist = np.histogram(angs, bins=6, range=(0, 180))[0]
                dom = int(np.argmax(hist)) * 30
                text += (f"The dominant orientation is ≈{dom}°–{dom+30}°, and the longest "
                         f"single segment measures {max(d.extra['length_m'] for d in L):.0f} m. "
                         f"A strongly peaked orientation histogram usually indicates a planned "
                         f"grid layout or a runway/major corridor.")
                ev_add("segment_count", len(L), "probabilistic Hough transform on Canny edges")
                ev_add("total_length_km", round(tot, 2), "Σ segment length × GSD")
            overlay = {"type": "detections", "kinds": ["linear"]}
            conf = 0.6

        elif "bright_target" in plan.objects:
            t = objs["bright_targets"]
            text = (f"**{len(t)} compact bright targets** detected outside water "
                    f"(vehicles, small structures, metal roofs or equipment).\n\n"
                    f"These are local-contrast anomalies against a median background; "
                    f"at {scene.gsd:g} m GSD each spans roughly "
                    f"{np.median([d.extra['length_m'] for d in t]):.0f} m if targets exist.")
            ev_add("target_count", len(t), "median-background top-hat, 99.3rd percentile")
            overlay = {"type": "detections", "kinds": ["bright_target"]}
            conf = 0.55

        else:
            cls = plan.entities[0] if plan.entities else "water"
            key = {"water": "water_regions", "dense_vegetation": "veg_regions",
                   "built_up": "built_regions"}.get(cls)
            if key:
                regs = objs[key]
                text = (f"**{len(regs)} distinct {CLASS_LABELS[cls].lower()} region(s)** "
                        f"found (8-connected components ≥ threshold).\n\n")
                if regs:
                    text += (f"The largest covers **{_fmt_area(regs[0].area_m2/1e6)}**; "
                             f"total across all regions is "
                             f"**{_fmt_area(sum(r.area_m2 for r in regs)/1e6)}**.")
                    ev_add("region_count", len(regs), "connected-component labelling")
                    ev_add("largest_region_km2", round(regs[0].area_m2 / 1e6, 4),
                           "component pixel count × pixel ground area")
                overlay = {"type": "class", "classes": [cls]}
                conf = _confidence_for(analysis, [cls])
            else:
                frac = lc.fractions.get(cls, 0)
                text = (f"**{CLASS_LABELS[cls]}** is not counted as discrete objects, but it "
                        f"covers **{_pct(frac)}** of the scene "
                        f"(**{_fmt_area(lc.areas_km2.get(cls,0))}**).")
                overlay = {"type": "class", "classes": [cls]}
                conf = _confidence_for(analysis, [cls])

    # ---------------- AREA ----------------
    elif plan.intent == "area":
        targets = plan.entities or [ _rank_classes(lc)[0][0] ]
        lines = []
        for c in targets:
            f = lc.fractions.get(c, 0.0)
            a = lc.areas_km2.get(c, 0.0)
            lines.append(f"- **{CLASS_LABELS[c]}** — {_pct(f)} of the scene, "
                         f"**{_fmt_area(a)}** ({a*100:.1f} ha)")
            ev_add(f"{c}_fraction", round(f, 4),
                   f"pixels labelled {c} ÷ total pixels")
            ev_add(f"{c}_area_km2", round(a, 4),
                   f"pixel count × ({scene.gsd:g} m)² ground sampling")
        head = (f"Scene footprint ≈ **{_fmt_area(total_km2)}** "
                f"({scene.orig_shape[1]}×{scene.orig_shape[0]} px @ {scene.gsd:g} m GSD).\n\n")
        text = head + "\n".join(lines)
        idx = analysis["features"].veg_index_name
        if any(c.endswith("vegetation") for c in targets):
            mask = class_mask(lc, [c for c in targets if c.endswith("vegetation")])
            if mask.any():
                text += (f"\n\nMean {idx} inside the vegetated mask is "
                         f"**{float(fs.veg_index[mask].mean()):+.3f}** "
                         f"(scene mean {float(fs.veg_index.mean()):+.3f}) — "
                         f"higher values indicate greater canopy vigour.")
                ev_add(f"mean_{idx}_in_mask", round(float(fs.veg_index[mask].mean()), 4),
                       "mean vegetation index within classified mask")
        overlay = {"type": "class", "classes": targets}
        conf = _confidence_for(analysis, targets)

    # ---------------- LOCATE ----------------
    elif plan.intent == "locate":
        if plan.objects:
            kind = plan.objects[0]
            pool = {"vessel": objs["vessels"], "linear": objs["linear"],
                    "bright_target": objs["bright_targets"]}[kind]
            if not pool:
                text = f"No **{kind.replace('_',' ')}** instances were localised in this scene."
                conf = 0.5
            else:
                text = (f"Localised **{len(pool)}** {kind.replace('_',' ')} instance(s). "
                        f"Top hits by confidence:\n\n")
                for d in pool[:5]:
                    x, y, w, h = d.bbox
                    text += (f"- pixel ({int(d.centroid[0])}, {int(d.centroid[1])}) "
                             f"{_quadrant(d.centroid, scene)} · "
                             f"{d.extra.get('length_m', 0):.0f} m · score {d.score:.2f}\n")
                ev_add("instances", len(pool), "detector output with pixel centroids")
                conf = 0.6
            overlay = {"type": "detections", "kinds": [kind]}
        else:
            targets = plan.entities or [_rank_classes(lc)[0][0]]
            c = targets[0]
            m = class_mask(lc, [c])
            if not m.any():
                text = (f"**{CLASS_LABELS[c]}** was not detected anywhere in this scene "
                        f"(0 pixels classified).")
                conf = 0.5
            else:
                ys, xs = np.nonzero(m)
                cx, cy = xs.mean(), ys.mean()
                text = (f"**{CLASS_LABELS[c]}** occupies {_pct(lc.fractions[c])} of the scene. "
                        f"Its centre of mass sits at pixel ({cx:.0f}, {cy:.0f}), "
                        f"in the **{_quadrant((cx, cy), scene)}** of the image, spanning "
                        f"rows {ys.min()}–{ys.max()} and columns {xs.min()}–{xs.max()}.\n\n"
                        f"The highlighted overlay marks every pixel assigned to this class.")
                key = {"water": "water_regions", "dense_vegetation": "veg_regions",
                       "built_up": "built_regions"}.get(c)
                if key and objs[key]:
                    big = objs[key][0]
                    text += (f" The single largest patch is centred at "
                             f"({big.centroid[0]:.0f}, {big.centroid[1]:.0f}) covering "
                             f"{_fmt_area(big.area_m2/1e6)}.")
                ev_add("centroid_px", [round(cx, 1), round(cy, 1)],
                       "first moment of the class mask")
                ev_add("bbox_px", [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       "extent of class mask")
                conf = _confidence_for(analysis, [c])
            overlay = {"type": "class", "classes": targets}

    # ---------------- PRESENCE ----------------
    elif plan.intent == "presence":
        if plan.objects:
            kind = plan.objects[0]
            pool = {"vessel": objs["vessels"], "linear": objs["linear"],
                    "bright_target": objs["bright_targets"]}[kind]
            yes = len(pool) > 0
            text = (f"**{'Yes' if yes else 'No'}** — "
                    + (f"{len(pool)} {kind.replace('_',' ')} candidate(s) detected."
                       if yes else
                       f"no {kind.replace('_',' ')} signatures found in this scene."))
            overlay = {"type": "detections", "kinds": [kind]}
            conf = 0.6
        else:
            targets = plan.entities or [_rank_classes(lc)[0][0]]
            parts = []
            for c in targets:
                f = lc.fractions.get(c, 0.0)
                if f < 0.005:
                    parts.append(f"**No** significant {CLASS_LABELS[c].lower()} "
                                 f"(only {_pct(f)} of pixels).")
                else:
                    parts.append(f"**Yes** — {CLASS_LABELS[c].lower()} is present, covering "
                                 f"{_pct(f)} ({_fmt_area(lc.areas_km2[c])}).")
                ev_add(f"{c}_fraction", round(f, 4), "class mask fraction")
            text = " ".join(parts)
            overlay = {"type": "class", "classes": targets}
            conf = _confidence_for(analysis, targets)

    # ---------------- SPECTRAL INDEX ----------------
    elif plan.intent == "spectral_index":
        from .features import compute_all_indices
        indices = compute_all_indices(scene, fs)
        avail = [v for v in indices.values() if v.available]
        unav = [v for v in indices.values() if not v.available]
        lines = [f"**Spectral indices** (basis: {fs.veg_index_name} + "
                 f"{fs.water_index_name}).", ""]
        for v in avail:
            s = v.stats or {}
            lines.append(f"- **{v.name}** `{v.formula}` — mean {s.get('mean', '—')}, "
                         f"median {s.get('median', '—')}, range "
                         f"[{s.get('min', '—')}, {s.get('max', '—')}], "
                         f"std {s.get('std', '—')}")
            ev_add(f"{v.name.lower()}_mean", s.get("mean"),
                   f"{v.formula} over all analysis pixels")
        if unav:
            lines.append("")
            lines.append("**Unavailable (required bands missing — "
                         "no substitution made):**")
            for v in unav:
                lines.append(f"- {v.name}: {v.reason}")
        text = "\n".join(lines)
        overlay = {"type": "landcover"}
        conf = 0.85 if scene.has_nir else 0.70

    # ---------------- SPATIAL STATISTICS ----------------
    elif plan.intent == "spatial_statistics":
        from .features import spatial_statistics
        st = spatial_statistics(fs)
        lines = ["**Spatial statistics (measured over the analysis raster):**", ""]
        for key in ("brightness", "saturation", "texture", "edge_density"):
            s = st.get(key, {})
            lines.append(f"- {key}: mean {s.get('mean', '—')}, "
                         f"std {s.get('std', '—')}, range "
                         f"[{s.get('min', '—')}, {s.get('max', '—')}]")
            ev_add(f"{key}_mean", s.get("mean"), "scene-wide pixel statistics")
        interp = st.get("interpretation", {})
        lines.append("")
        lines.append(f"Contrast: **{interp.get('contrast', '—')}**. "
                     f"{interp.get('note', '')}")
        text = "\n".join(lines)
        overlay = {"type": "landcover"}
        conf = 0.80

    # ---------------- LAND COVER ----------------
    elif plan.intent == "land_cover":
        ranked = _rank_classes(lc)
        lines = ["**Land-cover composition** (algorithmically inferred — "
                 "not ground-truth validated):", ""]
        for c, f in ranked:
            if f <= 0.0005:
                continue
            lines.append(f"- **{CLASS_LABELS[c]}**: {_pct(f)} · "
                         f"{_fmt_area(lc.areas_km2[c])}")
            ev_add(f"{c}_fraction", round(f, 4),
                   "k-means cluster → rule-based class assignment")
        q = getattr(lc, "quality", None) or {}
        if q:
            lines.append("")
            lines.append(f"Classification quality: **{q.get('label', '—')}** "
                         f"(mean confidence {q.get('mean_confidence', '—')}).")
            for r in q.get("rules", []):
                lines.append(f"  - {r}")
        text = "\n".join(lines)
        overlay = {"type": "landcover"}
        conf = float(q.get("mean_confidence", 0.6)) if q else 0.6

    # ---------------- COMPARISON ----------------
    elif plan.intent == "comparison":
        if not change:
            return {"answer": "I need a **second image** of the same area for a "
                              "side-by-side comparison. Load a 'Time 2' scene and "
                              "re-ask.",
                    "intent": "comparison", "evidence": [], "overlay": {"type": "none"},
                    "confidence": 0.0, "needs": "second_image"}
        cr = change["result"]
        lines = ["**Side-by-side class comparison (T1 → T2**, same classifier "
                 "on both dates):", ""]
        for c in CLASSES:
            a1 = cr.lc1.areas_km2.get(c, 0)
            a2 = cr.lc2.areas_km2.get(c, 0)
            if abs(a2 - a1) < 1e-4 and a1 < 1e-4:
                continue
            lines.append(f"- {CLASS_LABELS[c]}: {a1:.2f} → {a2:.2f} km² "
                         f"({(a2-a1):+.2f})")
            ev_add(f"compare_{c}", [round(a1, 4), round(a2, 4)],
                   "shared-classifier labelling of both dates")
        text = "\n".join(lines)
        overlay = {"type": "change"}
        conf = 0.72

    # ---------------- EVIDENCE EXPLANATION ----------------
    elif plan.intent == "evidence_explanation":
        text = ("**How answers are produced here:**\n\n"
                "1. Your question is parsed into intent + entities by the "
                "structured query planner.\n"
                "2. Only the required deterministic tools run (segmentation, "
                "detectors, indices, change).\n"
                "3. Every number is measured from pixels; each method is listed "
                "under Evidence with an ID.\n"
                "4. The overlay image marks exactly which pixels back the answer.\n\n"
                f"Current scene basis: {fs.veg_index_name} + {fs.water_index_name}"
                + (" with true NIR band." if scene.has_nir
                   else " (no NIR band — RGB proxies).")
                + "\n\nAsk a measurement question, then expand **Evidence** on "
                  "the reply (or open the Evidence panel) to audit it.")
        overlay = {"type": "landcover"}
        conf = 0.90

    # ---------------- DESCRIBE ----------------
    else:
        ranked = [(c, f) for c, f in _rank_classes(lc) if f > 0.012]
        dom = ranked[0] if ranked else ("bare_soil", 0)
        scene_type = _scene_type(lc, objs)
        parts = [f"**Scene interpretation — {scene_type}**\n",
                 f"Footprint ≈ {_fmt_area(total_km2)} at {scene.gsd:g} m GSD "
                 f"({scene.orig_shape[1]}×{scene.orig_shape[0]} px). "
                 f"Spectral basis: {fs.veg_index_name} + {fs.water_index_name}"
                 + (" with true NIR band." if scene.has_nir else " (no NIR band supplied).") + "\n",
                 "**Land-cover composition**"]
        for c, f in ranked:
            parts.append(f"- {CLASS_LABELS[c]}: {_pct(f)} · {_fmt_area(lc.areas_km2[c])}")
            ev_add(f"{c}_fraction", round(f, 4), "k-means cluster → rule-based class assignment")

        obs = []
        if objs["vessels"]:
            obs.append(f"{len(objs['vessels'])} probable vessel(s) on water")
        if objs["linear"]:
            tot = sum(d.extra["length_m"] for d in objs["linear"]) / 1000
            obs.append(f"{len(objs['linear'])} linear structures (~{tot:.1f} km traced)")
        if objs["water_regions"]:
            obs.append(f"{len(objs['water_regions'])} discrete water bodies")
        if objs["bright_targets"]:
            obs.append(f"{len(objs['bright_targets'])} compact bright targets")
        if obs:
            parts.append("\n**Structures detected:** " + "; ".join(obs) + ".")

        parts.append(f"\n**Reading:** the scene is dominated by "
                     f"{CLASS_LABELS[dom[0]].lower()} ({_pct(dom[1])}). " + _narrative(lc, objs, fs))
        # Honest reporting of known physical limits of the current input.
        caveats = []
        if not scene.has_nir:
            caveats.append("no NIR band supplied, so vegetation/water rely on RGB "
                           "proxies — **turbid or sediment-laden water can be "
                           "confused with bare soil**, since both are red-dominant "
                           "in visible light. Supply a 4-band product for true "
                           "NDVI/NDWI separation.")
        if lc.fractions.get("shadow", 0) > 0.15:
            caveats.append("large shadow fraction may mask true cover beneath.")
        if caveats:
            parts.append("\n**Limitations:** " + " ".join(caveats))

        if lc.fractions.get("cloud_snow", 0) > 0.12:
            parts.append(f"\n⚠️ **{_pct(lc.fractions['cloud_snow'])} cloud/snow cover** — "
                         f"obscured pixels reduce reliability of the other class estimates.")
        text = "\n".join(parts)
        overlay = {"type": "landcover"}
        conf = float(np.mean([r["confidence"] for r in lc.cluster_report])) if lc.cluster_report else 0.6
        if not scene.has_nir:
            conf *= 0.92

    return {"answer": text, "intent": plan.intent, "entities": plan.entities,
            "objects": plan.objects, "evidence": ev, "overlay": overlay,
            "confidence": round(float(conf), 3)}


def _quadrant(pt, scene) -> str:
    x, y = pt
    v = "north" if y < scene.height / 3 else ("south" if y > 2 * scene.height / 3 else "central")
    h = "west" if x < scene.width / 3 else ("east" if x > 2 * scene.width / 3 else "")
    if v == "central" and not h:
        return "centre"
    return (v + ("-" + h if h else "")).replace("central-", "")


def _scene_type(lc: LandCover, objs: dict) -> str:
    f = lc.fractions
    water, built = f.get("water", 0), f.get("built_up", 0)
    veg = f.get("dense_vegetation", 0) + f.get("sparse_vegetation", 0)
    bare, cloud = f.get("bare_soil", 0), f.get("cloud_snow", 0)
    if cloud > 0.45:
        return "heavily clouded / cryospheric scene"
    if water > 0.35 and built > 0.10:
        return "coastal or riverine urban area"
    if water > 0.40:
        return "predominantly aquatic scene (open water / large water body)"
    if built > 0.30:
        return "dense urban / built-up area"
    if veg > 0.50:
        return "vegetated landscape (forest or intensive agriculture)"
    if bare > 0.45:
        return "arid / barren terrain"
    if built > 0.12 and veg > 0.25:
        return "peri-urban mixed-use landscape"
    return "mixed-use terrain"


def _narrative(lc: LandCover, objs: dict, fs) -> str:
    f = lc.fractions
    bits = []
    veg = f.get("dense_vegetation", 0) + f.get("sparse_vegetation", 0)
    if veg > 0.25:
        vigour = "high" if f.get("dense_vegetation", 0) > f.get("sparse_vegetation", 0) else "moderate"
        bits.append(f"Vegetation vigour is {vigour} based on the {fs.veg_index_name} distribution.")
    if f.get("built_up", 0) > 0.10:
        d = objs.get("urban_edge_density", 0)
        dens = "high-density" if d > 0.14 else ("moderate-density" if d > 0.07 else "low-density")
        bits.append(f"Built-up texture indicates {dens} development.")
    if f.get("water", 0) > 0.05:
        n = len(objs.get("water_regions", []))
        bits.append(f"Water is distributed across {n} mapped {'body' if n == 1 else 'bodies'}.")
    if f.get("shadow", 0) > 0.10:
        bits.append("Extensive shadowing suggests low sun elevation or tall relief/structures.")
    return " ".join(bits) or "No further dominant structural signal."


# --------------------------------------------------------------------------- #
# v2: structured planner taxonomy (additive)
# --------------------------------------------------------------------------- #
#: v1 intent name -> v2 canonical name (answer() accepts both).
_ALIASES = {
    "change": "change", "temporal_change": "change",
    "comparison": "comparison",
    "count": "count", "object_count": "count",
    "area": "area", "area_estimation": "area",
    "locate": "locate", "object_location": "locate",
    "presence": "presence",
    "describe": "describe", "scene_summary": "describe",
    "land_cover": "land_cover", "spectral_index": "spectral_index",
    "spatial_statistics": "spatial_statistics",
    "evidence_explanation": "evidence_explanation",
}

_OPERATION_FOR = {
    "scene_summary": "summarize_scene", "land_cover": "classify",
    "spectral_index": "compute_indices", "object_count": "count",
    "object_location": "localize", "temporal_change": "area_delta",
    "comparison": "side_by_side", "area_estimation": "measure_area",
    "spatial_statistics": "compute_stats",
    "evidence_explanation": "trace_evidence", "presence": "test_presence",
    "change": "area_delta", "count": "count", "area": "measure_area",
    "locate": "localize", "describe": "summarize_scene",
}

_EVIDENCE_FOR = {
    "scene_summary": ["land_cover", "objects", "indices"],
    "land_cover": ["land_cover", "cluster_signatures", "uncertainty"],
    "spectral_index": ["indices", "band_availability"],
    "object_count": ["detections", "detector_method"],
    "object_location": ["detections", "class_mask"],
    "temporal_change": ["land_cover_A", "land_cover_B", "registration_quality",
                        "pixel_area", "change_mask"],
    "comparison": ["land_cover_A", "land_cover_B", "pixel_area"],
    "area_estimation": ["class_mask", "pixel_area"],
    "spatial_statistics": ["texture", "edges", "brightness"],
    "evidence_explanation": ["pipeline_trace"],
    "presence": ["class_mask", "detections"],
    "change": ["land_cover_A", "land_cover_B", "registration_quality",
               "pixel_area", "change_mask"],
    "count": ["detections", "detector_method"],
    "area": ["class_mask", "pixel_area"],
    "locate": ["detections", "class_mask"],
    "describe": ["land_cover", "objects", "indices"],
}

_TOOLS_FOR = {
    "scene_summary": ["landcover.segment", "objects.summarize", "features.compute"],
    "land_cover": ["landcover.segment", "landcover.quality"],
    "spectral_index": ["features.compute_all_indices"],
    "object_count": ["objects.summarize"],
    "object_location": ["objects.summarize", "landcover.class_mask"],
    "temporal_change": ["change.detect", "change.severity"],
    "comparison": ["change.detect(shared classifier)"],
    "area_estimation": ["landcover.class_mask", "scene.px_area"],
    "spatial_statistics": ["features.spatial_statistics"],
    "evidence_explanation": ["evidence.graph"],
    "presence": ["landcover.class_mask", "objects.summarize"],
    "change": ["change.detect", "change.severity"],
    "count": ["objects.summarize"],
    "area": ["landcover.class_mask", "scene.px_area"],
    "locate": ["objects.summarize", "landcover.class_mask"],
    "describe": ["landcover.segment", "objects.summarize", "features.compute"],
}


def _to_v2_intent(intent: str, ql: str, entities: List[str], objects: List[str]) -> str:
    """Map the lexicon pass onto the v2 structured intent taxonomy."""
    def _has(*phrases: str) -> bool:
        return any(re.search(r"\b" + re.escape(p) + r"\b", ql) for p in phrases)

    if ("how did you" in ql or "how was this" in ql or "explain the evidence" in ql
            or "explain evidence" in ql or "what method" in ql
            or "which method" in ql or "why did you" in ql
            or "show the evidence" in ql or "show evidence" in ql
            or "behind this result" in ql or "explain this result" in ql):
        return "evidence_explanation"
    if (_has("ndvi", "ndwi", "mndwi", "ndbi", "savi", "evi", "gndvi", "vari",
             "exg") or "vegetation index" in ql or "water index" in ql
            or "spectral index" in ql or "spectral indices" in ql):
        return "spectral_index"
    if (_has("texture", "brightness", "contrast", "histogram")
            or "edge density" in ql or "spatial statistic" in ql
            or "image statistic" in ql):
        return "spatial_statistics"
    if "land cover" in ql or "landcover" in ql or "classification" in ql:
        if intent in ("area", "describe"):
            return "land_cover"
    if intent == "change":
        if "compar" in ql and not any(k in ql for k in
                                      ("change", "changed", "loss", "lost", "gain",
                                       "deforest", "expansion", "shrink", "increase",
                                       "decrease", "convert")):
            return "comparison"
        return "temporal_change"
    return {"count": "object_count", "area": "area_estimation",
            "locate": "object_location", "describe": "scene_summary"}.get(intent, intent)


def plan_to_dict(plan: QueryPlan) -> dict:
    """Render the structured plan for the UI (QUERY → … → ANSWER trace)."""
    return {"intent": plan.intent, "target": plan.target,
            "operation": plan.operation,
            "required_evidence": plan.required_evidence,
            "tools": plan.tools, "entities": plan.entities,
            "objects": plan.objects, "superlative": plan.superlative}
