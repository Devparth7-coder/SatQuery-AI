"""Unit tests: structured query planner + grounded answering."""
import os

from core import objects as obj_mod
from core.change import detect
from core.features import compute_features, load_scene
from core.landcover import segment
from core.query import answer, parse_query, plan_to_dict

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _analysis(name="harbour_port.png"):
    s = load_scene(os.path.join(SAMPLES, name), gsd=10.0)
    fs = compute_features(s)
    lc = segment(s, fs, k=7)
    return {"scene": s, "features": fs, "landcover": lc,
            "objects": obj_mod.summarize(s, fs, lc)}


def test_planner_taxonomy():
    cases = {
        "Describe this scene": "scene_summary",
        "How much of the area is water?": "area_estimation",
        "Show the land cover classification": "land_cover",
        "What is the NDVI here?": "spectral_index",
        "How many ships are there?": "object_count",
        "Where is the built-up area?": "object_location",
        "What changed between the two dates?": "temporal_change",
        "Compare the land cover between the two dates": "comparison",
        "Show texture statistics": "spatial_statistics",
        "How did you compute this?": "evidence_explanation",
        "Is there any cloud cover?": "presence",
    }
    for q, intent in cases.items():
        p = parse_query(q)
        assert p.intent == intent, (q, p.intent)
        d = plan_to_dict(p)
        assert d["target"] and d["operation"] and d["required_evidence"] and d["tools"]


def test_answer_smoke_all_intents():
    a = _analysis()
    queries = ["Describe this scene", "How much water?", "What is the NDVI?",
               "How many ships?", "Where is built-up?", "Is there cloud?",
               "Show texture statistics", "How did you compute this?",
               "Show the land cover classification"]
    for q in queries:
        r = answer(parse_query(q), a, None)
        assert r["answer"] and len(r["answer"]) > 40, q
        assert 0.0 <= r["confidence"] <= 1.0
        assert "overlay" in r and "evidence" in r


def test_change_requires_second_image():
    a = _analysis()
    r = answer(parse_query("What changed?"), a, None)
    assert r.get("needs") == "second_image"
    assert r["confidence"] == 0.0


def test_change_answer_with_pair():
    s1 = load_scene(os.path.join(SAMPLES, "delta_t1.png"), gsd=10.0)
    s2 = load_scene(os.path.join(SAMPLES, "delta_t2.png"), gsd=10.0)
    fs1 = compute_features(s1)
    lc1 = segment(s1, fs1, k=7)
    a = {"scene": s1, "features": fs1, "landcover": lc1,
         "objects": obj_mod.summarize(s1, fs1, lc1)}
    cr = detect(s1, s2, k=7, lc1=lc1)
    r = answer(parse_query("What changed between the two dates?"), a, {"result": cr})
    assert "severity" in r["answer"].lower() or "Severity" in r["answer"]
    assert r["overlay"]["type"] == "change"
    assert any("changed_area_fraction" in e["metric"] for e in r["evidence"])
