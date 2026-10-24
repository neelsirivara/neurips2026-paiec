"""Smoke test for the shipped predictor and acquisition policy.

Calls predict() with the SAMPLE_SUBJECT / SAMPLE_ITEM shapes of the organisers'
tools/smoke_test.py at every budget (0, 1, 3, 7, 15, 31), with and without benchmark_id,
with malformed labels and malformed inputs, and asserts a finite float in [0, 1] and a
mean runtime below 50 ms per call.  Also exercises acquisition_function through the
organisers' AcquisitionHook call shapes.  Writes experiments/results/smoke_test.json.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# --dir DIR tests the model.py/labeling.py in DIR (e.g. ../../submission, the shipped files); default: this directory
_DIR = Path(sys.argv[sys.argv.index("--dir") + 1]).resolve() if "--dir" in sys.argv else HERE
sys.path.insert(0, str(_DIR))
import labeling  # noqa: E402
import model  # noqa: E402

SAMPLE_SUBJECT = {
    "normalized_name": "OpenAI GPT-4o", "provider": "OpenAI", "release_date": "2024-05-13",
    "access_date": "2025-11-01", "harness": "default", "reasoning_effort": "",
    "harness_version": "", "subject_features_extra": "",
}
SAMPLE_ITEM = {
    "item_content": "What is the derivative of f(x) = x^3 * sin(x)?",
    "item_features": "tier=medium;points=1", "interactors": "",
}
SMOKE_SUBJECT = {
    "normalized_name": "sample-model", "provider": "sample-org", "release_date": "2026-01-01",
    "access_date": "2026-02-01", "harness": "sample-harness", "reasoning_effort": "medium",
    "harness_version": "1.0", "subject_features_extra": "",
}
SMOKE_ITEM = {"item_content": "A sample yes/no evaluation item.", "item_features": "tier=near-term",
              "interactors": ""}
BUDGETS = [0, 1, 3, 7, 15, 31]
WORDS = ("integral derivative matrix prime graph compute sum product probability expected value "
         "sequence series limit function polynomial equation solve prove show find").split()


def make_labeled(subject, item, n, rng, with_bench=True):
    out = []
    for i in range(n):
        it = dict(item, item_content=" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 40))))
        if with_bench:
            it["benchmark_id"] = "bench_001"
        out.append([[copy.deepcopy(subject), it], rng.randint(0, 1)])
    return out


def check(value, label):
    assert not isinstance(value, bool), label
    v = float(value)
    assert math.isfinite(v) and 0.0 <= v <= 1.0, f"{label}: {value}"
    return v


def _global_list(n_subjects, n_benches, n_items, budget, rng, words=WORDS, item_words=12):
    """Synthetic GLOBAL labeled list as the official evaluator passes it (all pairs' labels)."""
    subjects = [dict(SMOKE_SUBJECT, normalized_name=f"model-{i % max(1, n_subjects // 2)}",
                     reasoning_effort=("low", "high")[i % 2]) for i in range(n_subjects)]
    items = {b: [{"benchmark_id": f"bench_{b:03d}",
                  "item_content": f"Q{b}.{k}: " + " ".join(rng.choice(words) for _ in range(item_words)),
                  "item_features": "", "interactors": ""} for k in range(n_items)] for b in range(n_benches)}
    labeled, targets = [], []
    for i, s in enumerate(subjects):
        for b in range(n_benches):
            if (i + b) % 3 == 2 and n_subjects > 2:  # not every subject on every benchmark
                continue
            idx = list(range(n_items))
            rng.shuffle(idx)
            half = n_items // 2
            for k in idx[:half][:budget]:
                labeled.append([[copy.deepcopy(s), dict(items[b][k])], int(rng.random() < 0.3 + 0.4 * (b % 2))])
            targets += [[s, items[b][k]] for k in idx[half:]]
    return subjects, items, labeled, targets


def _raise(*args, **kwargs):
    raise RuntimeError("simulated failure of the cross-pair fit")


def round4_checks(rng) -> dict:
    """Cross-pair predictor (model.XP_*) and the fresh-first acquisition policy under global labeled lists."""
    out = {"xp_enabled": bool(getattr(model, "XP_ENABLED", False))}
    if not hasattr(model, "predict_cross_pair"):
        return out
    import threading
    sys.path.insert(0, str(HERE / "validator" / "tools"))
    from streaming_ingestion import AcquisitionHook  # verbatim copy of the organisers' hook
    cases = 0
    for budget in (1, 3, 7, 15, 31):
        subjects, items, labeled, targets = _global_list(6, 3, 60, budget, rng)
        model.xp_clear_cache()
        extra = [[SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="bench_new")],             # unseen subject and benchmark
                 [dict(SMOKE_SUBJECT, normalized_name="unseen"), items[0][0]],             # unseen subject, seen item
                 [subjects[0], dict(items[1][0], item_content="new text")]]                # seen pair, new item
        for x in targets[:40] + extra:
            p = model.predict(x, labeled)
            check(p, f"xp budget {budget}")
            if model.XP_ENABLED:
                assert p == model.predict_cross_pair(x, labeled), "XP path not used"
            cases += 1
        bad = labeled + [None, 5, [], [[1, 2], 1], [["a"], 1], [[SAMPLE_SUBJECT, SAMPLE_ITEM], None],
                         [[SAMPLE_SUBJECT, SAMPLE_ITEM], "yes"], [["x", {"item_content": 3}], 0.2]]
        check(model.predict(targets[0], bad), "xp malformed")
        cases += 1
    # determinism: cache cleared, deep copy
    subjects, items, labeled, targets = _global_list(8, 4, 60, 15, random.Random(3))
    model.xp_clear_cache()
    a = [model.predict(x, labeled) for x in targets[:50]]
    model.xp_clear_cache()
    b = [model.predict(x, copy.deepcopy(labeled)) for x in targets[:50]]
    assert a == b, "cross-pair predictions are not deterministic"
    # thread safety: 4 threads on the same list give the sequential predictions
    model.xp_clear_cache()
    res = [None] * 4

    def work(k):
        res[k] = [model.predict(x, labeled) for x in targets[:50]]
    ths = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert all(r == a for r in res), "threaded predictions differ"
    # fallback: a failing cross-pair fit falls back to the per-pair predictor
    saved_build, saved_flag = model.xp_build, model.XP_ENABLED
    try:
        model.xp_build = _raise
        model.xp_clear_cache()
        p = check(model.predict(targets[0], labeled), "xp fallback")
        model.XP_ENABLED = False
        q = model.predict(targets[0], labeled)
        assert p == q, "fallback is not the per-pair predictor"
    finally:
        model.xp_build, model.XP_ENABLED = saved_build, saved_flag
        model.xp_clear_cache()
    # scale: 2,000 pairs x 31 labels = 62,000 entries (one checkpoint): fit once, then cache hits
    big_rng = random.Random(7)
    subjects = [dict(SMOKE_SUBJECT, normalized_name=f"model-{i % 400}", reasoning_effort=("low", "high", "")[i % 3])
                for i in range(2000)]
    bench_items = {b: [{"benchmark_id": f"bench_{b:03d}",
                        "item_content": f"Q{b}.{k}: " + " ".join(big_rng.choice(WORDS) for _ in range(250)),
                        "item_features": "tier=hard" if k % 4 == 0 else "", "interactors": ""} for k in range(100)]
                   for b in range(40)}
    big, big_targets = [], []
    for i, s in enumerate(subjects):
        b = i % 40
        idx = list(range(100))
        big_rng.shuffle(idx)
        big += [[[s, bench_items[b][k]], int(big_rng.random() < 0.5)] for k in idx[:31]]
        big_targets += [[s, bench_items[b][k]] for k in idx[50:]]
    model.xp_clear_cache()
    t0 = time.perf_counter()
    check(model.predict(big_targets[0], big), "big first call")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_calls = 3000
    for x in big_targets[1:1 + n_calls]:
        check(model.predict(x, big), "big")
    per_call = (time.perf_counter() - t0) / n_calls
    model.xp_clear_cache()
    out.update({"xp_cases": cases, "xp_big_entries": len(big), "xp_big_first_call_s": first,
                "xp_big_ms_per_call": 1000 * per_call, "xp_big_targets": len(big_targets),
                "xp_big_checkpoint_estimate_s": first + per_call * len(big_targets)})
    if model.XP_ENABLED:
        assert first <= 60.0 and 1000 * per_call <= 5.0, out
    # acquisition: evaluator call shape (no prediction parameter -> predict is not called per candidate)
    hook = AcquisitionHook(labeling.acquisition_function)
    out["acquisition_hook_mode"] = hook.mode
    out["acquisition_hook_keywords"] = list(hook.keywords)
    out["acquisition_needs_prediction"] = bool(hook.needs_prediction)
    assert hook.mode == "stream"
    subjects, items, labeled, targets = _global_list(4, 2, 40, 7, random.Random(5))
    ctx = {"subject_id": "s", "benchmark_id": "bench_000", "labels_acquired": 3, "labels_remaining": 28,
           "max_labels": 31, "items_remaining": 35}
    other = labeled[0][0][1]      # an item already labelled for subject 0
    fresh = dict(other, item_content="never labelled")
    newsub = dict(SMOKE_SUBJECT, normalized_name="someone-else")
    d_stale = hook([newsub, other], None, labeled, ctx)
    d_fresh = hook([newsub, fresh], None, labeled, ctx)
    d_urgent = hook([newsub, other], None, labeled, dict(ctx, items_remaining=28))
    d_none = hook([newsub, other], None, None, None)
    for d in (d_stale, d_fresh, d_urgent, d_none):
        assert type(d) is bool
    out["acquisition_fresh_checks"] = {"stale_item": d_stale, "fresh_item": d_fresh, "stale_but_urgent": d_urgent,
                                       "no_context": d_none}
    if getattr(labeling, "POLICY", "") == "fresh":
        assert d_stale is False and d_fresh is True and d_urgent is True
    return out


def main() -> int:
    rng = random.Random(0)
    report = {"cases": [], "ok": True}
    times = []
    for subject, item in ((SAMPLE_SUBJECT, SAMPLE_ITEM), (SMOKE_SUBJECT, SMOKE_ITEM)):
        for with_bench in (True, False):
            target = dict(item)
            if with_bench:
                target["benchmark_id"] = "bench_001"
            for n in BUDGETS:
                labeled = make_labeled(subject, item, n, rng, with_bench)
                t0 = time.perf_counter()
                p = model.predict([subject, target], labeled=labeled)
                times.append(time.perf_counter() - t0)
                v = check(p, f"budget {n} bench={with_bench}")
                report["cases"].append({"subject": subject["normalized_name"], "benchmark_id": with_bench,
                                        "budget": n, "prediction": v})
    # malformed labels and inputs must still give a valid probability
    bad_cases = {
        "labels_as_strings": [[[SAMPLE_SUBJECT, dict(SAMPLE_ITEM)], "1"], [[SAMPLE_SUBJECT, dict(SAMPLE_ITEM)], "no"]],
        "labels_none": [[[SAMPLE_SUBJECT, dict(SAMPLE_ITEM)], None]],
        "labels_float": [[[SAMPLE_SUBJECT, dict(SAMPLE_ITEM)], 0.7]],
        "entries_broken": [None, 5, [], [[1, 2], 1], [["a"], 1]],
        "labeled_none": None,
        "labeled_tuple": tuple([((SAMPLE_SUBJECT, SAMPLE_ITEM), 1)]),
    }
    for name, labeled in bad_cases.items():
        v = check(model.predict([SAMPLE_SUBJECT, SAMPLE_ITEM], labeled=labeled), name)
        report["cases"].append({"case": name, "prediction": v})
    for name, inp in {"empty_dicts": [{}, {}], "none_fields": [{"release_date": None}, {"item_content": None}],
                      "non_dicts": ["x", 3], "huge_text": [SAMPLE_SUBJECT, {"item_content": "word " * 200000}]}.items():
        v = check(model.predict(inp, labeled=[]), name)
        report["cases"].append({"case": name, "prediction": v})
    # item-text prior inputs: JSON contents (valid and truncated) and a malformed specification, which must
    # fall back to the hand-set prior without raising
    json_items = {"json_multimedia": '{"multimedia_elements": [{"content_type": "text/plain", "text": "What is 2+2?"}]}',
                  "json_messages": '{"messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}]}',
                  "json_truncated": '{"prompt": "unterminated', "json_list": '["a", 1, null, {"k": [true, 2.5]}]'}
    for name, content in json_items.items():
        for n in (0, 7):
            labeled = make_labeled(SAMPLE_SUBJECT, SAMPLE_ITEM, n, rng)
            v = check(model.predict([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, item_content=content, benchmark_id="bench_001")],
                                    labeled=labeled), name)
            report["cases"].append({"case": f"{name}_budget{n}", "prediction": v})
    if hasattr(model, "TEXT_PRIOR"):  # model.py versions with the (optional) item-text prior hooks
        saved = model.TEXT_PRIOR
        try:
            model.TEXT_PRIOR = {"cap": "broken"}
            model._TEXT_CACHE.clear()
            for n in (0, 7):
                v = check(model.predict([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="bench_001")],
                                        labeled=make_labeled(SAMPLE_SUBJECT, SAMPLE_ITEM, n, rng)), "malformed_spec")
                report["cases"].append({"case": f"malformed_text_spec_budget{n}", "prediction": v})
        finally:
            model.TEXT_PRIOR = saved
            model._TEXT_CACHE.clear()
            model._SET_CACHE.clear()
    report["text_prior_loaded"] = getattr(model, "TEXT_PRIOR", None) is not None
    report["tested_dir"] = str(_DIR)
    # determinism
    labeled = make_labeled(SAMPLE_SUBJECT, SAMPLE_ITEM, 15, random.Random(1))
    a = model.predict([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="bench_001")], labeled=labeled)
    b = model.predict([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="bench_001")], labeled=copy.deepcopy(labeled))
    assert a == b, "predict is not deterministic"
    # throughput with realistic long items at budget 31
    long_labeled = []
    for i in range(31):
        it = dict(SAMPLE_ITEM, benchmark_id="b", item_content=" ".join(rng.choice(WORDS) for _ in range(300)))
        long_labeled.append([[SAMPLE_SUBJECT, it], rng.randint(0, 1)])
    long_times = []
    for i in range(200):
        tgt = dict(SAMPLE_ITEM, benchmark_id="b", item_content=" ".join(rng.choice(WORDS) for _ in range(300)))
        t0 = time.perf_counter()
        check(model.predict([SAMPLE_SUBJECT, tgt], labeled=long_labeled), "long")
        long_times.append(time.perf_counter() - t0)
    # acquisition hook call shapes (mirrors tools/streaming_ingestion.AcquisitionHook)
    ctx = {"subject_id": "s", "benchmark_id": "b", "labels_acquired": 0, "labels_remaining": 31,
           "max_labels": 31, "items_remaining": 40}
    decisions = []
    full_shape = getattr(labeling, "acquisition_policy", labeling.acquisition_function)
    for lab in (None, [], long_labeled[:7]):
        for pred in (None, 0.5, 0.95):
            d = full_shape([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="b")], prediction=pred, labeled=lab, context=ctx)
            assert type(d) is bool
            decisions.append(d)
    assert type(full_shape([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], ctx)) is bool
    assert full_shape([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], dict(ctx, labels_remaining=0)) is False
    assert full_shape([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], dict(ctx, items_remaining=5)) is True
    assert type(full_shape([SAMPLE_SUBJECT, SAMPLE_ITEM], None, None, None)) is bool
    assert type(full_shape(["bad", None], "x", "y", {"labels_remaining": "q"})) is bool
    report.update(round4_checks(rng))
    report["mean_ms_short"] = 1000 * sum(times) / len(times)
    report["mean_ms_long_budget31"] = 1000 * sum(long_times) / len(long_times)
    report["max_ms_long_budget31"] = 1000 * max(long_times)
    report["calls_per_second_long_budget31"] = len(long_times) / sum(long_times)
    report["acquisition_decisions"] = decisions
    assert report["mean_ms_long_budget31"] < 50, report
    out = (Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv
           else HERE.parent / "results" / "smoke_test.json")
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=1))
    print(f"OK: {len(report['cases'])} cases; results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
