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
sys.path.insert(0, str(HERE))
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
    for lab in (None, [], long_labeled[:7]):
        for pred in (None, 0.5, 0.95):
            d = labeling.acquisition_function([SAMPLE_SUBJECT, dict(SAMPLE_ITEM, benchmark_id="b")],
                                              prediction=pred, labeled=lab, context=ctx)
            assert type(d) is bool
            decisions.append(d)
    assert type(labeling.acquisition_function([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], ctx)) is bool
    assert labeling.acquisition_function([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], dict(ctx, labels_remaining=0)) is False
    assert labeling.acquisition_function([SAMPLE_SUBJECT, SAMPLE_ITEM], 0.5, [], dict(ctx, items_remaining=5)) is True
    assert type(labeling.acquisition_function([SAMPLE_SUBJECT, SAMPLE_ITEM], None, None, None)) is bool
    assert type(labeling.acquisition_function(["bad", None], "x", "y", {"labels_remaining": "q"})) is bool
    report["mean_ms_short"] = 1000 * sum(times) / len(times)
    report["mean_ms_long_budget31"] = 1000 * sum(long_times) / len(long_times)
    report["max_ms_long_budget31"] = 1000 * max(long_times)
    report["calls_per_second_long_budget31"] = len(long_times) / sum(long_times)
    report["acquisition_decisions"] = decisions
    assert report["mean_ms_long_budget31"] < 50, report
    out = HERE.parent / "results" / "smoke_test.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=1))
    print(f"OK: {len(report['cases'])} cases; results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
