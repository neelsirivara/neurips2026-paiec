"""Sensitivity of the hand-set budget-0 prior to frontier-like subjects.

    python prior_sensitivity.py --suite real            # (a), (b) and the real-testbed offset sweep
    python prior_sensitivity.py --suite all --workers 2 # ... plus the synthetic offset sweep

Writes experiments/results/prior_sensitivity.json with three parts:

(a) frontier_grid: the prior probability p0 (no item cues, and with a ``tier=hard`` cue) that the
    shipped constants assign to subjects described by release year, reasoning effort, a size/name
    cue and the harness flag -- the inputs the official test set is likely to contain and the local
    testbeds do not (release dates 2025-2026, long reasoning efforts, very large models).
(b) miscalibration: the exact expected Brier excess over the pair-rate oracle, per budget and as ALC,
    of the shipped shrinkage rule (prior strength ALPHA0) when the prior rate p0 differs from the true
    pair rate r, with the number of successes among the n acquired labels binomial(n, r); and the ALC
    of the constant 0.5 on the same pair, so that the range of p0 for which shrinkage still beats the
    constant can be read off.
(c) offset_sweep: the two local testbeds re-run inside the harness with a global shift of the subject
    logit (PRIOR_INTERCEPT in {-1, -0.5, 0, +0.5, +1}), i.e. the ALC cost of a prior that is too
    pessimistic or too optimistic by up to one logit for every subject.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import model  # noqa: E402

RESULTS = HERE.parent / "results"
OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PRIORS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def frontier_grid() -> list:
    rows = []
    for year in ("2023-06-15", "2024-06-15", "2025-06-15", "2026-06-15"):
        for effort in ("", "low", "medium", "high"):
            for cue, name in (("none", "model-x"), ("mini", "model-x-mini"), ("pro", "model-x-pro"), ("70b", "model-x-70b"),
                              ("400b", "model-x-400b")):
                for harness_flag in ("default", "agent-scaffold"):
                    subject = {"normalized_name": name, "provider": "", "release_date": year, "access_date": "2026-10-01",
                               "harness": harness_flag, "harness_version": "", "reasoning_effort": effort,
                               "subject_features_extra": ""}
                    zs = model._subject_logit(subject)
                    p_plain = model._sigmoid(min(max(zs + model._item_logit({"item_content": "x" * 600, "item_features": ""}),
                                                     -model.PRIOR_LOGIT_CAP), model.PRIOR_LOGIT_CAP))
                    p_hard = model._sigmoid(min(max(zs + model._item_logit({"item_content": "x" * 600, "item_features": "tier=hard"}),
                                                    -model.PRIOR_LOGIT_CAP), model.PRIOR_LOGIT_CAP))
                    rows.append({"release_date": year, "reasoning_effort": effort, "name_cue": cue, "harness": harness_flag,
                                 "subject_logit": round(zs, 4), "p0": round(p_plain, 4), "p0_tier_hard": round(p_hard, 4)})
    return rows


NAMED_SUBJECTS = {
    # the organisers' two sample subjects (tools/smoke_test.py of the baseline repository)
    "sample_gpt4o": {"normalized_name": "OpenAI GPT-4o", "provider": "OpenAI", "release_date": "2024-05-13",
                     "access_date": "2025-11-01", "harness": "default", "reasoning_effort": "", "harness_version": "",
                     "subject_features_extra": ""},
    "smoke_2026_medium": {"normalized_name": "sample-model", "provider": "sample-org", "release_date": "2026-01-01",
                          "access_date": "2026-02-01", "harness": "sample-harness", "reasoning_effort": "medium",
                          "harness_version": "1.0", "subject_features_extra": ""},
    # local testbed subjects (real_testbed.py)
    "testbed_qwen_1_5b": {"normalized_name": "Qwen2.5-1.5B-Instruct", "provider": "Alibaba Cloud", "release_date": "2024-09-19",
                          "harness": "ollama-direct-answer", "reasoning_effort": ""},
    "testbed_llama_3b": {"normalized_name": "Llama-3.2-3B-Instruct", "provider": "Meta", "release_date": "2024-09-25",
                         "harness": "ollama-direct-answer", "reasoning_effort": ""},
}


def named_subjects() -> dict:
    out = {}
    for key, subject in NAMED_SUBJECTS.items():
        zs = model._subject_logit(subject)
        p0 = model._sigmoid(min(max(zs + model._item_logit({"item_content": "x" * 600, "item_features": ""}),
                                    -model.PRIOR_LOGIT_CAP), model.PRIOR_LOGIT_CAP))
        out[key] = {"subject": subject, "subject_logit": round(zs, 4), "p0": round(p0, 4)}
    return out


def miscalibration(alpha0: float) -> dict:
    """Exact expected Brier excess E[(rho_n - r)^2] over the oracle r(1-r), per budget and ALC."""
    out = {"alpha0": alpha0, "rates": list(RATES), "priors": list(PRIORS), "cells": []}
    for r in RATES:
        for p0 in PRIORS:
            excess = {}
            for n in harness.BUDGETS:
                e = 0.0
                for k in range(n + 1):
                    prob = math.comb(n, k) * r ** k * (1 - r) ** (n - k)
                    rho = (alpha0 * p0 + k) / (alpha0 + n) if n > 0 else p0
                    rho = min(max(rho, model.CLIP_LO), model.CLIP_HI)
                    e += prob * (rho - r) ** 2
                excess[n] = e
            alc_excess = harness.alc(excess)
            oracle = r * (1 - r)
            const_excess = (0.5 - r) ** 2
            out["cells"].append({"r": r, "p0": p0, "excess_by_budget": {str(n): round(v, 4) for n, v in excess.items()},
                                 "alc_excess": round(alc_excess, 4), "alc": round(alc_excess + oracle, 4),
                                 "alc_const": round(const_excess + oracle, 4),
                                 "beats_const": bool(alc_excess < const_excess)})
    return out


def offset_sweep(suite: str, workers: int) -> dict:
    import run_experiments as R
    out = {}
    datasets = {}
    if suite in ("real", "all"):
        import real_testbed
        raw = real_testbed.load_pairs()
        if raw:
            datasets["real"] = raw
    if suite in ("synthetic", "all"):
        import simulate
        datasets["synthetic"] = simulate.generate_pairs(0, R.N_SUBJECTS, R.N_BENCHMARKS, R.MAIN_SIGMA, R.MAIN_GAMMA)
    for name, raw in datasets.items():
        rec = {"offsets": {}}
        base = None
        for off in OFFSETS:
            s = R.summarize(R.run_dataset(raw, seed=0, params={"PRIOR_INTERCEPT": off}, workers=workers))
            rec["offsets"][f"{off:+.1f}"] = {m: {"alc": s["methods"][m]["alc"], "brier_by_budget": s["methods"][m]["brier_by_budget"],
                                                 "ece_pooled": s["methods"][m]["ece_pooled"]} for m in ("prior", "shrink", "full")}
            if off == 0.0:
                base = s
            rec["offsets"][f"{off:+.1f}"]["_per_pair"] = {m: s["methods"][m]["per_pair_alc"] for m in ("prior", "shrink", "full")}
            R.headline(s, f"{name} PRIOR_INTERCEPT={off:+.1f}")
        for key, r in rec["offsets"].items():
            r["paired_vs_zero"] = {m: harness.paired_bootstrap(r["_per_pair"][m], rec["offsets"]["+0.0"]["_per_pair"][m], R.N_BOOT, seed=11)
                                   for m in ("prior", "shrink", "full")}
        for r in rec["offsets"].values():
            r.pop("_per_pair", None)
        full = {k: v["full"]["alc"]["mean"] for k, v in rec["offsets"].items()}
        rec["full_alc_range"] = max(full.values()) - min(full.values())
        rec["best_offset_full"] = min(full, key=full.get)
        out[name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="real", choices=["none", "real", "synthetic", "all"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    path = RESULTS / "prior_sensitivity.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = {"constants": {"ALPHA0": model.ALPHA0, "SUBJECT_SHRINK": model.SUBJECT_SHRINK, "PRIOR_LOGIT_CAP": model.PRIOR_LOGIT_CAP,
                         "PRIOR_INTERCEPT": model.PRIOR_INTERCEPT},
           "frontier_grid": frontier_grid(), "named_subjects": named_subjects(), "miscalibration": miscalibration(model.ALPHA0),
           "offset_sweep": dict(prev.get("offset_sweep", {}))}
    if a.suite != "none":
        out["offset_sweep"].update(offset_sweep(a.suite, a.workers))
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    g = out["frontier_grid"]
    print(f"frontier grid: {len(g)} subjects, p0 from {min(r['p0'] for r in g):.3f} to {max(r['p0'] for r in g):.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
