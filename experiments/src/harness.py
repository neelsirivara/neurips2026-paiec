"""Local reproduction of the PAIEC evaluation protocol.

A *pair* is one subject evaluated on one benchmark.  Its items are split 50/50 into an
acquisition pool and an evaluation pool (pairs need >= 80 distinct items).  Acquisition
candidates arrive in a fixed random order; a streaming policy returns True/False per
candidate until 31 labels are acquired.  Nested label sets at budgets 0, 1, 3, 7, 15, 31
(prefixes of the one trajectory) are passed to predict() for every evaluation item.

Metric: Brier ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31, per pair, then
averaged over pairs (lower is better).  ECE (10 equal-width bins) is a diagnostic.
"""
from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import model  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]
ALC_WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
MAX_LABELS = 31
MIN_ITEMS = 80


def brier(preds, labels) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(labels)


def alc(brier_by_budget: dict) -> float:
    return sum(ALC_WEIGHTS[b] * brier_by_budget[b] for b in BUDGETS)


def ece(preds, labels, bins: int = 10) -> float:
    n = len(preds)
    if n == 0:
        return float("nan")
    tot = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, p in enumerate(preds) if (lo <= p < hi) or (k == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(preds[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(conf - acc)
    return tot


# ----------------------------------------------------------------------------- baselines
def empirical_mean_predict(input, labeled=None) -> float:
    """Organisers' empirical-mean baseline (copied logic: raw mean, 0.5 without labels)."""
    subject, item = input
    if not labeled:
        return 0.5
    benchmark = item.get("benchmark_id")
    successes = count = 0
    for (observed_subject, observed_item), label in labeled:
        if observed_subject != subject:
            continue
        if observed_item.get("benchmark_id") != benchmark:
            continue
        successes += int(label)
        count += 1
    return float(successes / count) if count else 0.5


# ----------------------------------------------------------------------------- policies
def random_policy_factory(rng: random.Random):
    """Evaluator default: accept with probability labels_remaining / items_remaining."""
    def policy(input, prediction, labeled, context):
        return rng.random() < min(1.0, context["labels_remaining"] / context["items_remaining"])
    return policy


def run_acquisition(pair: dict, policy, order: list, predictor=None, needs_prediction=True) -> list:
    """Stream the acquisition pool in ``order``; return the acquired indices (<= 31)."""
    subject = pair["subject"]
    acquired: list = []
    labeled: list = []
    items_remaining = len(order)
    for idx in order:
        if len(acquired) >= MAX_LABELS:
            break
        item, label = pair["acq"][idx]
        context = {"subject_id": pair["subject_id"], "benchmark_id": pair["benchmark_id"],
                   "labels_acquired": len(acquired), "labels_remaining": MAX_LABELS - len(acquired),
                   "max_labels": MAX_LABELS, "items_remaining": items_remaining}
        prediction = None
        if needs_prediction and predictor is not None:
            prediction = predictor([subject, item], copy.deepcopy(labeled))
        take = policy([subject, item], prediction, copy.deepcopy(labeled), context)
        if type(take) is not bool:
            raise ValueError("acquisition policy must return bool")
        if take:
            acquired.append(idx)
            labeled.append([[subject, item], label])
        items_remaining -= 1
    return acquired


# ----------------------------------------------------------------------------- evaluation
def evaluate_pair(pair: dict, acquired: list, methods: dict) -> dict:
    """methods: name -> callable(input, labeled) -> float or dict of named probabilities.

    A callable may return a dict {variant_name: p} so that several variants sharing
    computation (model.predict_components) are evaluated from one call.  Returns
    {method_name: {"brier": {budget: B}, "alc": float, "preds": {budget: [...]}}}.
    """
    subject = pair["subject"]
    eval_items = pair["eval"]
    labels = [y for _, y in eval_items]
    out: dict = {}
    for n in BUDGETS:
        labeled = [[[subject, pair["acq"][i][0]], pair["acq"][i][1]] for i in acquired[:n]]
        preds_by_method: dict = {}
        for name, fn in methods.items():
            for item, _ in eval_items:
                value = fn([subject, item], copy.deepcopy(labeled))
                if isinstance(value, dict):
                    for var, p in value.items():
                        preds_by_method.setdefault(f"{name}:{var}" if name else var, []).append(float(p))
                else:
                    preds_by_method.setdefault(name, []).append(float(value))
        for name, preds in preds_by_method.items():
            rec = out.setdefault(name, {"brier": {}, "preds": {}})
            rec["brier"][n] = brier(preds, labels)
            rec["preds"][n] = preds
    # oracle reference rows (not predictors): the evaluation-set success rate as a constant,
    # and, for synthetic data, the true item probabilities.
    rate = sum(labels) / len(labels)
    out["oracle_rate"] = {"brier": {n: brier([rate] * len(labels), labels) for n in BUDGETS},
                          "preds": {n: [rate] * len(labels) for n in BUDGETS}}
    if pair.get("eval_p") is not None:
        out["oracle_p"] = {"brier": {n: brier(pair["eval_p"], labels) for n in BUDGETS},
                           "preds": {n: list(pair["eval_p"]) for n in BUDGETS}}
    for rec in out.values():
        rec["alc"] = alc(rec["brier"])
    out["_labels"] = labels
    return out


def components_predictor(input, labeled=None) -> dict:
    """Variants of the shipped predictor sharing one computation."""
    comps = model.predict_components(input, labeled)
    return {"prior": model._clip(comps["prior"]), "shrink": model._clip(comps["pair"]),
            "shrink_ridge": model._clip(comps["ridge"]), "shrink_knn": model._clip(comps["knn"]),
            "full": model._clip(comps["full"])}


def constant_half(input, labeled=None) -> float:
    return 0.5


DEFAULT_METHODS = {
    "const": constant_half,
    "empirical_mean": empirical_mean_predict,
    "": components_predictor,
}


def make_pair(subject: dict, subject_id: str, benchmark_id: str, items: list, rng: random.Random) -> dict | None:
    """items: list of (item_dict, label).  Returns None if fewer than MIN_ITEMS items."""
    if len(items) < MIN_ITEMS:
        return None
    idx = list(range(len(items)))
    rng.shuffle(idx)
    half = len(idx) // 2
    acq = [(items[i][0], items[i][1]) for i in idx[:half]]
    ev = [(items[i][0], items[i][1]) for i in idx[half:]]
    eval_p = [items[i][2] for i in idx[half:]] if all(len(items[i]) > 2 for i in idx[half:]) else None
    for it, _ in acq + ev:
        it["benchmark_id"] = benchmark_id
    order = list(range(len(acq)))
    rng.shuffle(order)
    return {"subject": subject, "subject_id": subject_id, "benchmark_id": benchmark_id,
            "acq": acq, "eval": ev, "order": order, "eval_p": eval_p}


# ----------------------------------------------------------------------------- statistics
def bootstrap_mean_ci(values: list, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    boots = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return {"mean": mean, "ci_lo": boots[int(0.025 * n_boot)], "ci_hi": boots[int(0.975 * n_boot) - 1],
            "n": n, "n_boot": n_boot}


def paired_bootstrap(a: list, b: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """CI of mean(a - b) over pairs (negative favours a when lower is better)."""
    diffs = [x - y for x, y in zip(a, b)]
    res = bootstrap_mean_ci(diffs, n_boot, seed)
    res["frac_a_better"] = sum(d < 0 for d in diffs) / len(diffs)
    res["significant"] = bool(res["ci_hi"] < 0 or res["ci_lo"] > 0)
    return res
