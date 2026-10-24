"""Streaming acquisition policy for PAIEC.

acquisition_function(input, labeled=None, context=None) -> bool          (evaluator entry point)
acquisition_policy(input, prediction=None, labeled=None, context=None)   (same rule, full call shape for experiments)

Candidates arrive in a fixed random order; returning True reveals the candidate's label,
False skips it permanently.  The policy is stateless and never raises.

The evaluator's AcquisitionHook inspects the signature: because ``acquisition_function`` takes no
``prediction`` parameter, the evaluator does not call predict() for every candidate (round 4: with the
cross-pair predictor that call would refit the global model once per candidate; no shipped rule uses the
prediction).  The evaluator passes the GLOBAL ``labeled`` list (labels of all sampled pairs); the
novelty filter compares the candidate only with the labels of its own pair (identical subject attributes
and benchmark_id), as in the local per-pair validation.

Policies (select with POLICY):
  "novelty"  : accept the candidate unless it is a near-duplicate of an item already acquired for the same
               pair (TF-IDF cosine above DEDUP_SIM); always accept when the remaining candidates are needed
               to spend the remaining budget.  Stream order is uniformly random, so accepting in order yields a
               uniformly random sample, as the evaluator's default policy does.
  "fresh"    : round 4 "fresh-first": skip a candidate whose item (benchmark_id + item text) is already in the
               global list for another subject, while labels_remaining < items_remaining.
  "first"    : accept every candidate until the budget is spent (= random subset).
  "ble"      : the organisers' uncertainty rule, accept when |prediction - 0.5| <= 0.15 (needs prediction).
  "novelty_unc": novelty filter plus, from 7 labels on, skip candidates whose current prediction is already
               extreme (|prediction - 0.5| > UNC_SKIP) (needs prediction).
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_model_module():
    """Import the sibling model.py without relying on the evaluator's module naming."""
    try:
        spec = importlib.util.spec_from_file_location("paiec_submission_model", os.path.join(_HERE, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        pass
    try:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import model as mod  # type: ignore
        return mod
    except Exception:
        return None


_MODEL = _load_model_module()
similarities = getattr(_MODEL, "similarities", None)
subject_key = getattr(_MODEL, "xp_subject_key", None)
item_key = getattr(_MODEL, "xp_item_key", None)
label_value = getattr(_MODEL, "_label_value", None)

POLICY = "fresh"   # round 4 (experiments/results/round4/xpair_acq.json); rounds 1-3 shipped "novelty"
DEDUP_SIM = 0.6
UNC_SKIP = 0.45
UNC_MIN_N = 7
FRESH_MAX_ENTRIES = 20000   # larger global lists: no fresh-first check (accept, i.e. random)


def _int(x, default):
    try:
        return int(x)
    except Exception:
        return default


def acquisition_function(input, labeled=None, context=None) -> bool:
    try:
        return bool(_decide(input, None, labeled, context))
    except Exception:
        return True


def acquisition_policy(input, prediction=None, labeled=None, context=None) -> bool:
    try:
        return bool(_decide(input, prediction, labeled, context))
    except Exception:
        return True


def own_pair_items(subject, item, labeled) -> list:
    """Items of the entries of ``labeled`` that belong to the candidate's own pair."""
    if not labeled or subject_key is None:
        return []
    skey = subject_key(subject)
    bench = str(item.get("benchmark_id") or "")
    out = []
    for entry in labeled:
        try:
            (s, it), lab = entry[0], entry[1]
            if not isinstance(s, dict) or not isinstance(it, dict):
                continue
            if str(it.get("benchmark_id") or "") != bench:
                continue
            if s != subject and subject_key(s) != skey:
                continue
            if label_value is not None and label_value(lab) is None:
                continue
            out.append(it)
        except Exception:
            continue
    return out


def labelled_for_other_subject(subject, item, labeled) -> bool:
    if not labeled or subject_key is None or item_key is None or len(labeled) > FRESH_MAX_ENTRIES:
        return False
    ik = item_key(item)
    content, bench = item.get("item_content"), item.get("benchmark_id")
    skey = None
    for entry in labeled:
        try:
            (s, it), _ = entry[0], entry[1]
            if not isinstance(it, dict) or not isinstance(s, dict):
                continue
            # cheap pre-filter (identical strings compare fast), then the full keys
            if it.get("item_content") != content or it.get("benchmark_id") != bench or item_key(it) != ik:
                continue
            if s == subject:
                continue
            if skey is None:
                skey = subject_key(subject)
            if subject_key(s) != skey:
                return True
        except Exception:
            continue
    return False


def _decide(input, prediction, labeled, context) -> bool:
    if not isinstance(context, dict):
        return True
    remaining = _int(context.get("labels_remaining"), 1)
    items_left = _int(context.get("items_remaining"), 1)
    if remaining <= 0:
        return False
    if items_left <= remaining:
        return True
    policy = POLICY
    if policy == "first":
        return True
    if policy == "ble":
        if prediction is None:
            return True
        return abs(float(prediction) - 0.5) <= 0.15
    subject, item = input[0], input[1]
    subject = subject if isinstance(subject, dict) else {}
    item = item if isinstance(item, dict) else {}
    if policy == "fresh":
        return not labelled_for_other_subject(subject, item, labeled)
    # novelty-based policies
    if similarities is None or not labeled:
        return True
    items = own_pair_items(subject, item, labeled)
    n = len(items)
    if n == 0:
        return True
    # urgency: fraction of remaining candidates that must be taken anyway
    urgency = remaining / max(items_left, 1)
    sims = similarities(item, items)
    max_sim = max(sims) if sims else 0.0
    if max_sim > DEDUP_SIM and urgency < 0.9:
        return False
    if policy == "novelty_unc" and prediction is not None and n >= UNC_MIN_N:
        if abs(float(prediction) - 0.5) > UNC_SKIP and urgency < 0.5:
            return False
    return True
