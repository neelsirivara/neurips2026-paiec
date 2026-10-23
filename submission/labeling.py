"""Streaming acquisition policy for PAIEC: novelty-filtered pacing.

acquisition_function(input, prediction=None, labeled=None, context=None) -> bool

Candidates arrive in a fixed random order; returning True reveals the candidate's label,
False skips it permanently.  The policy is stateless and never raises.

Default policy ("novelty"): accept the candidate unless it is a near-duplicate of an
already acquired item (TF-IDF cosine above DEDUP_SIM), and always accept when the
remaining candidates are needed to spend the remaining budget.  Because the stream order
is uniformly random, accepting candidates in order yields a uniformly random sample of
the pool, as the evaluator's default policy does; the filter only removes redundant
items that the kNN component would learn little from.

Alternative policies kept for the ablation in the report (select with POLICY):
  "first"    : accept every candidate until the budget is spent (= random subset).
  "ble"      : the organisers' uncertainty rule, accept when |prediction - 0.5| <= 0.15.
  "novelty_unc": novelty filter plus, from 7 labels on, skip candidates whose current
               prediction is already extreme (|prediction - 0.5| > UNC_SKIP).
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
pair_evidence = getattr(_MODEL, "pair_evidence", None)
similarities = getattr(_MODEL, "similarities", None)

POLICY = "novelty"
DEDUP_SIM = 0.6
UNC_SKIP = 0.45
UNC_MIN_N = 7


def _int(x, default):
    try:
        return int(x)
    except Exception:
        return default


def acquisition_function(input, prediction=None, labeled=None, context=None) -> bool:
    try:
        return bool(_decide(input, prediction, labeled, context))
    except Exception:
        return True


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
    # novelty-based policies
    subject, item = input[0], input[1]
    if pair_evidence is None or similarities is None or not labeled:
        return True
    items, labels = pair_evidence(subject if isinstance(subject, dict) else {},
                                  item if isinstance(item, dict) else {}, labeled)
    n = len(labels)
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
