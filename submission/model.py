"""PAIEC predictor: hierarchical shrinkage with LOO-gated, text-aware test-time adaptation.

predict(input, labeled) -> float estimates P(subject answers item correctly).

Design (every constant below is a hand-set prior documented in the technical report; nothing
is fitted on competition data, and no state is kept between calls other than caches that
only speed up identical computations):

  1. Budget-0 prior.  A small logistic form over visible subject attributes (release
     date, reasoning effort, parameter count parsed from the name, harness) and item cues
     (tier/level/points in ``item_features``, item length), capped at +-1.5 logits.  The
     prior is a sum of coefficient * feature terms (``subject_terms``/``item_terms``) with
     an intercept and an optional per-provider table, so that coefficients fitted on the
     official training release can replace the hand-set constants one-to-one; the shipped
     values are the hand-set ones (intercept 0, empty provider table).
  2. Pair-level shrinkage.  Beta-binomial posterior mean of the subject-benchmark success
     rate given the acquired labels, with prior strength ALPHA0 centred on the prior.
  3. Within-pair difficulty regression (from RIDGE_MIN_N labels).  A ridge-penalised
     logistic regression on three cheap item features (log length, digit density, sentence
     count) fitted to the acquired labels; its adjustment is capped.
  4. Item-level similarity transfer.  TF-IDF (unigram+bigram, sublinear tf, in-call idf)
     cosine similarity between the target and the acquired items; a kernel-weighted
     residual correction anchored by TAU pseudo-counts and gated by n/(n+N0).
  5. Leave-one-out gating (from LOO_MIN_N labels; below that, no item-level adaptation).
     The kernel scale kappa (including "off") and the use of the ridge component are chosen
     per labeled set by leave-one-out Brier score on the acquired labels themselves, and the
     plain shrinkage estimate is kept unless the improvement exceeds LOO_SE_MULT standard
     errors, so adaptation is only applied when the pair's own labels show that it helps.
  6. Clipping to [0.01, 0.99].

Only the Python standard library is used.  predict never raises: any failure falls back to
the pair-level shrinkage estimate, then to a smoothed mean, then to 0.5.
"""
from __future__ import annotations

import math
import re

# ----------------------------------------------------------------------------- settings
ALPHA0 = 4.0           # prior strength (pseudo-counts) of the pair-level beta-binomial
TAU = 2.0              # pseudo-counts anchoring the kNN residual correction
KNN_N0 = 4.0           # gate n/(n+KNN_N0) on the kNN correction
KNN_KAPPA = 2.0        # default kernel scale: weight = (kappa * cosine)^2
KNN_KAPPAS = (1.0, 2.0, 4.0)  # candidate scales for leave-one-out selection
LOO_MIN_N = 5          # labels needed before leave-one-out selection is used
LOO_MARGIN = 0.0       # required absolute LOO Brier improvement to switch a component on
LOO_SE_MULT = 2.0      # ... plus this many standard errors of the per-item LOO difference
USE_LOO = True         # False: always-on components with default settings
RIDGE_MIN_N = 7        # minimum labels before the within-pair feature regression is used
RIDGE_LAMBDA = 4.0     # ridge penalty of that regression
RIDGE_CAP = 1.0        # cap (in logits) on its adjustment
PRIOR_LOGIT_CAP = 1.5  # cap on the budget-0 prior logit (p in [0.18, 0.82])
SUBJECT_SHRINK = 0.7   # multiplier on the subject part of the prior logit
# budget-0 prior coefficients (logits); hand-set, see the technical report
PRIOR_YEAR_SLOPE = 0.35      # per year of release date relative to mid-2024, clipped to +-2 years
PRIOR_EFFORT_LOW = 0.15      # reasoning_effort low / minimal
PRIOR_EFFORT_MID = 0.30      # any other non-empty reasoning_effort
PRIOR_EFFORT_HIGH = 0.45     # high / max / extended
PRIOR_SIZE_SLOPE = 0.25      # per doubling of the parameter count relative to 30B (clipped to +-4)
PRIOR_SMALL_NAME = -0.30     # name contains mini/nano/lite/small/haiku/tiny/flash (no explicit size)
PRIOR_LARGE_NAME = 0.20      # name contains pro/opus/ultra/large/max (no explicit size)
PRIOR_HARNESS = 0.10         # non-default harness
PRIOR_TIER_HARD = -0.60      # item_features tier/difficulty says hard/expert/...
PRIOR_TIER_EASY = 0.50       # ... easy/basic/...
PRIOR_TIER_NUM = 0.25        # per numeric level above 3 on a 1-5 scale
PRIOR_POINTS = 0.10          # per point above 1 (capped at 5)
PRIOR_LENGTH = 0.12          # per doubling of item length relative to 600 characters (clipped to +-3)
PRIOR_INTERCEPT = 0.0        # global shift of the prior logit (0 shipped: the knob to turn if the official
                             # formative feedback shows systematic over-/under-prediction, or the fitted
                             # intercept if a measurement-db fit is shipped; see fit_prior_from_measurement_db.py)
PRIOR_PROVIDER: dict = {}    # provider (lower-case) -> logit; empty unless fitted on measurement-db
TFIDF_BIGRAM_WEIGHT = 1.0  # weight of bigram terms relative to unigrams (0 = unigrams only)
TFIDF_RESTRICT_OOV = False  # drop target terms absent from the acquired set (query-side vocabulary)
CLIP_LO, CLIP_HI = 0.01, 0.99
MAX_TEXT_CHARS = 4000
_CACHE_LIMIT = 200000
_SET_CACHE_LIMIT = 4000

_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z])")
_KV_RE = re.compile(r"([a-z_ ]+)\s*[=:]\s*([^;,|]+)")
_DATE_RE = re.compile(r"(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?")
_SENT_RE = re.compile(r"[.!?]+")

_TOKEN_CACHE: dict = {}
_SET_CACHE: dict = {}

# Experiments only (never set in the shipped configuration): a callable text -> unit-norm
# dense vector (list of floats) or None.  When set, it replaces the TF-IDF representation in
# the similarity component so that pretrained sentence embeddings can be compared with TF-IDF
# under an otherwise identical predictor.
EMBED_LOOKUP = None


# ----------------------------------------------------------------------------- helpers
def _s(x) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _clip(p) -> float:
    try:
        p = float(p)
    except Exception:
        return 0.5
    if p != p or p in (float("inf"), float("-inf")):
        return 0.5
    return min(max(p, CLIP_LO), CLIP_HI)


def _year_fraction(date_text: str):
    m = _DATE_RE.search(_s(date_text))
    if not m:
        return None
    year = int(m.group(1))
    if year < 2015 or year > 2035:
        return None
    month = int(m.group(2) or 6)
    day = int(m.group(3) or 15)
    return year + (min(max(month, 1), 12) - 1) / 12.0 + (min(max(day, 1), 31) - 1) / 365.0


def _item_text(item: dict) -> str:
    content = _s(item.get("item_content"))[:MAX_TEXT_CHARS]
    feats = _s(item.get("item_features"))[:500]
    inter = _s(item.get("interactors"))[:300]
    return content + " \n " + feats + " \n " + inter


# ----------------------------------------------------------------------------- prior
# The prior logit is a sum of terms coefficient * value.  ``subject_terms`` and ``item_terms``
# return the ordered (coefficient name, value) pairs so that a coefficient table fitted on the
# official training release (fit_prior_from_measurement_db.py) can replace the hand-set
# constants one-to-one; the sums below are evaluated in the same order as the constants are
# listed, term by term.
SUBJECT_TERM_NAMES = ("PRIOR_YEAR_SLOPE", "PRIOR_EFFORT_LOW", "PRIOR_EFFORT_MID", "PRIOR_EFFORT_HIGH",
                      "PRIOR_SIZE_SLOPE", "PRIOR_SMALL_NAME", "PRIOR_LARGE_NAME", "PRIOR_HARNESS")
ITEM_TERM_NAMES = ("PRIOR_TIER_HARD", "PRIOR_TIER_EASY", "PRIOR_TIER_NUM", "PRIOR_POINTS", "PRIOR_LENGTH")
_EFFORT_OFF = ("none", "off", "0", "false", "no", "disabled", "n/a", "na", "-")
_EFFORT_LOW = ("low", "minimal", "min", "lowest")
_EFFORT_HIGH = ("high", "xhigh", "max", "maximum", "extended", "ultra", "highest")
_HARNESS_DEFAULT = ("default", "none", "null", "n/a", "na", "-")


def subject_terms(subject: dict) -> list:
    """Ordered (coefficient name, value) terms of the subject part of the prior logit."""
    terms = []
    yf = _year_fraction(subject.get("release_date"))
    if yf is not None:
        terms.append(("PRIOR_YEAR_SLOPE", min(max(yf - 2024.5, -2.0), 2.0)))
    effort = _s(subject.get("reasoning_effort")).strip().lower()
    if effort and effort not in _EFFORT_OFF:
        if effort in _EFFORT_LOW:
            terms.append(("PRIOR_EFFORT_LOW", 1.0))
        elif effort in _EFFORT_HIGH:
            terms.append(("PRIOR_EFFORT_HIGH", 1.0))
        else:
            terms.append(("PRIOR_EFFORT_MID", 1.0))
    name = _s(subject.get("normalized_name")).lower()
    sizes = [float(v) for v in _SIZE_RE.findall(name) if 0.05 <= float(v) <= 2000]
    if sizes:
        terms.append(("PRIOR_SIZE_SLOPE", min(max(math.log2(max(sizes) / 30.0), -4.0), 4.0)))
    else:
        if re.search(r"\b(mini|nano|lite|small|haiku|tiny|flash)\b", name):
            terms.append(("PRIOR_SMALL_NAME", 1.0))
        if re.search(r"\b(pro|opus|ultra|large|max)\b", name):
            terms.append(("PRIOR_LARGE_NAME", 1.0))
    harness = _s(subject.get("harness")).strip().lower()
    if harness and harness not in _HARNESS_DEFAULT:
        terms.append(("PRIOR_HARNESS", 1.0))
    return terms


def provider_key(subject: dict) -> str:
    """Lower-case provider string used to look up PRIOR_PROVIDER."""
    return _s(subject.get("provider")).strip().lower()


def item_terms(item: dict, content_length=None) -> list:
    """Ordered (coefficient name, value) terms of the item part of the prior logit.

    ``content_length`` may be given when the caller knows the length of ``item_content``
    without materialising the text (used when fitting on the training release).
    """
    terms = []
    feats = _s(item.get("item_features")).lower()
    if feats:
        pairs = _KV_RE.findall(feats)
        for key, val in pairs:
            key = key.strip()
            val = val.strip()
            if key in ("tier", "difficulty", "level", "hardness", "complexity"):
                if re.search(r"hard|expert|advanced|challeng|difficult|extreme|olympiad|phd|graduate", val):
                    terms.append(("PRIOR_TIER_HARD", 1.0))
                elif re.search(r"easy|basic|elementary|simple|trivial|beginner|intro", val):
                    terms.append(("PRIOR_TIER_EASY", 1.0))
                elif re.search(r"medium|intermediate|moderate|standard", val):
                    pass
                else:
                    nums = re.findall(r"\d+(?:\.\d+)?", val)
                    if nums:
                        num = float(nums[0])
                        if 1 <= num <= 5:
                            terms.append(("PRIOR_TIER_NUM", -(num - 3.0)))
            elif key in ("points", "score", "weight", "marks"):
                nums = re.findall(r"\d+(?:\.\d+)?", val)
                if nums:
                    terms.append(("PRIOR_POINTS", -min(max(float(nums[0]) - 1.0, 0.0), 5.0)))
        if not pairs:
            if re.search(r"hard|expert|advanced|challeng|difficult", feats):
                terms.append(("PRIOR_TIER_HARD", 0.67))
            elif re.search(r"easy|basic|elementary|simple", feats):
                terms.append(("PRIOR_TIER_EASY", 0.8))
    length = len(_s(item.get("item_content"))) if content_length is None else int(content_length)
    terms.append(("PRIOR_LENGTH", -min(max(math.log2(max(length, 50) / 600.0), -3.0), 3.0)))
    return terms


def _subject_logit(subject: dict) -> float:
    g = globals()
    z = 0.0
    for name, value in subject_terms(subject):
        z += g[name] * value
    z += PRIOR_PROVIDER.get(provider_key(subject), 0.0) if PRIOR_PROVIDER else 0.0
    return SUBJECT_SHRINK * z + PRIOR_INTERCEPT


def _item_logit(item: dict) -> float:
    g = globals()
    z = 0.0
    for name, value in item_terms(item):
        z += g[name] * value
    return z


def _prior_logit(subject: dict, item: dict) -> float:
    z = _subject_logit(subject) + _item_logit(item)
    return min(max(z, -PRIOR_LOGIT_CAP), PRIOR_LOGIT_CAP)


# ----------------------------------------------------------------------------- text
def _tokens(text: str) -> dict:
    key = hash(text)
    hit = _TOKEN_CACHE.get(key)
    if hit is not None and hit[0] == text[:64]:
        return hit[1]
    words = _TOKEN_RE.findall(text.lower())
    counts: dict = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    for a, b in zip(words, words[1:]):
        bg = a + "_" + b
        counts[bg] = counts.get(bg, 0) + 1
    if len(_TOKEN_CACHE) >= _CACHE_LIMIT:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[key] = (text[:64], counts)
    return counts


def _vectorize(counts: dict, df: dict, n: int) -> dict:
    vec = {}
    norm = 0.0
    for t, c in counts.items():
        d = df.get(t, 0)
        if TFIDF_RESTRICT_OOV and d == 0:
            continue
        v = (1.0 + math.log(c)) * (math.log((n + 1.0) / (d + 1.0)) + 1.0)
        if "_" in t:
            if TFIDF_BIGRAM_WEIGHT <= 0:
                continue
            v *= TFIDF_BIGRAM_WEIGHT
        vec[t] = v
        norm += v * v
    if norm > 0:
        inv = 1.0 / math.sqrt(norm)
        for t in vec:
            vec[t] *= inv
    return vec


def _cosine(a, b) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        if isinstance(a, dict) or isinstance(b, dict):
            return 0.0
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for t, v in a.items():
        w = b.get(t)
        if w is not None:
            s += v * w
    return max(0.0, min(1.0, s))


def _text_vec(text: str, stats: dict):
    """Representation of one item text: dense embedding if a lookup is set, else TF-IDF."""
    if EMBED_LOOKUP is not None:
        try:
            v = EMBED_LOOKUP(text)
        except Exception:
            v = None
        if v is not None:
            return list(v)
    return _vectorize(_tokens(text), stats["df"], stats["n"])


def similarities(target_item: dict, items: list) -> list:
    """Cosine TF-IDF similarity between the target item and each acquired item."""
    if not items:
        return []
    stats = _labeled_set_stats(items, None, None, None)
    tvec = _text_vec(_item_text(target_item), stats)
    return [_cosine(tvec, v) for v in stats["vecs"]]


# ----------------------------------------------------------------------------- features
def _features(item: dict) -> list:
    text = _s(item.get("item_content"))
    n = len(text)
    digits = sum(ch.isdigit() for ch in text)
    sents = len(_SENT_RE.findall(text))
    return [math.log(n + 1.0), digits / (n + 1.0), math.log1p(sents)]


def _solve(A: list, b: list):
    """Gaussian elimination for a small SPD system; returns None if singular."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f:
                    for cc in range(c, n + 1):
                        M[r][cc] -= f * M[c][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def _ridge_fit(Z: list, labels: list, offsets: list, k: int):
    beta = [0.0] * k
    for _ in range(6):
        grad = [-RIDGE_LAMBDA * b for b in beta]
        H = [[RIDGE_LAMBDA if i == j else 0.0 for j in range(k)] for i in range(k)]
        for row, y, off in zip(Z, labels, offsets):
            p = _sigmoid(off + sum(b * x for b, x in zip(beta, row)))
            w = p * (1.0 - p)
            for i in range(k):
                grad[i] += (y - p) * row[i]
                for j in range(k):
                    H[i][j] += w * row[i] * row[j]
        step = _solve(H, grad)
        if step is None:
            return None
        beta = [b + s for b, s in zip(beta, step)]
    return beta


def _ridge_adjust(beta, mean, sd, cols, feats: list) -> float:
    z = [min(max((feats[j] - mean[j]) / sd[j], -3.0), 3.0) for j in cols]
    adj = sum(b * x for b, x in zip(beta, z))
    return min(max(adj, -RIDGE_CAP), RIDGE_CAP)


# ----------------------------------------------------------------------------- labels
def _label_value(label):
    if isinstance(label, bool):
        return 1 if label else 0
    if isinstance(label, (int, float)):
        if label != label:
            return None
        return 1 if label >= 0.5 else 0
    text = _s(label).strip().lower()
    if text in ("1", "1.0", "true", "yes", "correct", "y", "t"):
        return 1
    if text in ("0", "0.0", "false", "no", "incorrect", "n", "f"):
        return 0
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError:
        return None


def pair_evidence(subject: dict, item: dict, labeled) -> tuple:
    """Return (items, labels) of acquired entries that belong to the target pair."""
    if not labeled:
        return [], []
    all_items, all_labels, same_items, same_labels = [], [], [], []
    bench = _s(item.get("benchmark_id")) if isinstance(item, dict) else ""
    name = _s(subject.get("normalized_name")) if isinstance(subject, dict) else ""
    for entry in labeled:
        try:
            pair, label = entry[0], entry[1]
            obs_subject, obs_item = pair[0], pair[1]
            if not isinstance(obs_item, dict):
                continue
            y = _label_value(label)
            if y is None:
                continue
            all_items.append(obs_item)
            all_labels.append(y)
            ok = True
            obs_bench = _s(obs_item.get("benchmark_id"))
            if bench and obs_bench and obs_bench != bench:
                ok = False
            if isinstance(obs_subject, dict):
                obs_name = _s(obs_subject.get("normalized_name"))
                if name and obs_name and obs_name != name:
                    ok = False
            if ok:
                same_items.append(obs_item)
                same_labels.append(y)
        except Exception:
            continue
    if same_items:
        return same_items, same_labels
    return all_items, all_labels


# ----------------------------------------------------------------------------- labeled-set statistics
def _knn_correction(weights: list, labels: list, anchor: float, n: int) -> float:
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0
    resid = sum(w * (y - anchor) for w, y in zip(weights, labels))
    return (n / (n + KNN_N0)) * resid / (TAU + wsum)


def _labeled_set_stats(items: list, labels, subject_logit, item_logits) -> dict:
    """Everything that depends only on the acquired set (not on the target), cached.

    Includes TF-IDF statistics, the ridge fit, and the leave-one-out selection of the
    kernel scale and of the ridge component.  ``labels`` may be None when only the text
    statistics are needed (acquisition policy).
    """
    texts = [_item_text(it) for it in items]
    key = (tuple(hash(t) for t in texts), tuple(labels) if labels is not None else None,
           round(subject_logit, 6) if subject_logit is not None else None, EMBED_LOOKUP is None,
           TFIDF_BIGRAM_WEIGHT, TFIDF_RESTRICT_OOV)
    hit = _SET_CACHE.get(key)
    if hit is not None and hit["first"] == (texts[0][:64] if texts else ""):
        return hit
    n = len(texts)
    docs = [_tokens(t) for t in texts]
    df: dict = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    rec = {"first": texts[0][:64] if texts else "", "n": n, "df": df,
           "kappa": KNN_KAPPA, "ridge": None, "use_ridge": False, "loo": None}
    vecs = [_text_vec(t, rec) for t in texts]
    rec["vecs"] = vecs
    if labels is not None and n > 0:
        mean_item_logit = sum(item_logits) / n
        k_succ = sum(labels)
        # pair-level logits (full set) and leave-one-out versions
        prior_rate = _sigmoid(min(max(subject_logit + mean_item_logit, -PRIOR_LOGIT_CAP), PRIOR_LOGIT_CAP))
        rate_logit = _logit((ALPHA0 * prior_rate + k_succ) / (ALPHA0 + n))
        offsets = [rate_logit + (zl - mean_item_logit) for zl in item_logits]
        # ridge fit on all acquired items
        feats = [_features(it) for it in items]
        k = len(feats[0])
        mean = [sum(row[j] for row in feats) / n for j in range(k)]
        sd = [math.sqrt(sum((row[j] - mean[j]) ** 2 for row in feats) / n) for j in range(k)]
        cols = [j for j in range(k) if sd[j] > 1e-9]
        ridge = None
        if n >= RIDGE_MIN_N and cols:
            Z = [[(row[j] - mean[j]) / sd[j] for j in cols] for row in feats]
            beta = _ridge_fit(Z, labels, offsets, len(cols))
            if beta is not None:
                ridge = (beta, mean, sd, cols)
        rec["ridge"] = ridge
        rec["use_ridge"] = ridge is not None
        if USE_LOO and n >= LOO_MIN_N:
            # leave-one-out predictions for every component configuration
            sims = [[_cosine(vecs[i], vecs[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
            loo_pair = []
            for j in range(n):
                m_j = (sum(item_logits) - item_logits[j]) / (n - 1)
                pr_j = _sigmoid(min(max(subject_logit + m_j, -PRIOR_LOGIT_CAP), PRIOR_LOGIT_CAP))
                rl_j = _logit((ALPHA0 * pr_j + k_succ - labels[j]) / (ALPHA0 + n - 1))
                loo_pair.append(rl_j + (item_logits[j] - m_j))
            briers = {}
            # ridge leave-one-out adjustments
            loo_ridge_adj = [0.0] * n
            if ridge is not None:
                for j in range(n):
                    idx = [i for i in range(n) if i != j]
                    Zj = [[(feats[i][c] - mean[c]) / sd[c] for c in cols] for i in idx]
                    beta_j = _ridge_fit(Zj, [labels[i] for i in idx], [loo_pair[j] if i == j else offsets[i] for i in idx],
                                        len(cols))
                    if beta_j is not None:
                        loo_ridge_adj[j] = _ridge_adjust(beta_j, mean, sd, cols, feats[j])
            sq_errors = {}
            for use_ridge in ((False, True) if ridge is not None else (False,)):
                for kappa in (0.0,) + tuple(KNN_KAPPAS):
                    errs = []
                    for j in range(n):
                        z = loo_pair[j] + (loo_ridge_adj[j] if use_ridge else 0.0)
                        p = _sigmoid(z)
                        if kappa > 0:
                            w = [(kappa * sims[j][i]) ** 2 if i != j else 0.0 for i in range(n)]
                            corr = _knn_correction([w[i] for i in range(n) if i != j],
                                                   [labels[i] for i in range(n) if i != j], p, n - 1)
                            p = _clip(p + corr)
                        errs.append((p - labels[j]) ** 2)
                    sq_errors[(use_ridge, kappa)] = errs
                    briers[(use_ridge, kappa)] = sum(errs) / n
            base = briers[(False, 0.0)]
            best = min(briers, key=lambda cfg: (briers[cfg], cfg[0], cfg[1]))
            # switch away from the plain shrinkage estimate only if the leave-one-out
            # improvement is clear: mean paired difference below -(margin + mult * SE)
            diffs = [a - b for a, b in zip(sq_errors[best], sq_errors[(False, 0.0)])]
            mean_d = sum(diffs) / n
            var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
            se_d = math.sqrt(var_d / n)
            if mean_d >= -(LOO_MARGIN + LOO_SE_MULT * se_d):
                best = (False, 0.0)
            rec["use_ridge"], rec["kappa"] = best
            rec["loo"] = {"base": base, "best": briers[best], "config": best, "mean_diff": mean_d, "se_diff": se_d}
    if len(_SET_CACHE) >= _SET_CACHE_LIMIT:
        _SET_CACHE.clear()
    _SET_CACHE[key] = rec
    return rec


# ----------------------------------------------------------------------------- predictor
def predict_components(input, labeled=None) -> dict:
    """Return intermediate estimates (used by the experiments for ablations).

    Keys: prior, pair (shrinkage only), ridge (shrinkage + always-on ridge), knn
    (shrinkage + always-on kNN at the default scale), full (LOO-gated combination).
    """
    subject, item = input[0], input[1]
    if not isinstance(subject, dict):
        subject = {}
    if not isinstance(item, dict):
        item = {}
    items, labels = pair_evidence(subject, item, labeled)
    n = len(labels)
    subject_logit = _subject_logit(subject)
    target_item_logit = _item_logit(item)
    prior = _sigmoid(min(max(subject_logit + target_item_logit, -PRIOR_LOGIT_CAP), PRIOR_LOGIT_CAP))
    out = {"n": n, "prior": prior}
    if n == 0:
        out.update(pair=prior, ridge=prior, knn=prior, full=prior)
        return out
    item_logits = [_item_logit(it) for it in items]
    mean_item_logit = sum(item_logits) / n
    prior_rate = _sigmoid(min(max(subject_logit + mean_item_logit, -PRIOR_LOGIT_CAP), PRIOR_LOGIT_CAP))
    rate_logit = _logit((ALPHA0 * prior_rate + sum(labels)) / (ALPHA0 + n))
    z_pair = rate_logit + (target_item_logit - mean_item_logit)
    p_pair = _sigmoid(z_pair)
    out["pair"] = p_pair
    stats = _labeled_set_stats(items, labels, subject_logit, item_logits)
    tvec = _text_vec(_item_text(item), stats)
    sims = [_cosine(tvec, v) for v in stats["vecs"]]
    tfeat = _features(item)
    ridge_adj = _ridge_adjust(*stats["ridge"], tfeat) if stats["ridge"] is not None else 0.0
    p_ridge = _sigmoid(z_pair + ridge_adj)
    out["ridge"] = p_ridge
    w_default = [(KNN_KAPPA * s_) ** 2 for s_ in sims]
    out["knn"] = _clip(p_pair + _knn_correction(w_default, labels, p_pair, n))
    # LOO-gated full model
    anchor = _sigmoid(z_pair + (ridge_adj if stats["use_ridge"] else 0.0))
    kappa = stats["kappa"]
    if USE_LOO and n < LOO_MIN_N:
        kappa = 0.0  # too few labels to test whether similarity helps: shrinkage only
    if kappa > 0:
        w = [(kappa * s_) ** 2 for s_ in sims]
        out["full"] = _clip(anchor + _knn_correction(w, labels, anchor, n))
    else:
        out["full"] = _clip(anchor)
    out["config"] = {"use_ridge": stats["use_ridge"], "kappa": kappa}
    return out


def predict(input, labeled=None) -> float:
    """Probability that the subject answers the item correctly.  Never raises."""
    try:
        return _clip(predict_components(input, labeled)["full"])
    except Exception:
        pass
    try:  # fallback 1: shrinkage only
        subject, item = input[0], input[1]
        subject = subject if isinstance(subject, dict) else {}
        item = item if isinstance(item, dict) else {}
        items, labels = pair_evidence(subject, item, labeled)
        z = _prior_logit(subject, item)
        if labels:
            n = len(labels)
            rate = (ALPHA0 * _sigmoid(z) + sum(labels)) / (ALPHA0 + n)
            return _clip(rate)
        return _clip(_sigmoid(z))
    except Exception:
        pass
    try:  # fallback 2: smoothed empirical mean
        k = sum(1 for e in labeled if _label_value(e[1]) == 1)
        n = sum(1 for e in labeled if _label_value(e[1]) is not None)
        return _clip((ALPHA0 * 0.5 + k) / (ALPHA0 + n))
    except Exception:
        return 0.5
