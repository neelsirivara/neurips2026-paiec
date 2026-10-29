"""W2 item layer: featurisation shared verbatim by the simulator and the deployed model.py (standard library only).

parse_features(item_features) -> list of (key, value) categories (at most FEAT_MAX_KEYS).
text_grams(item_content) -> {bucket: sublinear tf} of hashed character n-grams (unnormalised, no idf).
"""
import math
import re
import zlib

FEAT_MAX_KEYS = 8          # categories per item (first keys in the string's order)
FEAT_MAX_VALUE = 80        # characters of a value kept
FEAT_MAX_CHARS = 1000      # characters of item_features parsed
GRAM_N = 4                 # character n-gram length
GRAM_BITS = 18             # 2^18 hashed buckets
GRAM_HEAD = 1500           # characters from the start of the normalised text
GRAM_TAIL = 500            # ... plus this many from the end (questions often follow a long context)
_WS = re.compile(r"\s+")


def parse_features(feats) -> list:
    """'k1=v1;k2=v2' -> [('k1', 'v1'), ('k2', 'v2')]; a string without '=' is one category ('_raw', text)."""
    try:
        s = str(feats or "")[:FEAT_MAX_CHARS].strip()
    except Exception:
        return []
    if not s:
        return []
    out = []
    seen = set()
    if "=" not in s:
        return [("_raw", _WS.sub(" ", s.lower())[:FEAT_MAX_VALUE])]
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = _WS.sub(" ", k.strip().lower())[:40]
        v = _WS.sub(" ", v.strip().lower())[:FEAT_MAX_VALUE]
        if not k or k in seen:
            continue
        seen.add(k)
        out.append((k, v))
        if len(out) >= FEAT_MAX_KEYS:
            break
    return out


def norm_text(content) -> str:
    try:
        t = _WS.sub(" ", str(content or "").lower()).strip()
    except Exception:
        return ""
    if len(t) > GRAM_HEAD + GRAM_TAIL:
        t = t[:GRAM_HEAD] + " | " + t[-GRAM_TAIL:]
    return t


def text_grams(content) -> dict:
    """Hashed character n-grams of the normalised text with sublinear tf (1 + log count)."""
    t = norm_text(content)
    mask = (1 << GRAM_BITS) - 1
    counts: dict = {}
    for i in range(max(len(t) - GRAM_N + 1, 0)):
        h = zlib.crc32(t[i:i + GRAM_N].encode("utf-8")) & mask
        counts[h] = counts.get(h, 0) + 1
    return {h: 1.0 + math.log(c) for h, c in counts.items()}


EMB_HEAD = 600             # embedding text: the normalised text if <= 1,000 characters, else head 600 + tail 400
EMB_TAIL = 400


def embed_text(content) -> str:
    """Text given to the sentence-embedding model (lower-cased, whitespace-normalised, head + tail)."""
    t = norm_text(content)
    if len(t) > EMB_HEAD + EMB_TAIL:
        t = t[:EMB_HEAD] + " ... " + t[-EMB_TAIL:]
    return t
