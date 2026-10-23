"""Synthetic subject x benchmark x item generator (2-parameter IRT with topic clusters).

Subjects have an ability theta that correlates with visible attributes (release date,
reasoning effort, parameter count).  Each benchmark has a location mu_b, a discrimination
a_b, and K topic clusters with difficulty offsets c_k ~ N(0, sigma_topic^2); item difficulty
is b_i = mu_b + c_k(i) + gamma * z_len(i) + eps_i.  P(correct) = sigmoid(a_b (theta_s - b_i)).
Item text mixes benchmark template words, cluster-specific vocabulary, ambient vocabulary
and numbers, so TF-IDF similarity carries information about the cluster (hence difficulty)
only when sigma_topic > 0.  Half of the benchmarks expose a tier label in item_features
derived from the item-difficulty tercile.
"""
from __future__ import annotations

import math
import random

PROVIDERS = ["Acme AI", "Borealis", "Cerulean Labs", "Delta Systems"]
EFFORTS = ["", "", "low", "medium", "high"]
TEMPLATE_WORDS = ["question", "answer", "following", "choose", "correct", "option", "given", "which",
                  "statement", "true", "explain", "determine", "consider", "task", "provide", "final"]
_SYL_C = "b c d f g h j k l m n p r s t v z".split()
_SYL_V = "a e i o u".split()


def pseudo_word(rng: random.Random) -> str:
    """Pronounceable alphabetic pseudo-word (tokeniser-compatible: letters only)."""
    return "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(rng.randint(3, 4)))


def pseudo_vocab(rng: random.Random, n: int, taken: set) -> list:
    out = []
    while len(out) < n:
        w = pseudo_word(rng)
        if w not in taken:
            taken.add(w)
            out.append(w)
    return out


_AMBIENT_RNG = random.Random(12345)
_TAKEN: set = set(TEMPLATE_WORDS) if "TEMPLATE_WORDS" in globals() else set()
AMBIENT = pseudo_vocab(_AMBIENT_RNG, 400, _TAKEN)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def zipf_sample(rng: random.Random, pool: list, k: int) -> list:
    """Sample k words from pool with a Zipf-like distribution (rank^-1)."""
    weights = [1.0 / (r + 1) for r in range(len(pool))]
    return rng.choices(pool, weights=weights, k=k)


def make_subjects(rng: random.Random, n: int, attr_signal: float = 1.0) -> list:
    subjects = []
    for s in range(n):
        year = rng.uniform(2023.0, 2026.5)
        effort = rng.choice(EFFORTS)
        size = rng.choice([None, None, 7, 32, 70, 400])
        theta = (attr_signal * (0.6 * (year - 2024.75) + (0.4 if effort else 0.0)
                                + (0.2 * math.log2(size / 30.0) if size else 0.0))
                 + rng.gauss(0, 0.8))
        name = f"model-{s:02d}" + (f"-{size}b" if size else "")
        y = int(year)
        m = int((year - y) * 12) + 1
        subject = {"normalized_name": name, "provider": rng.choice(PROVIDERS),
                   "release_date": f"{y:04d}-{m:02d}-15", "access_date": "2026-09-01",
                   "harness": rng.choice(["default", "default", "agent-scaffold-v2"]),
                   "harness_version": "", "reasoning_effort": effort, "subject_features_extra": ""}
        subjects.append({"subject": subject, "theta": theta, "subject_id": f"subject_{s:03d}"})
    return subjects


def make_benchmark(rng: random.Random, b: int, sigma_topic: float, gamma: float, n_items: int,
                   k_clusters: int = 5, expose_tier: bool = False) -> dict:
    mu = rng.gauss(0.0, 1.0)
    a = math.exp(rng.gauss(0.0, 0.3))
    taken = set(AMBIENT) | set(TEMPLATE_WORDS)
    template = rng.sample(TEMPLATE_WORDS, 5) + pseudo_vocab(rng, 3, taken)
    clusters = []
    for k in range(k_clusters):
        vocab = pseudo_vocab(rng, 40, taken)
        clusters.append({"offset": rng.gauss(0.0, sigma_topic), "vocab": vocab})
    items = []
    lens = [rng.lognormvariate(math.log(45), 0.5) for _ in range(n_items)]
    mean_log = sum(math.log(x) for x in lens) / n_items
    sd_log = math.sqrt(sum((math.log(x) - mean_log) ** 2 for x in lens) / n_items) or 1.0
    for i in range(n_items):
        k = rng.randrange(k_clusters)
        n_words = max(8, int(lens[i]))
        z_len = (math.log(lens[i]) - mean_log) / sd_log
        difficulty = mu + clusters[k]["offset"] + gamma * z_len + rng.gauss(0, 0.5)
        n_cluster = max(2, int(0.35 * n_words))
        n_template = max(2, int(0.2 * n_words))
        n_ambient = max(1, n_words - n_cluster - n_template - 2)
        words = (zipf_sample(rng, clusters[k]["vocab"], n_cluster) + rng.choices(template, k=n_template)
                 + zipf_sample(rng, AMBIENT, n_ambient) + [str(rng.randint(1, 999)) for _ in range(2)])
        rng.shuffle(words)
        text = " ".join(words).capitalize() + "?"
        items.append({"item_content": text, "item_features": "", "interactors": "",
                      "difficulty": difficulty, "cluster": k})
    if expose_tier:
        sorted_d = sorted(it["difficulty"] for it in items)
        lo, hi = sorted_d[n_items // 3], sorted_d[2 * n_items // 3]
        for it in items:
            tier = "easy" if it["difficulty"] < lo else ("hard" if it["difficulty"] >= hi else "medium")
            it["item_features"] = f"tier={tier}"
    return {"benchmark_id": f"bench_{b:03d}", "mu": mu, "a": a, "items": items}


def generate_pairs(seed: int, n_subjects: int = 10, n_benchmarks: int = 16, sigma_topic: float = 0.7,
                   gamma: float = 0.3, items_range=(100, 240), attr_signal: float = 1.0) -> list:
    """Return list of raw pairs: (subject_dict, subject_id, benchmark_id, [(item, label, true_p), ...])."""
    rng = random.Random(seed)
    subjects = make_subjects(rng, n_subjects, attr_signal)
    pairs = []
    for b in range(n_benchmarks):
        n_items = rng.randint(*items_range)
        bench = make_benchmark(rng, b, sigma_topic, gamma, n_items, expose_tier=(b % 2 == 0))
        for subj in subjects:
            items = []
            for it in bench["items"]:
                p = _sigmoid(bench["a"] * (subj["theta"] - it["difficulty"]))
                y = 1 if rng.random() < p else 0
                item = {"item_content": it["item_content"], "item_features": it["item_features"],
                        "interactors": it["interactors"]}
                items.append((item, y, p))  # p = true probability, used only by the oracle rows
            pairs.append((dict(subj["subject"]), subj["subject_id"], bench["benchmark_id"], items))
    return pairs
