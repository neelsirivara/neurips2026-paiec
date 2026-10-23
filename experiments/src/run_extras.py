"""Extra ablations that reuse run_experiments: TF-IDF representation variants.

    python run_extras.py --workers 2

Writes experiments/results/tfidf_variants_{synthetic,real}.json: the always-on kNN and the
gated full model under (a) the shipped representation (unigrams + bigrams, no query-side
vocabulary restriction) and (b) a sharper unigram-only representation whose target terms are
restricted to the acquired vocabulary, with paired differences to (a).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import run_experiments as R  # noqa: E402
import simulate  # noqa: E402

VARIANTS = {
    "shipped_uni_bi": {},
    "unigram_restricted": {"TFIDF_BIGRAM_WEIGHT": 0.0, "TFIDF_RESTRICT_OOV": True},
    "uni_bi_restricted": {"TFIDF_RESTRICT_OOV": True},
}


def run(raw, seed, workers, key):
    out = {key: {}}
    per_pair = {}
    for name, params in VARIANTS.items():
        s = R.summarize(R.run_dataset(raw, seed=seed, params=params, workers=workers))
        per_pair[name] = {m: s["methods"][m]["per_pair_alc"] for m in ("shrink", "shrink_knn", "full")}
        out[key][name] = {m: {"alc": s["methods"][m]["alc"], "brier_by_budget": s["methods"][m]["brier_by_budget"]}
                          for m in ("shrink", "shrink_knn", "full")}
        R.headline(s, f"{key} tfidf {name}")
    out[key]["paired_vs_shipped"] = {
        name: {m: harness.paired_bootstrap(per_pair[name][m], per_pair["shipped_uni_bi"][m], R.N_BOOT, seed=8)
               for m in ("shrink_knn", "full")} for name in VARIANTS if name != "shipped_uni_bi"}
    out[key]["paired_knn_vs_shrink"] = {
        name: harness.paired_bootstrap(per_pair[name]["shrink_knn"], per_pair[name]["shrink"], R.N_BOOT, seed=9) for name in VARIANTS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--suite", default="all", choices=["synthetic", "real", "all"])
    a = ap.parse_args()
    if a.suite in ("synthetic", "all"):
        pairs = simulate.generate_pairs(0, R.N_SUBJECTS, R.N_BENCHMARKS, R.MAIN_SIGMA, R.MAIN_GAMMA)
        R.dump(run(pairs, 0, a.workers, "synthetic"), "tfidf_variants_synthetic.json")
    if a.suite in ("real", "all"):
        import real_testbed
        raw = real_testbed.load_pairs()
        if raw:
            R.dump(run(raw, 0, a.workers, "real"), "tfidf_variants_real.json")


if __name__ == "__main__":
    main()
