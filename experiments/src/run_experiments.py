"""Run all local experiments and write JSON results to experiments/results/.

    python run_experiments.py --suite synthetic      # main synthetic study + sweeps
    python run_experiments.py --suite real           # real testbed (needs real_testbed/ files)
    python run_experiments.py --suite all

Every experiment reuses one acquisition trajectory per pair for all predictors (paired
comparison) and reports Brier ALC with bootstrap 95% CIs over pairs (1000 resamples).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import labeling  # noqa: E402
import model  # noqa: E402
import simulate  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000
MAIN_SIGMA, MAIN_GAMMA = 0.7, 0.3
N_SUBJECTS, N_BENCHMARKS = 10, 16


# ----------------------------------------------------------------------------- one pair
def _set_params(params: dict | None):
    if not params:
        return
    for k, v in params.items():
        if k == "POLICY":
            labeling.POLICY = v
        elif k == "EMBED_NPZ":
            import embed_cache
            model.EMBED_LOOKUP = embed_cache.load_embed_lookup(v) if v else None
        else:
            setattr(model, k, v)


def eval_one_pair(args) -> dict:
    """(raw_pair, seed, policy_name, params) -> per-pair record. Runs in a worker process."""
    raw, seed, policy_name, params = args
    subject, subject_id, benchmark_id, items = raw[0], raw[1], raw[2], raw[3]
    _set_params(params)
    rng = random.Random(f"{seed}|{subject_id}|{benchmark_id}")
    pair = harness.make_pair(dict(subject), subject_id, benchmark_id, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
    if pair is None:
        return None
    if policy_name == "random":
        policy = harness.random_policy_factory(random.Random(f"acq|{seed}|{subject_id}|{benchmark_id}"))
        acquired = harness.run_acquisition(pair, policy, pair["order"], needs_prediction=False)
    else:
        labeling.POLICY = policy_name
        needs = policy_name in ("ble", "novelty_unc")
        acquired = harness.run_acquisition(pair, labeling.acquisition_function, pair["order"],
                                           predictor=model.predict, needs_prediction=needs)
    methods = dict(harness.DEFAULT_METHODS)
    res = harness.evaluate_pair(pair, acquired, methods)
    labels = res.pop("_labels")
    rec = {"subject_id": subject_id, "benchmark_id": benchmark_id, "n_acq_pool": len(pair["acq"]),
           "n_eval": len(pair["eval"]), "eval_rate": sum(labels) / len(labels),
           "acq_rate": sum(y for _, y in pair["acq"]) / len(pair["acq"]),
           "n_acquired": len(acquired), "methods": {}}
    for name, r in res.items():
        rec["methods"][name] = {"alc": r["alc"], "brier": {str(b): v for b, v in r["brier"].items()},
                                "preds": {str(b): [round(p, 4) for p in v] for b, v in r["preds"].items()}}
    rec["labels"] = labels
    return rec


def run_dataset(raw_pairs: list, seed: int, policy_name: str = "random", params: dict | None = None,
                workers: int = 4) -> list:
    jobs = [(raw[:4], seed, policy_name, params) for raw in raw_pairs]
    if workers <= 1:
        recs = [eval_one_pair(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(eval_one_pair, jobs, chunksize=2))
    return [r for r in recs if r is not None]


# ----------------------------------------------------------------------------- summaries
def summarize(recs: list, keep_preds: bool = False) -> dict:
    names = list(recs[0]["methods"].keys())
    out = {"n_pairs": len(recs), "methods": {}, "paired": {}}
    for name in names:
        alcs = [r["methods"][name]["alc"] for r in recs]
        briers = {b: sum(r["methods"][name]["brier"][b] for r in recs) / len(recs)
                  for b in recs[0]["methods"][name]["brier"]}
        ece_by_budget = {}
        for b in briers:
            preds = [p for r in recs for p in r["methods"][name]["preds"][b]]
            labels = [y for r in recs for y in r["labels"]]
            ece_by_budget[b] = harness.ece(preds, labels)
        pooled_preds = [p for r in recs for b in briers for p in r["methods"][name]["preds"][b]]
        pooled_labels = [y for r in recs for b in briers for y in r["labels"]]
        out["methods"][name] = {"alc": harness.bootstrap_mean_ci(alcs, N_BOOT, seed=1),
                                "brier_by_budget": briers, "ece_by_budget": ece_by_budget,
                                "ece_pooled": harness.ece(pooled_preds, pooled_labels),
                                "per_pair_alc": alcs}
    comparisons = [("full", "const"), ("full", "empirical_mean"), ("full", "shrink"), ("full", "shrink_knn"),
                   ("full", "shrink_ridge"), ("shrink", "const"), ("shrink", "empirical_mean"),
                   ("shrink_knn", "shrink"), ("shrink_ridge", "shrink"), ("prior", "const"),
                   ("shrink", "prior"), ("empirical_mean", "const")]
    for a, b in comparisons:
        if a in out["methods"] and b in out["methods"]:
            out["paired"][f"{a}_vs_{b}"] = harness.paired_bootstrap(out["methods"][a]["per_pair_alc"],
                                                                     out["methods"][b]["per_pair_alc"], N_BOOT, seed=2)
    # per-budget paired difference full vs const and full vs shrink
    for a, b in (("full", "const"), ("full", "shrink"), ("shrink", "const")):
        out["paired"][f"{a}_vs_{b}_by_budget"] = {
            bud: harness.paired_bootstrap([r["methods"][a]["brier"][bud] for r in recs],
                                          [r["methods"][b]["brier"][bud] for r in recs], N_BOOT, seed=3)
            for bud in recs[0]["methods"][a]["brier"]}
    if not keep_preds:
        for r in recs:
            for m in r["methods"].values():
                m.pop("preds", None)
            r.pop("labels", None)
    out["pairs"] = recs
    return out


def dump(obj: dict, name: str) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def headline(summary: dict, label: str) -> None:
    print(f"--- {label} (n_pairs={summary['n_pairs']})")
    for name, m in summary["methods"].items():
        a = m["alc"]
        print(f"  {name:16s} ALC {a['mean']:.4f} [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]  ECE {m['ece_pooled']:.3f}  "
              + " ".join(f"B{b}={v:.3f}" for b, v in m["brier_by_budget"].items()))


def loo_margin_sensitivity(raw_pairs: list, base_summary: dict, seed: int, workers: int, key: str) -> dict:
    """Full-model ALC when the leave-one-out switch requires 0, 1 (default) or 2 standard errors."""
    out = {key: {}}
    for mult in (0.0, 1.0, 2.0):
        if mult == model.LOO_SE_MULT:
            s = base_summary
        else:
            s = summarize(run_dataset(raw_pairs, seed=seed, params={"LOO_SE_MULT": mult}, workers=workers))
        out[key][str(mult)] = {"full": {"alc": s["methods"]["full"]["alc"], "ece_pooled": s["methods"]["full"]["ece_pooled"]},
                               "paired_full_vs_shrink": harness.paired_bootstrap(
                                   s["methods"]["full"]["per_pair_alc"], s["methods"]["shrink"]["per_pair_alc"], N_BOOT, seed=7)}
        print(f"LOO_SE_MULT={mult}: full {s['methods']['full']['alc']['mean']:.4f}")
    return out


# ----------------------------------------------------------------------------- suites
def suite_synthetic(workers: int) -> None:
    t0 = time.time()
    main_pairs = simulate.generate_pairs(0, N_SUBJECTS, N_BENCHMARKS, MAIN_SIGMA, MAIN_GAMMA)
    recs = run_dataset(main_pairs, seed=0, workers=workers)
    summ = summarize(recs)
    summ["config"] = {"seed": 0, "n_subjects": N_SUBJECTS, "n_benchmarks": N_BENCHMARKS, "sigma_topic": MAIN_SIGMA,
                      "gamma_length": MAIN_GAMMA, "policy": "random", "alpha0": model.ALPHA0, "tau": model.TAU,
                      "knn_n0": model.KNN_N0, "knn_kappa": model.KNN_KAPPA, "ridge_lambda": model.RIDGE_LAMBDA,
                      "loo_se_mult": model.LOO_SE_MULT, "loo_min_n": model.LOO_MIN_N, "ridge_min_n": model.RIDGE_MIN_N}
    headline(summ, "synthetic main")
    dump(summ, "synthetic_main.json")

    # replication over generator seeds
    rep = {"seeds": {}}
    for seed in (1, 2):
        pairs = simulate.generate_pairs(seed, N_SUBJECTS, N_BENCHMARKS, MAIN_SIGMA, MAIN_GAMMA)
        s = summarize(run_dataset(pairs, seed=seed, workers=workers))
        s.pop("pairs")
        rep["seeds"][str(seed)] = {k: {"alc": v["alc"], "ece_pooled": v["ece_pooled"]} for k, v in s["methods"].items()}
        rep["seeds"][str(seed)]["paired"] = {k: v for k, v in s["paired"].items() if not k.endswith("by_budget")}
        headline(s, f"synthetic seed {seed}")
    dump(rep, "synthetic_replication.json")

    # text-signal sweep
    sweep = {"settings": []}
    for sigma, gamma in ((0.0, 0.0), (0.0, 0.3), (0.35, 0.3), (0.7, 0.3), (1.0, 0.3), (0.7, 0.0)):
        pairs = simulate.generate_pairs(0, N_SUBJECTS, N_BENCHMARKS, sigma, gamma)
        s = summarize(run_dataset(pairs, seed=0, workers=workers))
        s.pop("pairs")
        sweep["settings"].append({"sigma_topic": sigma, "gamma_length": gamma,
                                  "methods": {k: {"alc": v["alc"], "brier_by_budget": v["brier_by_budget"],
                                                  "ece_pooled": v["ece_pooled"]} for k, v in s["methods"].items()},
                                  "paired": {k: v for k, v in s["paired"].items() if not k.endswith("by_budget")}})
        headline(s, f"sweep sigma={sigma} gamma={gamma}")
    dump(sweep, "synthetic_signal_sweep.json")

    # alpha0 sensitivity
    alpha = {"synthetic": {}}
    for a0 in (1.0, 2.0, 4.0, 8.0):
        s = summarize(run_dataset(main_pairs, seed=0, params={"ALPHA0": a0}, workers=workers))
        alpha["synthetic"][str(a0)] = {k: {"alc": v["alc"]} for k, v in s["methods"].items()}
        print(f"alpha0={a0}: shrink {s['methods']['shrink']['alc']['mean']:.4f} full {s['methods']['full']['alc']['mean']:.4f}")
    dump(alpha, "alpha0_sensitivity_synthetic.json")

    # leave-one-out gating ablation (always-on components at default settings)
    s_off = summarize(run_dataset(main_pairs, seed=0, params={"USE_LOO": False}, workers=workers))
    loo = {"synthetic": {"gated": {k: {"alc": v["alc"], "ece_pooled": v["ece_pooled"]} for k, v in summ["methods"].items()},
                         "always_on": {k: {"alc": v["alc"], "ece_pooled": v["ece_pooled"]} for k, v in s_off["methods"].items()},
                         "paired_full_gated_vs_always_on": harness.paired_bootstrap(
                             summ["methods"]["full"]["per_pair_alc"], s_off["methods"]["full"]["per_pair_alc"], N_BOOT, seed=5)}}
    headline(s_off, "synthetic LOO off")
    dump(loo, "loo_gating_ablation_synthetic.json")
    dump(loo_margin_sensitivity(main_pairs, summ, 0, workers, "synthetic"), "loo_margin_sensitivity_synthetic.json")

    # kNN hyper-parameter grid (smaller generator)
    grid_pairs = simulate.generate_pairs(3, 8, 10, MAIN_SIGMA, MAIN_GAMMA)
    grid = {"grid": []}
    for kappa in (1.0, 2.0, 4.0):
        for tau in (1.0, 2.0, 4.0):
            s = summarize(run_dataset(grid_pairs, seed=3, params={"KNN_KAPPA": kappa, "TAU": tau}, workers=workers))
            grid["grid"].append({"kappa": kappa, "tau": tau,
                                 "full": s["methods"]["full"]["alc"], "shrink_knn": s["methods"]["shrink_knn"]["alc"],
                                 "shrink": s["methods"]["shrink"]["alc"]})
            print(f"kappa={kappa} tau={tau}: full {s['methods']['full']['alc']['mean']:.4f}")
    dump(grid, "knn_hparam_grid_synthetic.json")

    # acquisition policies (full predictor)
    pol = {"synthetic": {}}
    per_pair = {}
    for policy in ("random", "first", "novelty", "novelty_unc", "ble"):
        s = summarize(run_dataset(main_pairs, seed=0, policy_name=policy, workers=workers))
        per_pair[policy] = {k: s["methods"][k]["per_pair_alc"] for k in ("full", "shrink", "empirical_mean")}
        pol["synthetic"][policy] = {k: {"alc": v["alc"], "brier_by_budget": v["brier_by_budget"]}
                                    for k, v in s["methods"].items() if k in ("full", "shrink", "empirical_mean")}
        headline(s, f"policy {policy}")
    pol["synthetic"]["paired_vs_random"] = {
        policy: {k: harness.paired_bootstrap(per_pair[policy][k], per_pair["random"][k], N_BOOT, seed=4)
                 for k in ("full", "shrink", "empirical_mean")} for policy in per_pair if policy != "random"}
    dump(pol, "acquisition_policies_synthetic.json")
    print(f"synthetic suite done in {time.time() - t0:.0f}s")


def suite_real(workers: int) -> None:
    import real_testbed
    raw = real_testbed.load_pairs()
    if not raw:
        print("no real testbed pairs found")
        return
    meta = [p[4] for p in raw]
    dump({"pairs": meta, "n_pairs": len(meta)}, "real_testbed_pairs.json")
    recs = run_dataset(raw, seed=0, workers=workers)
    summ = summarize(recs)
    summ["config"] = {"seed": 0, "policy": "random", "alpha0": model.ALPHA0, "pairs": meta}
    headline(summ, "real testbed")
    dump(summ, "real_testbed_main.json")

    # leave-one-out gating ablation
    s_off = summarize(run_dataset(raw, seed=0, params={"USE_LOO": False}, workers=workers))
    loo = {"real": {"gated": {k: {"alc": v["alc"], "ece_pooled": v["ece_pooled"]} for k, v in summ["methods"].items()},
                    "always_on": {k: {"alc": v["alc"], "ece_pooled": v["ece_pooled"]} for k, v in s_off["methods"].items()},
                    "paired_full_gated_vs_always_on": harness.paired_bootstrap(
                        summ["methods"]["full"]["per_pair_alc"], s_off["methods"]["full"]["per_pair_alc"], N_BOOT, seed=5)}}
    headline(s_off, "real LOO off")
    dump(loo, "loo_gating_ablation_real.json")
    dump(loo_margin_sensitivity(raw, summ, 0, workers, "real"), "loo_margin_sensitivity_real.json")

    # sentence-embedding similarity instead of TF-IDF (experiments only; needs embed_cache.py output)
    npz = RESULTS / "embeddings_real_testbed.npz"
    if npz.exists():
        s_emb = summarize(run_dataset(raw, seed=0, params={"EMBED_NPZ": str(npz)}, workers=workers))
        emb = {"real": {"tfidf": {k: {"alc": v["alc"], "brier_by_budget": v["brier_by_budget"], "ece_pooled": v["ece_pooled"]}
                                  for k, v in summ["methods"].items()},
                        "minilm": {k: {"alc": v["alc"], "brier_by_budget": v["brier_by_budget"], "ece_pooled": v["ece_pooled"]}
                                   for k, v in s_emb["methods"].items()},
                        "paired_minilm_vs_tfidf": {k: harness.paired_bootstrap(s_emb["methods"][k]["per_pair_alc"],
                                                                                summ["methods"][k]["per_pair_alc"], N_BOOT, seed=6)
                                                   for k in ("full", "shrink_knn")},
                        "model_id": "sentence-transformers/all-MiniLM-L6-v2"}}
        headline(s_emb, "real MiniLM similarity")
        dump(emb, "real_testbed_embedding.json")
    else:
        print("no embedding cache; skipping MiniLM comparison")

    alpha = {"real": {}}
    for a0 in (1.0, 2.0, 4.0, 8.0):
        s = summarize(run_dataset(raw, seed=0, params={"ALPHA0": a0}, workers=workers))
        alpha["real"][str(a0)] = {k: {"alc": v["alc"]} for k, v in s["methods"].items()}
        print(f"alpha0={a0}: shrink {s['methods']['shrink']['alc']['mean']:.4f} full {s['methods']['full']['alc']['mean']:.4f}")
    dump(alpha, "alpha0_sensitivity_real.json")

    pol = {"real": {}}
    per_pair = {}
    for policy in ("random", "first", "novelty", "novelty_unc", "ble"):
        s = summarize(run_dataset(raw, seed=0, policy_name=policy, workers=workers))
        per_pair[policy] = {k: s["methods"][k]["per_pair_alc"] for k in ("full", "shrink", "empirical_mean")}
        pol["real"][policy] = {k: {"alc": v["alc"], "brier_by_budget": v["brier_by_budget"]}
                               for k, v in s["methods"].items() if k in ("full", "shrink", "empirical_mean")}
        headline(s, f"real policy {policy}")
    pol["real"]["paired_vs_random"] = {
        policy: {k: harness.paired_bootstrap(per_pair[policy][k], per_pair["random"][k], N_BOOT, seed=4)
                 for k in ("full", "shrink", "empirical_mean")} for policy in per_pair if policy != "random"}
    dump(pol, "acquisition_policies_real.json")

    # multiple acquisition seeds on the real testbed (trajectory variability)
    multi = {"seeds": {}}
    for seed in (1, 2, 3, 4):
        s = summarize(run_dataset(raw, seed=seed, workers=workers))
        multi["seeds"][str(seed)] = {k: {"alc": v["alc"]} for k, v in s["methods"].items()}
        multi["seeds"][str(seed)]["paired"] = {k: v for k, v in s["paired"].items() if not k.endswith("by_budget")}
    dump(multi, "real_testbed_seeds.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["synthetic", "real", "all"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.suite in ("synthetic", "all"):
        suite_synthetic(a.workers)
    if a.suite in ("real", "all"):
        suite_real(a.workers)
