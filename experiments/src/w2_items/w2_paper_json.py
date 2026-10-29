"""Short results file for the paper / report agent -> results/w2_items/w2_results.json

    python -I w2_paper_json.py

Collects (no new computation): the pre-registration hash, the tune-fold selection, the confirm-fold per-scenario ALC
differences with 95% cluster-bootstrap CIs and the per-budget Brier differences (with CIs), the gate, the exploratory
confirm-fold results of the other grid settings (if finished), the candidate checks, and the real formative draws
(draws.json, if present).  All values are copied from the files named in "sources".
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "w2_items"
ROOT = HERE.parents[2]
BUDGETS = [0, 1, 3, 7, 15, 31]


def ci(r):
    return {"mean": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}


tune = json.loads((RES / "tune.json").read_text(encoding="utf-8"))
conf = json.loads((RES / "confirm.json").read_text(encoding="utf-8"))
cand_dir = RES / "candidates" / "w2_f05s1_k1q2"
info = json.loads((cand_dir / "candidate_info.json").read_text(encoding="utf-8"))
eq = json.loads((cand_dir / "equivalence.json").read_text(encoding="utf-8"))
smoke = json.loads((cand_dir / "smoke_test.json").read_text(encoding="utf-8"))
loop = json.loads((cand_dir / "official_loop_check.json").read_text(encoding="utf-8"))
sel = tune["selected"]["label"]
out = {
    "what": "W2 item layer (item_features category effects + char-4-gram text kernel, learned online within each benchmark "
            "from the evaluator's labeled list) on top of xp_level_m050; differences are candidate minus m050 "
            "(negative = better), Brier-ALC, simulator xpair_sim.py (config A, R=300 per fold), 95% cluster bootstrap over benchmarks",
    "prereg": {"file": "experiments/results/w2_items/PREREG.md",
               "sha256": hashlib.sha256((RES / "PREREG.md").read_bytes()).hexdigest(), "registered": "2026-10-29 10:58 PDT"},
    "candidate": {"zip": "submission/candidates/w2_f05s1_k1q2.zip", "sha256": info["sha256"], "bytes": info["bytes"],
                  "constants": info["constants"], "base": "xp_level_m050 (sha256 " + info["base_zip_sha256"] + ")"},
    "tune": {"folds": tune["folds"], "selected": tune["selected"],
             "criterion_top": sorted(([k, v] for k, v in tune["criterion"].items()), key=lambda kv: kv[1])[:6],
             "per_scenario_selected": {s: ci(tune["scenarios"][s]["variants"][sel]["minus_m050"]) for s in tune["scenarios"]}},
    "confirm": {"folds": conf["folds"], "gate": conf["gate"], "scenarios": {}},
    "checks": {"validator_exit": info.get("validator_exit"), "smoke_exit": info.get("smoke_exit"),
               "official_loop_exit": info.get("official_loop_exit"), "official_loop_alc_real_testbed": loop["alc_mean"],
               "official_loop_acquisition_mode": loop["acquisition_mode"],
               "equivalence_max_abs_diff": eq["max_abs_diff_vs_numpy"], "equivalence_n_predictions": eq["n_predictions"],
               "zero_strength_equals_m050": eq["zero_strength_equals_m050"],
               "smoke_big_first_call_s": smoke.get("xp_big_first_call_s"), "smoke_big_ms_per_call": smoke.get("xp_big_ms_per_call")},
    "sources": ["tune.json", "confirm.json", "candidates/w2_f05s1_k1q2/{candidate_info,equivalence,smoke_test,official_loop_check}.json",
                "draws.json"],
}
for s, r in conf["scenarios"].items():
    v = r["variants"][sel]
    out["confirm"]["scenarios"][s] = {
        "n_benchmarks": r["n_bench"], "n_units": r["n_units"], "m050_alc": r["base_alc"], "w2_alc": v["alc"],
        "alc_diff": ci(v["minus_m050"]),
        "brier_diff_by_budget": {str(b): ci(v["brier_minus_m050_ci"][i]) for i, b in enumerate(BUDGETS)}}
expl = conf.get("exploratory_other_settings")
if expl:
    out["confirm"]["exploratory_other_settings"] = {
        s: {k: ci(v["minus_m050"]) for k, v in r["variants"].items()} for s, r in expl.items()}
else:
    out["confirm"]["exploratory_other_settings"] = "not finished when this file was written"
dp = RES / "draws.json"
if dp.exists():
    d = json.loads(dp.read_text(encoding="utf-8"))
    out["real_draws"] = {"note": d["note"], "missing_logs": d["missing"],
                         "per_draw": {k: {"arm": v["arm"], "alc": v["alc"], "n_pairs": v["n_pairs"],
                                          "brier_by_budget": v["brier_by_budget"]} for k, v in d["draws"].items()},
                         "arms": d["arms"], "arm_difference_w2_minus_m050": d.get("arm_difference_w2_minus_m050"),
                         "benchmark_matched_w2_minus_m050": d.get("benchmark_matched_w2_minus_m050"),
                         "structure": d.get("structure")}
cs = RES / "confirm_sem.json"
if cs.exists():
    c = json.loads(cs.read_text(encoding="utf-8"))
    ts_ = json.loads((RES / "tune_sem.json").read_text(encoding="utf-8"))
    dg = json.loads((RES / "diag_sem.json").read_text(encoding="utf-8"))["aggregate"]
    out["w2_sem"] = {
        "what": "W2 + semantic text kernel (all-MiniLM-L6-v2 sentence embeddings); pre-registered in PREREG_SEM.md; "
                "504-setting grid on tune folds 0-1, selected variant confirmed on folds 2-4 against m050 and against W2",
        "prereg_sha256": hashlib.sha256((RES / "PREREG_SEM.md").read_bytes()).hexdigest(),
        "diag_infinite_label_r2_bench_mean": {k: dg[k]["bench_mean"] for k in ("r2_cat", "r2_char_q2", "r2_minilm_q8_l0.5",
                                                                               "r2_minilm_q16_l0.5", "r2_minilm_q8_l0.5_top20",
                                                                               "r2_comb_cat+char", "r2_comb_cat+char+minilm")},
        "tune": {"selected": ts_["selected"], "w2_criterion": ts_["w2_criterion"], "best_overall": ts_["best_overall"],
                 "best_char_only": ts_["best_char_only"]},
        "confirm": {s_: {"selected_minus_m050": ci(r["selected_minus_m050"]), "W2_minus_m050": ci(r["W2_minus_m050"]),
                         "selected_minus_W2": ci(r["selected_minus_W2"]), "n_benchmarks": r["n_bench"]}
                    for s_, r in c["scenarios"].items()},
        "gate": c["gate"], "decision": "not built (gate condition d failed); W2 stays the candidate" if not c["gate"]["pass"] else "passed"}
(RES / "w2_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["confirm"]["scenarios"], indent=1)[:1500])
