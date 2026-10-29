"""Re-derive every directional claim of the report from the result files and write paper/AUDIT.md.

    python make_audit.py

Each claim is checked programmatically against experiments/results/*.json; the audit lists the
claim, the artifact and the values, and flags claims that do NOT hold so that the text can be
corrected before submission.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PAPER = HERE.parents[1] / "paper"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def g(obj, *path, default=None):
    for k in path:
        if obj is None:
            return default
        obj = obj.get(str(k)) if isinstance(obj, dict) else None
    return default if obj is None else obj


def main() -> None:
    syn, real = load("synthetic_main.json"), load("real_testbed_main.json")
    emb = load("real_testbed_embedding.json")
    pol_s, pol_r = load("acquisition_policies_synthetic.json"), load("acquisition_policies_real.json")
    rep, seeds = load("synthetic_replication.json"), load("real_testbed_seeds.json")
    alpha_s, alpha_r = load("alpha0_sensitivity_synthetic.json"), load("alpha0_sensitivity_real.json")
    loo_s, loo_r = load("loo_gating_ablation_synthetic.json"), load("loo_gating_ablation_real.json")
    sweep = load("synthetic_signal_sweep.json")
    smoke = load("smoke_test.json")
    vtxt = (RESULTS / "validator_output.txt").read_text(encoding="utf-8") if (RESULTS / "validator_output.txt").exists() else ""

    rows = []

    def claim(text, artifact, values, holds):
        rows.append((text, artifact, values, holds))

    def paired(summ, key):
        d = g(summ, "paired", key)
        return d, (f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], significant={d['significant']}, "
                   f"frac_better={d['frac_a_better']:.2f}" if d else "missing")

    for name, summ, file in (("synthetic", syn, "synthetic_main.json"), ("real", real, "real_testbed_main.json")):
        if summ is None:
            claim(f"[{name}] results exist", file, "missing", False)
            continue
        m = summ["methods"]
        d, v = paired(summ, "full_vs_empirical_mean")
        claim(f"[{name}] full model beats the empirical mean (paired ALC difference < 0)", file + " paired.full_vs_empirical_mean", v, bool(d and d["mean"] < 0))
        claim(f"[{name}] ... and the difference is significant (95% CI excludes 0)", file, v, bool(d and d["significant"] and d["mean"] < 0))
        d, v = paired(summ, "full_vs_const")
        claim(f"[{name}] full model beats the constant 0.5, significantly", file + " paired.full_vs_const", v, bool(d and d["significant"] and d["mean"] < 0))
        b1 = m["empirical_mean"]["brier_by_budget"]["1"]
        claim(f"[{name}] empirical mean is worse than the constant at n=1 (B1 > 0.25)", file + " methods.empirical_mean.brier_by_budget.1", f"B1 = {b1:.4f}", b1 > 0.25)
        bb = m["empirical_mean"]["brier_by_budget"]
        first = next((b for b in ("1", "3", "7", "15", "31") if bb[b] < 0.25), None)
        claim(f"[{name}] empirical mean overtakes the constant at n={first} (the budget the text quotes) and not earlier", file,
              ", ".join(f"B{b}={bb[b]:.4f}" for b in ("1", "3", "7")), first is not None and all(bb[b] >= 0.25 for b in ("1", "3", "7", "15", "31") if int(b) < int(first)))
        d, v = paired(summ, "prior_vs_const")
        claim(f"[{name}] prior alone improves on the constant", file + " paired.prior_vs_const", v, bool(d and d["mean"] < 0))
        d, v = paired(summ, "shrink_vs_prior")
        claim(f"[{name}] shrinkage improves on the prior", file + " paired.shrink_vs_prior", v, bool(d and d["mean"] < 0))
        for key in ("shrink_knn_vs_shrink", "shrink_ridge_vs_shrink", "full_vs_shrink"):
            d, v = paired(summ, key)
            claim(f"[{name}] item-level effect {key} is small (|mean| < 0.005 ALC)", file + f" paired.{key}", v, bool(d and abs(d["mean"]) < 0.005))
        d, v = paired(summ, "full_vs_shrink")
        claim(f"[{name}] gated full model is not significantly worse than shrinkage", file + " paired.full_vs_shrink", v, bool(d and not (d["significant"] and d["mean"] > 0)))
        s31, o31 = m["shrink"]["brier_by_budget"]["31"], m["oracle_rate"]["brier_by_budget"]["31"]
        claim(f"[{name}] shrinkage at n=31 is within 0.01 Brier of the rate oracle", file, f"B31 shrink {s31:.4f} vs oracle_rate {o31:.4f}", s31 - o31 < 0.01)
        claim(f"[{name}] full-model ECE below empirical-mean ECE and constant ECE", file + " methods.*.ece_pooled",
              f"full {m['full']['ece_pooled']:.3f}, emp {m['empirical_mean']['ece_pooled']:.3f}, const {m['const']['ece_pooled']:.3f}",
              m["full"]["ece_pooled"] < m["empirical_mean"]["ece_pooled"] and m["full"]["ece_pooled"] < m["const"]["ece_pooled"])
    if real:
        m = real["methods"]
        claim("[real] empirical-mean ALC is above the constant's 0.25", "real_testbed_main.json methods.empirical_mean.alc.mean", f"{m['empirical_mean']['alc']['mean']:.4f}", m["empirical_mean"]["alc"]["mean"] > 0.25)
    if syn:
        m = syn["methods"]
        gap = m["oracle_rate"]["alc"]["mean"] - m["oracle_p"]["alc"]["mean"]
        claim("[synthetic] item-level headroom (oracle_rate - oracle_p) is about 0.02 ALC (0.01..0.03)", "synthetic_main.json methods.oracle_*", f"gap = {gap:.4f}", 0.01 <= gap <= 0.03)
    if alpha_s and alpha_r:
        bs = min(alpha_s["synthetic"], key=lambda a: alpha_s["synthetic"][a]["shrink"]["alc"]["mean"])
        br = min(alpha_r["real"], key=lambda a: alpha_r["real"][a]["shrink"]["alc"]["mean"])
        claim("alpha0 preferred by shrinkage differs between testbeds", "alpha0_sensitivity_*.json", f"synthetic best {bs}, real best {br}", bs != br)
        rng_s = max(v["full"]["alc"]["mean"] for v in alpha_s["synthetic"].values()) - min(v["full"]["alc"]["mean"] for v in alpha_s["synthetic"].values())
        claim("alpha0 range (1..8) moves the full-model ALC by less than 0.01 on synthetic", "alpha0_sensitivity_synthetic.json", f"range {rng_s:.4f}", rng_s < 0.01)
    if emb:
        d = emb["real"]["paired_minilm_vs_tfidf"]["full"]
        claim("[real] MiniLM embeddings do not significantly improve the full model over TF-IDF", "real_testbed_embedding.json paired_minilm_vs_tfidf.full",
              f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] sig={d['significant']}", not (d["significant"] and d["mean"] < 0))
    for name, pol, key in (("synthetic", pol_s, "synthetic"), ("real", pol_r, "real")):
        if not pol:
            continue
        for p in ("ble", "novelty", "novelty_unc", "first"):
            d = pol[key]["paired_vs_random"][p]["full"]
            claim(f"[{name}] policy '{p}' is not significantly better than random", f"acquisition_policies_{name}.json paired_vs_random.{p}.full",
                  f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] sig={d['significant']}", not (d["significant"] and d["mean"] < 0))
        d = pol[key]["paired_vs_random"]["novelty"]["full"]
        claim(f"[{name}] shipped novelty policy is not significantly worse than random", f"acquisition_policies_{name}.json", f"mean {d['mean']:+.4f} sig={d['significant']}", not (d["significant"] and d["mean"] > 0))
    if rep:
        for s in ("1", "2"):
            r = rep["seeds"][s]
            claim(f"[synthetic seed {s}] full < empirical mean and full < 0.25", "synthetic_replication.json", f"full {r['full']['alc']['mean']:.4f} emp {r['empirical_mean']['alc']['mean']:.4f}",
                  r["full"]["alc"]["mean"] < r["empirical_mean"]["alc"]["mean"] and r["full"]["alc"]["mean"] < 0.25)
    if seeds:
        n_sig = sum(1 for r in seeds["seeds"].values() if r["paired"]["full_vs_empirical_mean"]["significant"] and r["paired"]["full_vs_empirical_mean"]["mean"] < 0)
        claim("[real seeds] full beats empirical mean in every trajectory (mean difference < 0)", "real_testbed_seeds.json",
              f"{n_sig}/{len(seeds['seeds'])} significant; means " + ", ".join(f"{r['paired']['full_vs_empirical_mean']['mean']:+.4f}" for r in seeds["seeds"].values()),
              all(r["paired"]["full_vs_empirical_mean"]["mean"] < 0 for r in seeds["seeds"].values()))
    if loo_s and loo_r:
        for name, loo, key in (("synthetic", loo_s, "synthetic"), ("real", loo_r, "real")):
            d = loo[key]["paired_full_gated_vs_always_on"]
            claim(f"[{name}] gated model is not significantly worse than always-on components", f"loo_gating_ablation_{name}.json", f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] sig={d['significant']}", not (d["significant"] and d["mean"] > 0))
    if sweep:
        st = sweep["settings"]
        gaps = [(s["sigma_topic"], s["methods"]["oracle_rate"]["alc"]["mean"] - s["methods"]["oracle_p"]["alc"]["mean"]) for s in st if s["gamma_length"] == 0.3]
        gaps.sort()
        claim("[sweep] oracle headroom grows with sigma_topic", "synthetic_signal_sweep.json", ", ".join(f"sigma {a}: {b:.4f}" for a, b in gaps), all(gaps[i][1] <= gaps[i + 1][1] + 1e-9 for i in range(len(gaps) - 1)))
    if smoke:
        claim("predict runs at > 100 calls/s at budget 31 with long items, mean < 50 ms", "smoke_test.json", f"{smoke['calls_per_second_long_budget31']:.0f} calls/s, mean {smoke['mean_ms_long_budget31']:.2f} ms, max {smoke['max_ms_long_budget31']:.1f} ms", smoke["calls_per_second_long_budget31"] > 100 and smoke["mean_ms_long_budget31"] < 50)
        claim("smoke test passed all cases", "smoke_test.json", f"{len(smoke['cases'])} cases, ok={smoke['ok']}", bool(smoke["ok"]))
    claim("official validator accepts the ZIP", "validator_output.txt", vtxt.strip().splitlines()[1] if vtxt.strip() else "missing", "OK:" in vtxt and "[exit code 0]" in vtxt)
    # iteration 2026-10-22: prior sensitivity, training-release listing, fitting dry run
    sens, lst, dry = load("prior_sensitivity.json"), load("measurement_db_listing.json"), load("prior_fit_dry_run.json")
    fit, tb = load("prior_fit.json"), load("prior_fit_testbeds.json")
    if sens:
        p0s = [g["p0"] for g in sens["frontier_grid"]]
        lo, hi = 1 / (1 + 2.718281828 ** 1.5), 1 / (1 + 2.718281828 ** -1.5)
        claim("[sens] frontier-like subjects: p0 stays inside the cap [0.18, 0.82] and spans more than 0.3", "prior_sensitivity.json frontier_grid",
              f"p0 in [{min(p0s):.3f}, {max(p0s):.3f}] over {len(p0s)} subjects (values rounded to 4 decimals)",
              lo - 1e-3 <= min(p0s) and max(p0s) <= hi + 1e-3 and max(p0s) - min(p0s) > 0.3)
        cells = {(c["r"], c["p0"]): c for c in sens["miscalibration"]["cells"]}
        c = cells.get((0.3, 0.7))
        claim("[sens] alpha0=4: a prior of 0.7 on a pair with true rate 0.3 loses to the constant 0.5", "prior_sensitivity.json miscalibration",
              f"ALC excess {c['alc_excess']:.4f} vs constant excess {(0.5 - 0.3) ** 2:.4f}" if c else "missing", bool(c and not c["beats_const"]))
        ok_cells = [c for (r, p), c in cells.items() if abs(r - 0.5) >= 0.2 - 1e-9 and abs(p - r) <= 0.3 + 1e-9]
        claim("[sens] alpha0=4: shrinkage beats the constant whenever |r-0.5| >= 0.2 and |p0-r| <= 0.3", "prior_sensitivity.json miscalibration",
              f"{sum(1 for c in ok_cells if c['beats_const'])}/{len(ok_cells)} such cells beat the constant", bool(ok_cells) and all(c["beats_const"] for c in ok_cells))
        near = [c for (r, p), c in cells.items() if abs(r - 0.5) < 0.15]
        claim("[sens] alpha0=4: near r=0.5 (|r-0.5| < 0.15) no prior makes shrinkage beat the constant", "prior_sensitivity.json miscalibration",
              f"{sum(1 for c in near if c['beats_const'])}/{len(near)} cells beat the constant", bool(near) and not any(c["beats_const"] for c in near))
        sw = sens.get("offset_sweep", {}).get("real")
        if sw:
            pv = {k: v["paired_vs_zero"]["full"] for k, v in sw["offsets"].items()}
            claim("[real] shifts of +0.5 and +1 are significantly worse than the shipped 0", "prior_sensitivity.json offset_sweep.real",
                  ", ".join(f"{k}: {pv[k]['mean']:+.4f} sig={pv[k]['significant']}" for k in ("+0.5", "+1.0")),
                  all(pv[k]["significant"] and pv[k]["mean"] > 0 for k in ("+0.5", "+1.0")))
            d = pv["-0.5"]
            claim("[real] a shift of -0.5 is not significantly different from 0", "prior_sensitivity.json offset_sweep.real",
                  f"(-0.5) - (0): mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] sig={d['significant']}", not d["significant"])
        sw = sens.get("offset_sweep", {}).get("synthetic")
        if sw:
            pv = {k: v["paired_vs_zero"]["full"] for k, v in sw["offsets"].items()}
            claim("[synthetic] 0 is the best shift and both +-0.5 are significantly worse", "prior_sensitivity.json offset_sweep.synthetic",
                  ", ".join(f"{k}: {pv[k]['mean']:+.4f} sig={pv[k]['significant']}" for k in ("-0.5", "+0.5")),
                  sw["best_offset_full"] == "+0.0" and all(pv[k]["significant"] and pv[k]["mean"] > 0 for k in ("-0.5", "+0.5")))
        if fit and fit.get("diagnostics"):
            dg = fit["diagnostics"]
            claim("[release] the hand-set prior is not optimistic in the large (mean p0 <= mean pair rate)", "prior_fit.json diagnostics",
                  f"mean p0 {dg['hand_p0_mean']:.3f} vs mean pair rate {dg['pair_rate_mean']:.3f}", dg["hand_p0_mean"] <= dg["pair_rate_mean"])
    if lst:
        claim("measurement-db listing: 281 benchmark directories; core tables under 30 GB; traces and assets over 100 GB", "measurement_db_listing.json",
              f"{lst['n_benchmark_dirs']} dirs; core {lst['core_tables_gb']} GB; traces {lst['table_gb']['traces']} GB; assets {lst['table_gb']['assets']} GB",
              lst["n_benchmark_dirs"] == 281 and lst["core_tables_gb"] < 30 and lst["table_gb"]["traces"] > 100 and lst["table_gb"]["assets"] > 100)
    if dry and dry.get("dry_run_recovery"):
        rec = dry["dry_run_recovery"]
        claim("[dry run] fitting pipeline recovers the sign of every generating slope and each within 0.15 logits", "prior_fit_dry_run.json dry_run_recovery",
              f"sign agreement {rec['sign_agreement_slopes']}/{rec['n_slopes']}, max |error| {rec['max_abs_error_slopes']}",
              rec["sign_agreement_slopes"] == rec["n_slopes"] and rec["max_abs_error_slopes"] < 0.15)
    if fit and fit.get("cv"):
        h4 = fit["cv"]["hand_alpha4"]
        claim("[release] hand-set prior + shrinkage beats the empirical mean and the constant on the training release (closed-form protocol)",
              "prior_fit.json cv.hand_alpha4", f"shrink {h4['alc']['mean']:.4f}, empirical mean {h4['alc_emp']['mean']:.4f}, constant {h4['alc_const']['mean']:.4f}",
              h4["alc"]["mean"] < h4["alc_emp"]["mean"] and h4["alc"]["mean"] < h4["alc_const"]["mean"])
        alphas = {a: fit["cv"][f"hand_alpha{a}"]["alc"]["mean"] for a in ("2", "4", "8")}
        claim("[release] alpha0 = 4 is the best of {2, 4, 8} for the hand-set prior on the training release", "prior_fit.json cv.hand_alpha*",
              ", ".join(f"alpha {a}: {v:.4f}" for a, v in alphas.items()), min(alphas, key=alphas.get) == "4")
        sel = fit["cv"].get("selected")
        if sel:
            key = f"fit_C{float(sel['C']):g}_cap{float(sel['cap']):g}_alpha{float(sel['alpha0']):g}"
            d = fit["cv"][key]["paired_b0_fit_minus_hand"]
            better = fit.get("cv_better_than_hand")
            claim("[release] the report states the CV outcome of the fitted prior correctly (shipped iff better under CV and on the real testbed)",
                  "prior_fit.json cv_better_than_hand + prior_fit_testbeds.json ship_rule",
                  f"cv_better={better}; B0 fit-hand {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]; ship={None if tb is None else tb.get('ship_rule', {}).get('ship')}",
                  tb is not None and tb.get("ship_rule", {}).get("ship") == bool(better and tb["ship_rule"].get("real_testbed_full_better")))
    nov = load("novelty_filter_stats.json")
    if nov and pol_r and pol_s:
        rec = nov.get("real")
        if rec:
            a, b = pol_r["real"]["novelty"]["full"]["alc"]["mean"], pol_r["real"]["first"]["full"]["alc"]["mean"]
            claim("[real] novelty filter fires on some pairs yet its full-model ALC is within 1e-4 of stream order", "novelty_filter_stats.json + acquisition_policies_real.json",
                  f"fired on {rec['n_pairs_fired']}/{rec['n_pairs']} pairs, {rec['total_skipped']} skipped; ALC novelty {a:.5f} vs first {b:.5f}",
                  rec["n_pairs_fired"] > 0 and abs(a - b) < 1e-4)
        rec = nov.get("synthetic")
        if rec:
            claim("[synthetic] novelty filter never fires (pseudo-word items never reach the 0.6 cosine threshold)", "novelty_filter_stats.json",
                  f"fired on {rec['n_pairs_fired']}/{rec['n_pairs']} pairs", rec["n_pairs_fired"] == 0)
    import model as _model
    hand_shipped = (not _model.PRIOR_PROVIDER) and _model.PRIOR_INTERCEPT == 0.0
    shipped_fit = bool(tb is not None and tb.get("ship_rule", {}).get("ship"))
    claim("model.py ships the hand-set prior unless the ship rule passed (consistency of code and result files)", "model.py + prior_fit_testbeds.json",
          f"hand-set constants in model.py: {hand_shipped}; ship rule passed: {shipped_fit}", hand_shipped == (not shipped_fit))

    # ------------------------------------------------------------------ round 3: item-text prior
    import hashlib
    cv3 = load("round3/text_prior_cv.json")
    if cv3 is None:
        claim("[round 3] text-prior CV results exist", "round3/text_prior_cv.json", "missing", False)
    else:
        pre = RESULTS / "round3" / "PREREG.md"
        sha = hashlib.sha256(pre.read_bytes()).hexdigest() if pre.exists() else None
        claim("[round 3] PREREG.md is unchanged since the CV ran (SHA-256 recorded in the result equals the file's)",
              "round3/PREREG.md + text_prior_cv.json", f"{(sha or 'missing')[:12]} vs {cv3['prereg_sha256'][:12]}", sha == cv3["prereg_sha256"])
        rep3 = cv3["reproduction"]
        claim("[round 3] the CV frame reproduces the iteration-2 hand-set numbers exactly", "text_prior_cv.json reproduction",
              f"B0 {rep3['b0']:.6f} vs {rep3['b0_iteration2']:.6f}; ALC {rep3['alc']:.6f} vs {rep3['alc_iteration2']:.6f}", bool(rep3["exact"]))
        d = cv3["summary"]["paired_vs_hand"]["text"]["alc_cluster"]
        claim("[round 3] the item-text prior does not significantly change the benchmark-held-out ALC (cluster CI includes 0; rule (a) fails)",
              "text_prior_cv.json summary.paired_vs_hand.text.alc_cluster", f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]",
              d["ci_lo"] <= 0 <= d["ci_hi"] and not cv3["rule_a_cv_alc_ci_below_zero"])
        folds = cv3["folds"]
        claim("[round 3] the out-of-fold calibration slope W is negative in every outer fold", "text_prior_cv.json folds[].calibration.W",
              ", ".join(f"{f['calibration']['W']:+.3f}" for f in folds), all(f["calibration"]["W"] < 0 for f in folds))
        amax = max(float(a) for a in cv3["settings"]["alpha_grid"])
        claim("[round 3] the inner CV selected the largest pre-registered penalty in every fold", "text_prior_cv.json folds[].alpha",
              ", ".join(f"{float(f['alpha']):g}" for f in folds), all(float(f["alpha"]) == amax for f in folds))
        gaps = cv3["summary"]["paired_vs_hand"]["text"]["gap_closed"]
        claim("[round 3] the text prior closes essentially none (|ratio| < 5%) of the gap to the pair-rate oracle at budgets 0, 1, 3",
              "text_prior_cv.json gap_closed", ", ".join(f"B{b}: {100 * gaps[b]['ratio']:+.1f}%" for b in ("0", "1", "3")),
              all(abs(gaps[b]["ratio"]) < 0.05 for b in ("0", "1", "3")))
        d = cv3["summary"]["paired_vs_hand"]["benchmark_mean_oracle"]["alc_cluster"]
        claim("[round 3] knowing the held-out benchmark's mean rate would improve the ALC significantly (headroom exists)",
              "text_prior_cv.json benchmark_mean_oracle", f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]", d["ci_hi"] < 0)
        dup = cv3["duplicates"]
        claim("[round 3] exact cross-benchmark duplicates are negligible (< 1% of frame items on average)", "text_prior_cv.json duplicates",
              f"mean duplicate fraction {100 * dup['mean_dup_fraction_pairs']:.3f}%", dup["mean_dup_fraction_pairs"] < 0.01)
    tb3 = load("round3/text_prior_testbeds.json")
    if tb3:
        for name in ("real", "synthetic"):
            d = tb3["testbeds"][name]["full"]["paired_text_minus_hand"]
            claim(f"[round 3, {name}] text prior vs hand-set (full model) is not significantly different", "text_prior_testbeds.json",
                  f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]", not d["significant"])
        claim("[round 3] rule (b) holds (real testbed not significantly worse)", "text_prior_testbeds.json rule_b",
              str(tb3["rule_b_real_not_significantly_worse"]), bool(tb3["rule_b_real_not_significantly_worse"]))
    ex3 = load("round3/text_prior_explore.json")
    if ex3:
        add = RESULTS / "round3" / "PREREG_ADDENDUM_1.md"
        sha = hashlib.sha256(add.read_bytes()).hexdigest() if add.exists() else None
        claim("[round 3] PREREG_ADDENDUM_1.md is unchanged since the exploratory runs", "round3/PREREG_ADDENDUM_1.md + text_prior_explore.json",
              f"{(sha or 'missing')[:12]} vs {ex3['addendum_sha256'][:12]}", sha == ex3["addendum_sha256"])
        for name, v in ex3["variants"].items():
            d = v["summary"]["paired_vs_hand"][name]["alc_cluster"]
            claim(f"[round 3, exploratory] {name} does not significantly improve on the hand-set prior", "text_prior_explore.json",
                  f"mean {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]", not (d["ci_hi"] < 0))
            claim(f"[round 3, exploratory] {name}: W negative in every fold", "text_prior_explore.json",
                  ", ".join(f"{f['calibration']['W']:+.3f}" for f in v["folds"]), all(f["calibration"]["W"] < 0 for f in v["folds"]))
    chk = load("round3/text_prior_check.json")
    if chk:
        claim("[round 3] stdlib featurisation is bit-identical to the fitting cache and scores agree with numpy (< 1e-6)",
              "text_prior_check.json", f"{chk['n_features_bit_identical_to_cache']}/{chk['n_items']} identical; max diff {chk['max_abs_score_diff_stdlib_vs_numpy']:.1e}",
              chk["n_features_bit_identical_to_cache"] == chk["n_items"] and chk["max_abs_score_diff_stdlib_vs_numpy"] < 1e-6)
    rt = load("round3/text_prior_runtime.json")
    if rt:
        claim("[round 3] the engineering gates would have passed (>= 200 calls/s, <= 5 MB, smoke test with the weights loaded)",
              "text_prior_runtime.json", f"min calls/s {min(v['calls_per_s'] for v in rt['throughput'].values()):.0f}; "
              f"{rt['weights_file_bytes'] / 1e6:.2f} MB; smoke exit {rt['smoke_exit_code']}",
              bool(rt["gate_throughput_200"] and rt["gate_weights_5mb"] and rt["gate_smoke"]))
    eq = load("round3/shipped_equivalence.json")
    if eq:
        ship3 = bool(cv3 and cv3["rule_a_cv_alc_ci_below_zero"] and tb3 and tb3["rule_b_real_not_significantly_worse"])
        present = (HERE / "text_prior.json").exists()
        claim("[round 3] ship decision is consistent with the files: no text_prior.json shipped and the ZIP unchanged iff the rule failed",
              "shipped_equivalence.json + experiments/src", f"rule passed: {ship3}; text_prior.json present: {present}; ZIP unchanged: {eq['zip_unchanged']}",
              (not ship3) == ((not present) and eq["zip_unchanged"]))
        claim("[round 3] the shipped model.py and the development model.py (dormant hooks, no weights) give bit-identical predictions",
              "shipped_equivalence.json", f"{eq['n_bit_identical']}/{eq['n_cases']}", eq["n_bit_identical"] == eq["n_cases"] > 0)

    # ------------------------------------------------------------------ round 4: cross-pair evidence
    import hashlib as _hl
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    cv4, add4 = load("round4/xpair_cv.json"), load("round4/xpair_addendum1.json")
    real4, acq4, racq4 = load("round4/xpair_real_testbed.json"), load("round4/xpair_acq.json"), load("round4/xpair_real_acq.json")
    rt4, eq4, pp4 = load("round4/xpair_runtime.json"), load("round4/xpair_equivalence.json"), load("round4/xpair_perpair.json")
    loop4, smoke4 = load("round4/official_loop_check.json"), load("round4/smoke_test_submission.json")
    pre4 = RESULTS / "round4" / "PREREG.md"
    sha4 = _hl.sha256(pre4.read_bytes()).hexdigest() if pre4.exists() else None
    files4 = {"xpair_cv": cv4, "xpair_addendum1": add4, "xpair_real_testbed": real4, "xpair_acq": acq4, "xpair_real_acq": racq4,
              "xpair_runtime": rt4, "xpair_equivalence": eq4, "xpair_perpair": pp4}
    claim("[round 4] PREREG.md hash recorded in every round-4 result file equals the file on disk", "round4/*.json",
          ", ".join(f"{k}: {'ok' if (v and v.get('prereg_sha256') == sha4) else 'MISMATCH'}" for k, v in files4.items()),
          bool(sha4) and all(v and v.get("prereg_sha256") == sha4 for v in files4.values()))

    def cb(d):
        return f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]" if d else "missing"

    confs = {c: g(cv4, "configs", c) for c in ("A", "B", "C", "D")}
    confs["E"] = g(add4, "configs", "E")
    dA = g(confs["A"], "paired", "XP_minus_S0", "alc")
    claim("[round 4, rule (a)] config A: XP - S0 ALC CI entirely below 0 (primary)", "xpair_cv.json configs.A.paired.XP_minus_S0",
          cb(dA), bool(dA and dA["ci_hi"] < 0))
    for c in ("B", "C", "D", "E"):
        d = g(confs[c], "paired", "XP_minus_S0", "alc")
        claim(f"[round 4] config {c}: XP better than S0 with CI below 0 (rule (c) for B-D; addendum E)",
              f"{'xpair_addendum1' if c == 'E' else 'xpair_cv'}.json configs.{c}.paired.XP_minus_S0", cb(d), bool(d and d["ci_hi"] < 0))
    for c in ("A", "B", "C", "D", "E"):
        d = g(confs[c], "paired", "XP_minus_S1", "alc")
        claim(f"[round 4] config {c}: XP better than S1 (correct per-pair filter), CI below 0", f"configs.{c}.paired.XP_minus_S1",
              cb(d), bool(d and d["ci_hi"] < 0))
    d = g(confs["A"], "paired", "S1_minus_S0", "alc")
    claim("[round 4] config A: S1 vs S0 (name pooling) not significant", "xpair_cv.json configs.A.paired.S1_minus_S0", cb(d),
          bool(d and not d["significant"]))
    d = g(confs["C"], "paired", "S1_minus_S0", "alc")
    claim("[round 4] config C: name pooling costs (S1 - S0 CI below 0)", "xpair_cv.json configs.C.paired.S1_minus_S0", cb(d),
          bool(d and d["ci_hi"] < 0))
    d = g(confs["D"], "paired", "XP_minus_S0_evalonly_pairs", "alc")
    claim("[round 4] config D: evaluation-only pairs gain (CI below 0)", "xpair_cv.json configs.D.paired.XP_minus_S0_evalonly_pairs", cb(d),
          bool(d and d["ci_hi"] < 0))
    for c in ("A", "E"):
        br = g(confs[c], "paired", "XP_minus_S0", "brier") or []
        ok = len(br) == 6 and all(b["ci_hi"] < 0 for b in br[1:])
        largest = len(br) == 6 and min(range(1, 6), key=lambda i: br[i]["mean"]) == 5
        claim(f"[round 4] config {c}: XP - S0 significant at every budget from n=1, largest at n=31", f"configs.{c}.paired.XP_minus_S0.brier",
              ", ".join(f"{b['mean']:+.4f}" for b in br), bool(ok and largest))
        xb = g(confs[c], "methods", "XP", "brier")
        ob = g(confs[c], "methods", "oracle_rate", "brier")
        claim(f"[round 4] config {c}: XP at n=31 below the pair-rate oracle", f"configs.{c}.methods", f"XP B31 {xb[5]:.4f} vs oracle {ob[5]:.4f}"
              if xb and ob else "missing", bool(xb and ob and xb[5] < ob[5]))
    for c, ref in (("A", "XP_allfold"), ("E", "XPrb_allfold")):
        s = confs[c]
        db = g(s, "paired", f"minus_bench_minus_{ref}", "alc")
        di = g(s, "paired", f"minus_item_minus_{ref}", "alc")
        claim(f"[round 4] config {c}: removing the benchmark effect and removing the item effect both hurt significantly",
              f"configs.{c}.paired.minus_*", f"bench {cb(db)}; item {cb(di)}", bool(db and di and db["ci_lo"] > 0 and di["ci_lo"] > 0))
        claim(f"[round 4] config {c}: benchmark and item contributions are about equal (ratio within 2/3..3/2)", f"configs.{c}.paired.minus_*",
              f"ratio {db['mean'] / di['mean']:.2f}" if db and di else "missing", bool(db and di and 2 / 3 <= db["mean"] / di["mean"] <= 1.5))
        small = [g(s, "paired", f"minus_{k}_minus_{ref}", "alc") for k in ("subj", "pair", "probit")]
        claim(f"[round 4] config {c}: subject, pair and probit components are small (|effect| < 0.002)", f"configs.{c}.paired",
              ", ".join(cb(x) for x in small), all(x and abs(x["mean"]) < 0.002 for x in small))
    a = confs["A"]
    s0 = g(a, "methods", "S0", "alc", "mean")
    bp = g(a, "methods", "bench_and_pair", "alc", "mean")
    xpm = g(a, "methods", "XP_allfold", "alc", "mean")
    nsp = g(a, "methods", "name_subj_and_pair", "alc", "mean")
    pit = g(a, "methods", "pair_and_item", "alc", "mean")
    if None not in (s0, bp, xpm, nsp, pit):
        claim("[round 4] config A: benchmark+pair recovers more than half of the gain; name/subject+pair and item+pair recover none",
              "xpair_cv.json configs.A.methods", f"S0 {s0:.4f}, XP {xpm:.4f}, bench+pair {bp:.4f}, name/subj+pair {nsp:.4f}, item+pair {pit:.4f}",
              (s0 - bp) / (s0 - xpm) > 0.5 and nsp >= s0 - 1e-4 and pit >= s0 - 1e-4)
    rob = g(add4, "selection_robust") or {}
    prof = rob.get("tau_item_profile", {})
    if prof:
        best = min(prof, key=lambda k: prof[k]["best_A"])
        claim("[round 4] extended grid: the best tau_item on A is interior (2.25, not 0 or 4)", "xpair_addendum1.json tau_item_profile",
              ", ".join(f"{k}: {v['best_A']:.4f}" for k, v in prof.items()), best == "2.25")
    profb = rob.get("tau_bench_profile", {})
    if profb:
        bestb = min(profb, key=lambda k: profb[k]["best_A"])
        claim("[round 4] extended grid: the best tau_bench on A is interior (1.5)", "xpair_addendum1.json tau_bench_profile",
              ", ".join(f"{k}: {v['best_A']:.4f}" for k, v in profb.items()), bestb == "1.5")
    claim("[round 4] addendum robust pick equals the pre-registered all-fold pick", "xpair_addendum1.json",
          f"{g(add4, 'selection_robust', 'all', 'label')} vs {g(cv4, 'selection', 'all', 'label')}",
          bool(add4 and cv4 and g(add4, "selection_robust", "all", "label") == g(cv4, "selection", "all", "label")))
    pf = [r["diff"] for r in (g(confs["A"], "per_fold") or [])]
    claim("[round 4] config A: XP - S0 negative in every outer fold", "xpair_cv.json configs.A.per_fold", ", ".join(f"{v:+.4f}" for v in pf),
          bool(pf) and all(v < 0 for v in pf))
    runs = g(real4, "runs") or []
    r0 = runs[0] if runs else None
    d = g(r0, "paired", "XP_minus_shipped_global")
    claim("[round 4, rule (b)] real testbed seed 0: XP - shipped (global list) not significantly positive", "xpair_real_testbed.json runs[0]",
          cb(d), bool(d and not d["ci_lo"] > 0))
    nneg = sum(1 for r in runs if g(r, "paired", "XP_minus_shipped_global", "ci_hi") < 0)
    claim("[round 4] real testbed: XP - shipped global significantly negative in 4 of 5 trajectories (the abstract's count)",
          "xpair_real_testbed.json runs", f"{nneg} of {len(runs)}", nneg == 4 and len(runs) == 5)
    d = g(r0, "paired", "XP_minus_shipped_perpair")
    claim("[round 4] real testbed seed 0: XP vs per-pair shipped predictor not significant", "xpair_real_testbed.json runs[0]", cb(d),
          bool(d and not d["significant"]))
    d = g(r0, "paired", "XP_perpair_minus_shipped_perpair")
    claim("[round 4] real testbed seed 0: XP with per-pair lists is significantly worse than the per-pair predictor (limitation)",
          "xpair_real_testbed.json runs[0]", cb(d), bool(d and d["ci_lo"] > 0))
    xb, sb = g(r0, "methods", "XP", "brier"), g(r0, "methods", "shipped_global", "brier")
    claim("[round 4] real testbed seed 0: XP is worse at n=1 than the deployed per-pair predictor", "xpair_real_testbed.json runs[0].methods",
          f"XP B1 {xb[1]:.4f} vs {sb[1]:.4f}" if xb and sb else "missing", bool(xb and sb and xb[1] > sb[1]))
    d = g(pp4, "summary", "paired", "XP_minus_S1", "alc")
    claim("[round 4] release, per-pair lists: XP better than S1 (CI below 0)", "xpair_perpair.json", cb(d), bool(d and d["ci_hi"] < 0))
    d = g(acq4, "fresh_minus_random", "alc")
    claim("[round 4] config A: fresh-first better than random under XP (CI below 0)", "xpair_acq.json fresh_minus_random", cb(d),
          bool(d and d["ci_hi"] < 0))
    br = g(acq4, "fresh_minus_random", "brier") or []
    claim("[round 4] fresh-first: the gain is at n=15 and n=31 (significant there, not at n=1..7)", "xpair_acq.json fresh_minus_random.brier",
          ", ".join(f"{b['mean']:+.5f}" for b in br),
          len(br) == 6 and br[4]["ci_hi"] < 0 and br[5]["ci_hi"] < 0 and not any(b["significant"] for b in br[1:4]))
    d = g(acq4, "S0_fresh_minus_random", "alc")
    claim("[round 4] fresh-first does not affect S0 significantly", "xpair_acq.json S0_fresh_minus_random", cb(d), bool(d and not d["significant"]))
    rr = g(racq4, "runs") or []
    d = g(rr[0] if rr else None, "fresh_minus_first")
    claim("[round 4] real testbed seed 0: fresh-first not significantly worse (acquisition ship rule)", "xpair_real_acq.json runs[0]", cb(d),
          bool(d and not d["ci_lo"] > 0))
    claim("[round 4] equivalence: stdlib model.py vs numpy simulator max |diff| < 1e-3", "xpair_equivalence.json",
          f"{g(eq4, 'max_abs_diff')} over {g(eq4, 'n_predictions')}", bool(eq4 and eq4["max_abs_diff"] < 1e-3))
    claim("[round 4] official client: stream mode, no predict calls during acquisition, ALC equals the simulator", "official_loop_check.json",
          f"mode {g(loop4, 'acquisition_mode')}, acq predict calls {g(loop4, 'acquisition_prediction_calls')}, match {g(loop4, 'matches_xpair_sim')}",
          bool(loop4 and loop4["acquisition_mode"] == "stream" and loop4["acquisition_prediction_calls"] == 0 and loop4.get("matches_xpair_sim")))
    sz = {r["n_pairs"]: r for r in (g(rt4, "sizes") or [])}
    claim("[round 4, rule (d)] 62,000-label checkpoint within 60 s; cache-hit calls <= 5 ms", "xpair_runtime.json sizes[2000]",
          f"checkpoint {sz[2000]['checkpoint_s']:.1f} s, {sz[2000]['ms_per_call']:.3f} ms" if 2000 in sz else "missing",
          bool(2000 in sz and sz[2000]["checkpoint_s"] <= 60 and sz[2000]["ms_per_call"] <= 5))
    claim("[round 4] 4,000 pairs per checkpoint well under a minute", "xpair_runtime.json sizes[4000]",
          f"{sz[4000]['checkpoint_s']:.1f} s" if 4000 in sz else "missing", bool(4000 in sz and sz[4000]["checkpoint_s"] < 60))
    claim("[round 4] smoke test of the shipped files passed with cross-pair cases", "round4/smoke_test_submission.json",
          f"ok={g(smoke4, 'ok')}, xp cases {g(smoke4, 'xp_cases')}, hook {g(smoke4, 'acquisition_hook_mode')}",
          bool(smoke4 and smoke4.get("ok") and smoke4.get("xp_cases", 0) > 0 and smoke4.get("acquisition_hook_mode") == "stream"))
    import model as _m
    import labeling as _l
    lbl = g(add4, "decision", "ship_label") or ""
    want = {}
    for part in lbl.split("|")[1:]:
        for key in ("namebench", "bench", "name", "subj", "pair", "item", "probit"):
            if part.startswith(key):
                want[key] = float(part[len(key):])
                break
    taus = dict(_m.XP_TAUS)
    match = (_m.XP_ENABLED and all(taus.get(k) == want.get(k) for k in ("bench", "name", "subj", "namebench", "pair"))
             and _m.XP_TAU_ITEM == want.get("item") and bool(_m.XP_PROBIT) == bool(want.get("probit")))
    claim("[round 4] shipped model.py constants equal the selected setting and XP is enabled", "experiments/src/model.py vs xpair_addendum1.json",
          f"{lbl}; XP_TAUS={taus}, item={_m.XP_TAU_ITEM}, probit={_m.XP_PROBIT}, enabled={_m.XP_ENABLED}", bool(match))
    acq_rule = bool(acq4 and rr and g(acq4, "fresh_minus_random", "alc", "ci_hi") < 0 and not g(rr[0], "fresh_minus_first", "ci_lo") > 0)
    claim("[round 4] labeling.POLICY is 'fresh' iff the acquisition ship rule holds", "experiments/src/labeling.py",
          f"POLICY={_l.POLICY}; rule holds={acq_rule}", (_l.POLICY == "fresh") == acq_rule)
    import zipfile as _zf
    zp = HERE.parents[1] / "submission" / "paiec_submission.zip"
    same = False
    if zp.exists():
        with _zf.ZipFile(zp) as z:
            same = (z.read("model.py") == (HERE / "model.py").read_bytes() and z.read("labeling.py") == (HERE / "labeling.py").read_bytes())
    claim("[round 4] the ZIP members equal the development model.py and labeling.py", "submission/paiec_submission.zip", f"identical={same}", same)

    # ------------------------------------------------------------------ selected submission 1010717 (2026-10-26)
    import ast
    import hashlib
    import zipfile
    sub = HERE.parents[1] / "submission"
    zsel, zship = sub / "candidates" / "xp_level_m050.zip", sub / "paiec_submission.zip"
    fr = load("deep_dive/formative_runs.json")
    runs = g(fr, "runs") or {}
    info = load("deep_dive/candidates/xp_level_m050/candidate_info.json")
    sha_sel = hashlib.sha256(zsel.read_bytes()).hexdigest() if zsel.exists() else None
    sha_ship = hashlib.sha256(zship.read_bytes()).hexdigest() if zship.exists() else None
    claim("[selected] the selected ZIP's SHA-256 equals the run record (formative_runs.json) and the build record",
          "submission/candidates/xp_level_m050.zip", f"{sha_sel}", bool(sha_sel and sha_sel == g(fr, "selected_submission", "sha256")
                                                                          and sha_sel == g(info, "sha256")))
    claim("[selected] the selected submission ID is 1010717 and its run used xp_level_m050.zip", "formative_runs.json",
          f"id {g(fr, 'selected_submission', 'id')}, run zip {g(runs, 'xp_level_m050', 'zip')}, run id {g(runs, 'xp_level_m050', 'submission_id')}",
          g(fr, "selected_submission", "id") == 1010717 and g(runs, "xp_level_m050", "submission_id") == 1010717
          and g(runs, "xp_level_m050", "zip") == "submission/candidates/xp_level_m050.zip")
    claim("[selected] the round-4 ZIP (1010370) is unchanged", "submission/paiec_submission.zip", f"{sha_ship}",
          sha_ship == "c76331d084f175fbf8ce272b00d0a2360d80e440dacf6ede380f3f9eba5172e1")

    def zconst(path):
        with zipfile.ZipFile(path) as z:
            src = z.read("model.py").decode("utf-8")
            lab = z.read("labeling.py")
            names = z.namelist()
        out = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        try:
                            out[t.id] = ast.literal_eval(node.value)
                        except Exception:
                            pass
        return src, lab, names, out, ast.parse(src)
    if zsel.exists() and zship.exists():
        s_src, s_lab, s_names, s_c, s_tree = zconst(zsel)
        p_src, p_lab, p_names, p_c, _ = zconst(zship)
        diff = [i for i, (a, b) in enumerate(zip(p_src.splitlines(), s_src.splitlines())) if a != b]
        same_len = len(p_src.splitlines()) == len(s_src.splitlines())
        claim("[selected] model.py differs from 1010370 in exactly three lines: SUBJECT_SHRINK 0.7->0, PRIOR_INTERCEPT 0->-0.5, tau_mu 0.5->2",
              "both ZIPs, model.py", f"changed lines {[i + 1 for i in diff]}; shrink {p_c.get('SUBJECT_SHRINK')}->{s_c.get('SUBJECT_SHRINK')}, "
              f"intercept {p_c.get('PRIOR_INTERCEPT')}->{s_c.get('PRIOR_INTERCEPT')}, mu {dict(p_c['XP_TAUS'])['mu']}->{dict(s_c['XP_TAUS'])['mu']}",
              same_len and len(diff) == 3 and p_c.get("SUBJECT_SHRINK") == 0.7 and s_c.get("SUBJECT_SHRINK") == 0.0
              and p_c.get("PRIOR_INTERCEPT") == 0.0 and s_c.get("PRIOR_INTERCEPT") == -0.5
              and dict(p_c["XP_TAUS"])["mu"] == 0.5 and dict(s_c["XP_TAUS"])["mu"] == 2.0)
        claim("[selected] labeling.py is byte-identical to 1010370's", "both ZIPs", f"identical={s_lab == p_lab}", s_lab == p_lab)
        stdlib = {"__future__", "array", "base64", "hashlib", "json", "math", "os", "re", "sys", "threading", "zlib", "importlib",
                  "random", "time", "collections", "itertools", "functools", "typing", "pathlib"}
        imps = set()
        for node in ast.walk(s_tree):
            if isinstance(node, ast.Import):
                imps |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imps.add(node.module.split(".")[0])
        claim("[selected] no pretrained model: no models.txt/requirements.txt in the ZIP, model.py imports only the standard library",
              "submission/candidates/xp_level_m050.zip", f"members {s_names}; imports {sorted(imps)}",
              "models.txt" not in s_names and "requirements.txt" not in s_names and imps <= stdlib)
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    claim("[report] author block is exactly Neel Sirivara / University of Washington / nsiriv@cs.washington.edu",
          "paper/main.tex", "author block", "Neel Sirivara\\\\\n  University of Washington\\\\\n  \\texttt{nsiriv@cs.washington.edu}" in tex)
    mtex = (PAPER / "results_macros.tex").read_text(encoding="utf-8")
    claim("[report] the selected submission ID printed in the paper is 1010717", "paper/results_macros.tex \\SelId",
          "\\newcommand{\\SelId}{1010717}" in mtex, "\\newcommand{\\SelId}{1010717}" in mtex and "\\SelId" in tex)
    # formative runs
    bud = {k: g(v, "brier_by_budget") or {} for k, v in runs.items()}
    claim("[formative] 1010370's budget-0 Brier is above the constant's 0.25", "formative_runs.json runs.shipped",
          f"B0 {bud.get('shipped', {}).get('0')}", (bud.get("shipped", {}).get("0") or 0) > 0.25)
    cands = ("xp_nosubj_mu2", "xp_level_m050", "xp_level_m100")
    sdr = load("deep_dive/selected_predictor_draws.json")
    draws = g(sdr, "draws") or []
    rec_b0 = {k: bud.get(k, {}).get("0") for k in cands}
    rec_b0.update({str(d["submission_id"]): d["brier_by_budget"]["0"] for d in draws[1:]})
    above = [k for k, v in rec_b0.items() if v is not None and v > 0.25]
    claim("[formative] of the 5 recalibrated draws (3 variant runs + 2 probe draws) exactly 1 has B0 above 0.25, and it is a draw of "
          "1010717's predictor (1010837)", "formative_runs.json + selected_predictor_draws.json",
          ", ".join(f"{k} {v:.4f}" for k, v in rec_b0.items()), len(rec_b0) == 5 and above == ["1010837"])
    e0 = {k: g(v, "ece_by_budget", "0") for k, v in runs.items()}
    e0_rec = [e0[k] for k in cands if k in e0] + [d["ece_by_budget"]["0"] for d in draws[1:]]
    claim("[formative] every recalibrated draw (5) has lower budget-0 ECE than 1010370", "formative_runs.json + selected_predictor_draws.json",
          f"max {max(e0_rec):.3f} vs {e0.get('shipped'):.3f}", len(e0_rec) == 5 and max(e0_rec) < e0["shipped"])
    # three draws of the selected predictor
    ev_same = g(sdr, "same_predictor_evidence") or {}
    import zipfile as _zf2
    ok_code = True
    with _zf2.ZipFile(zsel) as z:
        sel_m, sel_l = z.read("model.py"), z.read("labeling.py")
    for nm in ("probe_A", "probe_B"):
        zp = sub / "candidates" / f"{nm}.zip"
        if not zp.exists():
            ok_code = False
            continue
        with _zf2.ZipFile(zp) as z:
            m, l2 = z.read("model.py"), z.read("labeling.py")
        tail = m[len(sel_m):]
        ok_code = ok_code and m.startswith(sel_m) and l2 == sel_l and b"_core_predict = predict" in tail and b"return _core_predict(input, labeled)" in tail
    claim("[draws] probes 1010835/1010837 run the selected predictor: model.py = selected model.py + wrapper returning _core_predict, "
          "labeling.py identical (re-derived from the ZIPs and recorded in the JSON)", "submission/candidates/probe_A.zip, probe_B.zip",
          f"re-derived {ok_code}; recorded {[v.get('model_py_starts_with_selected') for v in ev_same.values()]}",
          ok_code and all(v and v["model_py_starts_with_selected"] and v["labeling_identical"] for v in ev_same.values()))
    pinfo = [load(f"deep_dive/candidates/{nm}/probe_info.json") for nm in ("probe_A", "probe_B")]
    lsel_alc = g(load("deep_dive/candidates/xp_level_m050/official_loop_check.json"), "alc_mean")
    claim("[draws] with the probe disabled, the probes give exactly the selected ZIP's local-testbed ALC in the organisers' client",
          "deep_dive/candidates/probe_*/probe_info.json", f"{[g(p, 'official_loop_alc') for p in pinfo]} vs {lsel_alc}",
          all(p is not None and g(p, "official_loop_alc") == lsel_alc for p in pinfo))
    d1 = draws[0] if draws else {}
    claim("[draws] the 1010717 draw in selected_predictor_draws.json equals formative_runs.json (two parses of the same log)",
          "selected_predictor_draws.json vs formative_runs.json", f"{d1.get('alc')} vs {g(runs, 'xp_level_m050', 'alc_mean')}",
          bool(d1) and abs(d1["alc"] - g(runs, "xp_level_m050", "alc_mean")) < 1e-12
          and all(abs(d1["brier_by_budget"][b] - g(runs, "xp_level_m050", "brier_by_budget", b)) < 1e-12 for b in d1["brier_by_budget"]))
    import statistics as _st
    alcs3 = [d["alc"] for d in draws]
    alc4 = [v["alc_mean"] for v in runs.values()]
    claim("[draws] the selected predictor's three-draw range exceeds the range of the four submission runs (so their gaps are draw noise)",
          "selected_predictor_draws.json + formative_runs.json",
          f"draws {[round(a, 4) for a in alcs3]} (range {max(alcs3) - min(alcs3):.4f}), four runs range {max(alc4) - min(alc4):.4f}",
          len(alcs3) == 3 and max(alcs3) - min(alcs3) > max(alc4) - min(alc4))
    claim("[draws] three-draw mean and sd as printed (0.1897, 0.026)", "selected_predictor_draws.json",
          f"mean {_st.mean(alcs3):.4f}, sd {_st.stdev(alcs3):.4f}", round(_st.mean(alcs3), 4) == 0.1897 and round(_st.stdev(alcs3), 3) == 0.026)
    mm2 = g(sdr, "benchmark_matched_alc") or {}
    same_max = max(abs(v["mean_diff"]) for v in mm2.values())
    between = [abs(v["ALC"]["mean_diff"]) for v in (g(fr, "benchmark_matched") or {}).values()]
    claim("[draws] the largest matched difference between draws of the same predictor exceeds 3 of the 6 between-submission differences",
          "selected_predictor_draws.json + formative_runs.json", f"same max {same_max:.4f}; between {[round(b, 4) for b in between]}",
          sum(1 for b in between if b < same_max) == 3)
    sc_lb = g(load("leaderboard_context.json"), "scores_sorted") or []
    claim("[draws] the three-draw mean would rank 46 on the public board (the best draw 22)", "leaderboard_context.json",
          f"mean rank {1 + sum(1 for v in sc_lb if v < _st.mean(alcs3))}, best-draw rank {1 + sum(1 for v in sc_lb if v < min(alcs3))}",
          1 + sum(1 for v in sc_lb if v < _st.mean(alcs3)) == 46 and 1 + sum(1 for v in sc_lb if v < min(alcs3)) == 22)
    # wherever the report states 1010717's score, the mean over its draws is stated too (main text, abstract, conclusion)
    main_part = tex.split("\\appendix")[0]
    paras = [p for p in main_part.split("\n\n") if "\\FmSelAlc" in p or "\\SdDrawOneAlc" in p]
    bad = [p[:60] for p in paras if "\\SdAlcMean" not in p]
    abst = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0] if "\\begin{abstract}" in tex else ""
    concl = tex.split("\\section{Conclusion}")[1].split("\\section{")[0] if "\\section{Conclusion}" in tex else ""
    claim("[draws] every main-text paragraph that states 1010717's score also states the three-draw mean; abstract and conclusion state it",
          "paper/main.tex", f"{len(paras)} paragraphs, without mean: {bad}; abstract {'\\SdAlcMean' in abst}; conclusion {'\\SdAlcMean' in concl}",
          bool(paras) and not bad and "\\SdAlcMean" in abst and "\\SdAlcMean" in concl)
    ece = load("deep_dive/ece_check.json")
    sims = [g(ece, s, "ece_by_budget")[0] for s in ("A", "INV", "HARD", "SHIFT") if g(ece, s, "ece_by_budget")]
    claim("[formative] 1010370's budget-0 ECE is above all four simulated populations' values", "ece_check.json",
          f"observed {e0.get('shipped')}, simulated {[round(x, 3) for x in sims]}", bool(sims) and e0.get("shipped", 0) > max(sims))
    sims2 = [g(ece, f"{s}_nosubj_d-0.5_mu2", "ece_by_budget")[0] for s in ("A", "INV", "HARD") if g(ece, f"{s}_nosubj_d-0.5_mu2", "ece_by_budget")]
    claim("[formative] 1010717's budget-0 ECE is above every simulated value for its constants (appendix wording 'also above')",
          "ece_check.json", f"observed {e0.get('xp_level_m050')}, simulated {[round(x, 3) for x in sims2]}",
          bool(sims2) and e0.get("xp_level_m050", 0) > max(sims2))
    p3 = load("deep_dive/proto_level3.json")
    inv_b = [g(p3, "scenarios", "INV", "methods", "shipped", "brier")]
    sb = bud.get("shipped", {})
    claim("[formative] 1010370's B1 and B3 are within 0.01 of the simulated INV values (text: 'in the range of')",
          "formative_runs.json + proto_level3.json INV", f"B1 {sb.get('1')} vs {inv_b[0][1] if inv_b[0] else None}, B3 {sb.get('3')} vs {inv_b[0][2] if inv_b[0] else None}",
          bool(inv_b[0]) and abs(sb["1"] - inv_b[0][1]) < 0.01 and abs(sb["3"] - inv_b[0][2]) < 0.01)
    claim("[formative] ALC rows equal the ALC recomputed from per-budget means (all runs)", "formative_runs.json",
          ", ".join(f"{k} {abs(v['alc_mean'] - v['alc_from_budget_means']):.1e}" for k, v in runs.items()),
          all(abs(v["alc_mean"] - v["alc_from_budget_means"]) < 1e-4 for v in runs.values()) and len(runs) == 4)
    ov = g(fr, "overlap") or {}
    claim("[formative] the same 7 benchmark IDs in every run and no pair repeated across runs", "formative_runs.json overlap",
          f"benchmarks/run {[v.get('n_benchmark_ids') for v in runs.values()]}, all-runs {g(fr, 'distinct_benchmark_ids_all_runs')}, "
          f"max shared pairs {max((v['shared_pairs'] for v in ov.values()), default=None)}",
          all(v.get("n_benchmark_ids") == 7 for v in runs.values()) and g(fr, "distinct_benchmark_ids_all_runs") == 7
          and all(v["shared_pairs"] == 0 for v in ov.values()))
    bm = g(fr, "benchmark_matched") or {}
    ts_alc = [abs(v["ALC"]["mean_diff"] / v["ALC"]["se"]) for v in bm.values() if "ALC" in v]
    claim("[formative] no benchmark-matched ALC difference between any two runs reaches two standard errors", "formative_runs.json benchmark_matched",
          f"|t| {[round(t, 2) for t in ts_alc]}", len(ts_alc) == 6 and max(ts_alc) < 2)
    ts_all = [abs(x["mean_diff"] / x["se"]) for v in bm.values() for x in v.values()]
    claim("[formative] no matched entry (any budget) reaches the 5% t critical value with 6 df (2.447)", "formative_runs.json benchmark_matched",
          f"max |t| {max(ts_all):.2f} over {len(ts_all)}", len(ts_all) == 42 and max(ts_all) < 2.447)
    d0 = g(bm, "xp_level_m050_minus_shipped", "0", "mean_diff")
    claim("[formative] revision rule not triggered: matched B0 of 1010717 minus 1010370 is negative", "formative_runs.json benchmark_matched",
          f"{d0}", d0 is not None and d0 < 0)
    scen = g(p3, "scenarios") or {}
    mx = max(abs(scen[s]["methods"]["nosubj_d-0.5_mu2"]["brier"][i] - scen[s]["methods"]["shipped"]["brier"][i]) for s in scen for i in range(1, 6))
    a, b = bud.get("xp_level_m050", {}), bud.get("shipped", {})
    raw = [a[k] - b[k] for k in ("1", "3", "7", "15", "31")]
    claim("[formative] observed raw B1..B31 differences (1010717 - 1010370) far exceed the simulated maximum change at n>=1 (draw noise)",
          "formative_runs.json + proto_level3.json", f"mean raw {sum(raw) / 5:+.4f}, simulated max |delta| {mx:.4f}", abs(sum(raw) / 5) > 3 * mx)
    # simulator scenarios
    def ms(s, v):
        return g(scen, s, "methods", v, "minus_shipped") or {}
    sel = "nosubj_d-0.5_mu2"
    claim("[simulator] 1010717 constants: CI below 0 on MID, HARD, SHIFT", "proto_level3.json",
          ", ".join(f"{s} {ms(s, sel).get('ci_hi'):+.4f}" for s in ("MID", "HARD", "SHIFT")), all(ms(s, sel).get("ci_hi", 1) < 0 for s in ("MID", "HARD", "SHIFT")))
    claim("[simulator] 1010717 constants: not significant on INV and INV25", "proto_level3.json",
          ", ".join(f"{s} [{ms(s, sel).get('ci_lo'):+.4f}, {ms(s, sel).get('ci_hi'):+.4f}]" for s in ("INV", "INV25")),
          all(not ms(s, sel).get("significant") for s in ("INV", "INV25")))
    claim("[simulator] 1010717 constants: significant cost on A and Y25 (release-like)", "proto_level3.json",
          ", ".join(f"{s} {ms(s, sel).get('ci_lo'):+.4f}" for s in ("A", "Y25")), all(ms(s, sel).get("ci_lo", -1) > 0 for s in ("A", "Y25")))
    claim("[simulator] 1010717 constants: no significant loss in any non-release-like population", "proto_level3.json",
          ", ".join(f"{s} lo {ms(s, sel).get('ci_lo'):+.4f}" for s in ("INV", "INV25", "MID", "HARD", "SHIFT")),
          all(not ms(s, sel).get("ci_lo", 1) > 0 for s in ("INV", "INV25", "MID", "HARD", "SHIFT")))
    m100, nos = "nosubj_d-1_mu2", "nosubj_d0_mu2"
    claim("[simulator] 1010735 constants more than double the release-like cost of 1010717 (A and Y25)", "proto_level3.json",
          ", ".join(f"{s} {ms(s, m100).get('mean'):+.4f} vs {ms(s, sel).get('mean'):+.4f}" for s in ("A", "Y25")),
          all(ms(s, m100)["mean"] > 2 * ms(s, sel)["mean"] for s in ("A", "Y25")))
    claim("[simulator] 1010715 constants change the ALC by less than 0.002 in magnitude outside release-like data", "proto_level3.json",
          ", ".join(f"{s} {ms(s, nos).get('mean'):+.4f}" for s in ("INV", "INV25", "MID", "HARD", "SHIFT")),
          all(abs(ms(s, nos)["mean"]) < 0.002 for s in ("INV", "INV25", "MID", "HARD", "SHIFT")))
    claim("[simulator] the level matters almost only at B0: per scenario |dB0| exceeds every |dB1..B31|, and the latter stay below 0.005",
          "proto_level3.json", f"max |dB1..31| {mx:.4f}",
          mx < 0.005 and all(abs(scen[s]["methods"][sel]["brier"][0] - scen[s]["methods"]["shipped"]["brier"][0]) >
                             max(abs(scen[s]["methods"][sel]["brier"][i] - scen[s]["methods"]["shipped"]["brier"][i]) for i in range(1, 6)) for s in scen))
    pop = load("deep_dive/population.json")
    gi = g(pop, "groups") or {}
    claim("[data] on inventory benchmarks the hand-set prior's B0 exceeds 0.25 (also for 2025+ subjects); inventory rate < non-inventory",
          "population.json", f"B0 {g(gi, 'inventory_benchmarks', 'B0_prior_approx')}, {g(gi, 'inventory_x_2025plus', 'B0_prior_approx')}; "
          f"rates {g(gi, 'inventory_benchmarks', 'mean_rate')} vs {g(gi, 'non_inventory_benchmarks', 'mean_rate')}",
          (g(gi, "inventory_benchmarks", "B0_prior_approx") or 0) > 0.25 and (g(gi, "inventory_x_2025plus", "B0_prior_approx") or 0) > 0.25
          and g(gi, "inventory_benchmarks", "mean_rate") < g(gi, "non_inventory_benchmarks", "mean_rate"))
    r2 = [v["r2_best_slope"] for v in (g(pop, "ability_transfer") or {}).values()]
    claim("[data] a subject-ability table by model name does not transfer (R^2 < 0.02 in every variant)", "population.json ability_transfer",
          f"{[round(x, 3) for x in r2]}", bool(r2) and max(r2) < 0.02)
    orc = load("deep_dive/proto_oracle.json")
    ok = all(abs(g(orc, "scenarios", s, "methods", "item_oracle_1", "minus_shipped", "mean")) >
             abs(g(orc, "scenarios", s, "methods", "bench_oracle", "minus_shipped", "mean")) for s in ("A", "HARD"))
    claim("[deep dive] item-level oracle gains exceed benchmark-level ones in A and HARD (most headroom is item-level)", "proto_oracle.json",
          "; ".join(f"{s}: item {g(orc, 'scenarios', s, 'methods', 'item_oracle_1', 'minus_shipped', 'mean'):+.4f}, bench "
                    f"{g(orc, 'scenarios', s, 'methods', 'bench_oracle', 'minus_shipped', 'mean'):+.4f}" for s in ("A", "HARD")), ok)
    gi2 = load("deep_dive/grid_inv.json")
    d = g(gi2, "scenarios", "INV", "nested_minus_shipped")
    claim("[deep dive] re-tuning the prior sds on INV gives no significant gain", "grid_inv.json", f"{d.get('mean'):+.4f} [{d.get('ci_lo'):+.4f}, {d.get('ci_hi'):+.4f}]" if d else "missing",
          bool(d) and not d.get("significant"))
    # candidate checks
    for name in ("xp_level_m050", "xp_level_m100", "xp_nosubj_mu2"):
        base = RESULTS / "deep_dive" / "candidates" / name
        vt = (base / "validator_output.txt").read_text(encoding="utf-8") if (base / "validator_output.txt").exists() else ""
        sm, eq, lp = load(f"deep_dive/candidates/{name}/smoke_test.json"), load(f"deep_dive/candidates/{name}/equivalence.json"), \
            load(f"deep_dive/candidates/{name}/official_loop_check.json")
        claim(f"[checks] {name}: validator OK, smoke ok, equivalence < 1e-3, stream mode with 0 predict calls during acquisition, 31 labels per pair",
              f"deep_dive/candidates/{name}/", f"validator {'OK' if 'OK:' in vt and '[exit code 0]' in vt else 'FAIL'}, smoke {g(sm, 'ok')}, "
              f"equiv {g(eq, 'max_abs_diff')}, mode {g(lp, 'acquisition_mode')}, acq calls {g(lp, 'acquisition_prediction_calls')}",
              "OK:" in vt and "[exit code 0]" in vt and bool(g(sm, "ok")) and (g(eq, "max_abs_diff") or 1) < 1e-3
              and g(lp, "acquisition_mode") == "stream" and g(lp, "acquisition_prediction_calls") == 0
              and set(g(lp, "labels_per_pair") or []) == {31})
    lsel, lship = load("deep_dive/candidates/xp_level_m050/official_loop_check.json"), load("round4/official_loop_check.json")
    claim("[checks] local testbed in the organisers' client: the level change hardly matters (|1010717 - 1010370| < 0.001 ALC)",
          "official_loop_check.json (both)", f"{g(lsel, 'alc_mean')} vs {g(lship, 'alc_mean')}",
          lsel is not None and lship is not None and abs(lsel["alc_mean"] - lship["alc_mean"]) < 0.001)
    lb = load("leaderboard_context.json")
    sc = g(lb, "scores_sorted") or []
    claim("[leaderboard] our best draw (1010717) is worse than the board's #10 ('far from the leaders')", "leaderboard_context.json",
          f"1010717 {g(runs, 'xp_level_m050', 'alc_mean'):.4f}, #10 {sc[9] if len(sc) > 9 else None}",
          len(sc) > 9 and g(runs, "xp_level_m050", "alc_mean") > sc[9])

    # ------------------------------------------------------------------ pre-registered item-difficulty study (2026-10-29)
    ev = load("deep_dive/items/eval_items_qwen3vl_4b.json")
    items_dir = RESULTS / "deep_dive" / "items"

    def fsha(name):
        p = items_dir / name
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    claim("[items] the result file records the SHA-256 of PREREG_items.md and of amendments 1 and 2, and they match the files",
          "deep_dive/items/eval_items_qwen3vl_4b.json", f"{g(ev, 'prereg_sha256', default='')[:12]}, {g(ev, 'amendment_sha256', default='')[:12]}, "
          f"{g(ev, 'amendment2_sha256', default='')[:12]}",
          bool(ev) and ev.get("prereg_sha256") == fsha("PREREG_items.md") and ev.get("amendment_sha256") == fsha("PREREG_items_AMENDMENT_1.md")
          and ev.get("amendment2_sha256") == fsha("PREREG_items_AMENDMENT_2.md"))
    import datetime as _dt
    mt = {n: _dt.datetime.fromtimestamp((items_dir / n).stat().st_mtime) for n in
          ("PREREG_items.md", "PREREG_items_AMENDMENT_1.md", "PREREG_items_AMENDMENT_2.md", "PREREG_items_AMENDMENT_3.md",
           "ratings_qwen3vl_4b.jsonl", "eval_items_qwen3vl_4b.json") if (items_dir / n).exists()}
    order = ["PREREG_items.md", "PREREG_items_AMENDMENT_1.md", "PREREG_items_AMENDMENT_2.md", "PREREG_items_AMENDMENT_3.md",
             "eval_items_qwen3vl_4b.json"]
    claim("[items] file times are in the stated order: pre-registration < amendments 1, 2, 3 < gate evaluation", "deep_dive/items/ (mtimes)",
          ", ".join(f"{n.split('_')[-1]} {mt[n]:%m-%d %H:%M}" for n in order if n in mt),
          len(mt) == 6 and all(mt[a] < mt[b] for a, b in zip(order, order[1:])))
    rated = g(ev, "n_items_rated")
    claim("[items] every needed item was rated (coverage 1.0 in INV and AS)", "eval_items_qwen3vl_4b.json",
          f"{rated}/{g(ev, 'n_items_needed')}, coverage {g(ev, 'scenarios', 'INV', 'rating_coverage')}/{g(ev, 'scenarios', 'AS', 'rating_coverage')}",
          rated == g(ev, "n_items_needed") and g(ev, "scenarios", "INV", "rating_coverage") == 1.0 and g(ev, "scenarios", "AS", "rating_coverage") == 1.0)
    inv1, as1 = g(ev, "scenarios", "INV", "variants", "s1", "minus_base") or {}, g(ev, "scenarios", "AS", "variants", "s1", "minus_base") or {}
    ictl, actl = g(ev, "scenarios", "INV", "ctrl_diff") or {}, g(ev, "scenarios", "AS", "ctrl_diff") or {}
    gate_a = inv1.get("ci_hi", 1) < 0 and ictl.get("ci_hi", 1) < 0
    gate_b = not as1.get("ci_lo", -1) > 0 and not actl.get("ci_lo", -1) > 0
    claim("[items] gate re-derived from the CIs: (a) fails vs base and vs control, (b) holds in both; the recorded verdicts agree",
          "eval_items_qwen3vl_4b.json", f"INV vs base hi {inv1.get('ci_hi'):+.4f}, vs ctrl hi {ictl.get('ci_hi'):+.4f}; AS vs base lo "
          f"{as1.get('ci_lo'):+.4f}, vs ctrl lo {actl.get('ci_lo'):+.4f}; recorded {g(ev, 'gate')}, {g(ev, 'gate_vs_b_only')}",
          (not inv1.get("ci_hi", 1) < 0) and (not ictl.get("ci_hi", 1) < 0) and gate_b and not gate_a
          and g(ev, "gate", "pass") is False and g(ev, "gate_vs_b_only", "pass") is False)
    ws = [f["beta"][0] for f in (g(ev, "fits", "s1") or [])]
    claim("[items] the s1 slope is positive in all 5 folds", "eval_items_qwen3vl_4b.json fits.s1", f"{[round(w, 3) for w in ws]}",
          len(ws) == 5 and all(w > 0 for w in ws))
    cc = g(ev, "within_benchmark_corr_s1_vs_item_effect") or {}
    claim("[items] within-benchmark correlation of s1 with the empirical item effect is small (medians < 0.1 in INV and AS)",
          "eval_items_qwen3vl_4b.json", f"INV {g(cc, 'INV', 'median')}, AS {g(cc, 'AS', 'median')}",
          bool(cc) and g(cc, "INV", "median") < 0.1 and g(cc, "AS", "median") < 0.1)
    bb_i = g(ev, "scenarios", "INV", "variants", "s1", "brier_minus_base") or []
    bb_a = g(ev, "scenarios", "AS", "variants", "s1", "brier_minus_base") or []
    claim("[items] differences to the base are mostly the budget-0 level: |0.1 dB0| exceeds half of |dALC| in INV and AS, and |dB1..31| <= 0.001",
          "eval_items_qwen3vl_4b.json brier_minus_base", f"INV dB0 {bb_i[0] if bb_i else None}, AS dB0 {bb_a[0] if bb_a else None}",
          bool(bb_i and bb_a) and abs(0.1 * bb_i[0]) > 0.5 * abs(inv1["mean"]) and abs(0.1 * bb_a[0]) > 0.5 * abs(as1["mean"])
          and max(abs(x) for x in bb_i[1:] + bb_a[1:]) <= 0.001)
    claim("[items] the exploratory s3 and s1+s3 variants also have INV CIs that include 0", "eval_items_qwen3vl_4b.json",
          ", ".join(f"{v} [{g(ev, 'scenarios', 'INV', 'variants', v, 'minus_base', 'ci_lo'):+.4f}, "
                    f"{g(ev, 'scenarios', 'INV', 'variants', v, 'minus_base', 'ci_hi'):+.4f}]" for v in ("s3", "s1+s3")),
          all(g(ev, "scenarios", "INV", "variants", v, "minus_base", "ci_lo") < 0 < g(ev, "scenarios", "INV", "variants", v, "minus_base", "ci_hi")
              for v in ("s3", "s1+s3")))
    rmodels = set()
    rp = items_dir / "ratings_qwen3vl_4b.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rmodels.add(json.loads(line).get("model"))
    claim("[items] all ratings come from Qwen/Qwen3-VL-4B-Instruct (no API backend)", "ratings_qwen3vl_4b.jsonl", f"{sorted(rmodels)}",
          rmodels == {"Qwen/Qwen3-VL-4B-Instruct"})
    claim("[items] no Claude-API rating exists (the API route was built, never run)", "deep_dive/items/",
          f"files: {sorted(p.name for p in items_dir.iterdir())}", not any("claude" in p.name.lower() for p in items_dir.iterdir()))
    pr = g(load("deep_dive/probes_summary.json"), "probes") or {}
    claim("[probes] 1010835/1010837/1010865 are probe_A/B/C, built on the selected ZIP with the rating model in models.txt; B scored, "
          "C stopped with PAIEC-PREDICT-002", "deep_dive/probes_summary.json",
          "; ".join(f"{k}: {v['zip']}, scored={v['scored']}, codes={v['platform_error_codes']}" for k, v in pr.items()),
          [pr.get(k, {}).get("zip") for k in ("1010835", "1010837", "1010865")] == ["probe_A.zip", "probe_B.zip", "probe_C.zip"]
          and all(v.get("base_zip_sha256") == g(fr, "selected_submission", "sha256") and v.get("models_txt") == "Qwen/Qwen3-VL-4B-Instruct"
                  for v in pr.values())
          and pr["1010837"]["scored"] and "PAIEC-PREDICT-002" in pr["1010865"]["platform_error_codes"])
    pc = (sub / "candidates" / "probe_C" / "model.py")
    pct = pc.read_text(encoding="utf-8") if pc.exists() else ""
    claim("[probes] probe C returns NaN (invalid output) only after a successful load on a GPU with >= 70 GB", "submission/candidates/probe_C/model.py",
          "branch present" if 'get("class") == "ok"' in pct and 'gpu_total_gb") or 0) >= 70.0' in pct else "branch missing",
          'get("class") == "ok"' in pct and 'gpu_total_gb") or 0) >= 70.0' in pct)
    claim("[selected] the selected ZIP still has no models.txt after the item study (1010717 unchanged)", "submission/candidates/xp_level_m050.zip",
          f"sha {sha_sel[:12] if sha_sel else None}", bool(sha_sel) and sha_sel == "cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715")

    n_fail = sum(1 for r in rows if not r[3])
    lines = ["# Claim audit (generated by experiments/src/make_audit.py)", "",
             "Every directional claim in paper/main.tex, re-derived from the result files. `HOLDS` means the",
             "artifact supports the claim as worded; `FAILS` means the text must be changed (or the claim dropped).", "",
             f"{len(rows)} claims checked, {n_fail} failing.", "",
             "| # | Claim | Artifact | Values | Status |", "|---|---|---|---|---|"]
    for i, (t, a, v, h) in enumerate(rows, 1):
        lines.append(f"| {i} | {t} | `{a}` | {v} | {'HOLDS' if h else 'FAILS'} |")
    lines += ["", "Number provenance: every number in the PDF is a macro from `paper/results_macros.tex`",
              "(generated by `make_macros.py`); tables are `paper/results_table_*.tex` from the same script;",
              "figures are generated by `make_figures.py`. Constants quoted in the Method section are read",
              "from the shipped `model.py`/`labeling.py` by `make_macros.py`.", "",
              "Selected submission (2026-10-26 revision of the report): the three level constants are parsed from",
              "`model.py` inside `submission/candidates/xp_level_m050.zip` and `submission/paiec_submission.zip` (ast, not",
              "executed); the real formative numbers come only from `experiments/results/deep_dive/formative_runs.json`",
              "(aggregates of the organisers' Scoring output, written by `deep_dive/formative_summary.py`); simulator",
              "scenarios from `deep_dive/proto_level3.json`, calibration from `ece_check.json` / `decision_stat.json`,",
              "release statistics from `population.json`, ceilings from `proto_oracle.json`, re-tuning from `grid_inv.json`,",
              "candidate checks from `deep_dive/candidates/*/`, and the public leaderboard from",
              "`experiments/results/leaderboard_context.json` (derived by `make_macros.py` from the raw Codabench dumps in",
              "`../leaderboards/`; scores only). Figure 1 is drawn by pgfplots from",
              "`paper/results_fig_formative.tex`, generated from the same JSON.", "",
              "Item-difficulty study (2026-10-29 addition, Section 5.3 and Table 8): every number comes from",
              "`experiments/results/deep_dive/items/eval_items_qwen3vl_4b.json`, the PREREG files next to it (hashes and the",
              "timestamps they state) and `ratings_qwen3vl_4b.jsonl` (model name, rating time); the deployment probes from",
              "`experiments/results/deep_dive/probes_summary.json` (derived by `make_macros.py` from the saved Codabench logs:",
              "zip name, whether a score exists, platform error codes) and `submission/candidates/probe_*/`.", "",
              "Not re-derived here (stated as such in the report or in REVIEW_2.md): the participants' resubmission noise",
              "estimates mentioned in DEEP_DIVE.md (not used in the report); whether `build_candidate.py` reproduces the",
              "selected ZIP byte for byte (not re-run, to leave the submission files untouched)."]
    (PAPER / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} claims, {n_fail} failing -> {PAPER / 'AUDIT.md'}")
    for t, a, v, h in rows:
        if not h:
            print("  FAILS:", t, "|", v)


if __name__ == "__main__":
    main()
