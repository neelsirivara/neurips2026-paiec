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
              "from the shipped `model.py`/`labeling.py` by `make_macros.py`."]
    (PAPER / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} claims, {n_fail} failing -> {PAPER / 'AUDIT.md'}")
    for t, a, v, h in rows:
        if not h:
            print("  FAILS:", t, "|", v)


if __name__ == "__main__":
    main()
