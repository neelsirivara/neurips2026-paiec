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
        b3 = m["empirical_mean"]["brier_by_budget"]["3"]
        claim(f"[{name}] empirical mean overtakes the constant from n=3 (B3 < 0.25)", file, f"B3 = {b3:.4f}", b3 < 0.25)
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
