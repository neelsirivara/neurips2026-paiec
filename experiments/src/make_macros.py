"""Emit paper/results_macros.tex and the generated table bodies from experiments/results/*.json.

Every number in the report comes from here.  Missing inputs produce \\textbf{??} so that the
PDF never silently shows a stale or invented value.

    python make_macros.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PAPER = HERE.parents[1] / "paper"
MISSING = r"\textbf{??}"
BUDGETS = ["0", "1", "3", "7", "15", "31"]
BUDGET_WORDS = {"0": "Zero", "1": "One", "3": "Three", "7": "Seven", "15": "Fifteen", "31": "ThirtyOne"}
METHOD_CODES = {"const": "Const", "empirical_mean": "Emp", "prior": "Prior", "shrink": "Shrink",
                "shrink_ridge": "Ridge", "shrink_knn": "Knn", "full": "Full", "oracle_rate": "OrRate", "oracle_p": "OrP"}
METHOD_LABELS = {"const": "constant 0.5", "empirical_mean": "empirical mean (organisers)",
                 "prior": "attribute prior only", "shrink": "prior + shrinkage",
                 "shrink_ridge": "shrinkage + ridge (always on)",
                 "shrink_knn": "shrinkage + $k$NN (always on)",
                 "full": "full: LOO-gated ridge + $k$NN (shipped)",
                 "oracle_rate": "oracle: pair success rate", "oracle_p": "oracle: true item probability"}
POLICY_LABELS = {"random": "random (evaluator default)", "first": "stream order",
                 "novelty": "novelty filter (shipped)", "novelty_unc": "novelty + uncertainty",
                 "ble": "uncertainty band (BLE rule)"}
ALPHA_WORDS = {"1.0": "One", "2.0": "Two", "3.0": "Three", "4.0": "Four", "6.0": "Six", "8.0": "Eight"}
MULT_WORDS = {"0.0": "Zero", "1.0": "One", "2.0": "Two"}

macros: list[str] = []


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get(obj, *path):
    for key in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(str(key)) if str(key) in obj else obj.get(key)
        elif isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except Exception:
                return None
        else:
            return None
    return obj


def fmt(v, nd=4, sign=False):
    if v is None:
        return MISSING
    try:
        v = float(v)
    except Exception:
        return MISSING
    if v != v:
        return MISSING
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fmt_int(v):
    if v is None:
        return MISSING
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return MISSING


def mac(name, value):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def ci(d, nd=4):
    return f"[{fmt(get(d, 'ci_lo'), nd)}, {fmt(get(d, 'ci_hi'), nd)}]"


def sig_word(d):
    s = get(d, "significant")
    return MISSING if s is None else ("significant" if s else "not significant")


def paired_macros(prefix, d, nd=4):
    mac(prefix + "Mean", fmt(get(d, "mean"), nd, sign=True))
    mac(prefix + "Lo", fmt(get(d, "ci_lo"), nd, sign=True))
    mac(prefix + "Hi", fmt(get(d, "ci_hi"), nd, sign=True))
    mac(prefix + "Sig", sig_word(d))
    fr = get(d, "frac_a_better")
    mac(prefix + "Frac", MISSING if fr is None else f"{100 * fr:.0f}\\%")


# ----------------------------------------------------------------------------- main summaries
def summary_macros(prefix, summ):
    mac(prefix + "NPairs", fmt_int(get(summ, "n_pairs")))
    for m, code in METHOD_CODES.items():
        rec = get(summ, "methods", m)
        mac(prefix + code + "Alc", fmt(get(rec, "alc", "mean")))
        mac(prefix + code + "AlcLo", fmt(get(rec, "alc", "ci_lo")))
        mac(prefix + code + "AlcHi", fmt(get(rec, "alc", "ci_hi")))
        mac(prefix + code + "Ece", fmt(get(rec, "ece_pooled"), 3))
        for b in BUDGETS:
            mac(prefix + code + "B" + BUDGET_WORDS[b], fmt(get(rec, "brier_by_budget", b), 3))
    pairs = {"FullConst": "full_vs_const", "FullEmp": "full_vs_empirical_mean", "FullShrink": "full_vs_shrink",
             "FullKnn": "full_vs_shrink_knn", "FullRidge": "full_vs_shrink_ridge", "ShrinkConst": "shrink_vs_const",
             "ShrinkEmp": "shrink_vs_empirical_mean", "KnnShrink": "shrink_knn_vs_shrink",
             "RidgeShrink": "shrink_ridge_vs_shrink", "PriorConst": "prior_vs_const", "ShrinkPrior": "shrink_vs_prior",
             "EmpConst": "empirical_mean_vs_const"}
    for code, key in pairs.items():
        paired_macros(prefix + "Pair" + code, get(summ, "paired", key))
    for code, key in (("FullShrink", "full_vs_shrink_by_budget"), ("ShrinkConst", "shrink_vs_const_by_budget"),
                      ("FullConst", "full_vs_const_by_budget")):
        for b in BUDGETS:
            paired_macros(prefix + "Pair" + code + "B" + BUDGET_WORDS[b], get(summ, "paired", key, b), 3)
    # headroom decomposition: rate estimation (shrink - rate oracle) and item-level (rate oracle - true-p oracle)
    def diff(a, b, nd=4):
        return MISSING if a is None or b is None else f"{float(a) - float(b):.{nd}f}"
    sh, orr, orp = get(summ, "methods", "shrink"), get(summ, "methods", "oracle_rate"), get(summ, "methods", "oracle_p")
    mac(prefix + "HeadroomRate", diff(get(sh, "alc", "mean"), get(orr, "alc", "mean")))
    mac(prefix + "HeadroomItem", diff(get(orr, "alc", "mean"), get(orp, "alc", "mean")))
    for b in BUDGETS:
        mac(prefix + "HeadroomRateB" + BUDGET_WORDS[b], diff(get(sh, "brier_by_budget", b), get(orr, "brier_by_budget", b), 3))
    mac(prefix + "FullVsOrRate", diff(get(summ, "methods", "full", "alc", "mean"), get(orr, "alc", "mean")))


def table_main(syn, real):
    rows = []
    for m in ("const", "empirical_mean", "prior", "shrink", "shrink_ridge", "shrink_knn", "full", "oracle_rate", "oracle_p"):
        s_rec, r_rec = get(syn, "methods", m), get(real, "methods", m)
        if m == "oracle_p":
            r_cells = "--- & ---"
        else:
            r_cells = f"{fmt(get(r_rec, 'alc', 'mean'))} {ci(get(r_rec, 'alc'))} & {fmt(get(r_rec, 'ece_pooled'), 3)}"
        label = METHOD_LABELS[m]
        if m == "full":
            label = "\\textbf{" + label + "}"
        rows.append(f"{label} & {fmt(get(s_rec, 'alc', 'mean'))} {ci(get(s_rec, 'alc'))} & {fmt(get(s_rec, 'ece_pooled'), 3)} & {r_cells} \\\\")
        if m == "shrink_knn":
            rows[-1] = rows[-1]
    # insert midrules: after empirical_mean (baselines), after full (ours), oracles last
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        if i in (1, 6):
            out.append("\\midrule")
    (PAPER / "results_table_main.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def table_budget(syn, real):
    rows = []
    for name, summ in (("synthetic", syn), ("real", real)):
        for m in ("const", "empirical_mean", "prior", "shrink", "full", "oracle_rate"):
            rec = get(summ, "methods", m)
            cells = " & ".join(fmt(get(rec, "brier_by_budget", b), 3) for b in BUDGETS)
            lab = METHOD_LABELS[m] if m.startswith("oracle") else METHOD_LABELS[m].split(" (")[0]
            if m == "full":
                lab = "full (shipped)"
            rows.append(f"{name if m == 'const' else ''} & {lab} & {cells} & {fmt(get(rec, 'alc', 'mean'))} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    (PAPER / "results_table_budget.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- sweeps and ablations
def sweep_macros(sweep):
    rows = []
    settings = get(sweep, "settings") or []
    for i in range(6):
        st = settings[i] if i < len(settings) else None
        tag = f"Sweep{['A', 'B', 'C', 'D', 'E', 'F'][i]}"
        sigma, gamma = get(st, "sigma_topic"), get(st, "gamma_length")
        full, shrink = get(st, "methods", "full"), get(st, "methods", "shrink")
        d = get(st, "paired", "full_vs_shrink")
        if st is not None:
            rows.append(f"{sigma:.2f} & {gamma:.1f} & {fmt(get(shrink, 'alc', 'mean'))} & {fmt(get(full, 'alc', 'mean'))} {ci(get(full, 'alc'))} & "
                        f"{fmt(get(d, 'mean'), 4, True)} {ci(d)} & {fmt(get(st, 'methods', 'oracle_rate', 'alc', 'mean'))} & "
                        f"{fmt(get(st, 'methods', 'oracle_p', 'alc', 'mean'))} \\\\")
        mac("S" + tag + "Sigma", fmt(sigma, 2))
        mac("S" + tag + "Gamma", fmt(gamma, 1))
        mac("S" + tag + "FullAlc", fmt(get(full, "alc", "mean")))
        mac("S" + tag + "ShrinkAlc", fmt(get(shrink, "alc", "mean")))
        mac("S" + tag + "OrRateAlc", fmt(get(st, "methods", "oracle_rate", "alc", "mean")))
        mac("S" + tag + "OrPAlc", fmt(get(st, "methods", "oracle_p", "alc", "mean")))
        orr, orp = get(st, "methods", "oracle_rate", "alc", "mean"), get(st, "methods", "oracle_p", "alc", "mean")
        mac("S" + tag + "HeadroomItem", MISSING if orr is None or orp is None else f"{float(orr) - float(orp):.4f}")
        paired_macros("S" + tag + "PairFullShrink", d)
    if not rows:
        rows = [" & ".join([MISSING] * 7) + " \\\\"]
    (PAPER / "results_table_sweep.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mac("SSweepNSettings", fmt_int(len(settings)) if settings else MISSING)


def alpha_macros(prefix, data, key):
    d = get(data, key) or {}
    best_shrink, best_full = None, None
    for a, word in ALPHA_WORDS.items():
        rec = d.get(a)
        mac(prefix + "Alpha" + word + "ShrinkAlc", fmt(get(rec, "shrink", "alc", "mean")))
        mac(prefix + "Alpha" + word + "FullAlc", fmt(get(rec, "full", "alc", "mean")))
        sa, fa = get(rec, "shrink", "alc", "mean"), get(rec, "full", "alc", "mean")
        if sa is not None and (best_shrink is None or sa < best_shrink[1]):
            best_shrink = (a, sa)
        if fa is not None and (best_full is None or fa < best_full[1]):
            best_full = (a, fa)
    mac(prefix + "AlphaBestShrink", MISSING if best_shrink is None else f"{float(best_shrink[0]):g}")
    mac(prefix + "AlphaBestFull", MISSING if best_full is None else f"{float(best_full[0]):g}")
    vals = [get(rec, "full", "alc", "mean") for rec in d.values() if get(rec, "full", "alc", "mean") is not None]
    mac(prefix + "AlphaFullRange", MISSING if not vals else f"{max(vals) - min(vals):.4f}")


def loo_macros(prefix, abl, mult, key):
    mac(prefix + "LooGatedFullAlc", fmt(get(abl, key, "gated", "full", "alc", "mean")))
    mac(prefix + "LooAlwaysFullAlc", fmt(get(abl, key, "always_on", "full", "alc", "mean")))
    mac(prefix + "LooGatedFullEce", fmt(get(abl, key, "gated", "full", "ece_pooled"), 3))
    mac(prefix + "LooAlwaysFullEce", fmt(get(abl, key, "always_on", "full", "ece_pooled"), 3))
    paired_macros(prefix + "LooPairGatedAlways", get(abl, key, "paired_full_gated_vs_always_on"))
    for m, word in MULT_WORDS.items():
        mac(prefix + "LooMult" + word + "FullAlc", fmt(get(mult, key, m, "full", "alc", "mean")))
        paired_macros(prefix + "LooMult" + word + "PairFullShrink", get(mult, key, m, "paired_full_vs_shrink"))


def embedding_macros(emb):
    mac("REmbModel", get(emb, "real", "model_id") or MISSING)
    for rep, code in (("tfidf", "Tfidf"), ("minilm", "Minilm")):
        for m, mcode in (("full", "Full"), ("shrink_knn", "Knn")):
            mac("REmb" + code + mcode + "Alc", fmt(get(emb, "real", rep, m, "alc", "mean")))
            mac("REmb" + code + mcode + "AlcLo", fmt(get(emb, "real", rep, m, "alc", "ci_lo")))
            mac("REmb" + code + mcode + "AlcHi", fmt(get(emb, "real", rep, m, "alc", "ci_hi")))
    paired_macros("REmbPairFull", get(emb, "real", "paired_minilm_vs_tfidf", "full"))
    paired_macros("REmbPairKnn", get(emb, "real", "paired_minilm_vs_tfidf", "shrink_knn"))


def tfidf_macros(prefix, data, key):
    for var, code in (("unigram_restricted", "Uni"), ("uni_bi_restricted", "UniBi")):
        for m, mcode in (("shrink_knn", "Knn"), ("full", "Full")):
            mac(prefix + "Tfidf" + code + mcode + "Alc", fmt(get(data, key, var, m, "alc", "mean")))
            paired_macros(prefix + "Tfidf" + code + mcode + "VsShipped", get(data, key, "paired_vs_shipped", var, m))
    for m, mcode in (("shrink_knn", "Knn"), ("full", "Full")):
        mac(prefix + "TfidfShipped" + mcode + "Alc", fmt(get(data, key, "shipped_uni_bi", m, "alc", "mean")))
    for var, code in (("shipped_uni_bi", "Shipped"), ("unigram_restricted", "Uni")):
        paired_macros(prefix + "Tfidf" + code + "KnnVsShrink", get(data, key, "paired_knn_vs_shrink", var))


def policy_macros(syn, real):
    rows = []
    for pol in ("random", "first", "novelty", "novelty_unc", "ble"):
        cells = []
        for prefix, data, key in (("S", syn, "synthetic"), ("R", real, "real")):
            rec = get(data, key, pol, "full")
            d = get(data, key, "paired_vs_random", pol, "full")
            code = {"random": "Random", "first": "First", "novelty": "Novelty", "novelty_unc": "NoveltyUnc", "ble": "Ble"}[pol]
            mac(prefix + "Pol" + code + "FullAlc", fmt(get(rec, "alc", "mean")))
            mac(prefix + "Pol" + code + "FullAlcLo", fmt(get(rec, "alc", "ci_lo")))
            mac(prefix + "Pol" + code + "FullAlcHi", fmt(get(rec, "alc", "ci_hi")))
            mac(prefix + "Pol" + code + "ShrinkAlc", fmt(get(data, key, pol, "shrink", "alc", "mean")))
            mac(prefix + "Pol" + code + "EmpAlc", fmt(get(data, key, pol, "empirical_mean", "alc", "mean")))
            if pol != "random":
                paired_macros(prefix + "Pol" + code + "VsRandom", d)
                cells.append(f"{fmt(get(rec, 'alc', 'mean'))} {ci(get(rec, 'alc'))} & {fmt(get(d, 'mean'), 4, True)} {ci(d)}")
            else:
                cells.append(f"{fmt(get(rec, 'alc', 'mean'))} {ci(get(rec, 'alc'))} & ---")
        label = POLICY_LABELS[pol]
        if pol == "novelty":
            label = "\\textbf{" + label + "}"
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    (PAPER / "results_table_policies.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def replication_macros(rep):
    for seed, word in (("1", "One"), ("2", "Two")):
        for m, code in (("full", "Full"), ("shrink", "Shrink"), ("empirical_mean", "Emp"), ("const", "Const")):
            mac("SSeed" + word + code + "Alc", fmt(get(rep, "seeds", seed, m, "alc", "mean")))
        paired_macros("SSeed" + word + "PairFullShrink", get(rep, "seeds", seed, "paired", "full_vs_shrink"))
        paired_macros("SSeed" + word + "PairFullEmp", get(rep, "seeds", seed, "paired", "full_vs_empirical_mean"))


def real_seeds_macros(ms):
    seeds = get(ms, "seeds") or {}
    for m, code in (("full", "Full"), ("shrink", "Shrink"), ("empirical_mean", "Emp")):
        vals = [get(rec, m, "alc", "mean") for rec in seeds.values() if get(rec, m, "alc", "mean") is not None]
        mac("RSeeds" + code + "AlcMin", fmt(min(vals)) if vals else MISSING)
        mac("RSeeds" + code + "AlcMax", fmt(max(vals)) if vals else MISSING)
        mac("RSeeds" + code + "AlcMean", fmt(sum(vals) / len(vals)) if vals else MISSING)
    sig = [get(rec, "paired", "full_vs_empirical_mean", "significant") for rec in seeds.values()]
    mac("RSeedsNFullEmpSig", fmt_int(sum(1 for s in sig if s)) if sig else MISSING)
    mac("RSeedsN", fmt_int(len(seeds)) if seeds else MISSING)


def grid_macros(grid):
    cells = get(grid, "grid") or []
    if not cells:
        for n in ("SGridBestKappa", "SGridBestTau", "SGridBestFullAlc", "SGridWorstFullAlc", "SGridRange"):
            mac(n, MISSING)
        return
    best = min(cells, key=lambda c: c["full"]["mean"])
    worst = max(cells, key=lambda c: c["full"]["mean"])
    mac("SGridBestKappa", f"{best['kappa']:g}")
    mac("SGridBestTau", f"{best['tau']:g}")
    mac("SGridBestFullAlc", fmt(best["full"]["mean"]))
    mac("SGridWorstFullAlc", fmt(worst["full"]["mean"]))
    mac("SGridRange", fmt(worst["full"]["mean"] - best["full"]["mean"]))
    mac("SGridShrinkAlc", fmt(cells[0]["shrink"]["mean"]))


def testbed_macros(pairs_meta):
    pairs = get(pairs_meta, "pairs") or []
    rows = []
    bench_names = {"gsm8k": "GSM8K (test, first 200)", "svamp": "SVAMP (test, first 200)",
                   "bbh_mix": "BBH mix (10 subtasks $\\times$ 30)"}
    harness_names = {"ollama-direct-answer": "direct answer", "ollama-cot-8shot": "8-shot CoT"}
    for p in sorted(pairs, key=lambda p: (p["harness"], p["bench"], p["model"])):
        rows.append(f"{p['model']} & {harness_names.get(p['harness'], p['harness'])} & {bench_names.get(p['bench'], p['bench'])} & "
                    f"{p['n_items']} & {100 * p['accuracy']:.1f} \\\\")
    if not rows:
        rows = [" & ".join([MISSING] * 5) + " \\\\"]
    (PAPER / "results_table_testbed.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mac("RNItemsTotal", fmt_int(sum(p["n_items"] for p in pairs)) if pairs else MISSING)
    mac("RNModels", fmt_int(len({p["model"] for p in pairs})) if pairs else MISSING)
    mac("RNBenches", fmt_int(len({p["bench"] for p in pairs})) if pairs else MISSING)
    mac("RNHarnesses", fmt_int(len({p["harness"] for p in pairs})) if pairs else MISSING)
    accs = [p["accuracy"] for p in pairs]
    mac("RAccMin", f"{100 * min(accs):.1f}" if accs else MISSING)
    mac("RAccMax", f"{100 * max(accs):.1f}" if accs else MISSING)
    mac("RNPairsMeta", fmt_int(len(pairs)) if pairs else MISSING)


def smoke_macros(smoke):
    mac("SmokeMsShort", fmt(get(smoke, "mean_ms_short"), 2))
    mac("SmokeMsLong", fmt(get(smoke, "mean_ms_long_budget31"), 2))
    mac("SmokeMaxMsLong", fmt(get(smoke, "max_ms_long_budget31"), 1))
    cps = get(smoke, "calls_per_second_long_budget31")
    mac("SmokeCallsPerSec", fmt_int(cps))
    mac("SmokeNCases", fmt_int(len(get(smoke, "cases") or [])) if smoke else MISSING)
    cps = float(cps) if cps else None
    mac("SmokeCallsPerHour", MISSING if not cps else f"{cps * 3600 / 1e6:.1f}")


def zip_macros(info, validator_text):
    mac("ZipKB", MISSING if info is None else f"{info['bytes'] / 1000:.1f}")
    mac("ZipShaShort", MISSING if info is None else info["sha256"][:12])
    mac("ZipMembers", MISSING if info is None else ", ".join(f"\\texttt{{{m}}}" for m in info["members"]))
    if validator_text is None:
        mac("ValidatorStatus", MISSING)
    else:
        mac("ValidatorStatus", "passes" if "OK:" in validator_text and "[exit code 0]" in validator_text else "FAILS")


def config_macros(syn):
    cfg = get(syn, "config") or {}
    mac("SNSubjects", fmt_int(cfg.get("n_subjects")))
    mac("SNBenchmarks", fmt_int(cfg.get("n_benchmarks")))
    mac("SSigma", fmt(cfg.get("sigma_topic"), 2))
    mac("SGamma", fmt(cfg.get("gamma_length"), 1))
    mac("CfgAlpha", fmt(cfg.get("alpha0"), 0))
    mac("CfgTau", fmt(cfg.get("tau"), 0))
    mac("CfgKnnN", fmt(cfg.get("knn_n0"), 0))
    mac("CfgKappa", fmt(cfg.get("knn_kappa"), 0))
    mac("CfgRidgeLambda", fmt(cfg.get("ridge_lambda"), 0))
    mac("CfgLooSeMult", fmt(cfg.get("loo_se_mult"), 0))
    mac("CfgLooMinN", fmt_int(cfg.get("loo_min_n")))
    mac("CfgRidgeMinN", fmt_int(cfg.get("ridge_min_n")))


def model_constant_macros():
    """Constants read directly from the shipped model.py / labeling.py so the report cannot drift."""
    import importlib.util
    import sys
    sys.path.insert(0, str(HERE))
    import model
    import labeling
    def num(v, nd=2):
        v = float(v)
        return f"{v:g}" if nd is None else (f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:g}")
    consts = {"CAlpha": model.ALPHA0, "CTau": model.TAU, "CKnnN": model.KNN_N0, "CKappa": model.KNN_KAPPA,
              "CLooMinN": model.LOO_MIN_N, "CLooSeMult": model.LOO_SE_MULT, "CRidgeMinN": model.RIDGE_MIN_N,
              "CRidgeLambda": model.RIDGE_LAMBDA, "CRidgeCap": model.RIDGE_CAP, "CPriorCap": model.PRIOR_LOGIT_CAP,
              "CSubjectShrink": model.SUBJECT_SHRINK, "CClipLo": model.CLIP_LO, "CClipHi": model.CLIP_HI,
              "CMaxTextChars": model.MAX_TEXT_CHARS, "CYearSlope": model.PRIOR_YEAR_SLOPE,
              "CEffortLow": model.PRIOR_EFFORT_LOW, "CEffortMid": model.PRIOR_EFFORT_MID, "CEffortHigh": model.PRIOR_EFFORT_HIGH,
              "CSizeSlope": model.PRIOR_SIZE_SLOPE, "CSmallName": model.PRIOR_SMALL_NAME, "CLargeName": model.PRIOR_LARGE_NAME,
              "CHarness": model.PRIOR_HARNESS, "CTierHard": model.PRIOR_TIER_HARD, "CTierEasy": model.PRIOR_TIER_EASY,
              "CTierNum": model.PRIOR_TIER_NUM, "CPoints": model.PRIOR_POINTS, "CLength": model.PRIOR_LENGTH,
              "CBigramWeight": model.TFIDF_BIGRAM_WEIGHT, "CDedupSim": labeling.DEDUP_SIM, "CUncSkip": labeling.UNC_SKIP,
              "CUncMinN": labeling.UNC_MIN_N}
    for name, v in consts.items():
        mac(name, num(v))
    mac("CKappas", ", ".join(num(k) for k in model.KNN_KAPPAS))
    mac("CRestrictOov", "yes" if model.TFIDF_RESTRICT_OOV else "no")
    mac("CPolicy", labeling.POLICY.replace("_", "\\_"))
    mac("CPriorPLo", f"{1 / (1 + 2.718281828 ** model.PRIOR_LOGIT_CAP):.2f}")
    mac("CPriorPHi", f"{1 / (1 + 2.718281828 ** -model.PRIOR_LOGIT_CAP):.2f}")


def loo_usage_macros(prefix, summ):
    """Share of (pair, budget>=5) evaluations in which the LOO gate kept the plain shrinkage estimate."""
    # not stored per prediction; derived from the gated-vs-always-on ablation instead (see loo_macros)
    return


def main() -> None:
    syn, real = load("synthetic_main.json"), load("real_testbed_main.json")
    mac("NBoot", fmt_int(get(syn, "methods", "full", "alc", "n_boot") or get(real, "methods", "full", "alc", "n_boot")))
    summary_macros("S", syn)
    summary_macros("R", real)
    config_macros(syn)
    table_main(syn, real)
    table_budget(syn, real)
    sweep_macros(load("synthetic_signal_sweep.json"))
    alpha_macros("S", load("alpha0_sensitivity_synthetic.json"), "synthetic")
    alpha_macros("R", load("alpha0_sensitivity_real.json"), "real")
    loo_macros("S", load("loo_gating_ablation_synthetic.json"), load("loo_margin_sensitivity_synthetic.json"), "synthetic")
    loo_macros("R", load("loo_gating_ablation_real.json"), load("loo_margin_sensitivity_real.json"), "real")
    embedding_macros(load("real_testbed_embedding.json"))
    tfidf_macros("S", load("tfidf_variants_synthetic.json"), "synthetic")
    tfidf_macros("R", load("tfidf_variants_real.json"), "real")
    policy_macros(load("acquisition_policies_synthetic.json"), load("acquisition_policies_real.json"))
    replication_macros(load("synthetic_replication.json"))
    real_seeds_macros(load("real_testbed_seeds.json"))
    grid_macros(load("knn_hparam_grid_synthetic.json"))
    testbed_macros(load("real_testbed_pairs.json"))
    smoke_macros(load("smoke_test.json"))
    model_constant_macros()
    vpath = RESULTS / "validator_output.txt"
    zip_macros(load("submission_zip_info.json"), vpath.read_text(encoding="utf-8") if vpath.exists() else None)
    text = "% generated by experiments/src/make_macros.py -- do not edit\n" + "\n".join(macros) + "\n"
    (PAPER / "results_macros.tex").write_text(text, encoding="utf-8")
    n_missing = text.count(MISSING)
    print(f"wrote {PAPER / 'results_macros.tex'}: {len(macros)} macros, {n_missing} missing values")


if __name__ == "__main__":
    main()
