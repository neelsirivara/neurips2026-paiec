"""Emit paper/results_macros.tex and the generated table bodies from experiments/results/*.json.

Every number in the report comes from here.  Missing inputs produce \\textbf{??} so that the
PDF never silently shows a stale or invented value.

    python make_macros.py
"""
from __future__ import annotations

import hashlib
import json
import math
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
                 "full": "full: LOO-gated ridge + $k$NN (rounds 1--3)",
                 "oracle_rate": "oracle: pair success rate", "oracle_p": "oracle: true item probability"}
POLICY_LABELS = {"random": "random (evaluator default)", "first": "stream order",
                 "novelty": "novelty filter (rounds 1--3)", "novelty_unc": "novelty + uncertainty",
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


_MAC_NAMES: set = set()


def mac(name, value):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    if name in _MAC_NAMES:
        raise ValueError(f"duplicate macro {name}")
    _MAC_NAMES.add(name)
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
    # first positive budget at which the empirical mean beats the constant 0.5
    emp = get(summ, "methods", "empirical_mean", "brier_by_budget") or {}
    first = next((b for b in BUDGETS[1:] if emp.get(b) is not None and float(emp[b]) < 0.25), None)
    mac(prefix + "EmpOvertake", MISSING if first is None else first)
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
              "CUncMinN": labeling.UNC_MIN_N, "CIntercept": model.PRIOR_INTERCEPT}
    for name, v in consts.items():
        mac(name, num(v))
    mac("CNProvider", fmt_int(len(model.PRIOR_PROVIDER)))
    mac("CNSubjectTerms", fmt_int(len(model.SUBJECT_TERM_NAMES)))
    mac("CNItemTerms", fmt_int(len(model.ITEM_TERM_NAMES)))
    mac("CKappas", ", ".join(num(k) for k in model.KNN_KAPPAS))
    mac("CRestrictOov", "yes" if model.TFIDF_RESTRICT_OOV else "no")
    mac("CPolicy", labeling.POLICY.replace("_", "\\_"))
    mac("CPriorPLo", f"{1 / (1 + 2.718281828 ** model.PRIOR_LOGIT_CAP):.2f}")
    mac("CPriorPHi", f"{1 / (1 + 2.718281828 ** -model.PRIOR_LOGIT_CAP):.2f}")


def listing_macros(lst):
    """Public file listing of the gated training release (measurement_db_listing.json)."""
    mac("MdbNDirs", fmt_int(get(lst, "n_benchmark_dirs")))
    mac("MdbRevisionShort", (get(lst, "revision") or "??????????")[:10] if lst else MISSING)
    mac("MdbGated", str(get(lst, "gated")) if lst else MISSING)
    for t, code in (("items", "Items"), ("responses", "Resp"), ("traces", "Traces"), ("assets", "Assets")):
        mac("Mdb" + code + "Gb", fmt(get(lst, "table_gb", t), 1))
    mac("MdbCoreGb", fmt(get(lst, "core_tables_gb"), 1))
    mac("MdbCoreNoItemsGb", fmt(get(lst, "core_tables_without_items_gb"), 1))
    mac("MdbNTraces", fmt_int(get(lst, "table_file_counts", "traces")))


def dry_run_macros(dry):
    """Synthetic dry run of the fitting pipeline (prior_fit_dry_run.json)."""
    mac("DryNBench", fmt_int(get(dry, "n_benchmarks")))
    mac("DryNPairs", fmt_int(get(dry, "n_pairs")))
    mac("DryNRows", fmt_int(get(dry, "n_rows")))
    rec = get(dry, "dry_run_recovery")
    mac("DryMaxAbsErr", fmt(get(rec, "max_abs_error_slopes"), 2))
    mac("DryMeanAbsErr", fmt(get(rec, "mean_abs_error_slopes"), 2))
    mac("DrySignAgree", fmt_int(get(rec, "sign_agreement_slopes")))
    mac("DryNSlopes", fmt_int(get(rec, "n_slopes")))
    mac("DryBenchSd", fmt(get(dry, "benchmark_effect_sd"), 2))
    sel = get(dry, "cv", "selected")
    mac("DrySelC", MISSING if get(sel, "C") is None else f"{float(sel['C']):g}")
    mac("DrySelAlpha", MISSING if get(sel, "alpha0") is None else f"{float(sel['alpha0']):g}")
    key = None if sel is None else f"fit_C{float(sel['C']):g}_cap{float(sel['cap']):g}_alpha{float(sel['alpha0']):g}"
    paired_macros("DryPairBZero", get(dry, "cv", key, "paired_b0_fit_minus_hand") if key else None)
    paired_macros("DryPairAlc", get(dry, "cv", key, "paired_alc_fit_minus_hand") if key else None)


def fit_macros(fit, tb):
    """Fit on the real release, if it was run (prior_fit.json / prior_fit_testbeds.json)."""
    mac("FitRan", "yes" if fit and get(fit, "fitted") else "no")
    mac("FitNBench", fmt_int(get(fit, "n_benchmarks")))
    mac("FitNPairs", fmt_int(get(fit, "n_pairs")))
    mac("FitNRows", fmt_int(get(fit, "n_rows")))
    sel = get(fit, "cv", "selected")
    key = None if sel is None else f"fit_C{float(sel['C']):g}_cap{float(sel['cap']):g}_alpha{float(sel['alpha0']):g}"
    paired_macros("FitPairBZero", get(fit, "cv", key, "paired_b0_fit_minus_hand") if key else None)
    paired_macros("FitPairAlc", get(fit, "cv", key, "paired_alc_fit_minus_hand") if key else None)
    for m, code in (("full", "Full"), ("shrink", "Shrink"), ("prior", "Prior")):
        paired_macros("FitRealPair" + code, get(tb, "testbeds", "real", m, "paired_fit_minus_hand"))
        paired_macros("FitSynPair" + code, get(tb, "testbeds", "synthetic", m, "paired_fit_minus_hand"))
        mac("FitReal" + code + "Alc", fmt(get(tb, "testbeds", "real", m, "fitted", "mean")))
        mac("FitSyn" + code + "Alc", fmt(get(tb, "testbeds", "synthetic", m, "fitted", "mean")))
    ship = get(tb, "ship_rule", "ship")
    mac("FitShip", MISSING if ship is None else ("shipped" if ship else "not shipped"))
    # the training release evaluated with the closed-form protocol (hand-set prior, references)
    hand4 = get(fit, "cv", "hand_alpha4")
    mac("FitRelNPairsAlc", fmt_int(get(hand4, "n_pairs_alc")))
    mac("FitRelNPairsBZero", fmt_int(get(hand4, "n_pairs_b0")))
    mac("FitRelHandBZero", fmt(get(hand4, "b0", "mean")))
    mac("FitRelHandAlc", fmt(get(hand4, "alc", "mean")))
    mac("FitRelHandAlcLo", fmt(get(hand4, "alc", "ci_lo")))
    mac("FitRelHandAlcHi", fmt(get(hand4, "alc", "ci_hi")))
    mac("FitRelConstAlc", fmt(get(hand4, "alc_const", "mean")))
    mac("FitRelEmpAlc", fmt(get(hand4, "alc_emp", "mean")))
    mac("FitRelEmpAlcLo", fmt(get(hand4, "alc_emp", "ci_lo")))
    mac("FitRelEmpAlcHi", fmt(get(hand4, "alc_emp", "ci_hi")))
    mac("FitRelPriorOnlyAlc", fmt(get(hand4, "alc_prior_only", "mean")))
    for a, word in (("2", "Two"), ("4", "Four"), ("8", "Eight")):
        mac("FitRelHandAlpha" + word + "Alc", fmt(get(fit, "cv", f"hand_alpha{a}", "alc", "mean")))
    mac("FitRelIntOnlyBZero", fmt(get(fit, "cv", "intercept_only_alpha4", "b0", "mean")))
    mac("FitRelIntOnlyAlc", fmt(get(fit, "cv", "intercept_only_alpha4", "alc", "mean")))
    sel = get(fit, "cv", "selected")
    mac("FitSelC", MISSING if get(sel, "C") is None else f"{float(sel['C']):g}")
    mac("FitSelCap", MISSING if get(sel, "cap") is None else f"{float(sel['cap']):g}")
    mac("FitSelAlpha", MISSING if get(sel, "alpha0") is None else f"{float(sel['alpha0']):g}")
    mac("FitSelBZero", fmt(get(sel, "cv_b0")))
    mac("FitSelAlc", fmt(get(sel, "cv_alc")))
    # the least-regularised fitted prior in the grid, for the trend statement
    cs = sorted({float(k.split("_")[1][1:]) for k in (get(fit, "cv") or {}) if k.startswith("fit_C")})
    if cs:
        kmin, kmax = f"fit_C{cs[0]:g}_cap1.5_alpha4", f"fit_C{cs[-1]:g}_cap1.5_alpha4"
        mac("FitCMin", f"{cs[0]:g}")
        mac("FitCMax", f"{cs[-1]:g}")
        mac("FitCMinBZero", fmt(get(fit, "cv", kmin, "b0", "mean")))
        mac("FitCMaxBZero", fmt(get(fit, "cv", kmax, "b0", "mean")))
        mac("FitCMinAlc", fmt(get(fit, "cv", kmin, "alc", "mean")))
        mac("FitCMaxAlc", fmt(get(fit, "cv", kmax, "alc", "mean")))
    else:
        for n in ("FitCMin", "FitCMax", "FitCMinBZero", "FitCMaxBZero", "FitCMinAlc", "FitCMaxAlc"):
            mac(n, MISSING)
    diag = get(fit, "diagnostics")
    mac("FitRateMean", fmt(get(diag, "pair_rate_mean"), 3))
    mac("FitRateSd", fmt(get(diag, "pair_rate_sd"), 3))
    mac("FitHandPMean", fmt(get(diag, "hand_p0_mean"), 3))
    mac("FitHandPSd", fmt(get(diag, "hand_p0_sd"), 3))
    mac("FitBenchRateSd", fmt(get(diag, "benchmark_rate_sd"), 3))
    mac("FitBenchEffectSd", fmt(get(fit, "benchmark_effect_sd"), 2))
    mac("FitNProviders", fmt_int(len(get(fit, "providers") or [])) if fit else MISSING)
    mac("FitRevisionShort", (get(fit, "revision") or "??????????")[:10] if fit else MISSING)
    mac("FitNEligible", fmt_int(get(fit, "n_eligible_dirs")))
    mac("FitMaxItemsPerPair", fmt_int(get(fit, "max_items_per_pair")))
    codes = {"PRIOR_YEAR_SLOPE": "Year", "PRIOR_EFFORT_LOW": "EffortLow", "PRIOR_EFFORT_MID": "EffortMid", "PRIOR_EFFORT_HIGH": "EffortHigh",
             "PRIOR_SIZE_SLOPE": "Size", "PRIOR_SMALL_NAME": "SmallName", "PRIOR_LARGE_NAME": "LargeName", "PRIOR_HARNESS": "Harness",
             "PRIOR_TIER_HARD": "TierHard", "PRIOR_TIER_EASY": "TierEasy", "PRIOR_TIER_NUM": "TierNum", "PRIOR_POINTS": "Points",
             "PRIOR_LENGTH": "Length", "PRIOR_INTERCEPT": "Intercept"}
    for name, code in codes.items():
        mac("FitCoef" + code, fmt(get(fit, "fitted", name), 2, sign=True))
        # effective hand-set value (subject terms carry the SUBJECT_SHRINK multiplier)
        hv = get(fit, "hand_set", name)
        if hv is not None and name in ("PRIOR_YEAR_SLOPE", "PRIOR_EFFORT_LOW", "PRIOR_EFFORT_MID", "PRIOR_EFFORT_HIGH", "PRIOR_SIZE_SLOPE",
                                       "PRIOR_SMALL_NAME", "PRIOR_LARGE_NAME", "PRIOR_HARNESS"):
            hv = float(hv) * float(get(fit, "hand_set", "SUBJECT_SHRINK") or 1.0)
        mac("HandEffCoef" + code, fmt(hv, 2, sign=True))
    prov = get(fit, "fitted", "PRIOR_PROVIDER") or {}
    if prov:
        hi = max(prov, key=prov.get)
        lo = min(prov, key=prov.get)
        mac("FitProvMaxName", hi.replace("_", "\\_"))
        mac("FitProvMax", fmt(prov[hi], 2, sign=True))
        mac("FitProvMinName", lo.replace("_", "\\_"))
        mac("FitProvMin", fmt(prov[lo], 2, sign=True))
    else:
        for n in ("FitProvMaxName", "FitProvMax", "FitProvMinName", "FitProvMin"):
            mac(n, MISSING)
    counts = get(fit, "provider_pair_counts") or {}
    mac("FitFracPairsNoProvider", MISSING if not counts or not get(fit, "n_pairs") else f"{100 * counts.get('', 0) / fit['n_pairs']:.0f}\\%")


def sensitivity_macros(sens):
    """Frontier-like subjects, analytic miscalibration cost and the offset sweep (prior_sensitivity.json)."""
    grid = get(sens, "frontier_grid") or []
    years = ("2023-06-15", "2024-06-15", "2025-06-15", "2026-06-15")
    efforts = ("", "low", "medium", "high")
    rows = []
    for y in years:
        cells = []
        for e in efforts:
            r = [g for g in grid if g["release_date"] == y and g["reasoning_effort"] == e and g["name_cue"] == "none" and g["harness"] == "default"]
            cells.append(fmt(r[0]["p0"], 2) if r else MISSING)
        for cue in ("mini", "70b", "400b"):
            r = [g for g in grid if g["release_date"] == y and g["reasoning_effort"] == "high" and g["name_cue"] == cue and g["harness"] == "agent-scaffold"]
            cells.append(fmt(r[0]["p0"], 2) if r else MISSING)
        rows.append(f"{y[:4]} & " + " & ".join(cells) + " \\\\")
    if not rows:
        rows = [" & ".join([MISSING] * 8) + " \\\\"]
    (PAPER / "results_table_frontier.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    p0s = [g["p0"] for g in grid]
    mac("SensPMin", fmt(min(p0s), 2) if p0s else MISSING)
    mac("SensPMax", fmt(max(p0s), 2) if p0s else MISSING)
    mac("SensNGrid", fmt_int(len(grid)) if grid else MISSING)
    hard = [g for g in grid if g["release_date"] == "2026-06-15" and g["reasoning_effort"] == "high" and g["name_cue"] == "400b" and g["harness"] == "agent-scaffold"]
    mac("SensPMaxTierHard", fmt(hard[0]["p0_tier_hard"], 2) if hard else MISSING)
    for key, code in (("sample_gpt4o", "SampleGptFourO"), ("smoke_2026_medium", "SmokeSubject"), ("testbed_qwen_1_5b", "TestbedQwenSmall"),
                      ("testbed_llama_3b", "TestbedLlama")):
        mac("SensP" + code, fmt(get(sens, "named_subjects", key, "p0"), 2))
    # miscalibration table: rows r, columns p0 (ALC excess over the oracle; dagger when the constant 0.5 is better)
    mis = get(sens, "miscalibration")
    cells = {(c["r"], c["p0"]): c for c in (get(mis, "cells") or [])}
    rates = get(mis, "rates") or []
    priors = get(mis, "priors") or []
    rows = []
    for r in rates:
        if r in (0.1, 0.9):
            continue
        vals = []
        for p in priors:
            c = cells.get((r, p))
            if c is None:
                vals.append(MISSING)
            else:
                vals.append(fmt(c["alc_excess"], 3) + ("" if c["beats_const"] else "$^{\\dagger}$"))
        rows.append(f"{r:.1f} & " + " & ".join(vals) + f" & {fmt((0.5 - r) ** 2, 3)} \\\\")
    if not rows:
        rows = [" & ".join([MISSING] * 8) + " \\\\"]
    (PAPER / "results_table_miscal.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mac("MiscalAlpha", fmt(get(mis, "alpha0"), 0))
    words = {0.2: "Two", 0.3: "Three", 0.4: "Four", 0.5: "Five", 0.6: "Six", 0.7: "Seven", 0.8: "Eight"}
    for (r, p), code in (((0.3, 0.7), "RThreePSeven"), ((0.5, 0.7), "RFivePSeven"), ((0.7, 0.7), "RSevenPSeven"), ((0.3, 0.5), "RThreePFive"),
                         ((0.5, 0.5), "RFivePFive"), ((0.3, 0.3), "RThreePThree"), ((0.5, 0.6), "RFivePSix"), ((0.3, 0.6), "RThreePSix")):
        c = cells.get((r, p))
        mac("Miscal" + code, fmt(get(c, "alc_excess"), 3))
        mac("Miscal" + code + "BZero", fmt(get(c, "excess_by_budget", "0"), 3))
        mac("Miscal" + code + "BSeven", fmt(get(c, "excess_by_budget", "7"), 3))
    mac("MiscalConstRThree", fmt(get(cells.get((0.3, 0.5)), "alc_const"), 3) if cells else MISSING)
    mac("MiscalConstRThreeExcess", fmt((0.5 - 0.3) ** 2, 3))
    # offset sweep
    offs = (("-1.0", "MinusOne"), ("-0.5", "MinusHalf"), ("+0.0", "Zero"), ("+0.5", "PlusHalf"), ("+1.0", "PlusOne"))
    rows = []
    for off, word in offs:
        cells_ = []
        for name, prefix in (("real", "R"), ("synthetic", "S")):
            rec = get(sens, "offset_sweep", name, "offsets", off)
            for m, mcode in (("prior", "Prior"), ("shrink", "Shrink"), ("full", "Full")):
                mac(prefix + "Off" + word + mcode + "Alc", fmt(get(rec, m, "alc", "mean")))
                paired_macros(prefix + "Off" + word + mcode + "VsZero", get(rec, "paired_vs_zero", m))
                cells_.append(fmt(get(rec, m, "alc", "mean")))
            mac(prefix + "Off" + word + "FullBZero", fmt(get(rec, "full", "brier_by_budget", "0"), 3))
            mac(prefix + "Off" + word + "FullBOne", fmt(get(rec, "full", "brier_by_budget", "1"), 3))
        rows.append(f"${off.replace('+', '+')}$ & " + " & ".join(cells_) + " \\\\")
    (PAPER / "results_table_offset.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for name, prefix in (("real", "R"), ("synthetic", "S")):
        mac(prefix + "OffFullRange", fmt(get(sens, "offset_sweep", name, "full_alc_range")))
        best = get(sens, "offset_sweep", name, "best_offset_full")
        mac(prefix + "OffBest", MISSING if best is None else f"${best}$")


def novelty_macros(nov):
    """Where the shipped novelty filter fires (novelty_filter_stats.json)."""
    for name, prefix in (("real", "R"), ("synthetic", "S")):
        rec = get(nov, name)
        mac(prefix + "NovPairsFired", fmt_int(get(rec, "n_pairs_fired")))
        mac(prefix + "NovNPairs", fmt_int(get(rec, "n_pairs")))
        mac(prefix + "NovSkipped", fmt_int(get(rec, "total_skipped")))
        mac(prefix + "NovMaxSkipped", fmt_int(get(rec, "max_skipped_per_pair")))
        mac(prefix + "NovMaxSim", fmt(get(rec, "max_sim_skipped"), 2))
        benches = get(rec, "benches_fired")
        names = {"gsm8k": "GSM8K", "svamp": "SVAMP", "bbh_mix": "BBH"}
        mac(prefix + "NovBenches", MISSING if benches is None else (", ".join(names.get(b, b.replace("_", "\\_")) for b in benches) or "none"))


def loo_usage_macros(prefix, summ):
    """Share of (pair, budget>=5) evaluations in which the LOO gate kept the plain shrinkage estimate."""
    # not stored per prediction; derived from the gated-vs-always-on ablation instead (see loo_macros)
    return


def _pct(v, nd=0):
    if v is None:
        return MISSING
    try:
        return f"{100 * float(v):.{nd}f}"
    except Exception:
        return MISSING


def cluster_macros(prefix, d, nd=4):
    """mean [lo, hi] and a significance word of a cluster-bootstrap result (text_prior.cluster_boot)."""
    mac(prefix + "Mean", fmt(get(d, "mean"), nd, sign=True))
    mac(prefix + "Lo", fmt(get(d, "ci_lo"), nd, sign=True))
    mac(prefix + "Hi", fmt(get(d, "ci_hi"), nd, sign=True))
    s = get(d, "significant")
    mac(prefix + "Sig", MISSING if s is None else ("significant" if s else "not significant"))
    mac(prefix + "FracNeg", _pct(get(d, "frac_negative")))


def round3_extra_macros(cv, explore, runtime, equiv, addendum_ok):
    """Round 3: decomposition, exploratory addendum, engineering gates, shipped-file equivalence, data list."""
    p = get(cv, "summary", "paired_vs_hand") or {}
    cluster_macros("TpDiffIntShift", get(p, "intercept_shift_only", "alc_cluster"))
    cluster_macros("TpDiffIntOnly", get(p, "intercept_only", "alc_cluster"))
    cluster_macros("TpDiffBenchOracle", get(p, "benchmark_mean_oracle", "alc_cluster"))
    m = get(cv, "summary", "methods") or {}
    mac("TpIntShiftAlc", fmt(get(m, "intercept_shift_only", "alc", "mean")))
    mac("TpIntShiftBZero", fmt(get(m, "intercept_shift_only", "b0", "mean")))
    g = get(p, "benchmark_mean_oracle", "gap_closed")
    for b, w in (("0", "Zero"), ("1", "One"), ("3", "Three")):
        mac(f"TpBenchOracleGap{w}", _pct(get(g, b, "ratio")))
    mac("TpNClusters", fmt_int(get(p, "text", "alc_cluster", "n_clusters")))
    folds = get(cv, "folds") or []
    mac("TpNFoldsNegW", f"{sum(1 for f in folds if f['calibration']['W'] < 0)}" if folds else MISSING)
    mac("TpNFoldsAlphaMax", f"{sum(1 for f in folds if float(f['alpha']) == max(float(a) for a in cv['settings']['alpha_grid']))}"
        if folds else MISSING)
    mac("TpAlphaGridMax", f"{max(float(a) for a in cv['settings']['alpha_grid']):g}" if cv else MISSING)
    ib = [f.get("intercept_alone_B") for f in folds if f.get("intercept_alone_B") is not None]
    mac("TpIntShiftBMin", fmt(min(ib), 3, sign=True) if ib else MISSING)
    mac("TpIntShiftBMax", fmt(max(ib), 3, sign=True) if ib else MISSING)
    mac("TpAddendumConsistent", "verified unchanged" if addendum_ok else r"\textbf{changed}")
    for name, code in (("E1_heavier_penalty", "EOne"), ("E2_statistics_only", "ETwo")):
        v = get(explore, "variants", name)
        cluster_macros(f"Tp{code}Diff", get(v, "summary", "paired_vs_hand", name, "alc_cluster"))
        mac(f"Tp{code}Alc", fmt(get(v, "summary", "methods", name, "alc", "mean")))
        mac(f"Tp{code}BZero", fmt(get(v, "summary", "methods", name, "b0", "mean")))
        fl = get(v, "folds") or []
        mac(f"Tp{code}Alphas", ", ".join(f"{a:g}" for a in sorted({float(f['alpha']) for f in fl})) if fl else MISSING)
        mac(f"Tp{code}NegW", f"{sum(1 for f in fl if f['calibration']['W'] < 0)}" if fl else MISSING)
        mac(f"Tp{code}Grid", ", ".join(f"{float(a):g}" for a in get(v, "alpha_grid") or []) if v else MISSING)
    thr = get(runtime, "throughput") or {}
    for key, code in (("short15", "Short"), ("words300", "Mid"), ("words2000", "Long")):
        mac(f"TpCps{code}", fmt_int(get(thr, key, "calls_per_s")))
    mac("TpLoadMs", fmt(get(runtime, "load_ms"), 0))
    mac("TpWeightsMB", fmt((get(runtime, "weights_file_bytes") or 0) / 1e6, 2) if runtime else MISSING)
    mac("TpSmokeCases", fmt_int(get(runtime, "smoke", "n_cases")))
    gates = [get(runtime, k) for k in ("gate_throughput_200", "gate_weights_5mb", "gate_smoke")]
    mac("TpGates", MISSING if any(x is None for x in gates) else ("pass" if all(gates) else "fail"))
    mac("TpEquivN", fmt_int(get(equiv, "n_cases")))
    mac("TpEquivSame", fmt_int(get(equiv, "n_bit_identical")))
    mac("TpZipUnchanged", MISSING if get(equiv, "zip_unchanged") is None else ("unchanged" if equiv["zip_unchanged"] else "changed"))
    mac("TpZipShaShort", (get(equiv, "zip_sha256") or "????????")[:12] if equiv else MISSING)
    fit = load("prior_fit.json")
    dirs = get(cv, "folds_dirs")
    names = sorted({d for f in dirs for d in f}) if dirs else (get(fit, "data_sources") or [])
    mac("TpBenchList", ", ".join(r"\texttt{" + d.replace("_", r"\_") + "}" for d in names) if names else MISSING)


def round3_macros(cv, inter, tb, final, check, prereg_ok):
    """Round 3: item-text difficulty prior (results/round3/*.json)."""
    mac("TpPreregSha", (get(cv, "prereg_sha256") or "??????????")[:10] if cv else MISSING)
    mac("TpPreregConsistent", "verified unchanged" if prereg_ok else r"\textbf{changed}")
    mac("TpNPairs", fmt_int(get(cv, "n_pairs")))
    mac("TpNBench", fmt_int(get(cv, "n_benchmarks")))
    mac("TpNItems", fmt_int(get(cv, "n_items_featurised")))
    mac("TpNTargetItems", fmt_int(get(cv, "n_target_items")))
    mac("TpNnz", fmt_int(get(cv, "X_nnz")))
    mac("TpBits", MISSING if get(cv, "settings", "bits") is None else f"{2 ** int(cv['settings']['bits']):,}")
    rep = get(cv, "reproduction")
    mac("TpReproExact", MISSING if rep is None else ("exactly" if rep.get("exact") else "not exactly"))
    m = get(cv, "summary", "methods")
    for key, code in (("hand", "Hand"), ("text", "Text"), ("text_alpha2", "TextAlphaTwo"), ("text_alpha8", "TextAlphaEight"),
                      ("intercept_only", "IntOnly"), ("benchmark_mean_oracle", "BenchOracle")):
        mac(f"Tp{code}BZero", fmt(get(m, key, "b0", "mean")))
        mac(f"Tp{code}Alc", fmt(get(m, key, "alc", "mean")))
        mac(f"Tp{code}AlcLo", fmt(get(m, key, "alc", "ci_lo")))
        mac(f"Tp{code}AlcHi", fmt(get(m, key, "alc", "ci_hi")))
        for b, w in (("0", "Zero"), ("1", "One"), ("3", "Three"), ("7", "Seven"), ("15", "Fifteen"), ("31", "ThirtyOne")):
            mac(f"Tp{code}Bud{w}", fmt(get(m, key, "brier_by_budget", b)))
    mac("TpOracleRate", fmt(get(m, "hand", "oracle_rate_brier")))
    mac("TpNPairsAlc", fmt_int(get(m, "hand", "n_pairs_alc")))
    mac("TpNPairsBZero", fmt_int(get(m, "hand", "n_pairs_b0")))
    p = get(cv, "summary", "paired_vs_hand", "text")
    cluster_macros("TpDiffAlc", get(p, "alc_cluster"))
    cluster_macros("TpDiffBZero", get(p, "b0_cluster"))
    paired_macros("TpDiffAlcPairBoot", get(p, "alc_pair_boot"))
    for b, w in (("0", "Zero"), ("1", "One"), ("3", "Three"), ("7", "Seven"), ("15", "Fifteen"), ("31", "ThirtyOne")):
        cluster_macros(f"TpDiffBud{w}", get(p, "by_budget_cluster", b))
        g = get(p, "gap_closed", b)
        mac(f"TpGap{w}", _pct(get(g, "ratio"), 1))
        mac(f"TpGap{w}Lo", _pct(get(g, "ci_lo"), 1))
        mac(f"TpGap{w}Hi", _pct(get(g, "ci_hi"), 1))
    for key, code in (("text_alpha2", "AlphaTwo"), ("text_alpha8", "AlphaEight")):
        cluster_macros(f"TpDiff{code}", get(cv, "summary", "paired_vs_hand", key, "alc_cluster"))
    folds = get(cv, "folds") or []
    if folds:
        alphas = sorted({float(f["alpha"]) for f in folds})
        caps = sorted({float(f["cap"]) for f in folds})
        lams = sorted({float(f["lambda"]) for f in folds})
        ws = [float(f["calibration"]["W"]) for f in folds]
        bs = [float(f["calibration"]["B"]) for f in folds]
        mac("TpFoldAlphas", ", ".join(f"{a:g}" for a in alphas))
        mac("TpFoldCaps", ", ".join(f"{c:g}" for c in caps))
        mac("TpFoldLambdas", ", ".join(f"{c:g}" for c in lams))
        mac("TpFoldWMin", fmt(min(ws), 2))
        mac("TpFoldWMax", fmt(max(ws), 2))
        mac("TpFoldBMin", fmt(min(bs), 2, sign=True))
        mac("TpFoldBMax", fmt(max(bs), 2, sign=True))
        r = [1 - f["test_target_mse"] / f["test_target_mse_constant"] for f in folds if f.get("test_target_mse_constant")]
        mac("TpFoldRsqMin", fmt(min(r), 2) if r else MISSING)
        mac("TpFoldRsqMax", fmt(max(r), 2) if r else MISSING)
        ri = [1 - min(f["inner_mse"].values()) / f["inner_mse_constant"] for f in folds]
        mac("TpInnerRsqMin", fmt(min(ri), 2))
        mac("TpInnerRsqMax", fmt(max(ri), 2))
    else:
        for n in ("TpFoldAlphas", "TpFoldCaps", "TpFoldLambdas", "TpFoldWMin", "TpFoldWMax", "TpFoldBMin", "TpFoldBMax",
                  "TpFoldRsqMin", "TpFoldRsqMax", "TpInnerRsqMin", "TpInnerRsqMax"):
            mac(n, MISSING)
    pf = get(cv, "per_fold") or []
    nbetter = sum(1 for f in pf if f["diff"]["mean"] < 0)
    mac("TpFoldsBetter", f"{nbetter}" if pf else MISSING)
    mac("TpNFolds", f"{len(pf)}" if pf else MISSING)
    nsig = sum(1 for f in pf if f["diff"]["ci_hi"] < 0)
    mac("TpFoldsSigBetter", f"{nsig}" if pf else MISSING)
    pb = get(cv, "per_benchmark") or {}
    if pb:
        diffs = {k: v["text_alc"] - v["hand_alc"] for k, v in pb.items()}
        mac("TpBenchBetter", f"{sum(1 for v in diffs.values() if v < 0)}")
        mac("TpBenchN", f"{len(diffs)}")
    else:
        mac("TpBenchBetter", MISSING)
        mac("TpBenchN", MISSING)
    dup = get(cv, "duplicates")
    mac("TpDupMean", _pct(get(dup, "mean_dup_fraction_pairs"), 2))
    mac("TpDupPairsOverFive", _pct(get(dup, "frac_pairs_dup_over_5pct")))
    mac("TpDupPairsOverHalf", _pct(get(dup, "frac_pairs_dup_over_50pct")))
    cluster_macros("TpDiffLowDup", get(dup, "alc_diff_low_dup_pairs"))
    mac("TpLowDupN", fmt_int(get(dup, "alc_diff_low_dup_pairs", "n_pairs")))
    cluster_macros("TpDiffHighDup", get(dup, "alc_diff_high_dup_pairs"))
    mac("TpHighDupN", fmt_int(get(dup, "alc_diff_high_dup_pairs", "n_pairs")))
    rule_a = get(cv, "rule_a_cv_alc_ci_below_zero")
    mac("TpRuleA", MISSING if rule_a is None else ("holds" if rule_a else "fails"))
    # secondary interaction
    pi = get(inter, "summary", "paired_vs_text", "text_interaction", "alc_cluster")
    mac("TpInterRan", "yes" if inter else "no")
    cluster_macros("TpDiffInter", pi)
    # testbeds
    for name, code in (("real", "R"), ("synthetic", "S")):
        t = get(tb, "testbeds", name)
        for meth, mc in (("full", "Full"), ("shrink", "Shrink"), ("prior", "Prior")):
            mac(f"Tp{code}{mc}Hand", fmt(get(t, meth, "hand", "mean")))
            mac(f"Tp{code}{mc}Text", fmt(get(t, meth, "text", "mean")))
            paired_macros(f"Tp{code}Pair{mc}", get(t, meth, "paired_text_minus_hand"))
        paired_macros(f"Tp{code}TextVsEmp", get(t, "paired_text_full_vs_empirical_mean"))
        mac(f"Tp{code}NPairs", fmt_int(get(t, "n_pairs")))
    paired_macros("TpRNoBbh", get(tb, "testbeds", "real", "no_bbh", "full"))
    mac("TpRNoBbhN", fmt_int(get(tb, "testbeds", "real", "no_bbh", "n_pairs")))
    rule_b = get(tb, "rule_b_real_not_significantly_worse")
    mac("TpRuleB", MISSING if rule_b is None else ("holds" if rule_b else "fails"))
    # final fit and check
    mac("TpFinalAlpha", MISSING if get(final, "alpha") is None else f"{float(final['alpha']):g}")
    mac("TpFinalW", fmt(get(final, "calibration", "W"), 2))
    mac("TpFinalB", fmt(get(final, "calibration", "B"), 2, sign=True))
    mac("TpFinalCap", MISSING if get(final, "cap") is None else f"{float(final['cap']):g}")
    mac("TpFinalLambda", MISSING if get(final, "lambda") is None else f"{float(final['lambda']):g}")
    mac("TpFinalNnzW", fmt_int(get(final, "n_nonzero_weights")))
    mac("TpFinalMB", fmt((get(final, "weights_file_bytes") or 0) / 1e6, 2) if final else MISSING)
    mac("TpCheckN", fmt_int(get(check, "n_items")))
    mac("TpCheckIdentical", fmt_int(get(check, "n_features_bit_identical_to_cache")))
    v = get(check, "max_abs_score_diff_stdlib_vs_numpy")
    if v is None:
        mac("TpCheckMaxDiff", MISSING)
    else:
        mant, ex = f"{float(v):.1e}".split("e")
        mac("TpCheckMaxDiff", f"${mant}" + r"\times" + f"10^{{{int(ex)}}}$")
    ship = None if rule_a is None or rule_b is None else bool(rule_a and rule_b)
    mac("TpShip", MISSING if ship is None else ("shipped" if ship else "not shipped"))


def table_textprior(cv, explore=None):
    """CV table: hand-set vs text prior vs references (B0, B1, B3, B7, ALC)."""
    m = dict(get(cv, "summary", "methods") or {})
    for name in ("E1_heavier_penalty", "E2_statistics_only"):
        r = get(explore, "variants", name, "summary", "methods", name)
        if r is not None:
            m[name] = r
    rows = [("hand", "hand-set prior (shipped)"), ("text", "hand-set + item-text prior (pre-registered)"),
            ("text_alpha2", "\\quad same, $\\alpha_0=2$"), ("text_alpha8", "\\quad same, $\\alpha_0=8$"),
            ("E1_heavier_penalty", "\\quad exploratory E1: penalty up to $10^4$"),
            ("E2_statistics_only", "\\quad exploratory E2: 16 statistics only"),
            ("intercept_shift_only", "hand-set + fitted intercept shift only"),
            ("intercept_only", "intercept only (training mean)"),
            ("benchmark_mean_oracle", "oracle: held-out benchmark mean")]
    lines = []
    for key, label in rows:
        r = m.get(key)
        cells = [fmt(get(r, "b0", "mean"))] + [fmt(get(r, "brier_by_budget", b)) for b in ("1", "3", "7")] + [fmt(get(r, "alc", "mean"))]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    hand = m.get("hand")
    orc = fmt(get(hand, "oracle_rate_brier"))
    lines.append(f"oracle: pair success rate & -- & {orc} & {orc} & {orc} & -- \\\\")
    (PAPER / "results_table_textprior.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================= round 4: cross-pair evidence
R4_BUDGET_IDX = {"Zero": 0, "One": 1, "Three": 2, "Seven": 3, "Fifteen": 4, "ThirtyOne": 5}
R4_CONFIG_WORD = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}


def _sha_ok(path: Path, recorded) -> bool:
    return bool(path.exists() and recorded and hashlib.sha256(path.read_bytes()).hexdigest() == recorded)


def _sci(v):
    if v is None:
        return MISSING
    mant, ex = f"{float(v):.1e}".split("e")
    return f"${mant}" + r"\times" + f"10^{{{int(ex)}}}$"


def _tau_label(label: str) -> dict:
    out = {}
    for part in (label or "").split("|")[1:]:
        for key in ("bench", "namebench", "name", "subj", "pair", "item", "probit"):
            if part.startswith(key) and part[len(key):].replace(".", "", 1).isdigit():
                if key == "name" and part.startswith("namebench"):
                    continue
                out[key] = float(part[len(key):])
                break
    return out


def round4_macros(cv, add, real, acq, realacq, runtime, equiv, perpair, frame, loop, smoke_sub) -> None:
    pre = RESULTS / "round4" / "PREREG.md"
    mac("XpPreregSha", MISSING if not pre.exists() else hashlib.sha256(pre.read_bytes()).hexdigest()[:12])
    recs = [cv, add, real, acq, realacq, runtime, equiv, perpair]
    ok = all(_sha_ok(pre, get(r, "prereg_sha256")) for r in recs if r is not None) and all(r is not None for r in recs)
    mac("XpPreregConsistent", "consistent" if ok else "INCONSISTENT")
    addp = RESULTS / "round4" / "PREREG_ADDENDUM_1.md"
    mac("XpAddendumConsistent", "consistent" if _sha_ok(addp, get(add, "addendum_sha256")) else "INCONSISTENT")
    # frame
    for key, name in (("n_pairs", "FrNPairs"), ("n_pairs_80", "FrNPairsEighty"), ("n_rows", "FrNRows"), ("n_items", "FrNItems"),
                      ("n_benchmarks", "FrNBench"), ("n_subject_keys", "FrNSubjKeys"), ("n_names", "FrNNames"),
                      ("max_items_per_pair", "FrMaxItems"), ("names_on_2plus_benchmarks", "FrNamesMulti"),
                      ("subject_keys_on_2plus_benchmarks", "FrSubjMulti"), ("pairs_per_benchmark_max", "FrPairsPerBenchMax")):
        mac("Xp" + name, fmt_int(get(frame, key)))
    mac("XpFrPairsPerBenchMedian", fmt(get(frame, "pairs_per_benchmark_median"), 1))
    mac("XpFrNamesPerBenchMedian", fmt(get(frame, "names_per_benchmark_median"), 0))
    mac("XpFrShareNameBench", _pct(get(frame, "frac_pairs_sharing_name_and_benchmark"), 1))
    mac("XpFrItemShared", _pct(get(frame, "frac_rows_on_item_answered_by_2plus_pairs"), 1))
    # selection
    sel = get(cv, "selection") or {}
    mac("XpGridN", fmt_int(get(sel, "n_settings")))
    rob = get(add, "selection_robust") or {}
    mac("XpGridExtN", fmt_int(get(rob, "n_settings")))
    ship = get(add, "decision", "ship_label")
    t = _tau_label(ship)
    for key, name in (("bench", "Bench"), ("name", "Name"), ("subj", "Subj"), ("namebench", "Namebench"), ("pair", "Pair"),
                      ("item", "Item")):
        v = t.get(key)
        mac("XpTau" + name, MISSING if v is None else f"{v:g}")
    mac("XpProbit", MISSING if "probit" not in t else ("on" if t["probit"] else "off"))
    folds = get(sel, "per_fold") or []
    mac("XpSelNFoldsSame", fmt_int(sum(1 for r in folds if r.get("label") == get(sel, "all", "label"))))
    mac("XpSelNFolds", fmt_int(len(folds)))
    rfolds = get(rob, "per_fold") or []
    mac("XpRobNFoldsSame", fmt_int(sum(1 for r in rfolds if r.get("label") == get(rob, "all", "label"))))
    mac("XpRobSameAsPrereg", "the same setting" if ship and ship == get(sel, "all", "label") else "a different setting")
    mac("XpShipSource", MISSING if add is None else str(get(add, "decision", "ship_source")))
    for v, word in (("0", "Zero"), ("0.75", "ZeroSevenFive"), ("1.5", "OneFive"), ("2.25", "TwoTwoFive"), ("3", "Three"), ("4", "Four")):
        mac("XpProfItem" + word + "A", fmt(get(rob, "tau_item_profile", v, "best_A")))
        mac("XpProfItem" + word + "E", fmt(get(rob, "tau_item_profile", v, "best_E")))
    for v, word in (("0.5", "ZeroFive"), ("1", "One"), ("1.5", "OneFive"), ("2", "Two"), ("3", "Three")):
        mac("XpProfBench" + word + "A", fmt(get(rob, "tau_bench_profile", v, "best_A")))
    # configurations
    confs = {c: get(cv, "configs", c) for c in ("A", "B", "C", "D")}
    confs["E"] = get(add, "configs", "E")
    for c, s in confs.items():
        p = "XpC" + c
        mac(p + "NUnits", fmt_int(get(s, "n_units")))
        mac(p + "NBench", fmt_int(get(s, "n_benchmarks")))
        mac(p + "NTestsets", fmt_int(get(s, "n_testsets")))
        mac(p + "NLabels", fmt_int(get(s, "n_labels_budget31")))
        for m, code in (("S0", "SZero"), ("S1", "SOne"), ("XP", "Xp"), ("oracle_rate", "Or"), ("prior", "Prior")):
            r = get(s, "methods", m)
            mac(p + code + "Alc", fmt(get(r, "alc", "mean")))
            mac(p + code + "AlcLo", fmt(get(r, "alc", "ci_lo")))
            mac(p + code + "AlcHi", fmt(get(r, "alc", "ci_hi")))
            for bw, bi in R4_BUDGET_IDX.items():
                mac(p + code + "B" + bw, fmt(get(r, "brier", bi), 4))
        for key, code in (("XP_minus_S0", "DiffSZero"), ("XP_minus_S1", "DiffSOne"), ("S1_minus_S0", "SOneMinusSZero")):
            cluster_macros(p + code, get(s, "paired", key, "alc"))
            for bw, bi in R4_BUDGET_IDX.items():
                if bw == "Zero":
                    continue
                d = get(s, "paired", key, "brier", bi)
                mac(p + code + "B" + bw + "Mean", fmt(get(d, "mean"), 4, sign=True))
                mac(p + code + "B" + bw + "Lo", fmt(get(d, "ci_lo"), 4, sign=True))
                mac(p + code + "B" + bw + "Hi", fmt(get(d, "ci_hi"), 4, sign=True))
        gap_s0 = get(s, "methods", "S0", "alc", "mean")
        gap_or = get(s, "methods", "oracle_rate", "alc", "mean")
        gap_xp = get(s, "methods", "XP", "alc", "mean")
        mac(p + "GapClosed", MISSING if None in (gap_s0, gap_or, gap_xp) else _pct((gap_s0 - gap_xp) / (gap_s0 - gap_or), 0))
    for k, code in (("XP_minus_S0_evalonly_pairs", "EvalOnly"), ("XP_minus_S0_acquired_pairs", "Acquired")):
        cluster_macros("XpCD" + code, get(confs["D"], "paired", k, "alc"))
    # pre-registered nested vs all-fold, robust vs prereg
    cluster_macros("XpCAAllfoldDiff", get(confs["A"], "paired", "XP_allfold_minus_S0", "alc"))
    cluster_macros("XpCARobVsPre", get(add, "configs", "A", "paired", "XPrb_minus_XP", "alc"))
    # components: config A (xpair_cv.json) and E (addendum)
    comps = (("minus_bench", "Bench"), ("minus_subj", "Subj"), ("minus_pair", "Pair"), ("minus_item", "Item"),
             ("minus_probit", "Probit"), ("bench_and_pair", "BenchPair"), ("name_subj_and_pair", "NameSubjPair"),
             ("pair_and_item", "PairItem"), ("only_pair", "OnlyPair"))
    for c, ref in (("A", "XP_allfold"), ("E", "XPrb_allfold")):
        s = confs[c]
        for key, code in comps:
            cluster_macros(f"XpComp{c}{code}", get(s, "paired", f"{key}_minus_{ref}", "alc"))
            mac(f"XpComp{c}{code}Alc", fmt(get(s, "methods", key, "alc", "mean")))
    # equivalence, per-pair robustness, official loop
    mac("XpEquivN", fmt_int(get(equiv, "n_predictions")))
    mac("XpEquivNTestsets", fmt_int(get(equiv, "n_testsets")))
    mac("XpEquivMaxDiff", _sci(get(equiv, "max_abs_diff")))
    cluster_macros("XpPerPair", get(perpair, "summary", "paired", "XP_minus_S1", "alc"))
    mac("XpLoopAcqCalls", fmt_int(get(loop, "acquisition_calls")))
    mac("XpLoopAcqPredCalls", fmt_int(get(loop, "acquisition_prediction_calls")))
    mac("XpLoopPredCalls", fmt_int(get(loop, "prediction_calls_count")))
    mac("XpLoopMode", MISSING if loop is None else str(loop.get("acquisition_mode")))
    mac("XpLoopMatch", MISSING if loop is None else ("equals" if loop.get("matches_xpair_sim") else "DIFFERS FROM"))
    mac("XpLoopAlc", fmt(get(loop, "alc_mean")))
    # real testbed (global list), seed 0 and seeds summary
    runs = get(real, "runs") or []
    r0 = runs[0] if runs else None
    mac("XpRealNPairs", fmt_int(get(r0, "n_pairs")))
    mac("XpRealNSeeds", fmt_int(len(runs)))
    for m, code in (("XP", "Xp"), ("shipped_global", "ShipGlobal"), ("shipped_perpair", "ShipPerPair"), ("XP_perpair", "XpPerPair")):
        mac("XpReal" + code + "Alc", fmt(get(r0, "methods", m, "alc", "mean")))
        mac("XpReal" + code + "AlcLo", fmt(get(r0, "methods", m, "alc", "ci_lo")))
        mac("XpReal" + code + "AlcHi", fmt(get(r0, "methods", m, "alc", "ci_hi")))
        for bw, bi in R4_BUDGET_IDX.items():
            mac("XpReal" + code + "B" + bw, fmt(get(r0, "methods", m, "brier", bi), 4))
    for key, code in (("XP_minus_shipped_global", "DiffGlobal"), ("XP_minus_shipped_perpair", "DiffPerPair"),
                      ("shipped_perpair_minus_shipped_global", "PerPairVsGlobal"),
                      ("XP_perpair_minus_shipped_perpair", "XpPerPairVsPerPair")):
        paired_macros("XpReal" + code, get(r0, "paired", key))
        sig_neg = sum(1 for r in runs if get(r, "paired", key, "ci_hi") is not None and get(r, "paired", key, "ci_hi") < 0)
        sig_pos = sum(1 for r in runs if get(r, "paired", key, "ci_lo") is not None and get(r, "paired", key, "ci_lo") > 0)
        mac("XpReal" + code + "NSigNeg", fmt_int(sig_neg) if runs else MISSING)
        mac("XpReal" + code + "NSigPos", fmt_int(sig_pos) if runs else MISSING)
        means = [get(r, "paired", key, "mean") for r in runs]
        means = [float(x) for x in means if x is not None]
        mac("XpReal" + code + "SeedMin", fmt(min(means), 4, sign=True) if means else MISSING)
        mac("XpReal" + code + "SeedMax", fmt(max(means), 4, sign=True) if means else MISSING)
    rule_a = get(confs["A"], "paired", "XP_minus_S0", "alc", "ci_hi")
    rule_a = None if rule_a is None else bool(rule_a < 0)
    rule_b = get(r0, "paired", "XP_minus_shipped_global", "ci_lo")
    rule_b = None if rule_b is None else bool(not rule_b > 0)
    rule_c = None
    if all(confs[c] is not None for c in ("B", "C", "D")):
        rule_c = all(not (get(confs[c], "paired", "XP_minus_S0", "alc", "ci_lo") > 0) for c in ("B", "C", "D"))
    mac("XpRuleA", MISSING if rule_a is None else ("holds" if rule_a else "fails"))
    mac("XpRuleB", MISSING if rule_b is None else ("holds" if rule_b else "fails"))
    mac("XpRuleC", MISSING if rule_c is None else ("holds" if rule_c else "fails"))
    # acquisition
    cluster_macros("XpAcq", get(acq, "fresh_minus_random", "alc"))
    for bw in ("Fifteen", "ThirtyOne", "One", "Three", "Seven"):
        d = get(acq, "fresh_minus_random", "brier", R4_BUDGET_IDX[bw])
        mac("XpAcqB" + bw + "Mean", fmt(get(d, "mean"), 4, sign=True))
        mac("XpAcqB" + bw + "Lo", fmt(get(d, "ci_lo"), 4, sign=True))
        mac("XpAcqB" + bw + "Hi", fmt(get(d, "ci_hi"), 4, sign=True))
    mac("XpAcqRandomAlc", fmt(get(acq, "xp_random", "alc", "mean")))
    mac("XpAcqFreshAlc", fmt(get(acq, "xp_fresh", "alc", "mean")))
    mac("XpAcqSkips", fmt_int(get(acq, "n_skips_fresh")))
    cluster_macros("XpAcqSZero", get(acq, "S0_fresh_minus_random", "alc"))
    rruns = get(realacq, "runs") or []
    ra0 = rruns[0] if rruns else None
    paired_macros("XpRealAcq", get(ra0, "fresh_minus_first"))
    mac("XpRealAcqNSigNeg", fmt_int(sum(1 for r in rruns if get(r, "fresh_minus_first", "ci_hi") < 0)) if rruns else MISSING)
    mac("XpRealAcqNSigPos", fmt_int(sum(1 for r in rruns if get(r, "fresh_minus_first", "ci_lo") > 0)) if rruns else MISSING)
    means = [float(get(r, "fresh_minus_first", "mean")) for r in rruns]
    mac("XpRealAcqSeedMean", fmt(sum(means) / len(means), 4, sign=True) if means else MISSING)
    acq_ship = None
    if acq is not None and ra0 is not None:
        acq_ship = bool(get(acq, "fresh_minus_random", "alc", "ci_hi") < 0 and not get(ra0, "fresh_minus_first", "ci_lo") > 0)
    mac("XpAcqShip", MISSING if acq_ship is None else ("shipped" if acq_ship else "not shipped"))
    # runtime
    sizes = {int(r["n_pairs"]): r for r in (get(runtime, "sizes") or [])}
    for n, word in ((10, "Ten"), (100, "Hundred"), (1000, "Thousand"), (2000, "TwoThousand"), (4000, "FourThousand")):
        r = sizes.get(n)
        mac("XpRt" + word + "Labels", fmt_int(get(r, "n_labels")))
        mac("XpRt" + word + "Fit", fmt(get(r, "first_call_s"), 2))
        mac("XpRt" + word + "Ms", fmt(get(r, "ms_per_call"), 3))
        mac("XpRt" + word + "Ckpt", fmt(get(r, "checkpoint_s"), 1))
        mac("XpRt" + word + "Targets", fmt_int(get(r, "n_targets")))
    mac("XpSmokeXpCases", fmt_int(get(smoke_sub, "xp_cases")))
    mac("XpSmokeBigFit", fmt(get(smoke_sub, "xp_big_first_call_s"), 2))
    mac("XpSmokeBigMs", fmt(get(smoke_sub, "xp_big_ms_per_call"), 3))
    mac("XpSmokeHookMode", MISSING if smoke_sub is None else str(get(smoke_sub, "acquisition_hook_mode")))
    # shipped constants of the cross-pair model, read from model.py
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    import model as _model
    taus = dict(_model.XP_TAUS)
    mac("XpTauMu", f"{taus.get('mu', 0):g}")
    mac("XpSweeps", fmt_int(_model.XP_SWEEPS))
    mac("XpItemNewton", fmt_int(_model.XP_ITEM_NEWTON))
    shipped_match = (t.get("bench") == taus.get("bench") and t.get("subj") == taus.get("subj") and t.get("pair") == taus.get("pair")
                     and t.get("name") == taus.get("name") and t.get("namebench") == taus.get("namebench")
                     and t.get("item") == _model.XP_TAU_ITEM and bool(t.get("probit")) == bool(_model.XP_PROBIT) and _model.XP_ENABLED)
    mac("XpShippedMatches", "matches" if shipped_match else "DOES NOT MATCH")
    pf = [r.get("diff") for r in (get(confs["A"], "per_fold") or []) if r.get("diff") is not None]
    mac("XpCAFoldMin", fmt(min(pf), 4, sign=True) if pf else MISSING)
    mac("XpCAFoldMax", fmt(max(pf), 4, sign=True) if pf else MISSING)
    mac("XpCAFoldNNeg", fmt_int(sum(1 for v in pf if v < 0)) if pf else MISSING)
    # tables
    table_round4_configs(confs)
    table_round4_components(confs)
    table_round4_runtime(runtime)
    table_round4_budget(confs, r0)


def _cell(d, nd=4):
    if d is None:
        return MISSING
    return f"{fmt(get(d, 'mean'), nd, sign=True)} [{fmt(get(d, 'ci_lo'), nd, sign=True)}, {fmt(get(d, 'ci_hi'), nd, sign=True)}]"


def table_round4_configs(confs) -> None:
    rows = [("A", "A: formative-like (primary)"), ("E", "E: distinct models"), ("B", "B: one benchmark, all pairs"),
            ("C", "C: one fold, many benchmarks"), ("D", "D: A + eval-only small pairs")]
    lines = []
    for c, label in rows:
        s = confs.get(c)
        cells = [f"{fmt_int(get(s, 'n_units'))}/{fmt_int(get(s, 'n_benchmarks'))}",
                 fmt(get(s, "methods", "S0", "alc", "mean")), fmt(get(s, "methods", "S1", "alc", "mean")),
                 fmt(get(s, "methods", "XP", "alc", "mean")), fmt(get(s, "methods", "oracle_rate", "alc", "mean")),
                 _cell(get(s, "paired", "XP_minus_S0", "alc")), _cell(get(s, "paired", "XP_minus_S1", "alc"))]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_xp_configs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_round4_components(confs) -> None:
    rows = [("XP", "shipped setting (all effects)"), ("minus_bench", "$-$ benchmark effect"), ("minus_subj", "$-$ subject effect"),
            ("minus_pair", "$-$ subject--benchmark effect"), ("minus_item", "$-$ exact-item effect"),
            ("minus_probit", "$-$ probit integration"), ("bench_and_pair", "benchmark + pair effects only"),
            ("name_subj_and_pair", "name/subject + pair effects only"), ("pair_and_item", "pair + item effects only"),
            ("only_pair", "pair effect only")]
    ref = {"A": "XP_allfold", "E": "XPrb_allfold"}
    lines = []
    for key, label in rows:
        cells = []
        for c in ("A", "E"):
            s = confs.get(c)
            if key == "XP":
                cells += [fmt(get(s, "methods", ref[c], "alc", "mean")), "--"]
            else:
                cells += [fmt(get(s, "methods", key, "alc", "mean")), _cell(get(s, "paired", f"{key}_minus_{ref[c]}", "alc"))]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_xp_components.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_round4_runtime(runtime) -> None:
    lines = []
    for r in get(runtime, "sizes") or []:
        lines.append(f"{fmt_int(r['n_pairs'])} & {fmt_int(r['n_labels'])} & {fmt_int(r['n_targets'])} & {fmt(r['first_call_s'], 2)} & "
                     f"{fmt(r['ms_per_call'], 3)} & {fmt(r['checkpoint_s'], 1)} \\\\")
    (PAPER / "results_table_xp_runtime.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_round4_budget(confs, r0) -> None:
    """Brier by budget: configs A and E (S0, S1, XP, oracle) and the paired XP - S0 difference with its CI."""
    lines = []
    for c in ("A", "E"):
        s = confs.get(c)
        for m, label in (("S0", "S0 (rounds 1--3, as deployed)"), ("S1", "S1 (own pair's labels only)"),
                         ("XP", "XP (cross-pair, shipped)"), ("oracle_rate", "oracle: pair rate")):
            cells = [fmt(get(s, "methods", m, "brier", i)) for i in range(6)] + [fmt(get(s, "methods", m, "alc", "mean"))]
            lines.append(f"{c} & {label} & " + " & ".join(cells) + r" \\")
        diffs = ["--"] + [f"{fmt(get(s, 'paired', 'XP_minus_S0', 'brier', i, 'mean'), 4, sign=True)}" for i in range(1, 6)]
        diffs.append(fmt(get(s, "paired", "XP_minus_S0", "alc", "mean"), 4, sign=True))
        lines.append(f"{c} & XP $-$ S0 & " + " & ".join(diffs) + r" \\")
        cis = ["--"] + [f"[{fmt(get(s, 'paired', 'XP_minus_S0', 'brier', i, 'ci_lo'), 4, sign=True)}, "
                        f"{fmt(get(s, 'paired', 'XP_minus_S0', 'brier', i, 'ci_hi'), 4, sign=True)}]" for i in range(1, 6)]
        cis.append(f"[{fmt(get(s, 'paired', 'XP_minus_S0', 'alc', 'ci_lo'), 4, sign=True)}, "
                   f"{fmt(get(s, 'paired', 'XP_minus_S0', 'alc', 'ci_hi'), 4, sign=True)}]")
        lines.append(r" & \quad 95\% CI & " + " & ".join("{\\scriptsize " + x + "}" for x in cis) + r" \\")
        if c == "A":
            lines.append(r"\midrule")
    (PAPER / "results_table_xp_budget.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- selected submission (2026-10-26)
# Formative runs on Codabench (organiser-scored), the three-constant candidates built in the deep dive and the
# simulator scenarios behind the choice.  Sources: results/deep_dive/formative_runs.json (the single source for
# the real-run numbers, written by deep_dive/formative_summary.py), results/deep_dive/{proto_level3, ece_check,
# decision_stat, population, diagnose_A, proto_oracle, grid_inv}.json, results/deep_dive/candidates/*/ and the
# submission ZIPs themselves (constants are parsed from the ZIP members, never typed).
SUBMISSION = HERE.parents[1] / "submission"
RUN_CODES = {"shipped": "Ship", "xp_nosubj_mu2": "Nosubj", "xp_level_m050": "Sel", "xp_level_m100": "Mhund"}
RUN_ZIPS = {"shipped": SUBMISSION / "paiec_submission.zip", "xp_nosubj_mu2": SUBMISSION / "candidates" / "xp_nosubj_mu2.zip",
            "xp_level_m050": SUBMISSION / "candidates" / "xp_level_m050.zip",
            "xp_level_m100": SUBMISSION / "candidates" / "xp_level_m100.zip"}
SCEN_CODES = {"A": "A", "Y25": "Y", "INV": "Inv", "INV25": "InvY", "MID": "Mid", "HARD": "Hard", "SHIFT": "Shift"}
SCEN_LABELS = {"A": "A (release-like)", "Y25": "Y25 (2025+ subjects)",
               "INV": "INV (inventory)", "INV25": "INV25 (INV, 2025+)",
               "MID": "MID (rate $\\le$ 0.5)", "HARD": "HARD (rate $\\le$ 0.4)",
               "SHIFT": "SHIFT (A, prior +1)"}
SIM_KEYS = {"xp_nosubj_mu2": "nosubj_d0_mu2", "xp_level_m050": "nosubj_d-0.5_mu2", "xp_level_m100": "nosubj_d-1_mu2"}
MATCH_KEYS = {"xp_nosubj_mu2_minus_shipped": "NosubjShip", "xp_level_m050_minus_shipped": "SelShip",
              "xp_level_m100_minus_shipped": "MhundShip", "xp_level_m050_minus_xp_nosubj_mu2": "SelNosubj",
              "xp_level_m100_minus_xp_nosubj_mu2": "MhundNosubj", "xp_level_m100_minus_xp_level_m050": "MhundSel"}
BUD_IDX = [("0", "Zero"), ("1", "One"), ("3", "Three"), ("7", "Seven"), ("15", "Fifteen"), ("31", "ThirtyOne")]


def _zip_constants(path: Path) -> dict:
    """Literal top-level constants of model.py inside a submission ZIP (parsed with ast, not executed)."""
    import ast
    import zipfile
    out = {}
    if not path.exists():
        return out
    with zipfile.ZipFile(path) as z:
        src = z.read("model.py").decode("utf-8")
        out["_model_src"] = src
        out["_labeling"] = z.read("labeling.py")
        out["_members"] = sorted(z.namelist())
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    out["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    out["_bytes"] = path.stat().st_size
    return out


def _num(v):
    if v is None:
        return MISSING
    v = float(v)
    s = f"{v:g}"
    return s.replace("-", "\\ensuremath{-}") if v < 0 else s


def selected_constants_macros() -> dict:
    consts = {name: _zip_constants(p) for name, p in RUN_ZIPS.items()}
    for name, c in consts.items():
        code = RUN_CODES[name]
        taus = dict(c.get("XP_TAUS") or ())
        mac(code + "SubjectShrink", _num(c.get("SUBJECT_SHRINK")))
        mac(code + "Intercept", _num(c.get("PRIOR_INTERCEPT")))
        mac(code + "TauMu", _num(taus.get("mu")))
        mac(code + "ZipSha", MISSING if not c else c["_sha256"])
        mac(code + "ZipShaShort", MISSING if not c else c["_sha256"][:12])
        mac(code + "ZipKB", MISSING if not c else f"{c['_bytes'] / 1000:.1f}")
    ship, sel = consts.get("shipped") or {}, consts.get("xp_level_m050") or {}
    # plain-text values for the command line in the rerun instructions (alltt)
    mac("SelInterceptRaw", MISSING if not sel else f"{float(sel['PRIOR_INTERCEPT']):g}")
    mac("SelSubjectShrinkRaw", MISSING if not sel else f"{float(sel['SUBJECT_SHRINK']):g}")
    mac("SelTauMuRaw", MISSING if not sel else f"{float(dict(sel['XP_TAUS'])['mu']):g}")
    # exactly which lines differ between the shipped and the selected model.py, and is labeling.py identical?
    import difflib
    if ship and sel:
        a, b = ship["_model_src"].splitlines(), sel["_model_src"].splitlines()
        changed = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0) if ln.startswith("+") and not ln.startswith("+++")]
        mac("SelNChangedLines", fmt_int(len(changed)))
        mac("SelLabelingIdentical", "byte-identical" if ship["_labeling"] == sel["_labeling"] else "DIFFERENT")
        same_other = all(ship.get(k) == sel.get(k) for k in ship if not k.startswith("_")
                         and k not in ("SUBJECT_SHRINK", "PRIOR_INTERCEPT", "XP_TAUS"))
        mac("SelOtherConstantsSame", "unchanged" if same_other else "CHANGED")
        mac("SelMembers", ", ".join(f"\\texttt{{{m}}}" for m in sel["_members"]))
        # budget-0 prior of an item without cues (600 characters): sigma(intercept); shipped value for comparison
        mac("SelPZeroNoCue", f"{1 / (1 + math.exp(-float(sel['PRIOR_INTERCEPT']))):.2f}")
    else:
        for n in ("SelNChangedLines", "SelLabelingIdentical", "SelOtherConstantsSame", "SelMembers", "SelPZeroNoCue"):
            mac(n, MISSING)
    fr = load("deep_dive/formative_runs.json")
    want = get(fr, "selected_submission", "sha256")
    mac("SelId", MISSING if fr is None else str(get(fr, "selected_submission", "id")))
    mac("SelShaVerified", "verified" if sel and want == sel["_sha256"] else "MISMATCH")
    # constants table (results_table_constants.tex)
    lines = []
    rows = (("\\texttt{SUBJECT\\_SHRINK}", "weight of the hand-set subject terms", lambda c: c.get("SUBJECT_SHRINK")),
            ("\\texttt{PRIOR\\_INTERCEPT}", "shift of the prior logit", lambda c: c.get("PRIOR_INTERCEPT")),
            ("$\\tau_\\mu$ (\\texttt{XP\\_TAUS})", "prior sd of the global level $\\mu$", lambda c: dict(c.get("XP_TAUS") or ()).get("mu")))
    for label, role, f in rows:
        cells = [_num(f(consts[n])) if consts.get(n) else MISSING for n in ("shipped", "xp_nosubj_mu2", "xp_level_m050", "xp_level_m100")]
        cells[2] = "\\textbf{" + cells[2] + "}"
        lines.append(f"{label} & {role} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_constants.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return consts


def _t_crit(df: int, alpha: float = 0.05) -> float:
    """Two-sided critical value of Student's t (stdlib: Simpson integration of the density + bisection)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def tail(x, n=2000):  # P(T > x) = 0.5 - integral_0^x density
        h = x / n
        s = sum((1 if i in (0, n) else (4 if i % 2 else 2)) * c * (1 + (i * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(n + 1))
        return 0.5 - s * h / 3

    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 2 * tail(mid) > alpha else (lo, mid)
    return (lo + hi) / 2


def formative_macros(fr) -> None:
    runs = get(fr, "runs") or {}
    alcs = []
    for name, code in RUN_CODES.items():
        r = runs.get(name)
        p = "Fm" + code
        mac(p + "Id", MISSING if r is None else str(r.get("submission_id")))
        mac(p + "Alc", fmt(get(r, "alc_mean")))
        mac(p + "Ece", fmt(get(r, "ece_summary_mean")))
        mac(p + "D", fmt(get(r, "D_B0_minus_B31")))
        mac(p + "NPairs", fmt_int(get(r, "n_pairs")))
        mac(p + "NBench", fmt_int(get(r, "n_benchmark_ids")))
        mac(p + "NItems", fmt_int(get(r, "n_eval_subject_items")))
        for b, w in BUD_IDX:
            mac(p + "B" + w, fmt(get(r, "brier_by_budget", b)))
            mac(p + "EceB" + w, fmt(get(r, "ece_by_budget", b), 3))
        if get(r, "alc_mean") is not None:
            alcs.append(float(r["alc_mean"]))
        # consistency: organiser ALC rows vs our ALC recomputed from the per-budget means
        a, b2 = get(r, "alc_mean"), get(r, "alc_from_budget_means")
        mac(p + "AlcConsistent", MISSING if None in (a, b2) else ("consistent" if abs(a - b2) < 1e-4 else "INCONSISTENT"))
    import statistics as st
    mac("FmNRuns", fmt_int(len(alcs)))
    mac("FmAlcMin", fmt(min(alcs)) if alcs else MISSING)
    mac("FmAlcMax", fmt(max(alcs)) if alcs else MISSING)
    mac("FmAlcRange", fmt(max(alcs) - min(alcs)) if alcs else MISSING)
    mac("FmAlcSd", fmt(st.stdev(alcs), 3) if len(alcs) > 1 else MISSING)
    npairs = [int(get(runs, n, "n_pairs")) for n in RUN_CODES if get(runs, n, "n_pairs") is not None]
    mac("FmNPairsMin", fmt_int(min(npairs)) if npairs else MISSING)
    mac("FmNPairsMax", fmt_int(max(npairs)) if npairs else MISSING)
    nit = [int(get(runs, n, "n_eval_subject_items")) for n in RUN_CODES if get(runs, n, "n_eval_subject_items") is not None]
    mac("FmNItemsMin", fmt_int(min(nit)) if nit else MISSING)
    mac("FmNItemsMax", fmt_int(max(nit)) if nit else MISSING)
    sp, si = get(runs, "xp_level_m050", "n_pairs"), get(runs, "xp_level_m050", "n_eval_subject_items")
    mac("FmSelItemsPerPair", fmt_int(si / sp) if sp and si else MISSING)
    mac("FmNBenchAll", fmt_int(get(fr, "distinct_benchmark_ids_all_runs")))
    mac("FmNSubjAll", fmt_int(get(fr, "distinct_subject_ids_all_runs")))
    ov = get(fr, "overlap") or {}
    mac("FmSharedPairsMax", fmt_int(max((v.get("shared_pairs", 0) for v in ov.values()), default=None)) if ov else MISSING)
    mac("FmSharedBenchMin", fmt_int(min((v.get("shared_benchmark_ids", 0) for v in ov.values()), default=None)) if ov else MISSING)
    # how many runs predict budget 0 worse than the constant 0.5 (Brier 0.25 on every draw)
    nb0 = sum(1 for n in RUN_CODES if get(runs, n, "brier_by_budget", "0") is not None and runs[n]["brier_by_budget"]["0"] > 0.25)
    mac("FmNBZeroAboveConst", fmt_int(nb0) if runs else MISSING)
    cand_b0 = [runs[n]["brier_by_budget"]["0"] for n in ("xp_nosubj_mu2", "xp_level_m050", "xp_level_m100") if n in runs]
    mac("FmCandBZeroMax", fmt(max(cand_b0)) if cand_b0 else MISSING)
    mac("FmCandBZeroMin", fmt(min(cand_b0)) if cand_b0 else MISSING)
    # raw (unmatched) run-to-run difference at budgets 1..31, selected minus shipped, averaged over the five budgets
    a, b = get(runs, "xp_level_m050", "brier_by_budget"), get(runs, "shipped", "brier_by_budget")
    later = [a[k] - b[k] for k in ("1", "3", "7", "15", "31")] if a and b else []
    mac("FmSelMinusShipLaterMean", fmt(sum(later) / len(later), 3, sign=True) if later else MISSING)
    mac("FmSelMinusShipBZeroRaw", fmt(a["0"] - b["0"], 3, sign=True) if a and b else MISSING)
    mac("FmSelMinusShipAlcRaw", fmt(get(runs, "xp_level_m050", "alc_mean") - get(runs, "shipped", "alc_mean"), 4, sign=True)
        if get(runs, "xp_level_m050", "alc_mean") is not None and get(runs, "shipped", "alc_mean") is not None else MISSING)
    # benchmark-matched differences (equal weight per shared benchmark; mean, sd, se over benchmarks)
    bm = get(fr, "benchmark_matched") or {}
    tmax, ses = 0.0, []
    for key, code in MATCH_KEYS.items():
        for sec, w in [("ALC", "Alc")] + [(b, "B" + ww) for b, ww in BUD_IDX]:
            d = get(bm, key, sec)
            p = "FmMatch" + code + w
            mac(p + "Mean", fmt(get(d, "mean_diff"), 4, sign=True))
            mac(p + "Se", fmt(get(d, "se"), 4))
            m, se = get(d, "mean_diff"), get(d, "se")
            mac(p + "T", MISSING if None in (m, se) or not se else f"{m / se:+.2f}")
            if sec == "ALC" and None not in (m, se) and se:
                tmax = max(tmax, abs(m / se))
                ses.append(se)
        mac("FmMatch" + code + "N", fmt_int(get(bm, key, "ALC", "n_benchmarks")))
    mac("FmMatchMaxAbsT", f"{tmax:.2f}" if ses else MISSING)
    # all entries (6 run pairs x (ALC + 6 budgets)): how many reach 2 se, the largest |t|, and the t critical value
    ts = []
    for key in MATCH_KEYS:
        for sec in ["ALC"] + [b for b, _ in BUD_IDX]:
            m, se = get(bm, key, sec, "mean_diff"), get(bm, key, sec, "se")
            if None not in (m, se) and se:
                ts.append(abs(m / se))
    mac("FmMatchNEntries", fmt_int(len(ts)) if ts else MISSING)
    mac("FmMatchNAboveTwo", fmt_int(sum(1 for t in ts if t >= 2)) if ts else MISSING)
    mac("FmMatchMaxAbsTAll", f"{max(ts):.2f}" if ts else MISSING)
    nb = get(bm, "xp_level_m050_minus_shipped", "ALC", "n_benchmarks")
    mac("FmMatchTCrit", f"{_t_crit(int(nb) - 1):.2f}" if nb else MISSING)
    mac("FmMatchNAboveCrit", fmt_int(sum(1 for t in ts if nb and t >= _t_crit(int(nb) - 1))) if ts and nb else MISSING)
    mac("FmMatchSeMin", fmt(min(ses), 3) if ses else MISSING)
    mac("FmMatchSeMax", fmt(max(ses), 3) if ses else MISSING)
    n_b = get(bm, "xp_level_m050_minus_shipped", "ALC", "n_benchmarks")
    mac("FmMatchDf", fmt_int(n_b - 1) if n_b else MISSING)
    # the pre-recorded rule for revising the choice: matched B0 of the selected run not worse than the shipped one
    d = get(bm, "xp_level_m050_minus_shipped", "0", "mean_diff")
    mac("FmSelRuleBZero", MISSING if d is None else ("better" if d < 0 else "worse"))
    # tables
    sid = {n: str(get(runs, n, "submission_id")) for n in RUN_CODES}
    labels = {"shipped": f"{sid['shipped']} (round 4, as shipped)", "xp_nosubj_mu2": f"{sid['xp_nosubj_mu2']} (subject terms off)",
              "xp_level_m050": f"\\textbf{{{sid['xp_level_m050']} (selected)}}", "xp_level_m100": f"{sid['xp_level_m100']} (stronger shift)"}
    lines = []
    for name in ("shipped", "xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"):
        r = runs.get(name)
        cells = [fmt_int(get(r, "n_pairs"))] + [fmt(get(r, "brier_by_budget", b)) for b, _ in BUD_IDX] + \
                [fmt(get(r, "alc_mean")), fmt(get(r, "ece_summary_mean"), 3), fmt(get(r, "ece_by_budget", "0"), 3)]
        if name == "xp_level_m050":
            cells = ["\\textbf{" + c + "}" for c in cells]
        lines.append(f"{labels[name]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("constant 0.5 (any draw) & -- & " + " & ".join(["0.2500"] * 6) + r" & 0.2500 & -- & -- \\")
    (PAPER / "results_table_formative.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # learning-curve plot lines for pgfplots (x = log2(n+1))
    styles = {"shipped": "black, mark=*", "xp_nosubj_mu2": "blue, mark=square*", "xp_level_m050": "red, thick, mark=triangle*",
              "xp_level_m100": "teal, mark=diamond*"}
    plot = []
    for name in ("shipped", "xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"):
        r = runs.get(name)
        if r is None:
            continue
        pts = " ".join(f"({i},{float(r['brier_by_budget'][b]):.4f})" for i, (b, _) in enumerate(BUD_IDX))
        plot.append(f"\\addplot[{styles[name]}] coordinates {{{pts}}};")
        plot.append(f"\\addlegendentry{{{r['submission_id']}}}")
    plot.append("\\addplot[gray, dashed, domain=0:5] {0.25};")
    plot.append("\\addlegendentry{constant 0.5}")
    (PAPER / "results_fig_formative.tex").write_text("\n".join(plot) + "\n", encoding="utf-8")
    # matched-difference table (appendix)
    lines = []
    i_ship, i_nos, i_sel, i_mh = (sid[n] for n in ("shipped", "xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"))
    lab = {"NosubjShip": f"{i_nos} $-$ {i_ship}", "SelShip": f"{i_sel} $-$ {i_ship}", "MhundShip": f"{i_mh} $-$ {i_ship}",
           "SelNosubj": f"{i_sel} $-$ {i_nos}", "MhundNosubj": f"{i_mh} $-$ {i_nos}", "MhundSel": f"{i_mh} $-$ {i_sel}"}
    for key, code in MATCH_KEYS.items():
        cells = []
        for sec, w in [(b, "B" + ww) for b, ww in BUD_IDX] + [("ALC", "Alc")]:
            d = get(bm, key, sec)
            cells.append(f"{fmt(get(d, 'mean_diff'), 3, sign=True)} ({fmt(get(d, 'se'), 3)})")
        lines.append(f"{lab[code]} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_matched.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scenario_macros(p3, ece, dec, fr) -> None:
    scen = get(p3, "scenarios") or {}
    for s, code in SCEN_CODES.items():
        r = scen.get(s)
        p = "Dd" + code
        mac(p + "NUnits", fmt_int(get(r, "n_units")))
        mac(p + "NBench", fmt_int(get(r, "n_bench")))
        mac(p + "MedRate", fmt(get(r, "unit_rate_median"), 2))
        mac(p + "Oracle", fmt(get(r, "oracle")))
        mac(p + "ShipAlc", fmt(get(r, "methods", "shipped", "alc")))
        for i, (_, w) in enumerate(BUD_IDX):
            mac(p + "ShipB" + w, fmt(get(r, "methods", "shipped", "brier", i)))
        for name, key in SIM_KEYS.items():
            m = get(r, "methods", key)
            q = p + RUN_CODES[name]
            mac(q + "Alc", fmt(get(m, "alc")))
            mac(q + "BZero", fmt(get(m, "brier", 0)))
            d = get(m, "minus_shipped")
            mac(q + "Mean", fmt(get(d, "mean"), 4, sign=True))
            mac(q + "Lo", fmt(get(d, "ci_lo"), 4, sign=True))
            mac(q + "Hi", fmt(get(d, "ci_hi"), 4, sign=True))
            mac(q + "Sig", sig_word(d))
            b1s, b1m = get(r, "methods", "shipped", "brier", 1), get(m, "brier", 1)
            mac(q + "DeltaBOne", MISSING if None in (b1s, b1m) else fmt(b1m - b1s, 4, sign=True))
    # cross-check: the deep-dive summary copies the same scenario numbers
    sc = get(fr, "simulator_scenarios") or {}
    ok = bool(sc) and all(abs(get(sc, s, "xp_level_m050", "minus_shipped", "mean") -
                              get(scen, s, "methods", "nosubj_d-0.5_mu2", "minus_shipped", "mean")) < 1e-12 for s in sc)
    mac("DdScenConsistent", "consistent" if ok else "INCONSISTENT")
    # summary ranges of the selected change: hidden-pool-like scenarios (INV, INV25, MID, HARD, SHIFT) vs release-like (A, Y25)
    def rng(keys, name="xp_level_m050"):
        vals = [get(scen, s, "methods", SIM_KEYS[name], "minus_shipped", "mean") for s in keys]
        vals = [v for v in vals if v is not None]
        return (min(vals), max(vals)) if vals else (None, None)
    lo, hi = rng(("INV", "INV25", "MID", "HARD", "SHIFT"))
    mac("DdSelGainBest", fmt(lo, 4, sign=True))
    mac("DdSelGainWorstHidden", fmt(hi, 4, sign=True))
    lo, hi = rng(("A", "Y25"))
    mac("DdSelCostMin", fmt(lo, 4, sign=True))
    mac("DdSelCostMax", fmt(hi, 4, sign=True))
    lo, hi = rng(("INV", "INV25", "MID", "HARD", "SHIFT"), "xp_nosubj_mu2")
    mac("DdNosubjGainBest", fmt(lo, 4, sign=True))
    mac("DdNosubjGainWorst", fmt(hi, 4, sign=True))
    extra = [get(scen, s, "methods", SIM_KEYS["xp_level_m100"], "alc") - get(scen, s, "methods", SIM_KEYS["xp_level_m050"], "alc")
             for s in ("INV", "INV25", "MID", "HARD", "SHIFT") if s in scen]
    mac("DdMhundMinusSelBest", fmt(min(extra), 4, sign=True) if extra else MISSING)
    mac("DdMhundMinusSelWorst", fmt(max(extra), 4, sign=True) if extra else MISSING)
    nsig = sum(1 for s in ("INV", "INV25", "MID", "HARD", "SHIFT")
               if get(scen, s, "methods", SIM_KEYS["xp_level_m050"], "minus_shipped", "ci_hi") is not None
               and scen[s]["methods"][SIM_KEYS["xp_level_m050"]]["minus_shipped"]["ci_hi"] < 0)
    mac("DdSelNSigGain", fmt_int(nsig))
    # largest |change| of B1..B31 between the selected setting and the shipped one in any scenario (level matters at B0)
    mx = 0.0
    for s in scen:
        a, b = get(scen, s, "methods", "shipped", "brier"), get(scen, s, "methods", SIM_KEYS["xp_level_m050"], "brier")
        if a and b:
            mx = max(mx, max(abs(b[i] - a[i]) for i in range(1, 6)))
    mac("DdSelMaxDeltaLater", fmt(mx, 4) if scen else MISSING)
    lines = []
    for s, code in SCEN_CODES.items():
        r = scen.get(s)
        cells = [fmt_int(get(r, "n_bench")), fmt(get(r, "unit_rate_median"), 2), fmt(get(r, "methods", "shipped", "alc")),
                 fmt(get(r, "methods", "shipped", "brier", 0), 3) + r"$\to$" + fmt(get(r, "methods", SIM_KEYS["xp_level_m050"], "brier", 0), 3)]
        for name in ("xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"):
            d = get(r, "methods", SIM_KEYS[name], "minus_shipped")
            # full CI for the selected variant; mean and a significance star for the other two (CIs in the appendix)
            c = _cell(d) if name == "xp_level_m050" else fmt(get(d, "mean"), 4, sign=True)
            if d is not None and d.get("significant"):
                c = c.replace(" [", "$^{*}$ [", 1) if name == "xp_level_m050" else c + "$^{*}$"
            cells.append(c)
        lines.append(f"{SCEN_LABELS[s]} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_scenarios.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # appendix version: all three variants with CIs
    lines = []
    for s, code in SCEN_CODES.items():
        r = scen.get(s)
        cells = [fmt(get(r, "oracle")), fmt(get(r, "methods", "shipped", "alc"))]
        for name in ("xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"):
            m = get(r, "methods", SIM_KEYS[name])
            cells.append(f"{fmt(get(m, 'alc'))} & {_cell(get(m, 'minus_shipped'))}")
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_scenarios_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = []
    for s, code in SCEN_CODES.items():
        r = scen.get(s)
        cells = [fmt_int(get(r, "n_bench")), fmt(get(r, "unit_rate_median"), 2), fmt(get(r, "methods", "shipped", "alc")),
                 fmt(get(r, "methods", "shipped", "brier", 0), 3) + r"$\to$" + fmt(get(r, "methods", SIM_KEYS["xp_level_m050"], "brier", 0), 3)]
        for name in ("xp_nosubj_mu2", "xp_level_m050", "xp_level_m100"):
            d = get(r, "methods", SIM_KEYS[name], "minus_shipped")
            c = _cell(d) if name == "xp_level_m050" else fmt(get(d, "mean"), 4, sign=True)
            if d is not None and d.get("significant"):
                c = c.replace(" [", "$^{*}$ [", 1) if name == "xp_level_m050" else c + "$^{*}$"
            cells.append(c)
        lines.append(f"{SCEN_LABELS[s]} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_scenarios.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # budget-0 ECE by simulated population (ece_check.json) and the D = B0 - B31 statistic (decision_stat.json)
    for s, code in (("A", "A"), ("INV", "Inv"), ("HARD", "Hard"), ("SHIFT", "Shift")):
        mac("DdEceBZeroShip" + code, fmt(get(ece, s, "ece_by_budget", 0), 3))
        mac("DdEceAlcShip" + code, fmt(get(ece, s, "ece_alc"), 3))
        mac("DdTestsetAlcSd" + code, fmt(get(ece, s, "testset_alc_sd"), 3))
    for s, code in (("A", "A"), ("INV", "Inv"), ("HARD", "Hard")):
        mac("DdEceBZeroSel" + code, fmt(get(ece, f"{s}_nosubj_d-0.5_mu2", "ece_by_budget", 0), 3))
    vals = [get(ece, s, "ece_by_budget", 0) for s in ("A", "INV", "HARD", "SHIFT")]
    vals = [v for v in vals if v is not None]
    mac("DdEceBZeroShipMax", fmt(max(vals), 3) if vals else MISSING)
    mac("DdEceBZeroShipMin", fmt(min(vals), 3) if vals else MISSING)
    vals = [get(ece, f"{s}_nosubj_d-0.5_mu2", "ece_by_budget", 0) for s in ("A", "INV", "HARD")]
    vals = [v for v in vals if v is not None]
    mac("DdEceBZeroSelMax", fmt(max(vals), 3) if vals else MISSING)
    for s, code in (("A", "A"), ("INV", "Inv"), ("MID", "Mid"), ("HARD", "Hard")):
        mac("DdD" + code, fmt(get(dec, s, "D_mean"), 3))
        mac("DdDSd" + code, fmt(get(dec, s, "D_sd_single_draw"), 3))
    sds = [get(dec, s, "D_sd_single_draw") for s in ("A", "INV", "MID", "HARD")]
    sds = [v for v in sds if v is not None]
    mac("DdDSdMin", fmt(min(sds), 3) if sds else MISSING)
    mac("DdDSdMax", fmt(max(sds), 3) if sds else MISSING)


def deep_dive_extra_macros(pop, diag, orc, ginv) -> None:
    gi = get(pop, "groups") or {}
    mac("DdPopNInvBench", fmt_int(get(pop, "n_inventory_benchmarks")))
    mac("DdPopNBench", fmt_int(get(pop, "n_benchmarks")))
    for g, code in (("inventory_benchmarks", "Inv"), ("non_inventory_benchmarks", "NonInv"), ("inventory_x_2025plus", "InvY"),
                    ("all", "All"), ("release_2025plus", "Y")):
        mac("DdPop" + code + "Pairs", fmt_int(get(gi, g, "pairs")))
        mac("DdPop" + code + "Rate", fmt(get(gi, g, "mean_rate"), 3))
        mac("DdPop" + code + "BZero", fmt(get(gi, g, "B0_prior_approx"), 4))
        mac("DdPop" + code + "PZero", fmt(get(gi, g, "mean_prior_p0"), 3))
    mac("DdPopEmptyName", _pct(get(pop, "empty_name_share")))
    at = get(pop, "ability_transfer") or {}
    r2 = [v.get("r2_best_slope") for v in at.values() if v.get("r2_best_slope") is not None]
    mac("DdAbilityRsqMax", fmt(max(r2), 3) if r2 else MISSING)
    cs = [v.get("corr") for v in at.values() if v.get("corr") is not None]
    mac("DdAbilityCorrMin", fmt(min(cs), 2, sign=True) if cs else MISSING)
    mac("DdAbilityCorrMax", fmt(max(cs), 2, sign=True) if cs else MISSING)
    # diagnosis of the shipped predictor in config A: weighted gap to the pair-rate oracle by budget
    gap = get(diag, "weighted_gap_to_oracle_by_budget") or []
    for i, (_, w) in enumerate(BUD_IDX):
        mac("DdGapB" + w, fmt(gap[i] if i < len(gap) else None, 4, sign=True))
    mac("DdGapTotal", fmt(sum(gap), 4) if gap else MISSING)
    mac("DdGapShareLow", _pct(sum(gap[:3]) / sum(gap)) if gap else MISSING)
    mac("DdDiagAlc", fmt(get(diag, "xp_shipped", "alc")))
    mac("DdDiagOracle", fmt(get(diag, "xp_shipped", "oracle_rate")))
    mac("DdDiagNTestsets", fmt_int(get(diag, "n_testsets")))
    mac("DdDiagTestsetSd", fmt(get(diag, "testset_alc", "sd"), 3))
    bins = get(diag, "by_rate_bin") or []
    for b, code in ((0, "Low"), (len(bins) - 1, "High")):
        r = bins[b] if bins else None
        mac("DdBin" + code + "Alc", fmt(get(r, "alc"), 3))
        mac("DdBin" + code + "Oracle", fmt(get(r, "oracle"), 3))
        mac("DdBin" + code + "BiasBZero", fmt(get(r, "bias", 0), 2, sign=True))
    # ceilings for side information inside the cross-pair model (proto_oracle.json)
    lines = []
    for s, code in (("A", "A"), ("HARD", "Hard")):
        r = get(orc, "scenarios", s)
        for m, mc in (("bench_oracle", "Bench"), ("item_oracle_0.5", "ItemHalf"), ("item_oracle_1", "Item"), ("both_1", "Both")):
            d = get(r, "methods", m)
            mac("DdOrc" + code + mc + "Alc", fmt(get(d, "alc")))
            cluster_macros("DdOrc" + code + mc, get(d, "minus_shipped"))
        mac("DdOrc" + code + "ShipAlc", fmt(get(r, "methods", "shipped", "alc")))
    lab = {"bench_oracle": "benchmark mean logit (with test pairs)", "item_oracle_0.5": "item effect from other subjects, $\\times0.5$",
           "item_oracle_1": "item effect from other subjects, $\\times1$", "both_1": "benchmark mean + item effect"}
    for m in ("bench_oracle", "item_oracle_0.5", "item_oracle_1", "both_1"):
        cells = []
        for s in ("A", "HARD"):
            d = get(orc, "scenarios", s, "methods", m)
            cells += [fmt(get(d, "alc")), _cell(get(d, "minus_shipped"))]
        lines.append(f"{lab[m]} & " + " & ".join(cells) + r" \\")
    (PAPER / "results_table_oracle.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cluster_macros("DdGridInv", get(ginv, "scenarios", "INV", "nested_minus_shipped"))
    cluster_macros("DdGridInvOnA", get(ginv, "scenarios", "A", "minus_shipped"))
    g = get(ginv, "grid") or {}
    n = 1
    for v in g.values():
        n *= len(v)
    mac("DdGridN", fmt_int(n) if g else MISSING)


def candidate_check_macros() -> None:
    base = RESULTS / "deep_dive" / "candidates"
    for name, code in (("xp_level_m050", "Sel"), ("xp_level_m100", "Mhund"), ("xp_nosubj_mu2", "Nosubj")):
        info, eq = load(f"deep_dive/candidates/{name}/candidate_info.json"), load(f"deep_dive/candidates/{name}/equivalence.json")
        loop, smoke = load(f"deep_dive/candidates/{name}/official_loop_check.json"), load(f"deep_dive/candidates/{name}/smoke_test.json")
        vpath = base / name / "validator_output.txt"
        vtxt = vpath.read_text(encoding="utf-8") if vpath.exists() else None
        p = "Ck" + code
        mac(p + "Validator", MISSING if vtxt is None else ("passes" if "OK:" in vtxt and "[exit code 0]" in vtxt else "FAILS"))
        mac(p + "EquivMaxDiff", _sci(get(eq, "max_abs_diff")))
        mac(p + "EquivN", fmt_int(get(eq, "n_predictions")))
        mac(p + "EquivNTestsets", fmt_int(get(eq, "n_testsets")))
        mac(p + "LoopAlc", fmt(get(loop, "alc_mean")))
        mac(p + "LoopMode", MISSING if loop is None else str(loop.get("acquisition_mode")))
        mac(p + "LoopAcqPredCalls", fmt_int(get(loop, "acquisition_prediction_calls")))
        mac(p + "LoopNPairs", fmt_int(get(loop, "n_pairs")))
        lpp = get(loop, "labels_per_pair") or []
        mac(p + "LoopLabels", fmt_int(min(lpp)) if lpp and min(lpp) == max(lpp) else MISSING)
        ref = get(loop, "alc_xpair_sim_same_policy")
        mac(p + "LoopMinusShip", MISSING if None in (ref, get(loop, "alc_mean")) else fmt(loop["alc_mean"] - ref, 5, sign=True))
        mac(p + "SmokeCases", fmt_int(len(get(smoke, "cases") or [])) if smoke else MISSING)
        mac(p + "SmokeXpCases", fmt_int(get(smoke, "xp_cases")))
        mac(p + "SmokeOk", MISSING if smoke is None else ("passes" if smoke.get("ok") else "FAILS"))
        mac(p + "BigFit", fmt(get(smoke, "xp_big_first_call_s"), 2))
        mac(p + "BigMs", fmt(get(smoke, "xp_big_ms_per_call"), 3))
        mac(p + "BigEntries", fmt_int(get(smoke, "xp_big_entries")))
        mac(p + "LabelingSame", MISSING if info is None else ("yes" if info.get("labeling_identical_to_shipped") else "NO"))
        mac(p + "ShipZipUnchanged", MISSING if info is None else
            ("unchanged" if info.get("shipped_zip_sha256_after") == _zip_constants(RUN_ZIPS["shipped"]).get("_sha256") else "CHANGED"))
    ship_loop = load("round4/official_loop_check.json")
    mac("CkShipLoopAlc", fmt(get(ship_loop, "alc_mean")))


def leaderboard_macros() -> None:
    """Public Codabench leaderboard of the PAIEC competition phase (29785): best draw per user.

    Derived from the raw API dumps in ../../leaderboards when present (cached to results/leaderboard_context.json,
    which is what the released repository ships); only scores are kept, no participant names except the
    organisers' own baseline entry."""
    import datetime
    import statistics as st
    src_dir = HERE.parents[2] / "leaderboards"
    files = [src_dir / "cb_phase_29785_now.json", src_dir / "cb_phase_29785_p2.json"]
    cache = RESULTS / "leaderboard_context.json"
    if all(f.exists() for f in files):
        subs = []
        for f in files:
            subs += json.loads(f.read_text(encoding="utf-8")).get("submissions", [])
        scores = sorted(float(next(x["score"] for x in s["scores"] if x["column_key"] == "brier")) for s in subs)
        aims = [float(next(x["score"] for x in s["scores"] if x["column_key"] == "brier")) for s in subs
                if s.get("owner") == "AIMS Foundations"]
        snap = datetime.datetime.fromtimestamp(files[0].stat().st_mtime).strftime("%Y-%m-%d")
        rec = {"phase": 29785, "source": [f.name for f in files], "snapshot_date": snap, "n_entries": len(scores),
               "n_unique_ids": len({s["id"] for s in subs}), "scores_sorted": scores,
               "organiser_baseline_aims_foundations": aims[0] if aims else None,
               "note": "public leaderboard: each user's best formative draw (Brier ALC); scores only"}
        cache.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    lb = load("leaderboard_context.json")
    sc = get(lb, "scores_sorted") or []
    mac("LbN", fmt_int(len(sc)) if sc else MISSING)
    mac("LbDate", MISSING if lb is None else str(lb.get("snapshot_date")))
    for k, w in ((1, "First"), (5, "Fifth"), (10, "Tenth"), (20, "Twentieth")):
        mac("Lb" + w, fmt(sc[k - 1]) if len(sc) >= k else MISSING)
    mac("LbMedian", fmt(st.median(sc)) if sc else MISSING)
    mac("LbAims", fmt(get(lb, "organiser_baseline_aims_foundations")))
    mac("LbNConst", fmt_int(sum(1 for v in sc if abs(v - 0.25) < 1e-9)) if sc else MISSING)
    fr = load("deep_dive/formative_runs.json")
    for name, code in RUN_CODES.items():
        a = get(fr, "runs", name, "alc_mean")
        mac("LbRank" + code, MISSING if a is None or not sc else fmt_int(1 + sum(1 for v in sc if v < a)))


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
    listing_macros(load("measurement_db_listing.json"))
    dry_run_macros(load("prior_fit_dry_run.json"))
    fit_macros(load("prior_fit.json"), load("prior_fit_testbeds.json"))
    sensitivity_macros(load("prior_sensitivity.json"))
    novelty_macros(load("novelty_filter_stats.json"))
    cv3 = load("round3/text_prior_cv.json")
    prereg = RESULTS / "round3" / "PREREG.md"
    prereg_ok = bool(cv3 and prereg.exists() and hashlib.sha256(prereg.read_bytes()).hexdigest() == cv3.get("prereg_sha256"))
    round3_macros(cv3, load("round3/text_prior_cv_interaction.json"), load("round3/text_prior_testbeds.json"),
                  load("round3/text_prior_final_fit.json"), load("round3/text_prior_check.json"), prereg_ok)
    exp3 = load("round3/text_prior_explore.json")
    add = RESULTS / "round3" / "PREREG_ADDENDUM_1.md"
    addendum_ok = bool(exp3 and add.exists() and hashlib.sha256(add.read_bytes()).hexdigest() == exp3.get("addendum_sha256"))
    round3_extra_macros(cv3, exp3, load("round3/text_prior_runtime.json"), load("round3/shipped_equivalence.json"), addendum_ok)
    table_textprior(cv3, exp3)
    round4_macros(load("round4/xpair_cv.json"), load("round4/xpair_addendum1.json"), load("round4/xpair_real_testbed.json"),
                  load("round4/xpair_acq.json"), load("round4/xpair_real_acq.json"), load("round4/xpair_runtime.json"),
                  load("round4/xpair_equivalence.json"), load("round4/xpair_perpair.json"), load("round4/xpair_frame_stats.json"),
                  load("round4/official_loop_check.json"), load("round4/smoke_test_submission.json"))
    model_constant_macros()
    # selected submission 1010717 and the formative runs (2026-10-26)
    selected_constants_macros()
    fr = load("deep_dive/formative_runs.json")
    formative_macros(fr)
    scenario_macros(load("deep_dive/proto_level3.json"), load("deep_dive/ece_check.json"), load("deep_dive/decision_stat.json"), fr)
    deep_dive_extra_macros(load("deep_dive/population.json"), load("deep_dive/diagnose_A.json"), load("deep_dive/proto_oracle.json"),
                           load("deep_dive/grid_inv.json"))
    candidate_check_macros()
    leaderboard_macros()
    vpath = RESULTS / "validator_output.txt"
    zip_macros(load("submission_zip_info.json"), vpath.read_text(encoding="utf-8") if vpath.exists() else None)
    text = "% generated by experiments/src/make_macros.py -- do not edit\n" + "\n".join(macros) + "\n"
    (PAPER / "results_macros.tex").write_text(text, encoding="utf-8")
    n_missing = text.count(MISSING)
    print(f"wrote {PAPER / 'results_macros.tex'}: {len(macros)} macros, {n_missing} missing values")


if __name__ == "__main__":
    main()
