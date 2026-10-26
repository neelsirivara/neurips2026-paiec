"""Deep dive step 3b: the same variants on competition-eligible-like subjects.

The curation guide (methodology doc, 5 Sep 2026) says competition-eligible base models are 2025-26 releases.
  Y25    config A restricted to pairs whose subject release date is >= 2025 (all benchmarks)
  INV25  Y25 restricted to inventory benchmarks
Writes results/deep_dive/proto_level2.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
inv = inventory_dirs(F)
keep_pair = pyear >= 2025
_orig = X._eligible
def _elig_filtered(F_, b, lo=80, hi=None):
    e = _orig(F_, b, lo, hi)
    return e[keep_pair[e]]
X._eligible = _elig_filtered

def sub_folds(keep):
    return [[d for d in f if d in keep] for f in folds]

scen = {}
for name, fl in (("Y25", folds), ("INV25", sub_folds(inv))):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    scen[name] = (S, ts)
    log(f"scenario {name}: {S.n_units} units, {len(np.unique(S.unit_bench))} benchmarks, {len(ts)} test sets, "
        f"pairs/test set {S.n_units/len(ts):.1f}, unit rate mean {S.unit_rate.mean():.3f} median {np.median(S.unit_rate):.3f}")
X._eligible = _orig

hand = F.pair_zs.copy()
base_t = dict(SHIPPED_TAUS)
variants = [("shipped", 1.0, 0.0, {}), ("mu2", 1.0, 0.0, {"mu": 2.0}),
            ("d-0.5", 1.0, -0.5, {}), ("d-1", 1.0, -1.0, {}),
            ("nosubj", 0.0, 0.0, {}), ("nosubj_mu2", 0.0, 0.0, {"mu": 2.0}),
            ("nosubj_d-0.5", 0.0, -0.5, {}), ("nosubj_d-0.5_mu2", 0.0, -0.5, {"mu": 2.0}),
            ("half_subj", 0.5, 0.0, {}), ("half_subj_d-0.5", 0.5, -0.5, {})]
res = {"scenarios": {}}
for sname, (S, ts) in scen.items():
    out = {}
    for vname, sh, d, tchg in variants:
        taus = dict(base_t); taus.update(tchg)
        out[vname], _ = eval_xp(S, ts, subj_logit_by_fold=[sh * hand + d] * 5, taus=taus)
    orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
    cl = S.unit_bench
    rec = {"n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))), "oracle": float(orc.mean()),
           "unit_rate_mean": float(S.unit_rate.mean()), "unit_rate_median": float(np.median(S.unit_rate)), "methods": {}}
    a0 = alc(out["shipped"])
    for vname, br in out.items():
        r = {"alc": float(alc(br).mean()), "brier": [float(x) for x in br.mean(0)]}
        if vname != "shipped":
            r["minus_shipped"] = paired(alc(br), a0, cl)
        rec["methods"][vname] = r
        log(f"[{sname}] {vname:20s} ALC {r['alc']:.4f}  B {np.round(br.mean(0),4)}" + (f"  diff {fmt(r['minus_shipped'])}" if vname != "shipped" else f"  oracle {orc.mean():.4f}"))
    res["scenarios"][sname] = rec
(OUT / "proto_level2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time()-t0:.0f}s)")
