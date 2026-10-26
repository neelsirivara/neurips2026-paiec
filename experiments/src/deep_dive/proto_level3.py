"""Deep dive step 7: the calibrated-level family (subject terms off, intercept d, tau_mu 2) across all scenarios,
plus MID = config A on benchmarks with mean pair rate <= 0.50 (median pair rate close to the 0.27 that one
participant reports for the hidden runs).  Writes results/deep_dive/proto_level3.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
NB = int(F.pair_dir.max()) + 1
cnt = np.bincount(F.pair_dir[elig], minlength=NB)
bm = np.bincount(F.pair_dir[elig], weights=rate[elig], minlength=NB) / np.maximum(cnt, 1)
inv = inventory_dirs(F)
hard = {F.dirs[b] for b in range(len(F.dirs)) if cnt[b] > 0 and bm[b] <= 0.40}
mid = {F.dirs[b] for b in range(len(F.dirs)) if cnt[b] > 0 and bm[b] <= 0.50}
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
keep25 = pyear >= 2025
_orig = X._eligible


def build(fl, only25=False):
    if only25:
        X._eligible = lambda F_, b, lo=80, hi=None: (lambda e: e[keep25[e]])(_orig(F_, b, lo, hi))
    try:
        ts = X.make_testsets(F, fl, "A", R=300)
        S = X.Streams(F, ts)
    finally:
        X._eligible = _orig
    return S, ts


scen = [("A", folds, False, 0.0), ("Y25", folds, True, 0.0), ("INV", sub(inv), False, 0.0), ("INV25", sub(inv), True, 0.0),
        ("MID", sub(mid), False, 0.0), ("HARD", sub(hard), False, 0.0), ("SHIFT", folds, False, 1.0)]
variants = [("shipped", 1.0, 0.0, 0.5)] + [(f"nosubj_d{d:g}_mu2", 0.0, d, 2.0) for d in (0.0, -0.25, -0.5, -0.75, -1.0)]
res = {"scenarios": {}}
for sname, fl, only25, shift in scen:
    S, ts = build(fl, only25)
    cl = S.unit_bench
    out = {}
    for vname, sh, d, mu in variants:
        taus = dict(SHIPPED_TAUS); taus["mu"] = mu
        out[vname], _ = eval_xp(S, ts, subj_logit_by_fold=[sh * F.pair_zs + d + shift] * 5, taus=taus)
    orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
    rec = {"n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))), "oracle": float(orc.mean()),
           "unit_rate_mean": float(S.unit_rate.mean()), "unit_rate_median": float(np.median(S.unit_rate)), "methods": {}}
    a0 = alc(out["shipped"])
    for vname, br in out.items():
        r = {"alc": float(alc(br).mean()), "brier": [float(x) for x in br.mean(0)]}
        if vname != "shipped":
            r["minus_shipped"] = paired(alc(br), a0, cl)
        rec["methods"][vname] = r
    res["scenarios"][sname] = rec
    log(f"[{sname}] units {S.n_units} benches {rec['n_bench']} median rate {rec['unit_rate_median']:.3f} oracle {orc.mean():.4f} shipped {a0.mean():.4f} | " +
        "  ".join(f"{v}: {fmt(rec['methods'][v]['minus_shipped'])}" for v, *_ in variants[1:]))
(OUT / "proto_level3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"finished ({time.time()-t0:.0f}s)")
