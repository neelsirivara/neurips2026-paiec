"""Deep dive step 3: level calibration / adaptation-speed variants of the shipped XP predictor across four scenarios.

Scenarios (all benchmark-held-out folds of round 2, one global labeled list per test set, random acquisition):
  A     config A as in round 4 (release-like population)
  INV   config A built only from benchmarks that are rows of the official benchmark inventory (recent D&B/ICLR/ICML papers)
  HARD  config A built only from benchmarks whose eligible pairs have mean success rate <= 0.40 (stress test; the
        participants' public write-ups report hidden-test pair rates well below the release's)
  SHIFT config A with every offset shifted by +1 logit (the prior is 1 logit too optimistic for every pair)
Writes paiec/experiments/results/deep_dive/proto_level.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
NB = int(F.pair_dir.max()) + 1
bm = np.bincount(F.pair_dir[elig], weights=rate[elig], minlength=NB) / np.maximum(np.bincount(F.pair_dir[elig], minlength=NB), 1)
inv = inventory_dirs(F)
hard = {F.dirs[b] for b in range(len(F.dirs)) if np.bincount(F.pair_dir[elig], minlength=NB)[b] > 0 and bm[b] <= 0.40}
log(f"inventory benchmarks {len(inv)}, hard benchmarks {len(hard)}")

def sub_folds(keep):
    return [[d for d in f if d in keep] for f in folds]

scen = {}
for name, fl, shift in (("A", folds, 0.0), ("INV", sub_folds(inv), 0.0), ("HARD", sub_folds(hard), 0.0), ("SHIFT", folds, 1.0)):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    scen[name] = (S, ts, shift)
    rr = S.unit_rate
    log(f"scenario {name}: {S.n_units} units, {len(np.unique(S.unit_bench))} benchmarks, unit rate mean {rr.mean():.3f} median {np.median(rr):.3f} ({time.time()-t0:.0f}s)")

hand = F.pair_zs.copy()
def subj_by_fold(scale_hand, delta):
    z = scale_hand * hand + delta
    return [z] * len(folds)

base_t = dict(SHIPPED_TAUS)
variants = [("shipped", 1.0, 0.0, {}),
            ("mu1", 1.0, 0.0, {"mu": 1.0}), ("mu2", 1.0, 0.0, {"mu": 2.0}),
            ("d-0.5", 1.0, -0.5, {}), ("d-1", 1.0, -1.0, {}),
            ("d-0.5_mu2", 1.0, -0.5, {"mu": 2.0}), ("d-1_mu2", 1.0, -1.0, {"mu": 2.0}),
            ("nosubj", 0.0, 0.0, {}), ("nosubj_mu2", 0.0, 0.0, {"mu": 2.0}),
            ("nosubj_d-0.5_mu2", 0.0, -0.5, {"mu": 2.0}), ("nosubj_d-1_mu2", 0.0, -1.0, {"mu": 2.0}),
            ("mu2_bench2", 1.0, 0.0, {"mu": 2.0, "bench": 2.0}), ("mu2_bench2_pair0.9", 1.0, 0.0, {"mu": 2.0, "bench": 2.0, "pair": 0.9}),
            ("nosubj_d-0.5_mu2_bench2", 0.0, -0.5, {"mu": 2.0, "bench": 2.0}),
            ]
res = {"scenarios": {}, "variants": [v[0] for v in variants]}
for sname, (S, ts, shift) in scen.items():
    out = {}
    for vname, sh, d, tchg in variants:
        taus = dict(base_t); taus.update(tchg)
        sbf = subj_by_fold(sh, d + shift)
        br, _ = eval_xp(S, ts, subj_logit_by_fold=sbf, taus=taus)
        out[vname] = br
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
        log(f"[{sname}] {vname:26s} ALC {r['alc']:.4f}  B {np.round(br.mean(0),4)}" + (f"  diff {fmt(r['minus_shipped'])}" if vname != "shipped" else f"  oracle {orc.mean():.4f}"))
    res["scenarios"][sname] = rec
(OUT / "proto_level.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time()-t0:.0f}s)")
