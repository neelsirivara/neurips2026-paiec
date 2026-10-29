"""W2 development run on the TUNE folds only (0, 1): correctness of the base and a first look at the layer.

    python -I explore_tune.py [SCEN ...]
Writes experiments/results/w2_items/explore_tune.json (exploratory; not part of the confirmation).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_sim import *  # noqa: F401,F403

TUNE = (0, 1)
t0 = time.time()
W = World(with_grams=True)
scens = sys.argv[1:] or ["INV", "A"]
res = {}
for sname in scens:
    S, ts = build_scenario(W, sname, TUNE)
    cl = S.unit_bench
    log(f"[{sname}] units {S.n_units} benches {len(np.unique(cl))} ({time.time() - t0:.0f}s)")
    br0 = eval_w2(W, S, ts)
    if sname == "INV":
        ref, _ = eval_xp(S, ts, subj_logit_by_fold=[np.full(W.F.n_pairs, -0.5)] * 5, taus=base_taus())
        log(f"  base vs dd_common.eval_xp max |diff| {np.max(np.abs(br0 - ref)):.2e}")
    a0 = alc(br0)
    rec = {"base_alc": float(a0.mean()), "base_brier": [float(x) for x in br0.mean(0)], "variants": {}}
    variants = [("feat0.5", dict(tau_f=0.5)), ("feat1", dict(tau_f=1.0)),
                ("kern1_q8", dict(tau_k=1.0, q=8)), ("kern1_q4", dict(tau_k=1.0, q=4))]
    for vname, kw in variants:
        t1 = time.time()
        br = eval_w2(W, S, ts, **kw)
        d = paired(alc(br), a0, cl)
        rec["variants"][vname] = {"alc": float(alc(br).mean()), "minus_base": d,
                                  "brier_minus_base": [float(x) for x in (br - br0).mean(0)]}
        log(f"  {vname:10s} ALC {alc(br).mean():.4f} diff {fmt(d)} per budget {np.round((br - br0).mean(0), 4)} ({time.time() - t1:.0f}s)")
    res[sname] = rec
(RES / "explore_tune.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
