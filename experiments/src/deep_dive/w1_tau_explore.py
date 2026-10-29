"""W1 exploration (DEV seed 1): should the early-budget estimate move faster in the hidden-run structure? Grid of the
m050 cross-pair prior sds (benchmark, pair, item) on H7 / H7-INV / INV / A, per budget, vs m050.
Writes experiments/results/deep_dive/w1/tau_explore_seed1.json
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npeb import *  # noqa: F401,F403

SEED = int(os.environ.get("W1_SEED", "1"))
R = int(os.environ.get("W1_R", "100"))
OUTW = OUT / "w1"
t0 = time.time()
F, folds = load()
inv = inventory_dirs(F)
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
base_sl = [np.full(F.n_pairs, -0.5)] * 5
grid = [dict(bench=b, pair=p, item=i) for b, p, i in itertools.product((1.5, 2.0, 2.5), (0.6, 0.9, 1.2), (2.25, 3.0))]
res = {"seed": SEED, "R": R, "grid": grid, "scenarios": {}}
for sname, ts in (("H7INV", make_testsets_H7(F, sub(inv), R=R, seed=SEED)), ("H7", make_testsets_H7(F, folds, R=R, seed=SEED)),
                  ("INV", X.make_testsets(F, sub(inv), "A", R=R, seed=SEED)), ("A", X.make_testsets(F, folds, "A", R=R, seed=SEED))):
    S = X.Streams(F, ts)
    cl = S.unit_bench
    taus0 = dict(SHIPPED_TAUS); taus0["mu"] = 2.0
    m050, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus0)
    rec = {"m050": {"alc": float(alc(m050).mean()), "brier": [float(x) for x in m050.mean(0)]}, "settings": {}}
    for g in grid:
        taus = dict(taus0); taus["bench"] = g["bench"]; taus["pair"] = g["pair"]
        br, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus, ti=g["item"])
        lab = f"bench{g['bench']}|pair{g['pair']}|item{g['item']}"
        d = paired(alc(br), alc(m050), cl)
        rec["settings"][lab] = {"alc": float(alc(br).mean()), "brier_minus_m050": [float(x) for x in (br - m050).mean(0)], "minus_m050": d}
    best = min(rec["settings"], key=lambda k: rec["settings"][k]["alc"])
    rec["best"] = best
    log(f"[{sname}] m050 {alc(m050).mean():.4f}; best {best} {rec['settings'][best]['alc']:.4f} diff {fmt(rec['settings'][best]['minus_m050'])} "
        f"per budget {np.round(rec['settings'][best]['brier_minus_m050'], 4)}")
    for lab in ("bench2.5|pair1.2|item2.25", "bench1.5|pair1.2|item2.25", "bench2.5|pair0.6|item2.25"):
        r = rec["settings"][lab]
        log(f"[{sname}]   {lab}: diff {fmt(r['minus_m050'])} per budget {np.round(r['brier_minus_m050'], 4)}")
    res["scenarios"][sname] = rec
(OUTW / f"tau_explore_seed{SEED}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
