"""W1 exploration (DEV seed 1, R = 100): NPEB variants vs xp_level_m050, per budget, on A / INV / H7 / H7-INV.
Writes experiments/results/deep_dive/w1/npeb_explore.json.  The confirmation runs use seed 0 and are pre-registered
in w1/PREREG_w1.md before they run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npeb import *  # noqa: F401,F403

SEED = int(os.environ.get("W1_SEED", "1"))
R = int(os.environ.get("W1_R", "100"))
OUTW = OUT / "w1"
OUTW.mkdir(parents=True, exist_ok=True)
t0 = time.time()
F, folds = load()
inv = inventory_dirs(F)
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
pri_all = fit_priors(F, folds)
pri_inv = fit_priors(F, folds, keep_dirs=inv)
log("priors fitted: all n_bench per fold " + str([p["n_bench"] for p in pri_all]) + "; inv " + str([p["n_bench"] for p in pri_inv]))
taus_c = dict(SHIPPED_TAUS); taus_c["mu"] = 2.0
base_sl = [np.full(F.n_pairs, -0.5)] * 5


def scen(name):
    if name == "A":
        return X.make_testsets(F, folds, "A", R=R, seed=SEED)
    if name == "INV":
        return X.make_testsets(F, sub(inv), "A", R=R, seed=SEED)
    if name == "H7":
        return make_testsets_H7(F, folds, R=R, seed=SEED)
    if name == "H7INV":
        return make_testsets_H7(F, sub(inv), R=R, seed=SEED)
    raise ValueError(name)


res = {"seed": SEED, "R": R, "scenarios": {}}
for sname in ("H7INV", "H7", "INV", "A"):
    ts = scen(sname)
    S = X.Streams(F, ts)
    cl = S.unit_bench
    m050, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c)
    out = {"m050": m050, "npeb_all": eval_npeb(S, ts, pri_all), "npeb_inv": eval_npeb(S, ts, pri_inv)}
    for k in ("npeb_all", "npeb_inv"):
        h = m050.copy()
        h[:, :3] = out[k][:, :3]
        out[k + "_hybrid03"] = h
        h2 = m050.copy()
        h2[:, 1:3] = out[k][:, 1:3]
        out[k + "_hybrid13"] = h2
    orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
    rec = {"n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))), "pairs_per_ts": float(S.n_units / len(ts)),
           "oracle": float(orc.mean()), "median_rate": float(np.median(S.unit_rate)), "methods": {}}
    a0 = alc(m050)
    for k, br in out.items():
        r = {"alc": float(alc(br).mean()), "brier": [float(x) for x in br.mean(0)]}
        if k != "m050":
            r["minus_m050"] = paired(alc(br), a0, cl)
            r["brier_minus_m050"] = [paired(br[:, i], m050[:, i], cl) for i in range(6)]
        rec["methods"][k] = r
        log(f"[{sname}] {k:18s} ALC {r['alc']:.4f} B {np.round(br.mean(0), 4)}" +
            (f" diff {fmt(r['minus_m050'])}" if k != "m050" else f" oracle {orc.mean():.4f} pairs/ts {S.n_units/len(ts):.1f}"))
    res["scenarios"][sname] = rec
(OUTW / f"npeb_explore_seed{SEED}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
