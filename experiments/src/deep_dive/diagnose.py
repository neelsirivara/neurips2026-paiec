"""Deep dive step 2: where the shipped XP predictor loses Brier (config A, random acquisition, R=300 as in round 4).
Writes paiec/experiments/results/deep_dive/diagnose_A.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
inv = inventory_dirs(F)
testsets = X.make_testsets(F, folds, "A", R=300)
S = X.Streams(F, testsets)
log(f"streams: {S.n_units} units, {len(S.ev['y'])} eval rows ({time.time()-t0:.0f}s)")
br, preds = eval_xp(S, testsets)
orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
a = alc(br)
cl = S.unit_bench
res = {"config": "A", "R": 300, "n_units": int(S.n_units), "n_testsets": len(testsets)}
res["xp_shipped"] = {"alc": float(a.mean()), "brier": [float(x) for x in br.mean(0)], "oracle_rate": float(orc.mean())}
log(f"XP shipped ALC {a.mean():.4f}  per budget {np.round(br.mean(0), 4)}  oracle {orc.mean():.4f}")
# weighted loss vs oracle per budget
gap = (br - orc[:, None]).mean(0) * X.ALC_W
res["weighted_gap_to_oracle_by_budget"] = [float(x) for x in gap]
log(f"weighted gap to pair-rate oracle by budget {np.round(gap, 4)} total {gap.sum():.4f}")
# bias: mean prediction - mean outcome per budget, overall and by pair-rate bin
ev_unit = S.ev["unit"]
ur = S.unit_rate
bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
res["by_rate_bin"] = []
for lo, hi in bins:
    um = (ur >= lo) & (ur < hi)
    rm = um[ev_unit]
    rec = {"bin": [lo, hi], "n_units": int(um.sum()), "alc": float(a[um].mean()), "oracle": float(orc[um].mean()),
           "brier": [float(x) for x in br[um].mean(0)],
           "bias": [float(preds[b][rm].mean() - S.ev["y"][rm].mean()) for b in X.BUDGETS]}
    res["by_rate_bin"].append(rec)
    log(f"rate [{lo:.1f},{hi:.1f}) units {um.sum():5d} ALC {rec['alc']:.4f} oracle {rec['oracle']:.4f} brier {np.round(rec['brier'],3)} bias {np.round(rec['bias'],3)}")
res["bias_overall"] = [float(preds[b].mean() - S.ev["y"].mean()) for b in X.BUDGETS]
log(f"overall bias by budget {np.round(res['bias_overall'], 4)}")
# by subject release year
py_u = pyear[S.units[:, 1]]
groups = {"no date": np.isnan(py_u), "<2024": py_u < 2024, "2024": (py_u >= 2024) & (py_u < 2025), ">=2025": py_u >= 2025}
res["by_release"] = {}
for k, m in groups.items():
    if m.sum():
        res["by_release"][k] = {"n_units": int(m.sum()), "alc": float(a[m].mean()), "oracle": float(orc[m].mean()),
                                "brier": [float(x) for x in br[m].mean(0)], "mean_rate": float(ur[m].mean())}
        log(f"release {k:8s} units {m.sum():5d} ALC {a[m].mean():.4f} oracle {orc[m].mean():.4f} B0 {br[m,0].mean():.4f} B1 {br[m,1].mean():.4f} mean rate {ur[m].mean():.3f}")
# inventory vs not
invu = np.array([F.dirs[b] in inv for b in S.unit_bench])
for k, m in (("inventory", invu), ("non-inventory", ~invu)):
    res.setdefault("by_inventory", {})[k] = {"n_units": int(m.sum()), "alc": float(a[m].mean()), "oracle": float(orc[m].mean()),
                                             "brier": [float(x) for x in br[m].mean(0)]}
    log(f"{k:14s} units {m.sum():5d} ALC {a[m].mean():.4f} oracle {orc[m].mean():.4f} brier {np.round(br[m].mean(0),4)}")
# by benchmark: total loss vs oracle (ALC units)
ub, inv_ = np.unique(S.unit_bench, return_inverse=True)
loss = np.bincount(inv_, weights=a - orc)
cnt = np.bincount(inv_)
order = np.argsort(-loss)
tot = loss.sum()
res["by_benchmark_top"] = [{"bench": F.dirs[ub[j]], "units": int(cnt[j]), "mean_alc": float(a[inv_ == j].mean()),
                            "mean_oracle": float(orc[inv_ == j].mean()), "share_of_gap": float(loss[j] / tot)} for j in order[:15]]
log("top benchmarks by share of total (ALC - oracle): " + "; ".join(f"{r['bench']} {r['share_of_gap']*100:.1f}% (ALC {r['mean_alc']:.3f} vs {r['mean_oracle']:.3f})" for r in res["by_benchmark_top"][:10]))
# per-benchmark (equal weight) gap distribution
bg = np.array([(a[inv_ == j] - orc[inv_ == j]).mean() for j in range(len(ub))])
res["bench_gap_quantiles"] = [float(np.percentile(bg, q)) for q in (10, 25, 50, 75, 90)]
# formative-noise proxy: ALC mean over the 10 pairs of each test set
ts_ids = S.units[:, 0]
uts, inv_t = np.unique(ts_ids, return_inverse=True)
ts_alc = np.bincount(inv_t, weights=a) / np.bincount(inv_t)
ts_orc = np.bincount(inv_t, weights=orc) / np.bincount(inv_t)
res["testset_alc"] = {"mean": float(ts_alc.mean()), "sd": float(ts_alc.std()), "p05": float(np.percentile(ts_alc, 5)),
                      "p95": float(np.percentile(ts_alc, 95)), "oracle_sd": float(ts_orc.std()),
                      "min_of_N_expected": {str(N): float(np.mean([np.min(np.random.RandomState(s).choice(ts_alc, N)) for s in range(2000)])) for N in (1, 5, 10, 20, 50)}}
log(f"test-set (10-pair) ALC: mean {ts_alc.mean():.4f} sd {ts_alc.std():.4f} 5-95% [{np.percentile(ts_alc,5):.4f}, {np.percentile(ts_alc,95):.4f}]; "
    f"expected best of N: {res['testset_alc']['min_of_N_expected']}")
(OUT / "diagnose_A.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
np.savez(OUT / "units_A_shipped.npz",
         br=br, orc=orc, units=S.units, unit_bench=S.unit_bench, unit_fold=S.unit_fold, unit_rate=S.unit_rate)
log(f"done ({time.time()-t0:.0f}s)")
