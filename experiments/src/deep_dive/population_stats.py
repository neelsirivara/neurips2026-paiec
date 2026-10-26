"""Deep dive step 0: population facts used in DEEP_DIVE.md -> results/deep_dive/population.json
(1) pair success rates and the hand-set budget-0 prior's Brier vs constant 0.5, by subject release year and by
    whether the benchmark is a row of the official benchmark inventory (eligible pairs, >= 80 items);
(2) does a subject-ability table by normalized_name, fitted on the other folds' benchmarks, predict within-benchmark
    residual pair logits on held-out benchmarks? (names collapsed per (benchmark, name) on the training side; empty
    names excluded)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
inv = inventory_dirs(F)
inv_pair = np.array([F.dirs[b] in inv for b in F.pair_dir])
zi_pair = np.add.reduceat(F.item_zi[F.row_item], F.pair_start[:-1]) / F.pair_n
p0 = 1 / (1 + np.exp(-np.clip(F.pair_zs + zi_pair, -1.5, 1.5)))
out = {"n_pairs_eligible": int(elig.sum()), "n_benchmarks": int(len(np.unique(F.pair_dir[elig]))),
       "n_inventory_benchmarks": len(inv), "groups": {}}


def summ(m, lab):
    m = m & elig
    r, q = rate[m], p0[m]
    out["groups"][lab] = {"pairs": int(m.sum()), "benchmarks": int(len(np.unique(F.pair_dir[m]))), "mean_rate": float(r.mean()),
                          "median_rate": float(np.median(r)), "mean_prior_p0": float(q.mean()), "V_mean_r1mr": float(np.mean(r * (1 - r))),
                          "B0_prior_approx": float(np.mean(r * (1 - r) + (q - r) ** 2)), "B0_const05_approx": float(np.mean(r * (1 - r) + (0.5 - r) ** 2))}


summ(np.ones(F.n_pairs, bool), "all")
summ(np.isnan(pyear), "no_release_date")
summ(pyear >= 2025, "release_2025plus")
summ(inv_pair, "inventory_benchmarks")
summ(inv_pair & (pyear >= 2025), "inventory_x_2025plus")
summ(~inv_pair, "non_inventory_benchmarks")
# (2) ability transfer
n = F.pair_n.astype(float)
L = np.log((rate * n + 0.5) / (n - rate * n + 0.5))
empty = np.array([k.strip() == "" for k in F.name_keys])[F.pair_name]
fod = fold_of_dir(F, folds)
pf = np.array([fod.get(int(b), -1) for b in F.pair_dir])
NB = int(F.pair_dir.max()) + 1
NN = int(F.pair_name.max()) + 1
key = F.pair_dir.astype(np.int64) * NN + F.pair_name
out["empty_name_share"] = float(np.mean(empty[elig]))
res = {}
for k0 in (1.0, 3.0):
    xs = {"all": ([], []), "2025plus": ([], [])}
    for f in range(5):
        tr = elig & ~empty & (pf != f) & (pf >= 0)
        uu, invk = np.unique(key[tr], return_inverse=True)
        Lbn = np.bincount(invk, weights=L[tr]) / np.bincount(invk)
        b_of, n_of = (uu // NN).astype(int), (uu % NN).astype(int)
        mb = np.bincount(b_of, weights=Lbn, minlength=NB) / np.maximum(np.bincount(b_of, minlength=NB), 1)
        e = Lbn - mb[b_of]
        a = np.bincount(n_of, weights=e, minlength=NN) / (np.bincount(n_of, minlength=NN) + k0)
        seen = (np.bincount(n_of, minlength=NN) > 0)[F.pair_name]
        te = elig & ~empty & (pf == f) & seen
        uu2, inv2 = np.unique(key[elig & ~empty & (pf == f)], return_inverse=True)
        mbt = np.bincount(F.pair_dir[elig & ~empty & (pf == f)], weights=L[elig & ~empty & (pf == f)], minlength=NB) / \
            np.maximum(np.bincount(F.pair_dir[elig & ~empty & (pf == f)], minlength=NB), 1)
        resid = L - mbt[F.pair_dir]
        for lab, m in (("all", te), ("2025plus", te & (pyear >= 2025))):
            xs[lab][0].append(a[F.pair_name][m]); xs[lab][1].append(resid[m])
    for lab, (x, y) in xs.items():
        x, y = np.concatenate(x), np.concatenate(y)
        res[f"k0_{k0:g}_{lab}"] = {"n": int(len(y)), "corr": float(np.corrcoef(x, y)[0, 1]),
                                   "r2_best_slope": float(1 - np.sum((y - (np.sum(x * y) / np.sum(x * x)) * x) ** 2) / np.sum(y ** 2))}
out["ability_transfer"] = res
(OUT / "population.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
