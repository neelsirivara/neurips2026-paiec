"""Deep dive step 5: which simulated population reproduces OUR observed formative numbers (Brier ALC 0.1934, ECE 0.169)?

ECE as on the competition page: per pair and budget, response-weighted mean |mean prediction - success rate| over
10 equal-width bins; summary = ALC-weighted.  Computed for the shipped predictor per scenario, per unit, then
per test set (a formative draw is one test set).  Writes results/deep_dive/ece_check.json
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
sub = lambda keep: [[d for d in f if d in keep] for f in folds]


def unit_ece(S, p):
    u = S.ev["unit"]; y = S.ev["y"]
    b = np.minimum((p * 10).astype(int), 9)
    key = u * 10 + b
    K = S.n_units * 10
    n = np.bincount(key, minlength=K).astype(float)
    sp = np.bincount(key, weights=p, minlength=K)
    sy = np.bincount(key, weights=y, minlength=K)
    gap = np.abs(sp - sy)  # = n * |mean p - mean y|
    return np.bincount(np.arange(K) // 10, weights=gap, minlength=S.n_units) / S.unit_count


res = {}
for name, fl, variant in (("A", folds, None), ("INV", sub(inv), None), ("HARD", sub(hard), None),
                          ("SHIFT", folds, "shift"), ("A_nosubj_d-0.5_mu2", folds, "lvl"), ("INV_nosubj_d-0.5_mu2", sub(inv), "lvl"),
                          ("HARD_nosubj_d-0.5_mu2", sub(hard), "lvl")):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    if variant is None:
        br, preds = eval_xp(S, ts)
    elif variant == "shift":
        br, preds = eval_xp(S, ts, subj_logit_by_fold=[F.pair_zs + 1.0] * 5)
    else:
        taus = dict(SHIPPED_TAUS); taus["mu"] = 2.0
        br, preds = eval_xp(S, ts, subj_logit_by_fold=[np.zeros(F.n_pairs) - 0.5] * 5, taus=taus)
    E = np.stack([unit_ece(S, preds[b]) for b in X.BUDGETS], axis=1)
    e_alc = E @ X.ALC_W
    a = alc(br)
    tsid = S.units[:, 0]
    uts, inv_t = np.unique(tsid, return_inverse=True)
    ts_a = np.bincount(inv_t, weights=a) / np.bincount(inv_t)
    ts_e = np.bincount(inv_t, weights=e_alc) / np.bincount(inv_t)
    near = np.mean((np.abs(ts_a - 0.1934) < 0.01) & (np.abs(ts_e - 0.169) < 0.02))
    res[name] = {"alc": float(a.mean()), "ece_alc": float(e_alc.mean()), "ece_by_budget": [float(x) for x in E.mean(0)],
                 "testset_alc_sd": float(ts_a.std()), "testset_ece_sd": float(ts_e.std()),
                 "testset_ece_p10_p90": [float(np.percentile(ts_e, 10)), float(np.percentile(ts_e, 90))],
                 "frac_testsets_near_observed": float(near),
                 "frac_testsets_ece_ge_0.169": float(np.mean(ts_e >= 0.169))}
    log(f"{name:24s} ALC {a.mean():.4f} ECE(ALC-wtd) {e_alc.mean():.4f} by budget {np.round(E.mean(0),3)}; test-set ECE p10-p90 "
        f"[{np.percentile(ts_e,10):.3f}, {np.percentile(ts_e,90):.3f}]; P(test-set ECE >= 0.169) {np.mean(ts_e >= 0.169):.3f}; "
        f"P(near observed) {near:.3f}")
(OUT / "ece_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time()-t0:.0f}s)")
