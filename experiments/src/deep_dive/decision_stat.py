"""Deep dive step 8: an aggregate, within-draw statistic that tells release-like from harder hidden pools.

For one formative draw (here: one simulated test set of up to 10 pairs), D = mean over pairs of (B0 - B31), using the
SHIPPED predictor.  Differencing within the draw removes most of the pair-rate variance that makes raw Brier noisy.
Reports, per scenario, the distribution of D over single draws and over pools of k draws (k = 1, 3, 5), and the
paired ALC difference (xp_level_m050 - shipped) so a threshold on D can be read as a decision rule.
Writes experiments/results/deep_dive/decision_stat.json
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
taus_c = dict(SHIPPED_TAUS); taus_c["mu"] = 2.0
res = {}
rng = np.random.RandomState(0)
for sname, fl in (("A", folds), ("INV", sub(inv)), ("MID", sub(mid)), ("HARD", sub(hard))):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    br_s, _ = eval_xp(S, ts)
    br_c, _ = eval_xp(S, ts, subj_logit_by_fold=[np.zeros(F.n_pairs) - 0.5] * 5, taus=taus_c)
    tsid = S.units[:, 0]
    u, invt = np.unique(tsid, return_inverse=True)
    cntt = np.bincount(invt)
    D = np.bincount(invt, weights=br_s[:, 0] - br_s[:, 5]) / cntt
    gain = np.bincount(invt, weights=alc(br_c) - alc(br_s)) / cntt
    rec = {"D_mean": float(D.mean()), "D_sd_single_draw": float(D.std()), "gain_m050_mean": float(gain.mean()),
           "corr_D_gain": float(np.corrcoef(D, gain)[0, 1])}
    for k in (3, 5):
        idx = rng.randint(0, len(D), size=(4000, k))
        rec[f"D_sd_pool{k}"] = float(D[idx].mean(1).std())
    for thr in (0.09, 0.10, 0.11):
        rec[f"P(D>{thr})_single"] = float(np.mean(D > thr))
        idx = rng.randint(0, len(D), size=(4000, 3))
        rec[f"P(D>{thr})_pool3"] = float(np.mean(D[idx].mean(1) > thr))
    res[sname] = rec
    log(f"[{sname}] D mean {D.mean():.4f} sd(1 draw) {D.std():.4f} sd(3) {rec['D_sd_pool3']:.4f} sd(5) {rec['D_sd_pool5']:.4f}; "
        f"m050-shipped ALC {gain.mean():+.4f}; P(D>0.10) single {rec['P(D>0.1)_single']:.2f} pool3 {rec['P(D>0.1)_pool3']:.2f}")
(OUT / "decision_stat.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"finished ({time.time()-t0:.0f}s)")
