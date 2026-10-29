"""W1 (2026-10-29): an exact-posterior (nonparametric empirical-Bayes) predictor for the early budgets, vs xp_level_m050.

Model, per benchmark of a test set (the global labelled list):
    logit p(pair p, item i) = b + d_p + z_i',   b ~ G_b (shared by the pairs on the benchmark),  d_p ~ G_d (iid),
    z_i' = hand-set item logit minus its training mean.
G_b and G_d are fixed from the TRAINING folds only (nested by outer fold):
  * G_b: Gaussian-kernel-smoothed distribution of training benchmarks' mean pair logits, one atom per benchmark;
  * G_d: smoothed distribution of within-benchmark pair deviations (pair logit minus its benchmark mean).
The posterior over (b, d_p) is computed exactly on a grid from the binomial likelihood of each pair's labels on that
benchmark.  The prediction for a target is E[sigmoid(b + d_p + z_i')] under that posterior, and at budget 0 it is the
prior mean.  No Gaussian/MAP approximation is used.

Also provides the H7 configuration (hidden-run structure, read from counts in our own formative logs): 9 pairs on 7
distinct benchmarks (two benchmarks carry 2 pairs), every subject name used once, 120-item cap.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *  # noqa: F401,F403

BGRID = np.linspace(-6.0, 6.0, 49)
DGRID = np.linspace(-5.0, 5.0, 41)
EGRID = np.linspace(-11.0, 11.0, 177)     # grid for eta = b + d (bin width 0.125)


def _pair_logits(F, min_items=80):
    rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
    n = F.pair_n.astype(float)
    L = np.log((rate * n + 0.5) / (n - rate * n + 0.5))
    return L, F.pair_n >= min_items


def _smooth(atoms, wts, grid, bw):
    k = np.exp(-0.5 * ((grid[:, None] - atoms[None, :]) / bw) ** 2) @ wts
    return k / k.sum()


def fit_priors(F, folds, keep_dirs=None, bw_b=0.35, bw_d=0.25):
    """Per outer fold f: (G_b, G_d, zbar) from benchmarks not in fold f (optionally restricted to keep_dirs)."""
    L, elig = _pair_logits(F)
    fod = fold_of_dir(F, folds)
    pf = np.array([fod.get(int(b), -1) for b in F.pair_dir])
    NB = int(F.pair_dir.max()) + 1
    keep = np.ones(F.n_pairs, bool) if keep_dirs is None else np.isin(np.array(F.dirs)[F.pair_dir], list(keep_dirs))
    out = []
    for f in range(len(folds)):
        tr = elig & (pf != f) & (pf >= 0) & keep
        cnt = np.bincount(F.pair_dir[tr], minlength=NB)
        mb = np.bincount(F.pair_dir[tr], weights=L[tr], minlength=NB) / np.maximum(cnt, 1)
        bs = np.where(cnt > 0)[0]
        Gb = _smooth(mb[bs], np.ones(len(bs)) / len(bs), BGRID, bw_b)
        dev = (L - mb[F.pair_dir])[tr]
        w = 1.0 / cnt[F.pair_dir[tr]]          # each benchmark weighs equally
        Gd = _smooth(dev, w / w.sum(), DGRID, bw_d)
        rows = np.repeat(np.arange(F.n_pairs), F.pair_n)
        rt = tr[rows]
        zbar = float(F.item_zi[F.row_item[rt]].mean())
        out.append({"Gb": Gb, "Gd": Gd, "zbar": zbar, "n_bench": int(len(bs))})
    return out


def make_testsets_H7(F, folds, R=300, seed=0, cap=120, n_bench=7, n_double=2):
    out = []
    for fi, f in enumerate(folds):
        bidx = [F.dirs.index(d) for d in f]
        elig = {b: X._eligible(F, b) for b in bidx}
        benches = [b for b in bidx if len(elig[b])]
        if not benches:
            continue
        rng = np.random.RandomState(seed * 100003 + fi * 7 + 21)
        for r in range(R):
            K = min(n_bench, len(benches))
            bs = [int(b) for b in rng.choice(benches, K, replace=False)]
            dbl = set(int(b) for b in rng.choice(bs, min(n_double, K), replace=False))
            pairs, used = [], set()
            for b in bs:
                cand = [int(p) for p in elig[b] if int(F.pair_name[p]) not in used]
                rng.shuffle(cand)
                take = []
                for p in cand:
                    nm = int(F.pair_name[p])
                    if nm in used or any(int(F.pair_name[q]) == nm for q in take):
                        continue
                    take.append(p)
                    if len(take) == (2 if b in dbl else 1):
                        break
                for p in take:
                    used.add(int(F.pair_name[p]))
                pairs += take
            out.append({"fold": fi, "pairs": pairs, "cap": cap, "evalonly": [False] * len(pairs), "benches": sorted(bs)})
    return out


def _lik_logs(eta, k, n):
    """log-likelihood of k successes in n trials at logit eta (broadcast)."""
    return k * (-np.logaddexp(0.0, -eta)) + (n - k) * (-np.logaddexp(0.0, eta))


def npeb_predict(S, testsets, priors, budget, item_off=None):
    """Exact grid posterior predictive for every evaluation row of S at a budget (0 = prior)."""
    F = S.F
    ts_fold = np.array([t["fold"] for t in testsets])
    V = S.ev
    zprime_v = V["zi"] - np.array([priors[f]["zbar"] for f in ts_fold[V["ts"]]])
    if item_off is not None:
        zprime_v = zprime_v + item_off(V)
    # unit = (ts, pair): label counts and mean item offset of its labels
    unit_of_pair = {(int(t), int(p)): u for u, (t, p) in enumerate(S.units)}
    nU = S.n_units
    k = np.zeros(nU)
    n = np.zeros(nU)
    zs = np.zeros(nU)
    if budget > 0:
        E = S.acq[budget]
        zprime_e = E["zi"] - np.array([priors[f]["zbar"] for f in ts_fold[E["ts"]]])
        if item_off is not None:
            zprime_e = zprime_e + item_off(E)
        ue = np.array([unit_of_pair[(int(t), int(p))] for t, p in zip(E["ts"], E["pair"])], dtype=np.int64)
        k = np.bincount(ue, weights=E["y"], minlength=nU)
        n = np.bincount(ue, minlength=nU).astype(float)
        zs = np.bincount(ue, weights=zprime_e, minlength=nU) / np.maximum(n, 1)
    unit_ts = S.units[:, 0]
    unit_b = S.unit_bench
    gkey = unit_ts * 1000 + unit_b
    ug, ginv = np.unique(gkey, return_inverse=True)
    Bb, Bd = len(BGRID), len(DGRID)
    eta = BGRID[:, None] + DGRID[None, :]                      # (Bb, Bd)
    folds_u = ts_fold[unit_ts]
    logGd = np.stack([np.log(priors[f]["Gd"] + 1e-300) for f in range(len(priors))])
    logGb = np.stack([np.log(priors[f]["Gb"] + 1e-300) for f in range(len(priors))])
    CH = 1500

    def lw_of(sl):
        ll = _lik_logs(eta[None, :, :] + zs[sl, None, None], k[sl, None, None], n[sl, None, None])
        return ll + logGd[folds_u[sl]][:, None, :]
    # pass 1: log m_p(b) per unit
    lm = np.zeros((nU, Bb))
    for lo in range(0, nU, CH):
        sl = slice(lo, min(lo + CH, nU))
        lm[sl] = np.logaddexp.reduce(lw_of(sl), axis=2)
    lsum = np.zeros((len(ug), Bb))
    np.add.at(lsum, ginv, lm)
    gfold = np.zeros(len(ug), dtype=np.int64)
    gfold[ginv] = folds_u
    lpb = lsum + logGb[gfold]
    lpb -= np.logaddexp.reduce(lpb, axis=1, keepdims=True)
    # pass 2: joint posterior per unit, collapsed to the eta grid
    idx = np.clip(np.searchsorted(EGRID, eta.ravel()), 0, len(EGRID) - 1)
    M = np.zeros((Bb * Bd, len(EGRID)))
    M[np.arange(Bb * Bd), idx] = 1.0
    W = np.zeros((nU, len(EGRID)))
    for lo in range(0, nU, CH):
        sl = slice(lo, min(lo + CH, nU))
        lj = lpb[ginv[sl]][:, :, None] + lw_of(sl) - lm[sl][:, :, None]
        W[sl] = np.exp(lj).reshape(lj.shape[0], -1) @ M
    # predictions for evaluation rows: sum_eta W(eta) sigmoid(eta + z')
    p = np.zeros(len(V["y"]))
    ev_u = V["unit"]
    for lo in range(0, len(p), 50000):
        hi = min(lo + 50000, len(p))
        sg = X._sig(EGRID[None, :] + zprime_v[lo:hi, None])
        p[lo:hi] = np.sum(W[ev_u[lo:hi]] * sg, axis=1)
    return np.clip(p, model.CLIP_LO, model.CLIP_HI)


def eval_npeb(S, testsets, priors, item_off=None):
    br = np.zeros((S.n_units, 6))
    for bi, b in enumerate(X.BUDGETS):
        br[:, bi] = X.unit_briers(S, npeb_predict(S, testsets, priors, b, item_off))
    return br
