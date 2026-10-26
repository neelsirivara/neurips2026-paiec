"""Shared helpers for the 2026-10-26 deep dive (scratch; reuses paiec/experiments/src/xpair_sim.py unchanged)."""
import json, sys, math, time, csv, re, pickle
from pathlib import Path
SRC = Path(r"C:\Users\neels\Downloads\gen_papers_2\neurips_challenges\paiec\experiments\src")
sys.path.insert(0, str(SRC))
import numpy as np
import xpair_sim as X
import model

OUT = Path(r"C:\Users\neels\Downloads\gen_papers_2\neurips_challenges\paiec\experiments\results\deep_dive")
OUT.mkdir(parents=True, exist_ok=True)
INV_CSV = Path(r"C:\Users\neels\Downloads\gen_papers_2\neurips_challenges\_downloads\paiec_deepdive_2026-10-26\intel\inventory_gid0.csv")
SHIPPED_TAUS = dict(model.XP_TAUS)
SHIPPED_TI = model.XP_TAU_ITEM


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load():
    F = X.Frame(X.DEFAULT_PAYLOAD)
    folds = X.load_folds()
    return F, folds


def pair_stats(F):
    rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
    subj = [json.loads(k) for k in F.subj_keys]
    yrs = np.array([np.nan if model._year_fraction(s.get("release_date")) is None else model._year_fraction(s.get("release_date")) for s in subj])
    return rate, yrs[F.pair_subj]


def inventory_dirs(F):
    inv = list(csv.DictReader(open(INV_CSV, encoding="utf-8")))
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    slugs = {norm(r["benchmark_name_slug"]) for r in inv if r.get("benchmark_name_slug")}
    return {d for d in F.dirs if norm(d) in slugs}


def fold_of_dir(F, folds):
    m = {}
    for fi, f in enumerate(folds):
        for d in f:
            m[F.dirs.index(d)] = fi
    return m


def ability_tables(F, folds, k0=2.0, min_items=80, weight="pair"):
    """Per outer fold f: name-level ability a[name] from benchmarks NOT in fold f (centred residual pair logits).

    Residual of pair p on benchmark b: logit((k + .5)/(n + 1)) - mean over b's pairs; a_name = sum / (count + k0).
    Returns list over folds of arrays over pairs (ability of the pair's normalized_name; 0 if unseen) and coverage.
    """
    rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
    n = F.pair_n.astype(float)
    L = np.log((rate * n + 0.5) / (n - rate * n + 0.5))
    elig = F.pair_n >= min_items
    fod = fold_of_dir(F, folds)
    pf = np.array([fod.get(int(b), -1) for b in F.pair_dir])
    NB = int(F.pair_dir.max()) + 1
    out, cov = [], []
    names = F.pair_name
    NN = int(names.max()) + 1
    for f in range(len(folds)):
        tr = elig & (pf != f) & (pf >= 0)
        bs = np.bincount(F.pair_dir[tr], weights=L[tr], minlength=NB)
        bc = np.bincount(F.pair_dir[tr], minlength=NB)
        mb = bs / np.maximum(bc, 1)
        e = L - mb[F.pair_dir]
        s_ = np.bincount(names[tr], weights=e[tr], minlength=NN)
        c_ = np.bincount(names[tr], minlength=NN).astype(float)
        a = s_ / (c_ + k0)
        a_pair = a[names]
        seen = c_[names] > 0
        out.append(np.where(seen, a_pair, 0.0))
        cov.append(seen)
    return out, cov


def eval_xp(S, testsets, subj_logit_by_fold=None, taus=None, ti=None, probit=True, cap=None, b0_mode="prior",
            extra_offset=None):
    """Per-unit Brier (n_units x 6) for the XP model with an optional replacement of the subject part of the offset.

    subj_logit_by_fold: list over folds of arrays over pairs (new subject logit incl. intercept); None = shipped zs.
    extra_offset: optional function(table) -> array added to o (e.g. item effects).
    """
    taus = SHIPPED_TAUS if taus is None else taus
    ti = SHIPPED_TI if ti is None else ti
    cap = X.CAP if cap is None else cap
    ts_fold = np.array([t["fold"] for t in testsets])

    def off(T):
        if subj_logit_by_fold is None:
            zs = T["zs"]
        else:
            fo = ts_fold[T["ts"]]
            zs = np.zeros(len(T["ts"]))
            for f in range(len(subj_logit_by_fold)):
                m = fo == f
                zs[m] = subj_logit_by_fold[f][T["pair"][m]]
        o = np.clip(zs + T["zi"], -cap, cap)
        if extra_offset is not None:
            o = o + extra_offset(T)
        return o

    V = dict(S.ev)
    V["o"] = off(S.ev)
    br = np.zeros((S.n_units, 6))
    p0 = np.clip(X._sig(V["o"]), model.CLIP_LO, model.CLIP_HI)
    br[:, 0] = X.unit_briers(S, p0)
    preds = {0: p0}
    for bi, b in enumerate(X.BUDGETS[1:], start=1):
        E = dict(S.acq[b])
        E["o"] = off(S.acq[b])
        fit = X.xp_fit_np(E, V, taus)
        p = X.xp_predict_np(E, V, fit, ti, probit)
        preds[b] = p
        br[:, bi] = X.unit_briers(S, p)
    return br, preds


def alc(br):
    return br @ X.ALC_W


def paired(a, b, cl, seed=0):
    return X.cluster_boot(a - b, cl, seed=seed)


def fmt(r):
    return f"{r['mean']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
