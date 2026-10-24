"""Round 3: item-text difficulty prior fitted on the official PAIEC training release (measurement-db).

Pre-registered in ``experiments/results/round3/PREREG.md`` (its SHA-256 is stored in every output).
Run with ``python -I`` (reads downloaded data).  Steps:

    python -I text_prior.py build  [--payload DIR]   # frame + item targets + hashed text features -> DIR/round3_cache
    python -I text_prior.py cv                       # 5-fold benchmark-held-out CV -> results/round3/text_prior_cv.json
    python -I text_prior.py final                    # fit on all 144 benchmarks -> results/round3/text_prior_candidate.json
    python -I text_prior.py check                    # stdlib (model.py) vs numpy scores on release items
    python -I text_prior.py testbeds [--workers 4]   # full harness, both local testbeds -> results/round3/text_prior_testbeds.json

The features are produced by ``model.text_prior_features`` itself (stdlib), so the hashing used for
fitting and at prediction time is the same code.  Only this script uses numpy/scipy/scikit-learn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_THREADS = os.environ.get("PAIEC_FIT_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
os.environ.setdefault("PAIEC_FIT_THREADS", _THREADS)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fit_prior_from_measurement_db as F  # noqa: E402
import harness  # noqa: E402
import model  # noqa: E402
model.XP_ENABLED = False  # rounds 1-3 experiments evaluate the per-pair predictor (round 4: xpair_sim.py)

RESULTS = HERE.parent / "results"
R3 = RESULTS / "round3"
PREREG = R3 / "PREREG.md"
DEFAULT_PAYLOAD = HERE.parents[2] / "_downloads" / "measurement_db_payload"
MAX_TARGET_ITEMS = 1000
ALPHA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CAP_GRID = (1.5, 2.5, 3.5)
LAMBDA_GRID = (0.0, 0.5, 1.0)
ALPHA0 = 4.0
INNER_FOLDS = 4
N_CLUSTER_BOOT = 2000
N_BOOT = 1000
TARGET_CLIP = 5.0
D_HASH = 1 << model.TEXT_PRIOR_BITS
N_DENSE = len(model.TEXT_PRIOR_DENSE)
FEATURES = "all"   # "all" (pre-registered) or "dense" (exploratory addendum E2: the 16 statistics only)
ADDENDUM = R3 / "PREREG_ADDENDUM_1.md"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else "missing"


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ============================================================================= build
def cmd_build(args) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from scipy import sparse
    payload = Path(args.payload).resolve()
    data_dir = payload / "data"
    cache = payload / "round3_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"loading the iteration-2 frame from {data_dir} (keep_ids, item statistics) ...")
    stats: dict = {}
    pairs = F.load_pairs_from_payload(data_dir, keep_ids=True, item_stats=stats)
    dirs = sorted({p["dir"] for p in pairs})
    log(f"  {len(pairs)} pairs, {sum(len(p['items']) for p in pairs)} rows, {len(dirs)} benchmarks ({time.time() - t0:.0f}s)")
    fit_prev = json.loads((RESULTS / "prior_fit.json").read_text(encoding="utf-8"))
    providers = fit_prev["providers"]
    D = F.build_design(pairs, providers)
    hand = F.hand_set_coefficients()
    zs_h, zi_h = F.logits_from_coefs(D, hand, providers)
    # items needed: every frame item + up to MAX_TARGET_ITEMS target items per benchmark
    need: dict = defaultdict(set)
    for p in pairs:
        need[p["dir"]].update(p["item_ids"])
    sample: dict = {}
    for d in dirs:
        df = stats.get(d)
        ids = sorted(str(i) for i in df.index.tolist()) if df is not None else []
        rng = random.Random(f"tp|{d}")
        chosen = ids if len(ids) <= MAX_TARGET_ITEMS else rng.sample(ids, MAX_TARGET_ITEMS)
        sample[d] = set(chosen)
        need[d].update(chosen)
    log(f"  items to featurise: {sum(len(v) for v in need.values())} (target sample {sum(len(v) for v in sample.values())})")
    indptr, indices, values = [0], [], []
    dense_rows, item_dir, item_ids, in_sample = [], [], [], []
    sum_y, n_eval, sum_zs, zi_item, text_hash = [], [], [], [], []
    content_hashes: dict = {}
    key_to_row: dict = {}
    for k, d in enumerate(dirs, 1):
        path = data_dir / d / "items.parquet"
        found: dict = {}
        hashes = set()
        if path.exists():
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            ccol, fcol = F._first_col(names, F.CONTENT_CANDIDATES), F._first_col(names, F.FEATURE_CANDIDATES)
            icols = ["item_id"] + [c for c in (ccol, fcol) if c]
            for batch in pf.iter_batches(batch_size=4000, columns=icols):
                ids = pc.cast(batch.column(0), pa.string()).to_pylist()
                cont = (pc.cast(batch.column(icols.index(ccol)), pa.string()).to_pylist() if ccol else [None] * len(ids))
                feats = (pc.cast(batch.column(icols.index(fcol)), pa.string()).to_pylist() if fcol else [None] * len(ids))
                for i_, c_, f_ in zip(ids, cont, feats):
                    c_ = c_ or ""
                    hashes.add(hashlib.blake2b(c_.encode("utf-8"), digest_size=8).digest())
                    if i_ in need[d]:
                        found[i_] = (c_, f_ or "")
        content_hashes[d] = hashes
        df = stats.get(d)
        st = df.to_dict("index") if df is not None else {}
        for iid in sorted(need[d]):
            c_, f_ = found.get(iid, ("", ""))
            item = {"item_content": c_, "item_features": f_}
            hashed, dense = model.text_prior_features(item)
            idx = np.fromiter(hashed.keys(), dtype=np.int32, count=len(hashed))
            val = np.fromiter(hashed.values(), dtype=np.float64, count=len(hashed))
            order = np.argsort(idx, kind="stable")
            indices.append(idx[order])
            values.append(val[order].astype(np.float32))
            indptr.append(indptr[-1] + len(idx))
            dense_rows.append(dense)
            item_dir.append(k - 1)
            item_ids.append(iid)
            in_sample.append(iid in sample[d])
            s = st.get(iid)
            sum_y.append(float(s["sum_y"]) if s else np.nan)
            n_eval.append(float(s["n"]) if s else 0.0)
            sum_zs.append(float(s["sum_zs"]) if s else np.nan)
            zi_item.append(float(model._item_logit(item)))
            text_hash.append(hashlib.blake2b(c_.encode("utf-8"), digest_size=8).digest())
            key_to_row[(d, iid)] = len(item_ids) - 1
        if k % 10 == 0 or k == len(dirs):
            log(f"  {k}/{len(dirs)} dirs featurised, {len(item_ids)} items ({time.time() - t0:.0f}s)")
    X = sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), np.array(indptr, dtype=np.int64)),
                          shape=(len(item_ids), D_HASH))
    sparse.save_npz(cache / "X_hash.npz", X, compressed=False)
    # frame rows -> item rows
    frow = np.empty(len(D["y"]), dtype=np.int64)
    k = 0
    for p in pairs:
        for iid in p["item_ids"]:
            frow[k] = key_to_row[(p["dir"], iid)]
            k += 1
    assert k == len(frow)
    pair_start = np.array([r[0] for r in D["pair_rows"]] + [len(D["y"])], dtype=np.int64)
    pair_dir = np.array([dirs.index(p["dir"]) for p in pairs], dtype=np.int32)
    pair_year = np.array([dict(model.subject_terms(p["subject"])).get("PRIOR_YEAR_SLOPE", 0.0) for p in pairs])
    np.savez(cache / "meta.npz", dense=np.array(dense_rows, dtype=np.float64), item_dir=np.array(item_dir, dtype=np.int32),
             in_sample=np.array(in_sample), sum_y=np.array(sum_y), n_eval=np.array(n_eval), sum_zs=np.array(sum_zs),
             zi_item=np.array(zi_item), text_hash=np.frombuffer(b"".join(text_hash), dtype=np.uint64),
             y=D["y"], zs=zs_h, zi=zi_h, frow=frow, pair_start=pair_start, pair_dir=pair_dir, pair_year=pair_year)
    with open(cache / "aux.pkl", "wb") as fh:
        pickle.dump({"dirs": dirs, "item_ids": item_ids, "providers": providers,
                     "content_hashes": {d: np.frombuffer(b"".join(sorted(h)), dtype=np.uint64) for d, h in content_hashes.items()},
                     "revision": json.loads((data_dir / "corpus_manifest.json").read_text(encoding="utf-8")).get("revision"),
                     "n_items_per_dir": {d: len(content_hashes[d]) for d in dirs}}, fh)
    log(f"wrote {cache} ({X.nnz} non-zeros, {X.shape[0]} items; {time.time() - t0:.0f}s)")


# ============================================================================= data
class Data:
    def __init__(self, payload: Path):
        from scipy import sparse
        cache = Path(payload).resolve() / "round3_cache"
        self.X = sparse.load_npz(cache / "X_hash.npz").tocsr()
        m = np.load(cache / "meta.npz")
        for k in m.files:
            setattr(self, k, m[k])
        with open(cache / "aux.pkl", "rb") as fh:
            aux = pickle.load(fh)
        self.dirs, self.item_ids, self.providers = aux["dirs"], aux["item_ids"], aux["providers"]
        self.content_hashes, self.revision = aux["content_hashes"], aux["revision"]
        self.n_items_per_dir = aux["n_items_per_dir"]
        self.n_pairs = len(self.pair_start) - 1
        self.row_pair = np.repeat(np.arange(self.n_pairs), np.diff(self.pair_start))
        self.pair_n = np.diff(self.pair_start)
        self.base = self.zs + self.zi
        has = self.n_eval > 0
        t = np.full(len(self.n_eval), np.nan)
        rate = (self.sum_y[has] + 0.5) / (self.n_eval[has] + 1.0)
        t[has] = np.clip(_logit(rate) - self.sum_zs[has] / self.n_eval[has] - self.zi_item[has], -TARGET_CLIP, TARGET_CLIP)
        self.target = t
        self.is_target = self.in_sample & has
        n_b = np.bincount(self.item_dir[self.is_target], minlength=len(self.dirs)).astype(float)
        self.weight = np.where(self.is_target, 1.0 / np.maximum(n_b[self.item_dir], 1.0), 0.0)


# ============================================================================= ridge
def _design(data: Data, rows: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    from scipy import sparse
    Dn = (data.dense[rows] - mu) / sd
    if FEATURES == "dense":
        return Dn
    return sparse.hstack([data.X[rows], sparse.csr_matrix(Dn)], format="csr")


def ridge_fit(data: Data, train_items: np.ndarray, alpha: float) -> dict:
    from sklearn.linear_model import Ridge
    mu = data.dense[train_items].mean(axis=0)
    sd = data.dense[train_items].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    A = _design(data, train_items, mu, sd)
    w = data.weight[train_items]
    if FEATURES == "dense":
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(A, data.target[train_items], sample_weight=w)
        coef = np.concatenate([np.zeros(D_HASH), reg.coef_])
    else:
        reg = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", tol=1e-5, max_iter=2000)
        reg.fit(A, data.target[train_items], sample_weight=w)
        coef = reg.coef_
    return {"w_hash": coef[:D_HASH].copy(), "w_dense": coef[D_HASH:].copy(), "intercept": float(reg.intercept_),
            "mu": mu, "sd": sd, "alpha": alpha}


def ridge_score(data: Data, fit: dict, items: np.ndarray) -> np.ndarray:
    if FEATURES == "dense":
        return ((data.dense[items] - fit["mu"]) / fit["sd"]) @ fit["w_dense"] + fit["intercept"]
    return (data.X[items] @ fit["w_hash"] + ((data.dense[items] - fit["mu"]) / fit["sd"]) @ fit["w_dense"]
            + fit["intercept"])


def _dir_mask(data: Data, dir_idx) -> np.ndarray:
    return np.isin(data.item_dir, np.asarray(sorted(dir_idx), dtype=np.int32))


def inner_select(data: Data, train_dirs: list, seed: int, log_prefix: str = "", alpha_grid=None) -> dict:
    """Inner benchmark-grouped CV over ALPHA_GRID; returns the selected alpha, the per-alpha inner MSEs
    and out-of-fold scores (selected alpha) for every item of the training benchmarks."""
    grid = tuple(alpha_grid or ALPHA_GRID)
    tr = list(train_dirs)
    random.Random(seed).shuffle(tr)
    groups = [tr[i::INNER_FOLDS] for i in range(INNER_FOLDS)]
    all_items = np.where(_dir_mask(data, train_dirs))[0]
    oof = {a: np.full(len(data.item_dir), np.nan) for a in grid}
    sse = {a: 0.0 for a in grid}
    wsum = 0.0
    for gi, g in enumerate(groups):
        held = _dir_mask(data, g)
        fit_items = np.where(~held & _dir_mask(data, train_dirs) & data.is_target)[0]
        held_items = np.where(held)[0]
        held_t = held_items[data.is_target[held_items]]
        wsum += data.weight[held_t].sum()
        for a in grid:
            t1 = time.time()
            fit = ridge_fit(data, fit_items, a)
            oof[a][held_items] = ridge_score(data, fit, held_items)
            sse[a] += float(np.sum(data.weight[held_t] * (oof[a][held_t] - data.target[held_t]) ** 2))
            log(f"{log_prefix} inner {gi + 1}/{INNER_FOLDS} alpha={a:g}: {time.time() - t1:.0f}s")
    mse = {a: sse[a] / wsum for a in grid}
    best = min(grid, key=lambda a: (mse[a], a))
    # reference: weighted MSE of a constant (the weighted mean of the training targets)
    tt = all_items[data.is_target[all_items]]
    const = float(np.sum(data.weight[tt] * data.target[tt]) / np.sum(data.weight[tt]))
    mse_const = float(np.sum(data.weight[tt] * (data.target[tt] - const) ** 2) / np.sum(data.weight[tt]))
    return {"alpha": best, "inner_mse": {f"{a:g}": mse[a] for a in grid}, "inner_mse_constant": mse_const,
            "oof": oof[best], "groups": groups}


# ============================================================================= calibration
def fit_wb(data: Data, pair_ids: np.ndarray, s_item: np.ndarray, interaction: bool = False) -> dict:
    """Pair-weighted logistic fit of y ~ sigmoid(base + W s + B [+ G s year]) (Newton, fractional y)."""
    rows = np.concatenate([np.arange(data.pair_start[j], data.pair_start[j + 1]) for j in pair_ids])
    s = s_item[data.frow[rows]]
    base = data.base[rows]
    y = data.y[rows]
    w = 1.0 / data.pair_n[data.row_pair[rows]]
    if not np.any(s):
        theta_b = 0.0
        for _ in range(50):
            p = _sig(base + theta_b)
            step = float(np.sum(w * (y - p)) / (np.sum(w * p * (1 - p)) + 1e-12))
            theta_b += step
            if abs(step) < 1e-12:
                break
        return {"W": 0.0, "B": float(theta_b)}
    cols = [s, np.ones_like(s)]
    if interaction:
        cols.append(s * data.pair_year[data.row_pair[rows]])
    Z = np.stack(cols, axis=1)
    theta = np.zeros(Z.shape[1])
    for _ in range(50):
        p = _sig(base + Z @ theta)
        g = Z.T @ (w * (y - p))
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        theta = theta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    out = {"W": float(theta[0]), "B": float(theta[1])}
    if interaction:
        out["G"] = float(theta[2])
    return out


def row_text(data: Data, s_item: np.ndarray, cal: dict) -> np.ndarray:
    s = s_item[data.frow]
    T = cal["W"] * s + cal["B"]
    if cal.get("G"):
        T = T + cal["G"] * s * data.pair_year[data.row_pair]
    return np.clip(np.nan_to_num(T, nan=0.0), -model.TEXT_PRIOR_ABS_CAP, model.TEXT_PRIOR_ABS_CAP)


# ============================================================================= closed-form protocol
def eval_pairs(data: Data, pair_ids, T_row: np.ndarray, cap: float, a0: float, lam: float, seed: int = 0,
               base_row: np.ndarray | None = None, zs_row=None, zi_row=None) -> dict:
    """Per-pair B0 (all rows) and closed-form shrinkage Brier by budget / ALC (pairs with >= 80 rows);
    identical to fit_prior_from_measurement_db.pair_metrics when T = 0."""
    zs = data.zs if zs_row is None else zs_row
    zi = data.zi if zi_row is None else zi_row
    out = {"pair": [], "b0": [], "alc": [], "brier": [], "oracle": [], "alc_pairs": []}
    for j in pair_ids:
        a, b = data.pair_start[j], data.pair_start[j + 1]
        y = data.y[a:b]
        zsp = zs[a]
        zir = zi[a:b]
        T = T_row[a:b]
        p0 = _sig(np.clip(zsp + zir + T, -cap, cap))
        out["pair"].append(int(j))
        out["b0"].append(float(np.mean((p0 - y) ** 2)))
        n_rows = b - a
        if n_rows < F.MIN_PAIR_ITEMS_ALC:
            continue
        rng = np.random.RandomState(seed + int(j))
        perm = rng.permutation(n_rows)
        half = n_rows // 2
        acq, ev = perm[:half], perm[half:]
        order = acq[rng.permutation(len(acq))][:harness.MAX_LABELS]
        yy = y[ev]
        rate_ev = float(np.mean(yy))
        briers = {}
        for n in harness.BUDGETS:
            if n == 0:
                p = _sig(np.clip(zsp + zir[ev] + T[ev], -cap, cap))
            else:
                got = order[:n]
                k = float(np.sum(y[got]))
                mr = float(np.mean(zir[got] + T[got]))
                mo = float(np.mean(zir[got] + lam * T[got]))
                prior_rate = _sig(np.clip(zsp + mr, -cap, cap))
                rate_logit = _logit((a0 * prior_rate + k) / (a0 + n))
                p = _sig(rate_logit + (zir[ev] + lam * T[ev]) - mo)
            p = np.clip(p, model.CLIP_LO, model.CLIP_HI)
            briers[n] = float(np.mean((p - yy) ** 2))
        out["alc_pairs"].append(int(j))
        out["alc"].append(harness.alc(briers))
        out["brier"].append([briers[n] for n in harness.BUDGETS])
        out["oracle"].append(float(np.mean((rate_ev - yy) ** 2)))
    return out


def b0_mean(data: Data, pair_ids, T_row, cap) -> float:
    rows = np.concatenate([np.arange(data.pair_start[j], data.pair_start[j + 1]) for j in pair_ids])
    p0 = _sig(np.clip(data.base[rows] + T_row[rows], -cap, cap))
    se = (p0 - data.y[rows]) ** 2
    per_pair = np.bincount(data.row_pair[rows], weights=se, minlength=data.n_pairs)[pair_ids] / data.pair_n[pair_ids]
    return float(per_pair.mean())


# ============================================================================= statistics
def cluster_boot(diff: np.ndarray, clusters: np.ndarray, seed: int = 0, n_boot: int = N_CLUSTER_BOOT) -> dict:
    """Mean of per-pair values with a percentile bootstrap over clusters (benchmarks)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    S = np.bincount(inv, weights=diff)
    N = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uniq), size=(n_boot, len(uniq)))
    stats = np.sort(S[idx].sum(axis=1) / N[idx].sum(axis=1))
    return {"mean": float(diff.mean()), "ci_lo": float(stats[int(0.025 * n_boot)]),
            "ci_hi": float(stats[int(0.975 * n_boot) - 1]), "n_pairs": int(len(diff)), "n_clusters": int(len(uniq)),
            "n_boot": n_boot, "significant": bool(stats[int(0.975 * n_boot) - 1] < 0 or stats[int(0.025 * n_boot)] > 0),
            "frac_negative": float(np.mean(diff < 0))}


def cluster_boot_ratio(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, seed: int = 0) -> dict:
    uniq, inv = np.unique(clusters, return_inverse=True)
    Sn, Sd = np.bincount(inv, weights=num), np.bincount(inv, weights=den)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uniq), size=(N_CLUSTER_BOOT, len(uniq)))
    r = np.sort(Sn[idx].sum(axis=1) / Sd[idx].sum(axis=1))
    return {"ratio": float(num.sum() / den.sum()), "ci_lo": float(r[int(0.025 * N_CLUSTER_BOOT)]),
            "ci_hi": float(r[int(0.975 * N_CLUSTER_BOOT) - 1])}


def summarize_methods(data: Data, per: dict, ref: str = "hand") -> dict:
    """per: method -> eval_pairs output over the same pairs (in the same order)."""
    out = {"methods": {}, "paired_vs_" + ref: {}}
    pairs_b0 = np.array(per[ref]["pair"])
    pairs_alc = np.array(per[ref]["alc_pairs"])
    cl_b0, cl_alc = data.pair_dir[pairs_b0], data.pair_dir[pairs_alc]
    for name, m in per.items():
        rec = {"b0": F._ci(m["b0"]), "alc": F._ci(m["alc"]), "n_pairs_b0": len(m["b0"]), "n_pairs_alc": len(m["alc"])}
        br = np.array(m["brier"])
        rec["brier_by_budget"] = {str(b): float(br[:, i].mean()) for i, b in enumerate(harness.BUDGETS)}
        rec["oracle_rate_brier"] = float(np.mean(m["oracle"]))
        out["methods"][name] = rec
        if name == ref:
            continue
        d_alc = np.array(m["alc"]) - np.array(per[ref]["alc"])
        d_b0 = np.array(m["b0"]) - np.array(per[ref]["b0"])
        rec_p = {"alc_cluster": cluster_boot(d_alc, cl_alc, seed=0), "b0_cluster": cluster_boot(d_b0, cl_b0, seed=0),
                 "alc_pair_boot": F._paired(list(m["alc"]), list(per[ref]["alc"])),
                 "b0_pair_boot": F._paired(list(m["b0"]), list(per[ref]["b0"]))}
        br_ref = np.array(per[ref]["brier"])
        orc = np.array(m["oracle"])
        rec_p["by_budget_cluster"] = {}
        rec_p["gap_closed"] = {}
        for i, b in enumerate(harness.BUDGETS):
            rec_p["by_budget_cluster"][str(b)] = cluster_boot(br[:, i] - br_ref[:, i], cl_alc, seed=0)
            rec_p["gap_closed"][str(b)] = cluster_boot_ratio(br_ref[:, i] - br[:, i], br_ref[:, i] - orc, cl_alc, seed=0)
        out["paired_vs_" + ref][name] = rec_p
    return out


# ============================================================================= one outer fold
def run_fold(data: Data, train_dirs: list, test_dirs: list | None, seed: int, tag: str, interaction: bool = False,
             alpha_grid=None) -> dict:
    t0 = time.time()
    didx = {d: i for i, d in enumerate(data.dirs)}
    tr_idx = [didx[d] for d in train_dirs]
    sel = inner_select(data, tr_idx, seed=seed, log_prefix=f"[{tag}]", alpha_grid=alpha_grid)
    oof = sel.pop("oof")
    tr_pairs = np.where(np.isin(data.pair_dir, tr_idx))[0]
    cal = fit_wb(data, tr_pairs, oof, interaction=interaction)
    T_oof = row_text(data, oof, cal)
    caps = {f"{c:g}": b0_mean(data, tr_pairs, T_oof, c) for c in CAP_GRID}
    cap = min(CAP_GRID, key=lambda c: (caps[f"{c:g}"], c))
    lams = {}
    for lam in LAMBDA_GRID:
        lams[f"{lam:g}"] = float(np.mean(eval_pairs(data, tr_pairs, T_oof, cap, ALPHA0, lam)["alc"]))
    lam = min(LAMBDA_GRID, key=lambda l_: (lams[f"{l_:g}"], -l_))
    train_items = np.where(_dir_mask(data, tr_idx) & data.is_target)[0]
    fit = ridge_fit(data, train_items, sel["alpha"])
    rec = {"tag": tag, "n_train_dirs": len(train_dirs), "alpha": sel["alpha"], "inner_mse": sel["inner_mse"],
           "inner_mse_constant": sel["inner_mse_constant"], "calibration": cal, "cap": cap, "train_b0_by_cap": caps,
           "lambda": lam, "train_alc_by_lambda": lams, "n_train_target_items": int(len(train_items)),
           "train_pairs": int(len(tr_pairs)), "seconds": None}
    if test_dirs is not None:
        te_idx = [didx[d] for d in test_dirs]
        te_items = np.where(_dir_mask(data, te_idx))[0]
        s = np.full(len(data.item_dir), np.nan)
        s[te_items] = ridge_score(data, fit, te_items)
        rec["test_s"] = s
        rec["test_pairs"] = np.where(np.isin(data.pair_dir, te_idx))[0]
        # in-sample fit quality on the test items with targets (weighted MSE of the residual target)
        tt = te_items[data.n_eval[te_items] > 0]
        wt = 1.0 / np.bincount(data.item_dir[tt], minlength=len(data.dirs))[data.item_dir[tt]]
        rec["test_target_mse"] = float(np.sum(wt * (s[tt] - data.target[tt]) ** 2) / wt.sum())
        cst = float(np.sum(data.weight[train_items] * data.target[train_items]) / data.weight[train_items].sum())
        rec["test_target_mse_constant"] = float(np.sum(wt * (cst - data.target[tt]) ** 2) / wt.sum())
    rec["fit"] = fit
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"[{tag}] alpha {sel['alpha']:g} (inner MSE {sel['inner_mse'][f'{sel['alpha']:g}']:.4f} vs constant "
        f"{sel['inner_mse_constant']:.4f}); W {cal['W']:.3f} B {cal['B']:.3f}{' G %.3f' % cal['G'] if 'G' in cal else ''}; "
        f"cap {cap:g}; lambda {lam:g}; {rec['seconds']:.0f}s")
    return rec


# ============================================================================= cv
def cmd_cv(args) -> None:
    data = Data(args.payload)
    R3.mkdir(parents=True, exist_ok=True)
    fit_prev = json.loads((RESULTS / "prior_fit.json").read_text(encoding="utf-8"))
    folds = fit_prev["cv"]["folds"]
    all_pairs = np.arange(data.n_pairs)
    zero = np.zeros(len(data.y))
    log("hand-set reference (reproduction check against prior_fit.json) ...")
    hand = eval_pairs(data, all_pairs, zero, model.PRIOR_LOGIT_CAP, ALPHA0, 1.0)
    ref = fit_prev["cv"]["hand_alpha4"]
    repro = {"b0": float(np.mean(hand["b0"])), "alc": float(np.mean(hand["alc"])),
             "b0_iteration2": ref["b0"]["mean"], "alc_iteration2": ref["alc"]["mean"]}
    repro["exact"] = bool(abs(repro["b0"] - repro["b0_iteration2"]) < 1e-12 and abs(repro["alc"] - repro["alc_iteration2"]) < 1e-12)
    log(f"  hand-set B0 {repro['b0']:.5f} (iteration 2: {repro['b0_iteration2']:.5f}); ALC {repro['alc']:.5f} "
        f"({repro['alc_iteration2']:.5f}); exact={repro['exact']}")
    T_test = np.zeros(len(data.y))
    B_only = np.zeros(len(data.y))
    fold_recs = []
    fold_cache = Path(args.payload).resolve() / "round3_cache" / "cv_folds.pkl"
    if args.resume and fold_cache.exists():
        with open(fold_cache, "rb") as fh:
            T_test, B_only, fold_recs = pickle.load(fh)
        log(f"resumed {len(fold_recs)} folds from {fold_cache}")
    for fi, f in enumerate(folds):
        if fi < len(fold_recs):
            continue
        test_dirs = sorted(f)
        train_dirs = sorted(d for d in data.dirs if d not in set(f))
        rec = run_fold(data, train_dirs, test_dirs, seed=1000 + fi, tag=f"fold{fi + 1}")
        rows = np.isin(data.row_pair, rec["test_pairs"])
        T_fold = row_text(data, rec["test_s"], rec["calibration"])
        T_test[rows] = T_fold[rows]
        # descriptive decomposition (not pre-registered): the calibration intercept alone, refitted with W = 0
        tr_pairs = np.where(~np.isin(data.pair_dir, [data.dirs.index(d) for d in test_dirs]))[0]
        b_alone = fit_wb(data, tr_pairs, np.zeros(len(data.item_dir)))["B"]
        B_only[rows] = b_alone
        rec_out = {k: v for k, v in rec.items() if k not in ("test_s", "test_pairs", "fit")}
        rec_out["test_dirs"] = test_dirs
        rec_out["n_test_pairs"] = int(len(rec["test_pairs"]))
        rec_out["intercept_alone_B"] = float(b_alone)
        fold_recs.append(rec_out)
        rec["fit"] = None
        with open(fold_cache, "wb") as fh:
            pickle.dump((T_test, B_only, fold_recs), fh)
    # per-pair evaluation with each fold's own cap and lambda
    per = {"hand": hand}
    text = {"pair": [], "b0": [], "alc": [], "brier": [], "oracle": [], "alc_pairs": []}
    a2 = {k: [] for k in text}
    a8 = {k: [] for k in text}
    bonly = {k: [] for k in text}
    pairs_by_fold = []
    for fi, f in enumerate(folds):
        te = np.where(np.isin(data.pair_dir, [data.dirs.index(d) for d in f]))[0]
        pairs_by_fold.append(te)
        cap, lam = fold_recs[fi]["cap"], fold_recs[fi]["lambda"]
        for dst, a0 in ((text, ALPHA0), (a2, 2.0), (a8, 8.0)):
            m = eval_pairs(data, te, T_test, cap, a0, lam)
            for k in dst:
                dst[k] += m[k]
        m = eval_pairs(data, te, B_only, model.PRIOR_LOGIT_CAP, ALPHA0, 1.0)
        for k in bonly:
            bonly[k] += m[k]
    order = np.argsort(text["pair"], kind="stable")
    order_alc = np.argsort(text["alc_pairs"], kind="stable")

    def _reorder(m):
        return {"pair": list(np.array(m["pair"])[order]), "b0": list(np.array(m["b0"])[order]),
                "alc_pairs": list(np.array(m["alc_pairs"])[order_alc]), "alc": list(np.array(m["alc"])[order_alc]),
                "brier": list(np.array(m["brier"])[order_alc]), "oracle": list(np.array(m["oracle"])[order_alc])}
    per["text"] = _reorder(text)
    per["text_alpha2"] = _reorder(a2)
    per["text_alpha8"] = _reorder(a8)
    per["intercept_shift_only"] = _reorder(bonly)
    assert per["text"]["pair"] == hand["pair"] and per["text"]["alc_pairs"] == hand["alc_pairs"]
    # references: intercept-only (train-fold mean) and the held-out benchmark-mean oracle
    z_int = np.zeros(len(data.y))
    z_bench = np.zeros(len(data.y))
    pair_rate = np.bincount(data.row_pair, weights=data.y) / data.pair_n
    for fi, f in enumerate(folds):
        te_idx = [data.dirs.index(d) for d in f]
        tr_rows = ~np.isin(data.pair_dir[data.row_pair], te_idx)
        w = 1.0 / data.pair_n[data.row_pair[tr_rows]]
        mean = float(np.sum(w * data.y[tr_rows]) / np.sum(w))
        rows = np.isin(data.pair_dir[data.row_pair], te_idx)
        z_int[rows] = _logit(np.array([mean]))[0]
    for b in range(len(data.dirs)):
        pj = np.where(data.pair_dir == b)[0]
        if len(pj):
            rows = np.isin(data.row_pair, pj)
            z_bench[rows] = _logit(np.array([float(pair_rate[pj].mean())]))[0]
    zero_i = np.zeros(len(data.y))
    per["intercept_only"] = eval_pairs(data, all_pairs, zero, 50.0, ALPHA0, 1.0, zs_row=z_int, zi_row=zero_i)
    per["benchmark_mean_oracle"] = eval_pairs(data, all_pairs, zero, 50.0, ALPHA0, 1.0, zs_row=z_bench, zi_row=zero_i)
    summ = summarize_methods(data, per, ref="hand")
    # exact-duplicate diagnostic: fraction of each test pair's items whose content also occurs in a training benchmark
    dup_frac = np.zeros(data.n_pairs)
    for fi, f in enumerate(folds):
        te_idx = set(data.dirs.index(d) for d in f)
        train_hash = np.unique(np.concatenate([data.content_hashes[d] for i, d in enumerate(data.dirs) if i not in te_idx]))
        rows = np.where(np.isin(data.row_pair, pairs_by_fold[fi]))[0]
        hit = np.isin(data.text_hash[data.frow[rows]], train_hash).astype(float)
        cnt = np.bincount(data.row_pair[rows], weights=hit, minlength=data.n_pairs)
        dup_frac[pairs_by_fold[fi]] = cnt[pairs_by_fold[fi]] / data.pair_n[pairs_by_fold[fi]]
    low_dup = np.array(hand["alc_pairs"])[dup_frac[np.array(hand["alc_pairs"])] < 0.05]
    sel_mask = np.isin(np.array(hand["alc_pairs"]), low_dup)
    d_alc = np.array(per["text"]["alc"]) - np.array(hand["alc"])
    dup = {"mean_dup_fraction_pairs": float(dup_frac.mean()), "frac_pairs_dup_over_5pct": float(np.mean(dup_frac >= 0.05)),
           "frac_pairs_dup_over_50pct": float(np.mean(dup_frac >= 0.5)),
           "dirs_with_dup_over_20pct": sorted({data.dirs[data.pair_dir[j]] for j in all_pairs if dup_frac[j] >= 0.2}),
           "alc_diff_low_dup_pairs": cluster_boot(d_alc[sel_mask], data.pair_dir[np.array(hand["alc_pairs"])[sel_mask]], seed=0)
           if sel_mask.any() else None,
           "alc_diff_high_dup_pairs": cluster_boot(d_alc[~sel_mask], data.pair_dir[np.array(hand["alc_pairs"])[~sel_mask]], seed=0)
           if (~sel_mask).any() else None}
    # per-fold paired differences
    per_fold = []
    alc_pairs = np.array(hand["alc_pairs"])
    for fi in range(len(folds)):
        msk = np.isin(alc_pairs, pairs_by_fold[fi])
        per_fold.append({"fold": fi + 1, "n_pairs_alc": int(msk.sum()), "hand_alc": float(np.mean(np.array(hand["alc"])[msk])),
                         "text_alc": float(np.mean(np.array(per["text"]["alc"])[msk])),
                         "diff": cluster_boot(d_alc[msk], data.pair_dir[alc_pairs[msk]], seed=0)})
    # per-benchmark differences (for the appendix figure / table)
    per_bench = {}
    for b in np.unique(data.pair_dir[alc_pairs]):
        msk = data.pair_dir[alc_pairs] == b
        per_bench[data.dirs[b]] = {"n_pairs": int(msk.sum()), "hand_alc": float(np.mean(np.array(hand["alc"])[msk])),
                                   "text_alc": float(np.mean(np.array(per["text"]["alc"])[msk])),
                                   "mean_pair_rate": float(pair_rate[alc_pairs[msk]].mean())}
    d_main = summ["paired_vs_hand"]["text"]["alc_cluster"]
    rule_a = bool(d_main["ci_hi"] < 0)
    out = {"prereg_sha256": prereg_sha(), "revision": data.revision, "created": time.strftime("%Y-%m-%d %H:%M"),
           "n_pairs": int(data.n_pairs), "n_benchmarks": len(data.dirs), "n_items_featurised": int(len(data.item_dir)),
           "n_target_items": int(data.is_target.sum()), "X_nnz": int(data.X.nnz),
           "settings": {"alpha_grid": ALPHA_GRID, "cap_grid": CAP_GRID, "lambda_grid": LAMBDA_GRID, "alpha0": ALPHA0,
                        "inner_folds": INNER_FOLDS, "max_target_items": MAX_TARGET_ITEMS, "target_clip": TARGET_CLIP,
                        "bits": model.TEXT_PRIOR_BITS, "word_chars": model.TEXT_PRIOR_WORD_CHARS,
                        "char_chars": model.TEXT_PRIOR_CHAR_CHARS, "n_cluster_boot": N_CLUSTER_BOOT},
           "reproduction": repro, "folds": fold_recs, "summary": summ, "per_fold": per_fold, "duplicates": dup,
           "per_benchmark": per_bench, "rule_a_cv_alc_ci_below_zero": rule_a, "folds_dirs": folds}
    path = R3 / "text_prior_cv.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    np.savez(R3 / "text_prior_cv_perpair.npz", pair=np.array(hand["alc_pairs"]), hand_alc=np.array(hand["alc"]),
             text_alc=np.array(per["text"]["alc"]), dirs=np.array(data.dirs), pair_dir=data.pair_dir[alc_pairs],
             hand_b0=np.array(hand["b0"]), text_b0=np.array(per["text"]["b0"]))
    m = summ["methods"]
    log(f"CV: hand B0 {m['hand']['b0']['mean']:.4f} ALC {m['hand']['alc']['mean']:.4f}; text B0 {m['text']['b0']['mean']:.4f} "
        f"ALC {m['text']['alc']['mean']:.4f}; diff ALC {d_main['mean']:+.4f} [{d_main['ci_lo']:+.4f}, {d_main['ci_hi']:+.4f}] "
        f"(cluster); rule (a) {'PASS' if rule_a else 'FAIL'}")
    log(f"wrote {path}")
    if rule_a and not args.no_interaction:
        cmd_cv_interaction(data, folds, per, hand, out)


def cmd_cv_interaction(data: Data, folds, per, hand, out) -> None:
    """Secondary (pre-registered, only if the primary passes rule (a)): text x release-year interaction."""
    T = np.zeros(len(data.y))
    recs = []
    texti = {"pair": [], "b0": [], "alc": [], "brier": [], "oracle": [], "alc_pairs": []}
    for fi, f in enumerate(folds):
        test_dirs = sorted(f)
        train_dirs = sorted(d for d in data.dirs if d not in set(f))
        rec = run_fold(data, train_dirs, test_dirs, seed=1000 + fi, tag=f"int-fold{fi + 1}", interaction=True)
        rows = np.isin(data.row_pair, rec["test_pairs"])
        T[rows] = row_text(data, rec["test_s"], rec["calibration"])[rows]
        m = eval_pairs(data, rec["test_pairs"], T, rec["cap"], ALPHA0, rec["lambda"])
        for k in texti:
            texti[k] += m[k]
        recs.append({k: v for k, v in rec.items() if k not in ("test_s", "test_pairs", "fit")})
    o1, o2 = np.argsort(texti["pair"], kind="stable"), np.argsort(texti["alc_pairs"], kind="stable")
    ti = {"pair": list(np.array(texti["pair"])[o1]), "b0": list(np.array(texti["b0"])[o1]),
          "alc_pairs": list(np.array(texti["alc_pairs"])[o2]), "alc": list(np.array(texti["alc"])[o2]),
          "brier": list(np.array(texti["brier"])[o2]), "oracle": list(np.array(texti["oracle"])[o2])}
    summ = summarize_methods(data, {"text": per["text"], "text_interaction": ti}, ref="text")
    res = {"prereg_sha256": prereg_sha(), "folds": recs, "summary": summ}
    path = R3 / "text_prior_cv_interaction.json"
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    d = summ["paired_vs_text"]["text_interaction"]["alc_cluster"]
    log(f"interaction vs primary: ALC diff {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]; wrote {path}")


# ============================================================================= exploratory addendum
EXPLORE = {"E1_heavier_penalty": ("all", (10.0, 100.0, 1000.0, 10000.0)),
           "E2_statistics_only": ("dense", (0.1, 1.0, 10.0, 100.0))}


def cmd_explore(args) -> None:
    """PREREG_ADDENDUM_1.md: exploratory variants (never used for the ship decision)."""
    global FEATURES
    data = Data(args.payload)
    fit_prev = json.loads((RESULTS / "prior_fit.json").read_text(encoding="utf-8"))
    folds = fit_prev["cv"]["folds"]
    hand = eval_pairs(data, np.arange(data.n_pairs), np.zeros(len(data.y)), model.PRIOR_LOGIT_CAP, ALPHA0, 1.0)
    out = {"addendum_sha256": hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(), "prereg_sha256": prereg_sha(),
           "status": "exploratory; not used for the ship decision", "variants": {}}
    names = [args.variant] if args.variant else list(EXPLORE)
    for name in names:
        FEATURES, grid = EXPLORE[name]
        T = np.zeros(len(data.y))
        res = {"pair": [], "b0": [], "alc": [], "brier": [], "oracle": [], "alc_pairs": []}
        recs = []
        for fi, f in enumerate(folds):
            test_dirs = sorted(f)
            train_dirs = sorted(d for d in data.dirs if d not in set(f))
            rec = run_fold(data, train_dirs, test_dirs, seed=1000 + fi, tag=f"{name}-fold{fi + 1}", alpha_grid=grid)
            rows = np.isin(data.row_pair, rec["test_pairs"])
            T[rows] = row_text(data, rec["test_s"], rec["calibration"])[rows]
            m = eval_pairs(data, rec["test_pairs"], T, rec["cap"], ALPHA0, rec["lambda"])
            for k in res:
                res[k] += m[k]
            recs.append({k: v for k, v in rec.items() if k not in ("test_s", "test_pairs", "fit")})
        o1, o2 = np.argsort(res["pair"], kind="stable"), np.argsort(res["alc_pairs"], kind="stable")
        r = {"pair": list(np.array(res["pair"])[o1]), "b0": list(np.array(res["b0"])[o1]),
             "alc_pairs": list(np.array(res["alc_pairs"])[o2]), "alc": list(np.array(res["alc"])[o2]),
             "brier": list(np.array(res["brier"])[o2]), "oracle": list(np.array(res["oracle"])[o2])}
        summ = summarize_methods(data, {"hand": hand, name: r}, ref="hand")
        out["variants"][name] = {"features": FEATURES, "alpha_grid": grid, "folds": recs, "summary": summ}
        d = summ["paired_vs_hand"][name]["alc_cluster"]
        log(f"{name}: ALC diff vs hand {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
    FEATURES = "all"
    path = R3 / "text_prior_explore.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prev.update({k: v for k, v in out.items() if k != "variants"})
    prev.setdefault("variants", {}).update(out["variants"])
    path.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    log(f"wrote {path}")


# ============================================================================= shipped equivalence
def cmd_equivalence(args) -> None:
    """The shipped submission/model.py vs experiments/src/model.py (round-3 hooks, no weights file):
    predict_components and predict on every budget of the real pairs and 60 synthetic pairs."""
    import importlib.util
    import zipfile
    import real_testbed
    import simulate
    root = HERE.parents[1]
    shipped_path = root / "submission" / "model.py"
    spec = importlib.util.spec_from_file_location("paiec_shipped_model", shipped_path)
    shipped = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shipped)
    model.load_text_prior(False)
    pairs = real_testbed.load_pairs() + simulate.generate_pairs(0, 10, 16, 0.7, 0.3)[:60]
    n_cases, n_same = 0, 0
    for raw in pairs:
        subject, sid, bid, items = raw[0], raw[1], raw[2], raw[3]
        rng = random.Random(f"0|{sid}|{bid}")
        pair = harness.make_pair(dict(subject), sid, bid, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
        if pair is None:
            continue
        pol = harness.random_policy_factory(random.Random(f"acq|0|{sid}|{bid}"))
        acq = harness.run_acquisition(pair, pol, pair["order"], needs_prediction=False)
        for n in harness.BUDGETS:
            labeled = [[[pair["subject"], pair["acq"][i][0]], pair["acq"][i][1]] for i in acq[:n]]
            for item, _ in pair["eval"][:12]:
                a = model.predict_components([pair["subject"], item], labeled)
                b = shipped.predict_components([pair["subject"], item], labeled)
                same = all(repr(a[k]) == repr(b[k]) for k in ("prior", "pair", "ridge", "knn", "full"))
                same = same and repr(model.predict([pair["subject"], item], labeled)) == repr(shipped.predict([pair["subject"], item], labeled))
                n_cases += 1
                n_same += int(same)
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    zpath = root / "submission" / "paiec_submission.zip"
    with zipfile.ZipFile(zpath) as zf:
        zip_model = zf.read("model.py")
    info = json.loads((RESULTS / "submission_zip_info.json").read_text(encoding="utf-8"))
    res = {"n_cases": n_cases, "n_bit_identical": n_same, "shipped_model_sha256": sha(shipped_path.read_bytes()),
           "zip_model_sha256": sha(zip_model), "zip_sha256": sha(zpath.read_bytes()),
           "zip_sha256_recorded_iteration2": info.get("sha256"),
           "zip_unchanged": sha(zpath.read_bytes()) == info.get("sha256"),
           "zip_member_equals_submission_model": sha(zip_model) == sha(shipped_path.read_bytes()),
           "dev_model_sha256": sha((HERE / "model.py").read_bytes()),
           "dev_text_prior_file_present": (HERE / model.TEXT_PRIOR_FILE).exists()}
    (R3 / "shipped_equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"equivalence: {res}")


# ============================================================================= runtime gate
def cmd_runtime(args) -> None:
    """Engineering gates of PREREG.md for the candidate: predict() throughput on one core with a new target item
    per call (budget 31; 15-, 300- and 2,000-word items), weight-file size and load time, and the smoke test
    of model.py + labeling.py with the candidate installed as text_prior.json (in a temporary directory)."""
    import shutil
    import subprocess
    import tempfile
    import smoke_test as S
    spec_path = Path(args.spec).resolve() if args.spec else (R3 / "text_prior_candidate.json").resolve()
    t0 = time.perf_counter()
    spec = model.load_text_prior(str(spec_path))
    load_ms = 1000 * (time.perf_counter() - t0)
    assert spec is not None
    rng = random.Random(0)
    out = {"spec": spec_path.name, "weights_file_bytes": spec_path.stat().st_size, "load_ms": load_ms, "throughput": {}}
    for label, nwords in (("short15", 15), ("words300", 300), ("words2000", 2000)):
        lab = []
        for _ in range(31):
            it = dict(S.SAMPLE_ITEM, benchmark_id="b", item_content=" ".join(rng.choice(S.WORDS) for _ in range(nwords)))
            lab.append([[S.SAMPLE_SUBJECT, it], rng.randint(0, 1)])
        ts = []
        for _ in range(200):
            tgt = dict(S.SAMPLE_ITEM, benchmark_id="b", item_content=" ".join(rng.choice(S.WORDS) for _ in range(nwords)))
            t1 = time.perf_counter()
            model.predict([S.SAMPLE_SUBJECT, tgt], labeled=lab)
            ts.append(time.perf_counter() - t1)
        out["throughput"][label] = {"calls_per_s": len(ts) / sum(ts), "mean_ms": 1000 * sum(ts) / len(ts), "max_ms": 1000 * max(ts)}
    model.load_text_prior(False)
    with tempfile.TemporaryDirectory(prefix="paiec_tp_") as tmp:
        for name in ("model.py", "labeling.py"):
            shutil.copyfile(HERE / name, Path(tmp) / name)
        shutil.copyfile(spec_path, Path(tmp) / model.TEXT_PRIOR_FILE)
        smoke_out = R3 / "text_prior_smoke_test.json"
        proc = subprocess.run([sys.executable, str(HERE / "smoke_test.py"), "--dir", tmp, "--out", str(smoke_out)],
                              capture_output=True, text=True)
        out["smoke_exit_code"] = proc.returncode
        if smoke_out.exists():
            sm = json.loads(smoke_out.read_text(encoding="utf-8"))
            out["smoke"] = {"n_cases": len(sm.get("cases", [])), "text_prior_loaded": sm.get("text_prior_loaded"),
                            "calls_per_second_long_budget31": sm.get("calls_per_second_long_budget31"),
                            "mean_ms_long_budget31": sm.get("mean_ms_long_budget31")}
    min_cps = min(v["calls_per_s"] for v in out["throughput"].values())
    out["gate_throughput_200"] = bool(min_cps >= 200)
    out["gate_weights_5mb"] = bool(out["weights_file_bytes"] <= 5e6)
    out["gate_smoke"] = bool(out["smoke_exit_code"] == 0 and out.get("smoke", {}).get("text_prior_loaded"))
    (R3 / "text_prior_runtime.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"runtime: {json.dumps(out)}")


# ============================================================================= final fit
def spec_from_fit(fit: dict, cal: dict, cap: float, lam: float, extra: dict) -> dict:
    import base64
    w32 = np.asarray(fit["w_hash"], dtype="<f4")
    spec = {"featurizer": {"version": model.TEXT_PRIOR_VERSION, "bits": model.TEXT_PRIOR_BITS,
                           "word_chars": model.TEXT_PRIOR_WORD_CHARS, "char_chars": model.TEXT_PRIOR_CHAR_CHARS,
                           "stat_chars": model.TEXT_PRIOR_STAT_CHARS, "dense": list(model.TEXT_PRIOR_DENSE)},
            "intercept": float(fit["intercept"]), "W": float(cal["W"]), "B": float(cal["B"]), "cap": float(cap),
            "within": float(lam), "dense_w": [float(x) for x in fit["w_dense"]],
            "dense_mean": [float(x) for x in fit["mu"]], "dense_sd": [float(x) for x in fit["sd"]],
            "provenance": extra, "weights_f32_le_b64": base64.b64encode(w32.tobytes()).decode("ascii")}
    return spec


def cmd_final(args) -> None:
    data = Data(args.payload)
    rec = run_fold(data, list(data.dirs), None, seed=999, tag="final")
    fit = rec["fit"]
    extra = {"source": "aims-foundations/measurement-db", "revision": data.revision, "benchmarks": list(data.dirs),
             "n_target_items": int(rec["n_train_target_items"]), "alpha": rec["alpha"], "inner_mse": rec["inner_mse"],
             "inner_mse_constant": rec["inner_mse_constant"], "fitted_on": time.strftime("%Y-%m-%d %H:%M"),
             "prereg_sha256": prereg_sha(), "script": "experiments/src/text_prior.py final"}
    spec = spec_from_fit(fit, rec["calibration"], rec["cap"], rec["lambda"], extra)
    path = R3 / "text_prior_candidate.json"
    path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    summary = {k: v for k, v in rec.items() if k != "fit"}
    summary["n_nonzero_weights"] = int(np.sum(fit["w_hash"] != 0))
    summary["weights_file_bytes"] = path.stat().st_size
    summary["prereg_sha256"] = prereg_sha()
    (R3 / "text_prior_final_fit.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


# ============================================================================= check
def cmd_check(args) -> None:
    """model.py (stdlib) features/scores vs the cached matrix and numpy on release items."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    data = Data(args.payload)
    spec_path = Path(args.spec) if args.spec else R3 / "text_prior_candidate.json"
    spec = model.load_text_prior(str(spec_path))
    assert spec is not None, "candidate spec did not load"
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    import base64
    w = np.frombuffer(base64.b64decode(raw["weights_f32_le_b64"]), dtype="<f4").astype(np.float64)
    rng = np.random.RandomState(0)
    pick = rng.choice(len(data.item_ids), size=min(args.n, len(data.item_ids)), replace=False)
    by_dir = defaultdict(list)
    for i in pick:
        by_dir[data.dirs[data.item_dir[i]]].append(i)
    payload = Path(args.payload).resolve() / "data"
    max_feat_diff, max_score_diff, n_checked, n_identical = 0.0, 0.0, 0, 0
    for d, rows in by_dir.items():
        want = {data.item_ids[i]: i for i in rows}
        pf = pq.ParquetFile(payload / d / "items.parquet")
        names = pf.schema_arrow.names
        ccol, fcol = F._first_col(names, F.CONTENT_CANDIDATES), F._first_col(names, F.FEATURE_CANDIDATES)
        icols = ["item_id"] + [c for c in (ccol, fcol) if c]
        found = {}
        for batch in pf.iter_batches(batch_size=4000, columns=icols):
            ids = pc.cast(batch.column(0), pa.string()).to_pylist()
            cont = pc.cast(batch.column(icols.index(ccol)), pa.string()).to_pylist() if ccol else [None] * len(ids)
            feats = pc.cast(batch.column(icols.index(fcol)), pa.string()).to_pylist() if fcol else [None] * len(ids)
            for i_, c_, f_ in zip(ids, cont, feats):
                if i_ in want:
                    found[i_] = (c_ or "", f_ or "")
        for iid, i in want.items():
            c_, f_ = found.get(iid, ("", ""))
            item = {"item_content": c_, "item_features": f_}
            hashed, dense = model.text_prior_features(item)
            row = data.X[i]
            cached = dict(zip(row.indices.tolist(), row.data.tolist()))
            same_keys = set(hashed) == set(cached)
            diff = max(abs(np.float32(hashed[k]) - cached[k]) for k in hashed) if hashed and same_keys else (0.0 if same_keys else 1.0)
            n_identical += int(same_keys and diff == 0.0 and np.allclose(dense, data.dense[i], rtol=0, atol=0))
            max_feat_diff = max(max_feat_diff, float(diff))
            s_np = float((row @ w)[0] + ((data.dense[i] - np.array(raw["dense_mean"])) / np.array(raw["dense_sd"])) @ np.array(raw["dense_w"])
                         + raw["intercept"])
            s_py = model.text_prior_score(item, spec)
            max_score_diff = max(max_score_diff, abs(s_np - s_py))
            n_checked += 1
    res = {"n_items": n_checked, "n_features_bit_identical_to_cache": n_identical, "max_abs_feature_diff_float32": max_feat_diff,
           "max_abs_score_diff_stdlib_vs_numpy": max_score_diff, "spec": str(spec_path.name)}
    (R3 / "text_prior_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"check: {res}")


# ============================================================================= testbeds
def cmd_testbeds(args) -> None:
    import run_experiments as R
    import real_testbed
    import simulate
    spec_path = str(Path(args.spec).resolve()) if args.spec else str((R3 / "text_prior_candidate.json").resolve())
    out = {"spec": Path(spec_path).name, "prereg_sha256": prereg_sha(), "testbeds": {}}
    datasets = {"real": real_testbed.load_pairs()}
    if not args.skip_synthetic:
        datasets["synthetic"] = simulate.generate_pairs(0, R.N_SUBJECTS, R.N_BENCHMARKS, R.MAIN_SIGMA, R.MAIN_GAMMA)
    for name, raw in datasets.items():
        s_hand = R.summarize(R.run_dataset(raw, seed=0, params={"TEXT_PRIOR_FILE": False}, workers=args.workers))
        s_text = R.summarize(R.run_dataset(raw, seed=0, params={"TEXT_PRIOR_FILE": spec_path}, workers=args.workers))
        rec = {"n_pairs": s_hand["n_pairs"]}
        for m in ("prior", "shrink", "full", "empirical_mean", "const"):
            rec[m] = {"hand": s_hand["methods"][m]["alc"], "text": s_text["methods"][m]["alc"],
                      "hand_brier_by_budget": s_hand["methods"][m]["brier_by_budget"],
                      "text_brier_by_budget": s_text["methods"][m]["brier_by_budget"],
                      "paired_text_minus_hand": harness.paired_bootstrap(s_text["methods"][m]["per_pair_alc"],
                                                                         s_hand["methods"][m]["per_pair_alc"], N_BOOT, seed=3)}
        rec["paired_text_full_vs_empirical_mean"] = harness.paired_bootstrap(
            s_text["methods"]["full"]["per_pair_alc"], s_text["methods"]["empirical_mean"]["per_pair_alc"], N_BOOT, seed=2)
        if name == "real":
            meta = [p[4] for p in raw if len(p[3]) >= harness.MIN_ITEMS]
            keep = [i for i, mt in enumerate(meta) if mt["bench"] != "bbh_mix"]
            rec["no_bbh"] = {"n_pairs": len(keep), "full": harness.paired_bootstrap(
                [s_text["methods"]["full"]["per_pair_alc"][i] for i in keep],
                [s_hand["methods"]["full"]["per_pair_alc"][i] for i in keep], N_BOOT, seed=3)}
            rec["per_pair"] = [{"bench": mt["bench"], "model": mt["model"], "harness": mt["harness"],
                                "hand_full": s_hand["methods"]["full"]["per_pair_alc"][i],
                                "text_full": s_text["methods"]["full"]["per_pair_alc"][i]} for i, mt in enumerate(meta)]
        out["testbeds"][name] = rec
        R.headline(s_hand, f"{name} hand-set"), R.headline(s_text, f"{name} text prior")
    d = out["testbeds"]["real"]["full"]["paired_text_minus_hand"]
    out["rule_b_real_not_significantly_worse"] = bool(not (d["ci_lo"] > 0))
    path = R3 / "text_prior_testbeds.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {path}; rule (b): {out['rule_b_real_not_significantly_worse']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build", "cv", "final", "check", "testbeds", "explore", "equivalence", "runtime"])
    ap.add_argument("--payload", default=str(DEFAULT_PAYLOAD))
    ap.add_argument("--spec", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--no-interaction", action="store_true")
    ap.add_argument("--variant", default=None, choices=sorted(EXPLORE))
    ap.add_argument("--resume", action="store_true", help="cv: reuse the folds cached in round3_cache/cv_folds.pkl")
    args = ap.parse_args()
    cmds = {"build": cmd_build, "cv": cmd_cv, "final": cmd_final, "check": cmd_check, "testbeds": cmd_testbeds,
            "explore": cmd_explore, "equivalence": cmd_equivalence, "runtime": cmd_runtime}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
