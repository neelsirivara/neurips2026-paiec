"""W2 item layer in the round-4 multi-pair simulator (numpy; same equations as the deployed stdlib code in w2_layer.py).

Base = xp_level_m050 (subject terms off, intercept -0.5, tau_mu 2; the other XP constants as shipped).  On top of it:

(a) category effects ("feat"): for every benchmark of the current list and every item_features key that is ACTIVE
    for it, a random effect f_(bench, key, value) ~ N(0, tau_f^2) is added to the stage-1 random-effects fit (joint
    MAP by backfitting, one effect type per key rank).  A key is active for a benchmark if, among the distinct
    labelled items of that benchmark that carry the key, it takes >= 2 values and at most share * (number of those
    items) values (mostly-singleton keys behave like item identities, which stage 2 already models).  Targets add
    the posterior mean of their own (bench, key, value) group and its posterior variance (probit); an unseen group
    adds nothing.
(b) text kernel ("kern"): after stage 2, for a target item t of benchmark b, the stage-1 residuals of the labelled
    entries of b are summed per distinct item j != t (G_j = sum(y - p), H_j = sum p(1-p)), and
    delta_t = sum_j K_tj G_j / (sum_j K_tj H_j + 1 / tau_k^2), K_tj = cos(t, j)^q over hashed character 4-gram
    TF-IDF (idf from the benchmark's distinct labelled items); the prediction is sigmoid(logit(p) + delta_t).

With tau_f = 0 and tau_k = 0 the prediction is exactly xp_level_m050.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_common import *  # noqa: F401,F403
from w2_common import X, model, np, json, time, log, CACHE, WF  # noqa: E402

BASE_ZS = -0.5
BASE_TAUS = None  # set in load_world
MAXK = WF.FEAT_MAX_KEYS


class World:
    """Frame + per-item features and hashed-gram rows for the items in w2_items_all.jsonl."""

    def __init__(self, with_grams=True):
        import scipy.sparse as sp
        self.F, self.folds = load()
        F = self.F
        NI = len(F.item_dir)
        recs = [json.loads(l) for l in open(CACHE / "w2_items_all.jsonl", encoding="utf-8")]
        self.has = np.zeros(NI, bool)
        keys_all = set()
        parsed = {}
        for r in recs:
            p = WF.parse_features(r["features"])
            parsed[r["idx"]] = p
            keys_all.update(k for k, _ in p)
            self.has[r["idx"]] = True
        self.key_vocab = sorted(keys_all)
        kid = {k: i for i, k in enumerate(self.key_vocab)}
        vals = {}
        self.fk = np.full((NI, MAXK), -1, dtype=np.int32)
        self.fv = np.full((NI, MAXK), -1, dtype=np.int64)
        for idx, p in parsed.items():
            for s, (k, v) in enumerate(p[:MAXK]):
                self.fk[idx, s] = kid[k]
                self.fv[idx, s] = vals.setdefault((k, v), len(vals))
        self.n_vals = len(vals)
        log(f"world: {len(recs)} items with text/features, {len(self.key_vocab)} keys, {len(vals)} (key, value) pairs")
        self.row_of = np.full(NI, -1, dtype=np.int64)
        if with_grams:
            rows, cols, data = [], [], []
            for j, r in enumerate(recs):
                self.row_of[r["idx"]] = j
                g = WF.text_grams(r["text"])
                rows.append(np.full(len(g), j, dtype=np.int32))
                cols.append(np.fromiter(g.keys(), dtype=np.int32, count=len(g)))
                data.append(np.fromiter(g.values(), dtype=np.float32, count=len(g)))
            self.G = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(len(recs), 1 << WF.GRAM_BITS), dtype=np.float32)
            log(f"grams: nnz {self.G.nnz}")
        del recs, parsed


# --------------------------------------------------------------------------- feature groups
def _active_slots(W, E, share):
    """For each labelled-list block (ts, bench): the active key ids sorted (= key rank -> key id).

    Returns dict {(ts, bench): [kid, ...]} computed from the distinct labelled items of E.
    """
    trip = np.unique(np.stack([E["ts"], E["bench"], E["item"]], 1), axis=0) if len(E["ts"]) else np.zeros((0, 3), np.int64)
    if not len(trip):
        return {}
    ts, b, it = trip[:, 0], trip[:, 1], trip[:, 2]
    k = W.fk[it]          # (n, MAXK)
    v = W.fv[it]
    rr = np.repeat(np.arange(len(trip)), MAXK)
    kk = k.ravel()
    vv = v.ravel()
    ok = kk >= 0
    rr, kk, vv = rr[ok], kk[ok], vv[ok]
    blk_ts, blk_b = ts[rr], b[rr]
    # per (ts, bench, key): number of items and of distinct values
    NK = len(W.key_vocab) + 1
    key_id = (blk_ts * 200 + blk_b) * NK + kk
    uk, inv_k, n_items = np.unique(key_id, return_inverse=True, return_counts=True)
    kv = np.unique(np.stack([inv_k, vv], 1), axis=0)
    n_vals = np.bincount(kv[:, 0], minlength=len(uk))
    act = (n_vals >= 2) & (n_vals <= share * n_items)
    out = {}
    for u in uk[act]:
        blk, kid = divmod(int(u), NK)
        t_, b_ = divmod(blk, 200)
        out.setdefault((t_, b_), []).append(kid)
    for key in out:
        out[key].sort()
    return out


def _feat_groups(W, T, slots, rank):
    """Group code per row of table T for key rank ``rank`` (-1 if the row's block has no such active key or the
    row's item lacks that key).  Code = (ts, bench, key, value) packed in one int64 (unique per group)."""
    n = len(T["ts"])
    code = np.full(n, -1, dtype=np.int64)
    if not slots:
        return code
    blk = T["ts"] * 200 + T["bench"]
    # key id assigned to this rank for each row's block
    want = np.full(n, -1, dtype=np.int64)
    ub, inv = np.unique(blk, return_inverse=True)
    kb = np.full(len(ub), -1, dtype=np.int64)
    for j, u in enumerate(ub):
        lst = slots.get(divmod(int(u), 200))
        if lst is not None and len(lst) > rank:
            kb[j] = lst[rank]
    want = kb[inv]
    has = want >= 0
    if not np.any(has):
        return code
    k = W.fk[T["item"]]
    v = W.fv[T["item"]]
    match = (k == want[:, None]) & has[:, None]
    anym = match.any(1)
    col = np.argmax(match, 1)
    val = v[np.arange(n), col]
    sel = anym & (val >= 0)
    code[sel] = (blk[sel] * (W.n_vals + 1) + val[sel])
    return code


def fit_w2(W, E, V, taus, tau_f=0.0, share=0.5, sweeps=None, tol=None):
    """Stage 1 of m050 plus the category effects; returns {"eta", "m", "var"} like X.xp_fit_np."""
    sweeps = model.XP_SWEEPS if sweeps is None else sweeps
    tol = model.XP_TOL if tol is None else tol
    types = []
    for name, tau in taus.items():
        if tau <= 0:
            continue
        ge, gv, k = X._group(E["keys"][name], V["keys"][name])
        types.append({"tau": tau, "prec": 1.0 / tau ** 2, "ge": ge, "gv": gv, "mem": None, "theta": np.zeros(k), "H": np.zeros(k),
                      "feat": False})
    if tau_f > 0 and len(E["ts"]):
        slots = _active_slots(W, E, share)
        nr = max((len(v) for v in slots.values()), default=0)
        for r in range(nr):
            ce = _feat_groups(W, E, slots, r)
            cv = _feat_groups(W, V, slots, r)
            mem = np.where(ce >= 0)[0]
            if not len(mem):
                continue
            uniq, inv = np.unique(ce[mem], return_inverse=True)
            pos = np.clip(np.searchsorted(uniq, cv), 0, len(uniq) - 1)
            gv = np.where((cv >= 0) & (uniq[pos] == cv), pos, -1)
            types.append({"tau": tau_f, "prec": 1.0 / tau_f ** 2, "ge": inv, "gv": gv, "mem": mem,
                          "theta": np.zeros(len(uniq)), "H": np.zeros(len(uniq)), "feat": True})
    eta = E["o"].astype(float).copy()
    y = E["y"]
    for _ in range(max(int(sweeps), 1)):
        mx = 0.0
        for t in types:
            k = len(t["theta"])
            if t["mem"] is None:
                p = X._sig(eta)
                G = np.bincount(t["ge"], weights=y - p, minlength=k)
                H = np.bincount(t["ge"], weights=p * (1 - p), minlength=k)
            else:
                mem = t["mem"]
                p = X._sig(eta[mem])
                G = np.bincount(t["ge"], weights=y[mem] - p, minlength=k)
                H = np.bincount(t["ge"], weights=p * (1 - p), minlength=k)
            step = (G - t["theta"] * t["prec"]) / (H + t["prec"])
            t["theta"] += step
            t["H"] = H
            if t["mem"] is None:
                eta += step[t["ge"]]
            else:
                eta[t["mem"]] += step[t["ge"]]
            if k:
                mx = max(mx, float(np.max(np.abs(step))))
        if mx < tol:
            break
    m = V["o"].astype(float).copy()
    var = np.zeros(len(m))
    for t in types:
        gv = t["gv"]
        seen = gv >= 0
        m[seen] += t["theta"][gv[seen]]
        if t["feat"]:
            var[seen] += 1.0 / (t["H"][gv[seen]] + t["prec"])
        else:
            var += np.where(seen, 1.0 / (t["H"][np.maximum(gv, 0)] + t["prec"]), t["tau"] ** 2)
    return {"eta": eta, "m": m, "var": var}


# --------------------------------------------------------------------------- text kernel
def kernel_delta(W, E, V, eta, tau_k, q, max_items=None):
    """delta per row of V (0 where no evidence)."""
    import scipy.sparse as sp
    out = np.zeros(len(V["ts"]))
    if tau_k <= 0 or not len(E["ts"]):
        return out
    lam = 1.0 / tau_k ** 2
    p = X._sig(eta)
    blkE = E["ts"] * 200 + E["bench"]
    blkV = V["ts"] * 200 + V["bench"]
    # aggregate per (block, item)
    keyE = blkE * (len(W.F.item_dir) + 1) + E["item"]
    uk, invE = np.unique(keyE, return_inverse=True)
    Gs = np.bincount(invE, weights=E["y"] - p)
    Hs = np.bincount(invE, weights=p * (1 - p))
    ublk = uk // (len(W.F.item_dir) + 1)
    uit = uk % (len(W.F.item_dir) + 1)
    # first-occurrence order of items within a block (deployment caps at max_items in list order)
    first = np.full(len(uk), np.iinfo(np.int64).max)
    np.minimum.at(first, invE, np.arange(len(keyE)))
    orderV = np.argsort(blkV, kind="stable")
    sV = blkV[orderV]
    bounds_V = np.searchsorted(sV, np.unique(blkV))
    ub_V = np.unique(blkV)
    bounds_V = list(bounds_V) + [len(sV)]
    lab_start = np.searchsorted(ublk, ub_V, side="left")
    lab_end = np.searchsorted(ublk, ub_V, side="right")
    for j, b in enumerate(ub_V):
        a0, a1 = lab_start[j], lab_end[j]
        if a1 <= a0:
            continue
        rowsV = orderV[bounds_V[j]:bounds_V[j + 1]]
        Lsel = np.arange(a0, a1)
        if max_items is not None and len(Lsel) > max_items:
            Lsel = Lsel[np.argsort(first[Lsel], kind="stable")[:max_items]]
        Litems = uit[Lsel]
        Lrows = W.row_of[Litems]
        okL = Lrows >= 0
        if not np.any(okL):
            continue
        Lsel, Litems, Lrows = Lsel[okL], Litems[okL], Lrows[okL]
        Tit, Tinv = np.unique(V["item"][rowsV], return_inverse=True)
        Trows = W.row_of[Tit]
        okT = Trows >= 0
        RL = W.G[Lrows]
        n = RL.shape[0]
        cols_u, cinv = np.unique(RL.indices, return_inverse=True)   # local column space of the block
        df_u = np.bincount(cinv, minlength=len(cols_u)).astype(np.float64)   # rows have distinct columns
        XL = sp.csr_matrix((RL.data * (np.log((n + 1.0) / (df_u[cinv] + 1.0)) + 1.0), cinv, RL.indptr),
                           shape=(n, len(cols_u)))
        nl = np.sqrt(np.asarray(XL.multiply(XL).sum(1)).ravel())
        nl[nl == 0] = 1.0
        XL = sp.diags(1.0 / nl) @ XL
        RT = W.G[np.where(okT, Trows, 0)]
        # target norm uses all its grams (unseen grams: df 0); the dot product only the shared ones
        pos = np.searchsorted(cols_u, RT.indices)
        pos_c = np.clip(pos, 0, max(len(cols_u) - 1, 0))
        shared = (len(cols_u) > 0) & (cols_u[pos_c] == RT.indices) if len(cols_u) else np.zeros(len(RT.indices), bool)
        dfT = np.where(shared, df_u[pos_c], 0.0)
        wT = RT.data * (np.log((n + 1.0) / (dfT + 1.0)) + 1.0)
        nt = np.sqrt(np.bincount(np.repeat(np.arange(RT.shape[0]), np.diff(RT.indptr)), weights=wT * wT,
                                 minlength=RT.shape[0]))
        nt[nt == 0] = 1.0
        rowT = np.repeat(np.arange(RT.shape[0]), np.diff(RT.indptr))
        XT = sp.csr_matrix(((wT / nt[rowT])[shared], (rowT[shared], pos_c[shared])), shape=(RT.shape[0], len(cols_u)))
        Sm = np.asarray((XT @ XL.T).todense())
        Sm = np.clip(Sm, 0.0, 1.0)
        Sm[~okT] = 0.0
        Sm[Tit[:, None] == Litems[None, :]] = 0.0
        K = Sm ** q
        d = (K @ Gs[Lsel]) / (K @ Hs[Lsel] + lam)
        out[rowsV] = d[Tinv]
    return out


# --------------------------------------------------------------------------- evaluation
def base_taus():
    t = dict(SHIPPED_TAUS)
    t["mu"] = 2.0
    return t


def eval_w2(W, S, ts, tau_f=0.0, share=0.5, tau_k=0.0, q=8, max_items=None, return_preds=False):
    """Per-unit Brier (n_units x 6) of m050 + W2 layer on the test sets of S."""
    taus = base_taus()

    def off(T):
        return np.clip(BASE_ZS + T["zi"], -X.CAP, X.CAP)

    V = dict(S.ev)
    V["o"] = off(S.ev)
    br = np.zeros((S.n_units, 6))
    p0 = np.clip(X._sig(V["o"]), model.CLIP_LO, model.CLIP_HI)
    br[:, 0] = X.unit_briers(S, p0)
    preds = {0: p0}
    for bi, b in enumerate(X.BUDGETS[1:], start=1):
        E = dict(S.acq[b])
        E["o"] = off(S.acq[b])
        fit = fit_w2(W, E, V, taus, tau_f=tau_f, share=share)
        p = X.xp_predict_np(E, V, fit, SHIPPED_TI, True)
        if tau_k > 0:
            d = kernel_delta(W, E, V, fit["eta"], tau_k, q, max_items)
            p = np.clip(X._sig(X._logit(p) + d), model.CLIP_LO, model.CLIP_HI)
        preds[b] = p
        br[:, bi] = X.unit_briers(S, p)
    return (br, preds) if return_preds else br


# --------------------------------------------------------------------------- scenarios
def scenario_dirs(W):
    F = W.F
    rate, pyear = pair_stats(F)
    elig = F.pair_n >= 80
    NB = int(F.pair_dir.max()) + 1
    cnt = np.bincount(F.pair_dir[elig], minlength=NB)
    bm = np.bincount(F.pair_dir[elig], weights=rate[elig], minlength=NB) / np.maximum(cnt, 1)
    inv = inventory_dirs(F)
    hard = {F.dirs[b] for b in range(len(F.dirs)) if cnt[b] > 0 and bm[b] <= 0.40}
    mid = {F.dirs[b] for b in range(len(F.dirs)) if cnt[b] > 0 and bm[b] <= 0.50}
    return {"A": set(F.dirs), "INV": set(inv), "MID": mid, "HARD": hard}


def make_sparse(F, folds_sub, R=300, seed=0, n_pairs=9, n_bench=7):
    """Formative-like test sets: up to 7 benchmarks of the fold, 9 pairs (1-2 per benchmark), cap 100."""
    out = []
    for fi, f in enumerate(folds_sub):
        if not f:
            continue
        bidx = [F.dirs.index(d) for d in f]
        elig = {b: X._eligible(F, b) for b in bidx}
        benches = [b for b in bidx if len(elig[b])]
        if not benches:
            continue
        rng = np.random.RandomState(seed * 100003 + fi * 7 + 21)
        for r in range(R):
            K = min(n_bench, len(benches))
            bs = sorted(int(b) for b in rng.choice(benches, K, replace=False))
            per = [1] * K
            for j in rng.choice(K, max(n_pairs - K, 0), replace=True):
                per[int(j)] += 1
            pairs = []
            for b, m in zip(bs, per):
                cand = elig[b]
                take = cand if len(cand) <= m else rng.choice(cand, m, replace=False)
                pairs += sorted(int(x) for x in take)
            out.append({"fold": fi, "pairs": pairs, "cap": 100, "evalonly": [False] * len(pairs), "benches": bs})
    return out


def build_scenario(W, name, fold_ids, R=300):
    """Test sets of scenario ``name`` restricted to the benchmark folds ``fold_ids`` (the fold index is kept)."""
    F = W.F
    sd = scenario_dirs(W)
    keep = sd["INV"] if name == "SPARSE" else sd[name]
    fl = [[d for d in f if d in keep] if i in fold_ids else [] for i, f in enumerate(W.folds)]
    if name == "SPARSE":
        ts = make_sparse(F, fl, R=R)
    else:
        ts = [t for t in X.make_testsets(F, fl, "A", R=R) if t["pairs"]]
    S = X.Streams(F, ts)
    return S, ts
