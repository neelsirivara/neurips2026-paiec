"""Round 4: cross-pair evidence in the official PAIEC evaluator -- frame builder and multi-pair simulator.

The official streaming client (paiec_baseline/tools/streaming_ingestion.py, run_streaming) passes ONE global
``labeled`` list (all labels acquired so far for ALL sampled subject-benchmark pairs) to every predict() call
of a budget checkpoint.  This script rebuilds that protocol on the official training release
(measurement-db) with benchmark-held-out folds.  Pre-registered in experiments/results/round4/PREREG.md.

Run with ``python -I`` (reads downloaded data):

    python -I xpair_sim.py build   [--payload DIR]     # frame -> DIR/round4_cache/frame.npz + aux.pkl
    python -I xpair_sim.py explore                     # structure only (no outcomes): overlap statistics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("PAIEC_FIT_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import model  # noqa: E402

RESULTS = HERE.parent / "results"
R4 = RESULTS / "round4"
PREREG = R4 / "PREREG.md"
DEFAULT_PAYLOAD = HERE.parents[2] / "_downloads" / "measurement_db_payload"
SUBJECT_COLS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
                "reasoning_effort", "subject_features_extra")
CONTENT_CANDIDATES = ("content", "item_content", "prompt", "question", "input", "text")
FEATURE_CANDIDATES = ("item_features", "features", "metadata")
RESPONSE_CANDIDATES = ("response", "score", "correct", "is_correct", "label")
MIN_PAIR_ITEMS = 20      # pairs entering the frame (as in iterations 2-3)
MAX_ITEMS_PER_PAIR = 200  # per pair: the 200 answered items with the lowest benchmark-level hash rank


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else "missing"


def _first_col(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def _rank(dir_name: str, key: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"r4|{dir_name}|{key}".encode("utf-8"), digest_size=8).digest(), "big")


def subject_key(subject: dict) -> str:
    return json.dumps({c: subject.get(c, "") for c in SUBJECT_COLS}, sort_keys=True, separators=(",", ":"))


def _hand_item_logit(features: str, length: int) -> float:
    return float(sum(getattr(model, n) * v for n, v in model.item_terms({"item_features": features}, content_length=length)))


# ============================================================================= build
def cmd_build(args) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    try:
        pa.set_cpu_count(int(_THREADS))
    except Exception:
        pass
    payload = Path(args.payload).resolve()
    data_dir = payload / "data"
    cache = payload / "round4_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    dirs_all = sorted(manifest["eligible_dirs"])
    t0 = time.time()
    subj_keys: dict = {}
    name_keys: dict = {}
    sid_keys: dict = {}
    item_keys: dict = {}
    item_dir, item_zi = [], []
    pair_dir, pair_subj, pair_name, pair_sid, pair_zs, pair_ntot = [], [], [], [], [], []
    row_pair, row_item, row_y, row_rank = [], [], [], []
    dirs_used = []
    for k, d in enumerate(dirs_all, 1):
        base = data_dir / d
        resp_path = base / "response.parquet"
        if not resp_path.exists():
            continue
        subj_tbl = pq.read_table(base / "subjects.parquet").to_pandas()
        subj_tbl["subject_id"] = subj_tbl["subject_id"].astype(str)
        subjects = {}
        for row in subj_tbl.to_dict("records"):
            subjects[row["subject_id"]] = {c: ("" if row.get(c) is None or (isinstance(row.get(c), float) and math.isnan(row.get(c)))
                                               else str(row.get(c))) for c in SUBJECT_COLS}
        names = pq.ParquetFile(resp_path).schema_arrow.names
        rcol = _first_col(names, RESPONSE_CANDIDATES)
        if rcol is None or "subject_id" not in names or "item_id" not in names:
            continue
        cols = ["subject_id", "item_id", rcol] + (["test_condition"] if "test_condition" in names else [])
        resp = pq.read_table(resp_path, columns=cols)
        col = resp.column(rcol)
        if pa.types.is_boolean(col.type) or pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
            y = np.clip(pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False), 0.0, 1.0)
        else:
            vals = col.to_pylist()
            y = np.full(len(vals), np.nan)
            for i, v in enumerate(vals):
                lab = model._label_value(v)
                if lab is not None:
                    y[i] = float(lab)
        sid = pc.cast(resp.column("subject_id"), pa.string()).to_numpy(zero_copy_only=False)
        iid = pc.cast(resp.column("item_id"), pa.string()).to_numpy(zero_copy_only=False)
        cond = (pc.fill_null(pc.cast(resp.column("test_condition"), pa.string()), "").to_numpy(zero_copy_only=False)
                if "test_condition" in names else np.array([""] * len(y), dtype=object))
        ok = ~np.isnan(y)
        df = pd.DataFrame({"sid": sid[ok], "iid": iid[ok], "cond": cond[ok], "y": y[ok]})
        agg = df.groupby(["sid", "cond", "iid"], sort=False)["y"].mean().reset_index()
        # item table: content hash (exact-item identity) and the hand-set item logit
        info = {}
        items_path = base / "items.parquet"
        if items_path.exists():
            inames = pq.ParquetFile(items_path).schema_arrow.names
            ccol, fcol = _first_col(inames, CONTENT_CANDIDATES), _first_col(inames, FEATURE_CANDIDATES)
            hcol = "content_hash" if "content_hash" in inames else None
            icols = ["item_id"] + [c for c in (ccol, fcol, hcol) if c]
            it = pq.read_table(items_path, columns=icols)
            ids = pc.cast(it.column("item_id"), pa.string()).to_pylist()
            if ccol:
                cont = pc.cast(it.column(ccol), pa.string())
                lengths = pc.fill_null(pc.utf8_length(cont), 0).to_pylist()
            else:
                cont, lengths = None, [0] * len(ids)
            feats = pc.fill_null(pc.cast(it.column(fcol), pa.string()), "").to_pylist() if fcol else [""] * len(ids)
            hashes = pc.fill_null(pc.cast(it.column(hcol), pa.string()), "").to_pylist() if hcol else [""] * len(ids)
            if cont is not None and any(h == "" for h in hashes):
                cl = cont.to_pylist()
                hashes = [h or hashlib.blake2b((c or "").encode("utf-8"), digest_size=8).hexdigest() for h, c in zip(hashes, cl)]
            for i_, L, f_, h_ in zip(ids, lengths, feats, hashes):
                info[i_] = (int(L or 0), f_ or "", h_ or ("id:" + i_))
        dir_idx = len(dirs_used)
        n_pairs_before = len(pair_dir)
        for (s, c), grp in agg.groupby(["sid", "cond"], sort=False):
            if len(grp) < MIN_PAIR_ITEMS:
                continue
            subject = dict(subjects.get(s, {col: "" for col in SUBJECT_COLS}))
            if c:
                subject["subject_features_extra"] = ((subject["subject_features_extra"] + ";") if subject["subject_features_extra"] else "") + "test_condition=" + c
            iids = grp["iid"].tolist()
            ys = grp["y"].tolist()
            ranks = [_rank(d, i_) for i_ in iids]
            order = sorted(range(len(iids)), key=lambda j: ranks[j])[:MAX_ITEMS_PER_PAIR]
            p = len(pair_dir)
            pair_dir.append(dir_idx)
            pair_subj.append(subj_keys.setdefault(subject_key(subject), len(subj_keys)))
            pair_name.append(name_keys.setdefault(subject["normalized_name"].strip().lower(), len(name_keys)))
            pair_sid.append(sid_keys.setdefault(s, len(sid_keys)))
            pair_zs.append(float(model._subject_logit(subject)))
            pair_ntot.append(len(iids))
            for r, j in enumerate(order):
                L, f_, h_ = info.get(iids[j], (0, "", "id:" + iids[j]))
                key = (dir_idx, h_)
                ix = item_keys.get(key)
                if ix is None:
                    ix = len(item_dir)
                    item_keys[key] = ix
                    item_dir.append(dir_idx)
                    item_zi.append(_hand_item_logit(f_, L))
                row_pair.append(p)
                row_item.append(ix)
                row_y.append(float(ys[j]))
                row_rank.append(r)
        if len(pair_dir) > n_pairs_before:
            dirs_used.append(d)
        if k % 10 == 0 or k == len(dirs_all):
            log(f"  {k}/{len(dirs_all)} dirs, {len(pair_dir)} pairs, {len(row_y)} rows, {len(item_dir)} items ({time.time() - t0:.0f}s)")
    pair_dir = np.array(pair_dir, dtype=np.int32)
    row_pair = np.array(row_pair, dtype=np.int32)
    pair_start = np.searchsorted(row_pair, np.arange(len(pair_dir) + 1)).astype(np.int64)
    np.savez(cache / "frame.npz", pair_dir=pair_dir, pair_subj=np.array(pair_subj, dtype=np.int32),
             pair_name=np.array(pair_name, dtype=np.int32), pair_sid=np.array(pair_sid, dtype=np.int32),
             pair_zs=np.array(pair_zs), pair_ntot=np.array(pair_ntot, dtype=np.int32), pair_start=pair_start,
             row_item=np.array(row_item, dtype=np.int32), row_y=np.array(row_y), row_rank=np.array(row_rank, dtype=np.int32),
             item_dir=np.array(item_dir, dtype=np.int32), item_zi=np.array(item_zi))
    with open(cache / "aux.pkl", "wb") as fh:
        pickle.dump({"dirs": dirs_used, "subj_keys": list(subj_keys), "name_keys": list(name_keys), "sid_keys": list(sid_keys),
                     "revision": manifest.get("revision")}, fh)
    log(f"wrote {cache}: {len(pair_dir)} pairs, {len(row_y)} rows, {len(item_dir)} items, {len(dirs_used)} benchmarks, "
        f"{len(subj_keys)} subject keys, {len(name_keys)} names ({time.time() - t0:.0f}s)")


class Frame:
    def __init__(self, payload: Path):
        cache = Path(payload).resolve() / "round4_cache"
        m = np.load(cache / "frame.npz")
        for k in m.files:
            setattr(self, k, m[k])
        with open(cache / "aux.pkl", "rb") as fh:
            aux = pickle.load(fh)
        self.dirs, self.subj_keys, self.name_keys = aux["dirs"], aux["subj_keys"], aux["name_keys"]
        self.revision = aux["revision"]
        self.n_pairs = len(self.pair_dir)
        self.pair_n = np.diff(self.pair_start)


# ============================================================================= simulator
BUDGETS = list(harness.BUDGETS)
ALC_W = np.array([harness.ALC_WEIGHTS[b] for b in BUDGETS])
MAX_LABELS = harness.MAX_LABELS
CAP = model.PRIOR_LOGIT_CAP
ALPHA0 = model.ALPHA0
N_CLUSTER_BOOT = 2000
GRID = {"bench": (0.5, 1.0, 1.5), "name": (0.0, 0.5), "subj": (0.0, 0.5), "namebench": (0.0, 0.5, 1.0),
        "pair": (0.3, 0.6, 0.9)}
TAU_MU = 0.5
TAU_ITEMS = (0.0, 0.75, 1.5, 2.25)
# PREREG_ADDENDUM_1.md: extended grid
GRID_EXT = dict(GRID, bench=(0.5, 1.0, 1.5, 2.0, 3.0))
TAU_ITEMS_EXT = (0.0, 0.75, 1.5, 2.25, 3.0, 4.0)
PROBITS = (True, False)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_folds() -> list:
    fit = json.loads((RESULTS / "prior_fit.json").read_text(encoding="utf-8"))
    return [sorted(f) for f in fit["cv"]["folds"]]


def _eligible(F: Frame, b: int, lo: int = 80, hi: int | None = None) -> np.ndarray:
    m = (F.pair_dir == b) & (F.pair_n >= lo)
    if hi is not None:
        m &= F.pair_n <= hi
    return np.where(m)[0]


def make_testsets(F: Frame, folds: list, config: str, R: int = 300, seed: int = 0) -> list:
    """List of test sets {fold, pairs, cap, evalonly}; see PREREG.md (configs A-D)."""
    out = []
    for fi, f in enumerate(folds):
        bidx = [F.dirs.index(d) for d in f]
        elig = {b: _eligible(F, b) for b in bidx}
        small = {b: _eligible(F, b, 20, 79) for b in bidx}
        benches = [b for b in bidx if len(elig[b])]
        if config == "E":
            rng = np.random.RandomState(seed * 100003 + fi * 7 + 15)
            for r in range(R):
                K = min(int(rng.randint(1, 4)), len(benches))
                bs = sorted(int(b) for b in rng.choice(benches, K, replace=False))
                per = [10 // K + (1 if i < 10 % K else 0) for i in range(K)]
                pairs, used = [], set()
                for b, m in zip(bs, per):
                    cand = elig[b]
                    names = sorted(set(int(F.pair_name[x]) for x in cand) - used)
                    take = names if len(names) <= m else [int(x) for x in rng.choice(names, m, replace=False)]
                    for nm in sorted(take):
                        opts = cand[F.pair_name[cand] == nm]
                        pairs.append(int(opts[rng.randint(len(opts))]))
                        used.add(nm)
                out.append({"fold": fi, "pairs": pairs, "cap": 100, "evalonly": [False] * len(pairs), "benches": bs})
            continue
        if config in ("A", "D"):
            rng = np.random.RandomState(seed * 100003 + fi * 7 + 11)
            rng_small = np.random.RandomState(seed * 100003 + fi * 7 + 12)
            for r in range(R):
                K = min(int(rng.randint(1, 4)), len(benches))
                bs = sorted(int(b) for b in rng.choice(benches, K, replace=False))
                per = [10 // K + (1 if i < 10 % K else 0) for i in range(K)]
                pairs = []
                for b, m in zip(bs, per):
                    cand = elig[b]
                    take = cand if len(cand) <= m else rng.choice(cand, m, replace=False)
                    pairs += sorted(int(x) for x in take)
                ts = {"fold": fi, "pairs": pairs, "cap": 100, "evalonly": [False] * len(pairs), "benches": bs}
                if config == "D":
                    pool = np.concatenate([small[b] for b in bs]) if bs else np.array([], dtype=int)
                    if len(pool):
                        k = min(2, len(pool))
                        extra = sorted(int(x) for x in rng_small.choice(pool, k, replace=False))
                        ts["pairs"] = pairs + extra
                        ts["evalonly"] = [False] * len(pairs) + [True] * len(extra)
                out.append(ts)
        elif config == "B":
            rng = np.random.RandomState(seed * 100003 + fi * 7 + 13)
            for b in benches:
                cand = elig[b]
                if len(cand) > 300:
                    cand = np.sort(rng.choice(cand, 300, replace=False))
                out.append({"fold": fi, "pairs": [int(x) for x in cand], "cap": 200, "evalonly": [False] * len(cand),
                            "benches": [int(b)]})
        elif config == "C":
            rng = np.random.RandomState(seed * 100003 + fi * 7 + 14)
            cand = np.concatenate([elig[b] for b in benches])
            if len(cand) > 2000:
                cand = np.sort(rng.choice(cand, 2000, replace=False))
            out.append({"fold": fi, "pairs": [int(x) for x in cand], "cap": 100, "evalonly": [False] * len(cand),
                        "benches": sorted(int(b) for b in benches)})
        else:
            raise ValueError(config)
    return out


def pair_split(F: Frame, p: int, cap: int):
    """Fixed per (pair, cap): (acquisition order (frame rows), evaluation rows)."""
    a = int(F.pair_start[p])
    n_used = int(min(F.pair_n[p], cap))
    rows = np.arange(a, a + n_used)
    rng = np.random.RandomState((p * 7919 + n_used) % (2 ** 31 - 1))
    perm = rng.permutation(n_used)
    half = n_used // 2
    acq, ev = rows[perm[:half]], rows[perm[half:]]
    order = acq[rng.permutation(half)]
    return order, ev


class Streams:
    """Global labeled lists (per budget) and evaluation rows of a list of test sets, as flat arrays."""

    def __init__(self, F: Frame, testsets: list, policy: str = "random"):
        self.F = F
        name_eff = F.pair_name.astype(np.int64).copy()
        empty = [i for i, k in enumerate(F.name_keys) if k == ""]
        NN = len(F.name_keys)
        if empty:
            m = np.isin(F.pair_name, empty)
            name_eff[m] = NN + F.pair_subj[m]
        self.name_eff = name_eff
        self.name_raw_empty = np.isin(F.pair_name, empty) if empty else np.zeros(F.n_pairs, bool)
        ev_ts, ev_pair, ev_row, ev_unit = [], [], [], []
        acq = {b: ([], [], []) for b in BUDGETS[1:]}
        units = []  # (ts, pair, evalonly)
        n_fresh_skips = 0
        self.acquired = []
        for t, ts in enumerate(testsets):
            orders = []
            for p, eo in zip(ts["pairs"], ts["evalonly"]):
                if eo:
                    a = int(F.pair_start[p])
                    ev = np.arange(a, a + int(min(F.pair_n[p], ts["cap"])))
                    order = np.array([], dtype=np.int64)
                else:
                    order, ev = pair_split(F, p, ts["cap"])
                u = len(units)
                units.append((t, p, eo))
                ev_ts.append(np.full(len(ev), t))
                ev_pair.append(np.full(len(ev), p))
                ev_row.append(ev)
                ev_unit.append(np.full(len(ev), u))
                orders.append(order)
            if policy == "random":
                acquired = [list(o[:MAX_LABELS]) for o in orders]
            elif policy == "fresh":
                acquired, skips = self._fresh(ts, orders)
                n_fresh_skips += skips
            else:
                raise ValueError(policy)
            self.acquired.append(acquired)
            for b in BUDGETS[1:]:
                for (p, got) in zip(ts["pairs"], acquired):
                    got = got[:b]
                    if got:
                        acq[b][0].append(np.full(len(got), t))
                        acq[b][1].append(np.full(len(got), p))
                        acq[b][2].append(np.asarray(got, dtype=np.int64))
        self.n_fresh_skips = n_fresh_skips
        self.units = np.array([(t, p) for t, p, _ in units], dtype=np.int64).reshape(-1, 2)
        self.unit_evalonly = np.array([eo for _, _, eo in units], dtype=bool)
        self.unit_bench = F.pair_dir[self.units[:, 1]]
        self.unit_fold = np.array([testsets[t]["fold"] for t, _ in self.units], dtype=np.int64)
        self.ev = self._table(np.concatenate(ev_ts), np.concatenate(ev_pair), np.concatenate(ev_row))
        self.ev["unit"] = np.concatenate(ev_unit)
        self.n_units = len(units)
        self.acq = {}
        for b in BUDGETS[1:]:
            if acq[b][0]:
                self.acq[b] = self._table(np.concatenate(acq[b][0]), np.concatenate(acq[b][1]), np.concatenate(acq[b][2]))
            else:
                z = np.array([], dtype=np.int64)
                self.acq[b] = self._table(z, z, z)
        cnt = np.bincount(self.ev["unit"], minlength=self.n_units).astype(float)
        self.unit_count = cnt
        self.unit_rate = np.bincount(self.ev["unit"], weights=self.ev["y"], minlength=self.n_units) / cnt

    def _fresh(self, ts: dict, orders: list):
        """Stage-wise round robin with the fresh-first rule (PREREG.md, acquisition, amended)."""
        F = self.F
        subj = [int(F.pair_subj[p]) for p in ts["pairs"]]
        labelled: dict = {}
        pos = [0] * len(orders)
        acquired = [[] for _ in orders]
        skips = 0
        for b in BUDGETS[1:]:
            for j, order in enumerate(orders):
                while len(acquired[j]) < b and pos[j] < len(order):
                    row = int(order[pos[j]])
                    items_remaining = len(order) - pos[j]
                    pos[j] += 1
                    labels_remaining = MAX_LABELS - len(acquired[j])
                    it = int(F.row_item[row])
                    others = labelled.get(it)
                    stale = others is not None and any(s_ != subj[j] for s_ in others)
                    if stale and labels_remaining < items_remaining:
                        skips += 1
                        continue
                    acquired[j].append(row)
                    labelled.setdefault(it, set()).add(subj[j])
        return acquired, skips

    def _table(self, ts, pair, row) -> dict:
        F = self.F
        ts = ts.astype(np.int64)
        pair = pair.astype(np.int64)
        row = row.astype(np.int64)
        item = F.row_item[row].astype(np.int64)
        y = F.row_y[row]
        zi = F.item_zi[item]
        zs = F.pair_zs[pair]
        o = np.clip(zs + zi, -CAP, CAP)
        bench = F.pair_dir[pair].astype(np.int64)
        name = self.name_eff[pair]
        subj = F.pair_subj[pair].astype(np.int64)
        NB = 200
        NS = len(F.subj_keys) + 1
        NN = len(F.name_keys) + NS
        NI = len(F.item_dir) + 1
        keys = {"mu": ts, "bench": ts * NB + bench, "name": ts * NN + name, "subj": ts * NS + subj,
                "namebench": (ts * NN + name) * NB + bench, "pair": (ts * NS + subj) * NB + bench}
        # the deployed model.pair_evidence: same benchmark_id and same normalized_name (no name filter if empty)
        ship_name = np.where(self.name_raw_empty[pair], -1, F.pair_name[pair].astype(np.int64))
        return {"ts": ts, "pair": pair, "row": row, "item": item, "y": y, "zi": zi, "zs": zs, "o": o, "bench": bench,
                "keys": keys, "ikey": ts * NI + item,
                "ship_key": (ts * NB + bench) * (len(F.name_keys) + 2) + (ship_name + 1),
                "ship_empty": self.name_raw_empty[pair], "ship_bench_key": ts * NB + bench,
                "pair_key": ts * (F.n_pairs + 1) + pair}


def _group(keys_e: np.ndarray, keys_v: np.ndarray):
    """Group index of entries; for evaluation rows, the index or -1 if unseen."""
    uniq, inv = np.unique(keys_e, return_inverse=True)
    if not len(uniq):
        return inv, np.full(len(keys_v), -1), 0
    pos = np.clip(np.searchsorted(uniq, keys_v), 0, len(uniq) - 1)
    seen = uniq[pos] == keys_v
    return inv, np.where(seen, pos, -1), len(uniq)


def xp_fit_np(E: dict, V: dict, taus: dict, sweeps: int = None, tol: float = None):
    """Vectorised stage 1 (same equations as model.xp_build; test sets are independent blocks)."""
    sweeps = model.XP_SWEEPS if sweeps is None else sweeps
    tol = model.XP_TOL if tol is None else tol
    types = []
    for name, tau in taus.items():
        if tau <= 0:
            continue
        ge, gv, k = _group(E["keys"][name], V["keys"][name])
        types.append({"name": name, "tau": tau, "prec": 1.0 / tau ** 2, "ge": ge, "gv": gv, "theta": np.zeros(k), "H": np.zeros(k)})
    eta = E["o"].astype(float).copy()
    y = E["y"]
    for _ in range(max(int(sweeps), 1)):
        mx = 0.0
        for t in types:
            p = _sig(eta)
            k = len(t["theta"])
            G = np.bincount(t["ge"], weights=y - p, minlength=k)
            H = np.bincount(t["ge"], weights=p * (1 - p), minlength=k)
            step = (G - t["theta"] * t["prec"]) / (H + t["prec"])
            t["theta"] += step
            t["H"] = H
            eta += step[t["ge"]]
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
        var += np.where(seen, 1.0 / (t["H"][np.maximum(gv, 0)] + t["prec"]), t["tau"] ** 2)
    return {"eta": eta, "m": m, "var": var}


def xp_predict_np(E: dict, V: dict, fit: dict, tau_item: float, probit: bool) -> np.ndarray:
    m = fit["m"] / np.sqrt(1.0 + (math.pi / 8.0) * fit["var"]) if probit else fit["m"].copy()
    p = _sig(m)
    if tau_item > 0 and len(E["y"]):
        uniq, inv = np.unique(E["ikey"], return_inverse=True)
        pos = np.clip(np.searchsorted(uniq, V["ikey"]), 0, len(uniq) - 1)
        hit = uniq[pos] == V["ikey"]
        if np.any(hit):
            c = math.sqrt(1.0 + (math.pi / 8.0) * tau_item ** 2)
            prec = 1.0 / tau_item ** 2
            k = len(uniq)
            ce = c * fit["eta"]
            d = np.zeros(k)
            h = np.full(k, prec)
            for _ in range(model.XP_ITEM_NEWTON):
                pe = _sig(ce + d[inv])
                g = np.bincount(inv, weights=E["y"] - pe, minlength=k) - d * prec
                h = np.bincount(inv, weights=pe * (1 - pe), minlength=k) + prec
                d = d + g / h
            j = pos[hit]
            p[hit] = _sig((c * m[hit] + d[j]) / np.sqrt(1.0 + (math.pi / 8.0) / h[j]))
    return np.clip(p, model.CLIP_LO, model.CLIP_HI)


def shipped_np(E: dict, V: dict, mode: str = "deployed") -> np.ndarray:
    """Closed-form shipped rule (prior + beta-binomial alpha0) with the deployed pair_evidence or per-pair evidence."""
    if not len(E["y"]):
        return np.clip(_sig(V["o"]), model.CLIP_LO, model.CLIP_HI)
    if mode == "deployed":
        ge, gv, k = _group(E["ship_key"], V["ship_key"])
        # entries with an empty name match every target of their benchmark, and targets with an empty name match
        # every entry of their benchmark (model.pair_evidence); both are rare and handled via the benchmark sums
        n = np.bincount(ge, minlength=k).astype(float)
        ksum = np.bincount(ge, weights=E["y"], minlength=k)
        zsum = np.bincount(ge, weights=E["zi"], minlength=k)
        nv = np.where(gv >= 0, n[np.maximum(gv, 0)], 0.0)
        kv_ = np.where(gv >= 0, ksum[np.maximum(gv, 0)], 0.0)
        zv = np.where(gv >= 0, zsum[np.maximum(gv, 0)], 0.0)
        gbe, gbv, kb = _group(E["ship_bench_key"], V["ship_bench_key"])
        e_empty = E["ship_empty"].astype(float)
        nb_empty = np.bincount(gbe, weights=e_empty, minlength=kb)
        kb_empty = np.bincount(gbe, weights=E["y"] * e_empty, minlength=kb)
        zb_empty = np.bincount(gbe, weights=E["zi"] * e_empty, minlength=kb)
        nb_all = np.bincount(gbe, minlength=kb).astype(float)
        kb_all = np.bincount(gbe, weights=E["y"], minlength=kb)
        zb_all = np.bincount(gbe, weights=E["zi"], minlength=kb)
        jb = np.maximum(gbv, 0)
        okb = gbv >= 0
        v_empty = V["ship_empty"]
        # non-empty target name: own-name entries plus empty-name entries of the benchmark
        add_n = np.where(okb & ~v_empty, nb_empty[jb], 0.0)
        nv = np.where(v_empty, np.where(okb, nb_all[jb], 0.0), nv + add_n)
        kv_ = np.where(v_empty, np.where(okb, kb_all[jb], 0.0), kv_ + np.where(okb & ~v_empty, kb_empty[jb], 0.0))
        zv = np.where(v_empty, np.where(okb, zb_all[jb], 0.0), zv + np.where(okb & ~v_empty, zb_empty[jb], 0.0))
        # no matching entry -> ALL entries of the (global) list
        gts, gvts, kts = _group(E["ts"], V["ts"])
        n_all = np.bincount(gts, minlength=kts).astype(float)
        k_all = np.bincount(gts, weights=E["y"], minlength=kts)
        z_all = np.bincount(gts, weights=E["zi"], minlength=kts)
        none = (nv == 0) & (gvts >= 0)
        jt = np.maximum(gvts, 0)
        nv = np.where(none, n_all[jt], nv)
        kv_ = np.where(none, k_all[jt], kv_)
        zv = np.where(none, z_all[jt], zv)
    else:
        ge, gv, k = _group(E["pair_key"], V["pair_key"])
        n = np.bincount(ge, minlength=k).astype(float)
        ksum = np.bincount(ge, weights=E["y"], minlength=k)
        zsum = np.bincount(ge, weights=E["zi"], minlength=k)
        nv = np.where(gv >= 0, n[np.maximum(gv, 0)], 0.0)
        kv_ = np.where(gv >= 0, ksum[np.maximum(gv, 0)], 0.0)
        zv = np.where(gv >= 0, zsum[np.maximum(gv, 0)], 0.0)
    has = nv > 0
    mean_z = np.where(has, zv / np.maximum(nv, 1), 0.0)
    prior_rate = _sig(np.clip(V["zs"] + mean_z, -CAP, CAP))
    rate = (ALPHA0 * prior_rate + kv_) / (ALPHA0 + np.maximum(nv, 0))
    p = np.where(has, _sig(_logit(rate) + V["zi"] - mean_z), _sig(V["o"]))
    return np.clip(p, model.CLIP_LO, model.CLIP_HI)


def unit_briers(S: Streams, preds: np.ndarray) -> np.ndarray:
    se = (preds - S.ev["y"]) ** 2
    return np.bincount(S.ev["unit"], weights=se, minlength=S.n_units) / S.unit_count


def cluster_boot(diff: np.ndarray, clusters: np.ndarray, seed: int = 0, n_boot: int = N_CLUSTER_BOOT) -> dict:
    uniq, inv = np.unique(clusters, return_inverse=True)
    S_ = np.bincount(inv, weights=diff)
    N = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uniq), size=(n_boot, len(uniq)))
    stats = np.sort(S_[idx].sum(axis=1) / N[idx].sum(axis=1))
    lo, hi = float(stats[int(0.025 * n_boot)]), float(stats[int(0.975 * n_boot) - 1])
    return {"mean": float(diff.mean()), "ci_lo": lo, "ci_hi": hi, "n_units": int(len(diff)), "n_clusters": int(len(uniq)),
            "n_boot": n_boot, "significant": bool(hi < 0 or lo > 0), "frac_negative": float(np.mean(diff < 0))}


def cluster_mean_ci(values: np.ndarray, clusters: np.ndarray, seed: int = 1) -> dict:
    r = cluster_boot(values, clusters, seed=seed)
    return {"mean": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n_units": r["n_units"], "n_clusters": r["n_clusters"]}


def taus_of(setting: dict) -> dict:
    return {"mu": TAU_MU, "bench": setting["bench"], "name": setting["name"], "subj": setting["subj"],
            "namebench": setting["namebench"], "pair": setting["pair"]}


def setting_label(setting: dict) -> str:
    return "xp|" + "|".join(f"{k}{setting[k]:g}" for k in GRID)


def grid_settings(ext: bool = False) -> list:
    import itertools
    g = GRID_EXT if ext else GRID
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[g[k] for k in keys])]


def eval_methods(S: Streams, stage1: list, baselines: bool = True) -> dict:
    """Per-unit Brier by budget (n_units x 6) for S0, S1, prior, oracle and XP variants.

    stage1: list of (label, taus dict, [(tau_item, probit), ...]).
    """
    out = {}
    p0 = np.clip(_sig(S.ev["o"]), model.CLIP_LO, model.CLIP_HI)
    b0 = unit_briers(S, p0)
    names = ["S0", "S1", "prior"] if baselines else []
    for k in names:
        out[k] = np.repeat(b0[:, None], 6, axis=1)
    if baselines:
        orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
        out["oracle_rate"] = np.repeat(orc[:, None], 6, axis=1)
    for label, taus, variants in stage1:
        for ti, pr in variants:
            out[f"{label}|item{ti:g}|probit{int(pr)}"] = np.repeat(b0[:, None], 6, axis=1)
    for bi, b in enumerate(BUDGETS[1:], start=1):
        E = S.acq[b]
        if baselines:
            out["S0"][:, bi] = unit_briers(S, shipped_np(E, S.ev, "deployed"))
            out["S1"][:, bi] = unit_briers(S, shipped_np(E, S.ev, "pair"))
        for label, taus, variants in stage1:
            fit = xp_fit_np(E, S.ev, taus)
            for ti, pr in variants:
                out[f"{label}|item{ti:g}|probit{int(pr)}"][:, bi] = unit_briers(S, xp_predict_np(E, S.ev, fit, ti, pr))
    return out


def alc_of(briers: np.ndarray) -> np.ndarray:
    return briers @ ALC_W


# ============================================================================= commands
def _cache_dir(args) -> Path:
    d = Path(args.payload).resolve() / "round4_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _grid_worker(job):
    payload, idx_list, R, config, ext = job
    F = Frame(payload)
    folds = load_folds()
    S = Streams(F, make_testsets(F, folds, config, R=R))
    settings = grid_settings(ext)
    variants = [(ti, pr) for ti in (TAU_ITEMS_EXT if ext else TAU_ITEMS) for pr in PROBITS]
    res = {}
    t0 = time.time()
    for i in idx_list:
        st = settings[i]
        out = eval_methods(S, [(setting_label(st), taus_of(st), variants)], baselines=False)
        for k, v in out.items():
            res[k] = alc_of(v).astype(np.float32)
        log(f"  setting {i + 1}/{len(settings)} {setting_label(st)} ({time.time() - t0:.0f}s)")
    return res


def cmd_grid(args) -> None:
    """Config-A grid (all 864 settings), per-unit ALC -> round4_cache/grid_A.npz."""
    from concurrent.futures import ProcessPoolExecutor
    F = Frame(args.payload)
    folds = load_folds()
    t0 = time.time()
    S = Streams(F, make_testsets(F, folds, args.config, R=args.R))
    log(f"config {args.config}: {S.n_units} units, {len(S.ev['y'])} evaluation rows, {len(S.acq[31]['y'])} labels at budget 31 "
        f"({time.time() - t0:.0f}s)")
    base = eval_methods(S, [], baselines=True)
    ext = args.grid == "ext"
    settings = grid_settings(ext)
    chunks = [list(range(i, len(settings), args.workers)) for i in range(args.workers)]
    res = {k: alc_of(v).astype(np.float32) for k, v in base.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for part in ex.map(_grid_worker, [(args.payload, c, args.R, args.config, ext) for c in chunks]):
            res.update(part)
    fname = "grid_A.npz" if (args.config == "A" and not ext) else f"grid_{args.config}_{args.grid}.npz"
    np.savez(_cache_dir(args) / fname, unit_bench=S.unit_bench, unit_fold=S.unit_fold, units=S.units,
             names=np.array(list(res)), alc=np.stack([res[k] for k in res]))
    log(f"grid done: {len(res)} methods ({time.time() - t0:.0f}s)")


def _parse_label(label: str) -> tuple:
    """'xp|bench1|name0.5|...|item1.5|probit1' -> (setting dict, tau_item, probit)."""
    parts = label.split("|")[1:]
    st = {}
    ti, pr = 0.0, True
    for p in parts:
        for k in GRID:
            if p.startswith(k) and p[len(k):].replace(".", "", 1).isdigit():
                st[k] = float(p[len(k):])
        if p.startswith("item"):
            ti = float(p[4:])
        if p.startswith("probit"):
            pr = bool(int(p[6:]))
    return st, ti, pr


def select_nested(grid: dict) -> dict:
    """Per outer fold: the XP setting with the lowest mean config-A ALC on the other folds; plus the all-fold pick."""
    names = list(grid["names"])
    alc = grid["alc"]
    fold = grid["unit_fold"]
    xp_idx = [i for i, n in enumerate(names) if n.startswith("xp|")]
    out = {"per_fold": [], "n_settings": len(xp_idx)}
    for f in range(int(fold.max()) + 1):
        tr = fold != f
        means = alc[xp_idx][:, tr].mean(axis=1)
        j = xp_idx[int(np.argmin(means))]
        out["per_fold"].append({"fold": f, "label": names[j], "train_alc": float(means.min())})
    means = alc[xp_idx].mean(axis=1)
    order = np.argsort(means)
    out["all"] = {"label": names[xp_idx[int(order[0])]], "alc": float(means[order[0]])}
    out["top10"] = [{"label": names[xp_idx[int(i)]], "alc": float(means[i])} for i in order[:10]]
    return out


def _summ(S: Streams, out: dict, ref: str = "S0", cmp: list | None = None) -> dict:
    """Mean ALC and Brier per budget (cluster CIs over benchmarks) and paired differences vs ref."""
    cl = S.unit_bench
    res = {"n_units": int(S.n_units), "n_benchmarks": int(len(np.unique(cl))), "methods": {}, "paired": {}}
    for k, br in out.items():
        a = alc_of(br)
        res["methods"][k] = {"alc": cluster_mean_ci(a, cl), "brier": [float(x) for x in br.mean(axis=0)],
                             "brier_ci": [cluster_mean_ci(br[:, i], cl) for i in range(6)]}
    for k in (cmp or [m for m in out if m != ref]):
        if k == ref or k not in out:
            continue
        d = alc_of(out[k]) - alc_of(out[ref])
        rec = {"alc": cluster_boot(d, cl, seed=0)}
        rec["brier"] = [cluster_boot(out[k][:, i] - out[ref][:, i], cl, seed=0) for i in range(6)]
        res["paired"][f"{k}_minus_{ref}"] = rec
    return res


def _fold_selected_out(S: Streams, sel: dict, extra: list | None = None) -> dict:
    """Briers of S0/S1/prior/oracle and the fold-selected XP (each unit uses its own fold's setting)."""
    base = eval_methods(S, [], baselines=True)
    xp = np.zeros((S.n_units, 6))
    for rec in sel["per_fold"]:
        st, ti, pr = _parse_label(rec["label"])
        m = S.unit_fold == rec["fold"]
        if not np.any(m):
            continue
        o = eval_methods(S, [("xp", taus_of(st), [(ti, pr)])], baselines=False)
        xp[m] = o[f"xp|item{ti:g}|probit{int(pr)}"][m]
    base["XP"] = xp
    st, ti, pr = _parse_label(sel["all"]["label"])
    o = eval_methods(S, [("xpall", taus_of(st), [(ti, pr)])], baselines=False)
    base["XP_allfold"] = o[f"xpall|item{ti:g}|probit{int(pr)}"]
    for label, taus, ti, pr in (extra or []):
        o = eval_methods(S, [(label, taus, [(ti, pr)])], baselines=False)
        base[label] = o[f"{label}|item{ti:g}|probit{int(pr)}"]
    return base


def components_of(st: dict, ti: float, pr: bool) -> list:
    """Ablations around the selected setting: each effect off in turn, and single-source variants."""
    full = taus_of(st)
    comps = []
    for k in ("bench", "name", "subj", "namebench", "pair"):
        if full[k] > 0:
            t = dict(full)
            t[k] = 0.0
            comps.append((f"minus_{k}", t, ti, pr))
    if ti > 0:
        comps.append(("minus_item", full, 0.0, pr))
    comps.append(("minus_probit", full, ti, not pr))
    off = {k: 0.0 for k in full}
    comps.append(("only_pair", dict(off, mu=0.0, pair=max(full["pair"], 0.3)), 0.0, pr))
    comps.append(("bench_and_pair", dict(off, mu=full["mu"], bench=full["bench"], pair=full["pair"]), 0.0, pr))
    comps.append(("name_subj_and_pair", dict(off, name=max(full["name"], 0.5), subj=max(full["subj"], 0.5),
                                             namebench=full["namebench"], pair=full["pair"]), 0.0, pr))
    comps.append(("pair_and_item", dict(off, pair=full["pair"]), max(ti, 1.5), pr))
    return comps


def cmd_cv(args) -> None:
    """Nested selection on the config-A grid; primary result; configs B, C, D; components -> results/round4/xpair_cv.json."""
    F = Frame(args.payload)
    folds = load_folds()
    grid = dict(np.load(_cache_dir(args) / "grid_A.npz", allow_pickle=False))
    sel = select_nested(grid)
    log("selected per fold: " + "; ".join(r["label"] for r in sel["per_fold"]) + f"; all folds: {sel['all']['label']}")
    st, ti, pr = _parse_label(sel["all"]["label"])
    comps = components_of(st, ti, pr)
    res = {"prereg_sha256": prereg_sha(), "revision": F.revision, "selection": sel, "configs": {}}
    for config in ("A", "B", "C", "D"):
        t0 = time.time()
        S = Streams(F, make_testsets(F, folds, config, R=args.R))
        out = _fold_selected_out(S, sel, extra=comps if config == "A" else None)
        summ = _summ(S, out, ref="S0")
        d1 = alc_of(out["XP"]) - alc_of(out["S1"])
        summ["paired"]["XP_minus_S1"] = {"alc": cluster_boot(d1, S.unit_bench, seed=0),
                                         "brier": [cluster_boot(out["XP"][:, i] - out["S1"][:, i], S.unit_bench, seed=0) for i in range(6)]}
        d2 = alc_of(out["S1"]) - alc_of(out["S0"])
        summ["paired"]["S1_minus_S0_check"] = {"alc": cluster_boot(d2, S.unit_bench, seed=0)}
        if config == "A":
            for label, *_ in comps:
                dd = alc_of(out[label]) - alc_of(out["XP_allfold"])
                summ["paired"][f"{label}_minus_XP_allfold"] = {"alc": cluster_boot(dd, S.unit_bench, seed=0)}
            # per-fold primary differences
            dd = alc_of(out["XP"]) - alc_of(out["S0"])
            summ["per_fold"] = [{"fold": f, "n_units": int(np.sum(S.unit_fold == f)),
                                 "diff": float(dd[S.unit_fold == f].mean()) if np.any(S.unit_fold == f) else None}
                                for f in range(len(folds))]
            if args.save_units:
                np.savez(_cache_dir(args) / "units_A.npz", **{k: v for k, v in out.items()}, unit_bench=S.unit_bench,
                         unit_fold=S.unit_fold, units=S.units)
        if config == "D":
            eo = S.unit_evalonly
            for nm, mask in (("acquired_pairs", ~eo), ("evalonly_pairs", eo)):
                if np.any(mask):
                    dd = alc_of(out["XP"])[mask] - alc_of(out["S0"])[mask]
                    summ["paired"][f"XP_minus_S0_{nm}"] = {"alc": cluster_boot(dd, S.unit_bench[mask], seed=0)}
        summ["n_labels_budget31"] = int(len(S.acq[31]["y"]))
        summ["n_testsets"] = int(len(np.unique(S.units[:, 0])))
        res["configs"][config] = summ
        p = summ["paired"]["XP_minus_S0"]["alc"]
        log(f"config {config}: {S.n_units} units, XP-S0 ALC {p['mean']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]; "
            f"S0 {summ['methods']['S0']['alc']['mean']:.4f}, S1 {summ['methods']['S1']['alc']['mean']:.4f}, "
            f"XP {summ['methods']['XP']['alc']['mean']:.4f} ({time.time() - t0:.0f}s)")
    R4.mkdir(parents=True, exist_ok=True)
    (R4 / "xpair_cv.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log(f"wrote {R4 / 'xpair_cv.json'}")


def cmd_acq(args) -> None:
    """Fresh-first vs random acquisition in config A under the shipped XP setting -> results/round4/xpair_acq.json."""
    F = Frame(args.payload)
    folds = load_folds()
    sel = json.loads((R4 / "xpair_cv.json").read_text(encoding="utf-8"))["selection"]
    st, ti, pr = _parse_label(sel["all"]["label"])
    res = {"prereg_sha256": prereg_sha(), "setting": sel["all"]["label"]}
    outs = {}
    for policy in ("random", "fresh"):
        S = Streams(F, make_testsets(F, folds, "A", R=args.R), policy=policy)
        o = eval_methods(S, [("xp", taus_of(st), [(ti, pr)])], baselines=True)
        outs[policy] = (S, o)
        res[f"n_skips_{policy}"] = int(S.n_fresh_skips)
    S = outs["random"][0]
    key = f"xp|item{ti:g}|probit{int(pr)}"
    a_r, a_f = outs["random"][1][key], outs["fresh"][1][key]
    res["xp_random"] = {"alc": cluster_mean_ci(alc_of(a_r), S.unit_bench), "brier": [float(x) for x in a_r.mean(axis=0)]}
    res["xp_fresh"] = {"alc": cluster_mean_ci(alc_of(a_f), S.unit_bench), "brier": [float(x) for x in a_f.mean(axis=0)]}
    res["fresh_minus_random"] = {"alc": cluster_boot(alc_of(a_f) - alc_of(a_r), S.unit_bench, seed=0),
                                 "brier": [cluster_boot(a_f[:, i] - a_r[:, i], S.unit_bench, seed=0) for i in range(6)]}
    s0r, s0f = outs["random"][1]["S0"], outs["fresh"][1]["S0"]
    res["S0_fresh_minus_random"] = {"alc": cluster_boot(alc_of(s0f) - alc_of(s0r), S.unit_bench, seed=0)}
    (R4 / "xpair_acq.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    d = res["fresh_minus_random"]["alc"]
    log(f"fresh - random (XP): {d['mean']:+.5f} [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]; skips {res['n_skips_fresh']}")


def _stdlib_preds(S: Streams, F: Frame, t: int, b: int, binary: bool = True) -> tuple:
    """model.predict_cross_pair on dict inputs for test set t at budget b; returns (stdlib, numpy-row index)."""
    E = S.acq[b]
    me = E["ts"] == t
    mv = S.ev["ts"] == t
    subj_cache = {}

    def subject(p):
        if p not in subj_cache:
            subj_cache[p] = json.loads(F.subj_keys[int(F.pair_subj[p])])
        return subj_cache[p]

    def item(i):
        return {"benchmark_id": F.dirs[int(F.item_dir[i])], "item_content": f"item {int(i)}", "item_features": "",
                "interactors": ""}
    labeled = []
    for p, i, y in zip(E["pair"][me], E["item"][me], E["y"][me]):
        labeled.append([[subject(int(p)), item(int(i))], (1 if y >= 0.5 else 0) if binary else float(y)])
    preds = [model.predict_cross_pair([subject(int(p)), item(int(i))], labeled) for p, i in zip(S.ev["pair"][mv], S.ev["item"][mv])]
    return np.array(preds, dtype=float), np.where(mv)[0]


def cmd_equivalence(args) -> None:
    """Pre-registered check: stdlib model.py vs the numpy simulator (selected setting, >= 20 config-A test sets)."""
    F = Frame(args.payload)
    folds = load_folds()
    sel = json.loads((R4 / "xpair_cv.json").read_text(encoding="utf-8"))["selection"]
    st, ti, pr = _parse_label(sel["all"]["label"])
    taus = taus_of(st)
    model.XP_TAUS = tuple((k, taus[k]) for k in ("mu", "bench", "name", "subj", "namebench", "pair"))
    model.XP_TAU_ITEM, model.XP_PROBIT = ti, pr
    zi_of = {f"item {i}": float(z) for i, z in enumerate(F.item_zi)}
    orig = model._item_logit
    model._item_logit = lambda it: zi_of.get(it.get("item_content"), 0.0)
    tsets = make_testsets(F, folds, "A", R=args.R)
    rng = np.random.RandomState(5)
    pick = sorted(int(x) for x in rng.choice(len(tsets), 24, replace=False))
    sub = [tsets[i] for i in pick]
    S = Streams(F, sub)
    for b in BUDGETS[1:]:
        S.acq[b]["y"] = (S.acq[b]["y"] >= 0.5).astype(float)
    max_diff, n = 0.0, 0
    t0 = time.time()
    for b in BUDGETS[1:]:
        fit = xp_fit_np(S.acq[b], S.ev, taus)
        p_np = xp_predict_np(S.acq[b], S.ev, fit, ti, pr)
        for t in range(len(sub)):
            model.xp_clear_cache()
            ps, idx = _stdlib_preds(S, F, t, b)
            max_diff = max(max_diff, float(np.max(np.abs(ps - p_np[idx]))))
            n += len(idx)
    model._item_logit = orig
    res = {"prereg_sha256": prereg_sha(), "setting": sel["all"]["label"], "n_testsets": len(sub), "n_predictions": n,
           "max_abs_diff": max_diff, "passes_1e-3": bool(max_diff < 1e-3), "seconds": time.time() - t0}
    (R4 / "xpair_equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"equivalence: {n} predictions, max |diff| {max_diff:.2e} ({time.time() - t0:.0f}s)")


def _apply_setting(label: str) -> None:
    st, ti, pr = _parse_label(label)
    taus = taus_of(st)
    model.XP_TAUS = tuple((k, taus[k]) for k in ("mu", "bench", "name", "subj", "namebench", "pair"))
    model.XP_TAU_ITEM, model.XP_PROBIT = ti, pr
    model.xp_clear_cache()


def real_testbed_global(label: str, seed: int = 0, extra_settings: dict | None = None) -> dict:
    """The 17-pair real Ollama testbed evaluated with ONE global labeled list per budget (PREREG.md, rule b).

    Per-pair splits and random acquisition exactly as run_experiments.eval_one_pair (same seeds).
    Methods: shipped full predictor on the global list (deployed), shipped full predictor on per-pair lists
    (= the local harness), XP on the global list; optional extra XP settings {name: label}.
    """
    import random
    import real_testbed
    raw = real_testbed.load_pairs()
    pairs = []
    for subject, sid, bid, items, meta in raw:
        rng = random.Random(f"{seed}|{sid}|{bid}")
        pair = harness.make_pair(dict(subject), sid, bid, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
        if pair is None:
            continue
        policy = harness.random_policy_factory(random.Random(f"acq|{seed}|{sid}|{bid}"))
        acquired = harness.run_acquisition(pair, policy, pair["order"], needs_prediction=False)
        pairs.append((pair, acquired, meta))
    settings = {"XP": label}
    settings.update(extra_settings or {})
    methods = ["shipped_global", "shipped_perpair"] + list(settings) + ["XP_perpair"]
    settings_pp = {"XP_perpair": label}
    briers = {m: [[None] * 6 for _ in pairs] for m in methods}
    labels_all = [[y for _, y in pr["eval"]] for pr, _, _ in pairs]
    saved = (model.XP_ENABLED, model.XP_TAUS, model.XP_TAU_ITEM, model.XP_PROBIT)
    t0 = time.time()
    try:
        for bi, b in enumerate(BUDGETS):
            glob = []
            for pr, acq, _ in pairs:
                glob += [[[pr["subject"], pr["acq"][i][0]], pr["acq"][i][1]] for i in acq[:b]]
            for m in methods:
                if m in settings or m in settings_pp:
                    model.XP_ENABLED = True
                    _apply_setting(settings.get(m) or settings_pp[m])
                else:
                    model.XP_ENABLED = False
                for j, (pr, acq, _) in enumerate(pairs):
                    if m in ("shipped_perpair", "XP_perpair"):
                        lab = [[[pr["subject"], pr["acq"][i][0]], pr["acq"][i][1]] for i in acq[:b]]
                    else:
                        lab = glob
                    preds = [model.predict([pr["subject"], it], lab) for it, _ in pr["eval"]]
                    briers[m][j][bi] = harness.brier(preds, labels_all[j])
    finally:
        model.XP_ENABLED, model.XP_TAUS, model.XP_TAU_ITEM, model.XP_PROBIT = saved
        model.xp_clear_cache()
    alcs = {m: [harness.alc(dict(zip(BUDGETS, briers[m][j]))) for j in range(len(pairs))] for m in methods}
    out = {"seed": seed, "n_pairs": len(pairs), "seconds": time.time() - t0,
           "pairs": [{"bench": meta["bench"], "model": meta["model"], "harness": meta["harness"]} for _, _, meta in pairs],
           "methods": {m: {"alc": harness.bootstrap_mean_ci(alcs[m], 1000, seed=1),
                           "brier": [sum(briers[m][j][k] for j in range(len(pairs))) / len(pairs) for k in range(6)],
                           "per_pair_alc": alcs[m]} for m in methods},
           "paired": {}}
    for m in methods:
        if m in ("shipped_global",):
            continue
        out["paired"][f"{m}_minus_shipped_global"] = harness.paired_bootstrap(alcs[m], alcs["shipped_global"], 1000, seed=0)
    out["paired"]["XP_perpair_minus_shipped_perpair"] = harness.paired_bootstrap(alcs["XP_perpair"], alcs["shipped_perpair"], 1000, seed=0)
    for m in settings:
        out["paired"][f"{m}_minus_shipped_perpair"] = harness.paired_bootstrap(alcs[m], alcs["shipped_perpair"], 1000, seed=0)
        out["paired"][f"{m}_minus_shipped_global_by_budget"] = [
            harness.bootstrap_mean_ci([briers[m][j][k] - briers["shipped_global"][j][k] for j in range(len(pairs))], 1000, seed=0)
            for k in range(6)]
    return out


def cmd_realtest(args) -> None:
    sel = json.loads((R4 / "xpair_cv.json").read_text(encoding="utf-8"))["selection"]
    label = args.label or sel["all"]["label"]
    res = {"prereg_sha256": prereg_sha(), "setting": label, "runs": []}
    for seed in range(args.seeds):
        r = real_testbed_global(label, seed=seed)
        res["runs"].append(r)
        d = r["paired"]["XP_minus_shipped_global"]
        log(f"real testbed seed {seed}: XP {r['methods']['XP']['alc']['mean']:.4f}, shipped global "
            f"{r['methods']['shipped_global']['alc']['mean']:.4f}, shipped per-pair {r['methods']['shipped_perpair']['alc']['mean']:.4f}; "
            f"XP - shipped global {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] ({r['seconds']:.0f}s)")
    out = R4 / (args.out or "xpair_real_testbed.json")
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log(f"wrote {out}")


def _load_grid(args, fname: str) -> dict:
    g = dict(np.load(_cache_dir(args) / fname, allow_pickle=False))
    g["index"] = {str(n): i for i, n in enumerate(g["names"])}
    return g


def robust_selection(gA: dict, gE: dict) -> dict:
    names = [str(n) for n in gA["names"] if str(n).startswith("xp|") and str(n) in gE["index"]]
    A = np.stack([gA["alc"][gA["index"][n]] for n in names])
    Eg = np.stack([gE["alc"][gE["index"][n]] for n in names])
    fa, fe = gA["unit_fold"], gE["unit_fold"]
    out = {"per_fold": [], "n_settings": len(names)}
    for f in range(int(fa.max()) + 1):
        crit = (A[:, fa != f].mean(axis=1) + Eg[:, fe != f].mean(axis=1)) / 2
        j = int(np.argmin(crit))
        out["per_fold"].append({"fold": f, "label": names[j], "train_crit": float(crit[j])})
    crit = (A.mean(axis=1) + Eg.mean(axis=1)) / 2
    order = np.argsort(crit)
    out["all"] = {"label": names[int(order[0])], "crit": float(crit[order[0]]), "alc_A": float(A[order[0]].mean()),
                  "alc_E": float(Eg[order[0]].mean())}
    out["top10"] = [{"label": names[int(i)], "crit": float(crit[i]), "alc_A": float(A[i].mean()), "alc_E": float(Eg[i].mean())}
                    for i in order[:10]]
    # tau_item profile: best mean ALC per tau_item value on A and on E
    prof = {}
    for ti in TAU_ITEMS_EXT:
        m = [i for i, n in enumerate(names) if _parse_label(n)[1] == ti]
        prof[f"{ti:g}"] = {"best_A": float(A[m].mean(axis=1).min()), "best_E": float(Eg[m].mean(axis=1).min()),
                           "best_avg": float(crit[m].min())}
    out["tau_item_profile"] = prof
    profb = {}
    for tb in GRID_EXT["bench"]:
        m = [i for i, n in enumerate(names) if _parse_label(n)[0].get("bench") == tb]
        profb[f"{tb:g}"] = {"best_A": float(A[m].mean(axis=1).min()), "best_E": float(Eg[m].mean(axis=1).min())}
    out["tau_bench_profile"] = profb
    return out


def cmd_addendum(args) -> None:
    """PREREG_ADDENDUM_1.md: extended grid, config E, robust pick -> results/round4/xpair_addendum1.json."""
    F = Frame(args.payload)
    folds = load_folds()
    gA, gE = _load_grid(args, "grid_A_ext.npz"), _load_grid(args, "grid_E_ext.npz")
    cv = json.loads((R4 / "xpair_cv.json").read_text(encoding="utf-8"))
    pre = cv["selection"]
    rob = robust_selection(gA, gE)
    log("robust pick per fold: " + "; ".join(r["label"] for r in rob["per_fold"]) + f"; all folds: {rob['all']['label']}")
    res = {"prereg_sha256": prereg_sha(), "addendum_sha256": hashlib.sha256((R4 / "PREREG_ADDENDUM_1.md").read_bytes()).hexdigest(),
           "selection_robust": rob, "selection_prereg": pre, "configs": {}}
    st, ti, pr = _parse_label(rob["all"]["label"])
    for config in ("A", "E"):
        S = Streams(F, make_testsets(F, folds, config, R=args.R))
        out = _fold_selected_out(S, pre)  # XP = pre-registered nested, XP_allfold = pre-registered all-fold pick
        rob_out = _fold_selected_out(S, rob)
        out["XPrb"] = rob_out["XP"]
        out["XPrb_allfold"] = rob_out["XP_allfold"]
        if config == "E":
            for label, taus, ti_, pr_ in components_of(st, ti, pr):
                o = eval_methods(S, [(label, taus, [(ti_, pr_)])], baselines=False)
                out[label] = o[f"{label}|item{ti_:g}|probit{int(pr_)}"]
        summ = _summ(S, out, ref="S0", cmp=[k for k in out if k not in ("S0",)])
        for a, b in (("XPrb", "XP"), ("XP", "S1"), ("XPrb", "S1"), ("XPrb_allfold", "XP_allfold")):
            d = alc_of(out[a]) - alc_of(out[b])
            summ["paired"][f"{a}_minus_{b}"] = {"alc": cluster_boot(d, S.unit_bench, seed=0),
                                               "brier": [cluster_boot(out[a][:, i] - out[b][:, i], S.unit_bench, seed=0) for i in range(6)]}
        if config == "E":
            for label, *_ in components_of(st, ti, pr):
                dd = alc_of(out[label]) - alc_of(out["XPrb_allfold"])
                summ["paired"][f"{label}_minus_XPrb_allfold"] = {"alc": cluster_boot(dd, S.unit_bench, seed=0)}
        summ["n_testsets"] = int(len(np.unique(S.units[:, 0])))
        summ["n_labels_budget31"] = int(len(S.acq[31]["y"]))
        res["configs"][config] = summ
        for k in ("XP_minus_S0", "XPrb_minus_S0", "XPrb_minus_XP"):
            p = summ["paired"][k]["alc"]
            log(f"config {config}: {k} {p['mean']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]")
    a_ok = not (res["configs"]["A"]["paired"]["XPrb_minus_XP"]["alc"]["ci_lo"] > 0)
    e_ok = res["configs"]["E"]["paired"]["XPrb_minus_S0"]["alc"]["ci_hi"] < 0
    res["decision"] = {"robust_not_worse_on_A": bool(a_ok), "robust_better_than_S0_on_E": bool(e_ok),
                       "ship_label": rob["all"]["label"] if (a_ok and e_ok) else pre["all"]["label"],
                       "ship_source": "robust pick (addendum 1)" if (a_ok and e_ok) else "pre-registered pick"}
    log(f"decision: ship {res['decision']['ship_label']} ({res['decision']['ship_source']})")
    (R4 / "xpair_addendum1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def cmd_recheck(args) -> None:
    """Rules (b)/(c) for the setting chosen by addendum 1: configs B, C, D with its nested picks; real testbed."""
    F = Frame(args.payload)
    folds = load_folds()
    add = json.loads((R4 / "xpair_addendum1.json").read_text(encoding="utf-8"))
    label = add["decision"]["ship_label"]
    sel = add["selection_robust"] if add["decision"]["ship_source"].startswith("robust") else add["selection_prereg"]
    res = {"prereg_sha256": prereg_sha(), "ship_label": label, "configs": {}}
    for config in ("B", "C", "D"):
        S = Streams(F, make_testsets(F, folds, config, R=args.R))
        out = _fold_selected_out(S, sel)
        summ = _summ(S, out, ref="S0", cmp=["XP", "XP_allfold", "S1"])
        d1 = alc_of(out["XP"]) - alc_of(out["S1"])
        summ["paired"]["XP_minus_S1"] = {"alc": cluster_boot(d1, S.unit_bench, seed=0)}
        if config == "D":
            eo = S.unit_evalonly
            for nm, mask in (("acquired_pairs", ~eo), ("evalonly_pairs", eo)):
                if np.any(mask):
                    dd = alc_of(out["XP"])[mask] - alc_of(out["S0"])[mask]
                    summ["paired"][f"XP_minus_S0_{nm}"] = {"alc": cluster_boot(dd, S.unit_bench[mask], seed=0)}
        res["configs"][config] = summ
        p = summ["paired"]["XP_minus_S0"]["alc"]
        log(f"config {config}: XP - S0 {p['mean']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]")
    (R4 / "xpair_recheck.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def real_testbed_acq(label: str, seed: int = 0) -> dict:
    """Fresh-first vs in-order (random-stream) acquisition on the real testbed, both with XP and ONE global list.

    Stage-wise round robin as in the simulator: for b in 1, 3, 7, 15, 31 each pair streams its (fixed, random)
    candidate order until it holds b labels; labeling.acquisition_function is called with the global list.
    """
    import random
    import labeling
    import real_testbed
    raw = real_testbed.load_pairs()
    pairs = []
    for subject, sid, bid, items, meta in raw:
        rng = random.Random(f"{seed}|{sid}|{bid}")
        pair = harness.make_pair(dict(subject), sid, bid, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
        if pair is not None:
            pairs.append((pair, meta))
    saved = (labeling.POLICY, model.XP_ENABLED, model.XP_TAUS, model.XP_TAU_ITEM, model.XP_PROBIT)
    res = {}
    try:
        model.XP_ENABLED = True
        _apply_setting(label)
        acq_by_policy = {}
        for policy in ("first", "fresh"):
            labeling.POLICY = policy
            glob = []
            pos = [0] * len(pairs)
            got = [[] for _ in pairs]
            skips = 0
            for b in BUDGETS[1:]:
                for j, (pr, _) in enumerate(pairs):
                    order = pr["order"]
                    while len(got[j]) < b and pos[j] < len(order):
                        idx = order[pos[j]]
                        ctx = {"subject_id": pr["subject_id"], "benchmark_id": pr["benchmark_id"],
                               "labels_acquired": len(got[j]), "labels_remaining": MAX_LABELS - len(got[j]),
                               "max_labels": MAX_LABELS, "items_remaining": len(order) - pos[j]}
                        pos[j] += 1
                        item, y = pr["acq"][idx]
                        take = labeling.acquisition_function([pr["subject"], item], labeled=glob, context=ctx)
                        if take:
                            got[j].append(idx)
                            glob.append([[pr["subject"], item], y])
                        else:
                            skips += 1
            acq_by_policy[policy] = (got, skips)
        briers = {pol: [[None] * 6 for _ in pairs] for pol in acq_by_policy}
        for pol, (got, _) in acq_by_policy.items():
            for bi, b in enumerate(BUDGETS):
                glob = []
                for j, (pr, _) in enumerate(pairs):
                    glob += [[[pr["subject"], pr["acq"][i][0]], pr["acq"][i][1]] for i in got[j][:b]]
                model.xp_clear_cache()
                for j, (pr, _) in enumerate(pairs):
                    preds = [model.predict([pr["subject"], it], glob) for it, _ in pr["eval"]]
                    briers[pol][j][bi] = harness.brier(preds, [y for _, y in pr["eval"]])
        alcs = {pol: [harness.alc(dict(zip(BUDGETS, briers[pol][j]))) for j in range(len(pairs))] for pol in briers}
        res = {"seed": seed, "n_pairs": len(pairs), "skips_fresh": acq_by_policy["fresh"][1],
               "alc_first": harness.bootstrap_mean_ci(alcs["first"], 1000, seed=1),
               "alc_fresh": harness.bootstrap_mean_ci(alcs["fresh"], 1000, seed=1),
               "fresh_minus_first": harness.paired_bootstrap(alcs["fresh"], alcs["first"], 1000, seed=0)}
    finally:
        labeling.POLICY, model.XP_ENABLED, model.XP_TAUS, model.XP_TAU_ITEM, model.XP_PROBIT = saved
        model.xp_clear_cache()
    return res


def cmd_realacq(args) -> None:
    add = json.loads((R4 / "xpair_addendum1.json").read_text(encoding="utf-8"))
    label = add["decision"]["ship_label"]
    out = {"prereg_sha256": prereg_sha(), "setting": label, "runs": []}
    for seed in range(args.seeds):
        r = real_testbed_acq(label, seed)
        out["runs"].append(r)
        d = r["fresh_minus_first"]
        log(f"real testbed acquisition seed {seed}: fresh - random {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
            f"(skips {r['skips_fresh']})")
    (R4 / "xpair_real_acq.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def cmd_runtime(args) -> None:
    """Runtime of the shipped stdlib predictor on global lists of several sizes -> results/round4/xpair_runtime.json.

    Synthetic dict inputs shaped like the official ones (250-word items, 40 benchmarks, subjects with several
    efforts); per size: one call that parses and fits the list (first call of a checkpoint), then cache-hit calls
    with new target items.  One process, one core.
    """
    import random as _random
    words = ("integral derivative matrix prime graph compute sum product probability expected value sequence series "
             "limit function polynomial equation solve prove show find the of is a which").split()
    out = {"prereg_sha256": prereg_sha(), "xp_taus": dict(model.XP_TAUS), "xp_tau_item": model.XP_TAU_ITEM, "sizes": []}
    saved = model.XP_ENABLED
    model.XP_ENABLED = True
    try:
        for n_pairs in (10, 100, 1000, 2000, 4000):
            rng = _random.Random(n_pairs)
            n_b = 40
            items = {b: [{"benchmark_id": f"bench_{b:03d}", "item_content": f"Q{b}.{k}: " + " ".join(rng.choice(words) for _ in range(250)),
                          "item_features": "tier=hard" if k % 4 == 0 else "", "interactors": ""} for k in range(100)] for b in range(n_b)}
            labeled, targets = [], []
            for i in range(n_pairs):
                s = {"normalized_name": f"model-{i % 400}", "provider": "p", "release_date": "2025-06-01", "access_date": "",
                     "harness": "", "harness_version": "", "reasoning_effort": ("low", "high", "")[i % 3],
                     "subject_features_extra": ""}
                b = i % n_b
                idx = list(range(100))
                rng.shuffle(idx)
                labeled += [[[s, items[b][k]], int(rng.random() < 0.5)] for k in idx[:31]]
                targets += [[s, items[b][k]] for k in idx[50:]]
            model.xp_clear_cache()
            t0 = time.perf_counter()
            model.predict(targets[0], labeled)
            first = time.perf_counter() - t0
            state = model._XP_CACHE["entry"][1]
            k = min(len(targets) - 1, 5000)
            t0 = time.perf_counter()
            for x in targets[1:1 + k]:
                model.predict(x, labeled)
            per = (time.perf_counter() - t0) / k
            t0 = time.perf_counter()
            model._xp_fingerprint(labeled)
            fp = time.perf_counter() - t0
            rec = {"n_pairs": n_pairs, "n_labels": len(labeled), "n_targets": len(targets), "first_call_s": first,
                   "ms_per_call": 1000 * per, "fingerprint_ms": 1000 * fp,
                   "checkpoint_s": first + per * len(targets), "n_groups": {t["name"]: len(t["theta"]) for t in state["types"]}}
            out["sizes"].append(rec)
            log(f"runtime {n_pairs} pairs / {len(labeled)} labels: fit {first:.2f}s, {1000 * per:.3f} ms per call, "
                f"checkpoint {rec['checkpoint_s']:.1f}s")
    finally:
        model.XP_ENABLED = saved
        model.xp_clear_cache()
    import platform
    out["machine"] = {"python": platform.python_version(), "processor": platform.processor(), "system": platform.system()}
    (R4 / "xpair_runtime.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_perpair(args) -> None:
    """Robustness (descriptive): every config-A pair evaluated ALONE, i.e. as if `labeled` held only the target
    pair's labels (the reading of the interface used in rounds 1-3) -> results/round4/xpair_perpair.json."""
    F = Frame(args.payload)
    folds = load_folds()
    add = json.loads((R4 / "xpair_addendum1.json").read_text(encoding="utf-8"))
    label = add["decision"]["ship_label"]
    st, ti, pr = _parse_label(label)
    tsA = make_testsets(F, folds, "A", R=args.R)
    single = [{"fold": t["fold"], "pairs": [p], "cap": t["cap"], "evalonly": [False], "benches": t["benches"]}
              for t in tsA for p in t["pairs"]]
    S = Streams(F, single)
    out = eval_methods(S, [("xp", taus_of(st), [(ti, pr)])], baselines=True)
    out["XP"] = out.pop(f"xp|item{ti:g}|probit{int(pr)}")
    summ = _summ(S, out, ref="S1", cmp=["XP"])
    res = {"prereg_sha256": prereg_sha(), "setting": label, "summary": summ}
    (R4 / "xpair_perpair.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    d = summ["paired"]["XP_minus_S1"]["alc"]
    log(f"per-pair lists: XP - S1 {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")


def cmd_explore(args) -> None:
    """Structure of the round-4 frame (no labels, no predictions) -> results/round4/xpair_frame_stats.json."""
    from collections import Counter, defaultdict
    F = Frame(args.payload)
    n = F.pair_n
    nb = len(F.dirs)
    pairs_per_b = np.bincount(F.pair_dir, minlength=nb)
    names_per_b = np.array([len(set(F.pair_name[F.pair_dir == b].tolist())) for b in range(nb)])
    nb_of_name, nb_of_subj = defaultdict(set), defaultdict(set)
    for p in range(F.n_pairs):
        nb_of_name[int(F.pair_name[p])].add(int(F.pair_dir[p]))
        nb_of_subj[int(F.pair_subj[p])].add(int(F.pair_dir[p]))
    cnb = Counter(zip(F.pair_name.tolist(), F.pair_dir.tolist()))
    v = np.array(list(cnb.values()))
    cnt = np.bincount(F.row_item)
    res = {"revision": F.revision, "n_pairs": int(F.n_pairs), "n_pairs_80": int((n >= 80).sum()), "n_rows": int(len(F.row_y)),
           "n_items": int(len(F.item_dir)), "n_benchmarks": int(nb), "n_subject_keys": int(len(F.subj_keys)),
           "n_names": int(len(F.name_keys)), "max_items_per_pair": MAX_ITEMS_PER_PAIR,
           "pairs_per_benchmark_median": float(np.median(pairs_per_b)), "pairs_per_benchmark_max": int(pairs_per_b.max()),
           "names_per_benchmark_median": float(np.median(names_per_b)),
           "names_on_2plus_benchmarks": int(sum(len(s) >= 2 for s in nb_of_name.values())),
           "subject_keys_on_2plus_benchmarks": int(sum(len(s) >= 2 for s in nb_of_subj.values())),
           "frac_pairs_sharing_name_and_benchmark": float(v[v > 1].sum() / F.n_pairs),
           "frac_rows_on_item_answered_by_2plus_pairs": float((cnt[F.row_item] >= 2).mean()),
           "folds_n_benchmarks": [len(f) for f in load_folds()]}
    (R4 / "xpair_frame_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--payload", default=str(DEFAULT_PAYLOAD))
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-units", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--config", default="A")
    ap.add_argument("--grid", default="prereg")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    {"build": cmd_build, "grid": cmd_grid, "cv": cmd_cv, "acq": cmd_acq, "equivalence": cmd_equivalence, "realtest": cmd_realtest,
     "addendum": cmd_addendum, "recheck": cmd_recheck,
     "realacq": cmd_realacq, "runtime": cmd_runtime, "perpair": cmd_perpair, "explore": cmd_explore}[args.cmd](args)


if __name__ == "__main__":
    main()
