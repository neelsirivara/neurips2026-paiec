"""Item-rating study, step 1: recover the text of every frame item the pre-registered scenarios need.

    python -I items_extract.py

The round-4 frame stores item indices but not item text.  This replays xpair_sim.cmd_build's item-index assignment
for the needed benchmarks only (indices are contiguous per benchmark, in build order), verifies the replay against
the frame (item count, hand-set item logit and row-to-item map of every row of those benchmarks), and writes
  _downloads/measurement_db_payload/round4_cache/deep_dive_items.jsonl   (one line per needed item: idx, dir,
      content_hash, item_id, content[:1500], features[:300]; outside the repository, derived from the gated release)
  experiments/results/deep_dive/items/items_manifest.json             (scenario benchmark lists, counts, checks)
Needed items = all distinct items of the INV and AS test sets (PREREG_items.md).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *
import hashlib
import math

PREREG = OUT / "items" / "PREREG_items.md"
CACHE = Path(X.DEFAULT_PAYLOAD).resolve() / "round4_cache"
t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
inv = inventory_dirs(F)
with_elig = sorted({F.dirs[b] for b in np.unique(F.pair_dir[elig])})
non_inv = [d for d in with_elig if d not in inv]
as_dirs = [non_inv[i] for i in np.random.RandomState(2026).choice(len(non_inv), 25, replace=False)]
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
need = {}
for name, keep in (("INV", set(inv)), ("AS", set(as_dirs))):
    ts = X.make_testsets(F, sub(keep), "A", R=300)
    S = X.Streams(F, ts)
    its = np.unique(np.concatenate([S.ev["item"], S.acq[31]["item"]]))
    need[name] = its
    log(f"{name}: {len(its)} items, {len(np.unique(F.item_dir[its]))} benchmarks")
need_all = np.unique(np.concatenate(list(need.values())))
dirs_needed = sorted({F.dirs[int(F.item_dir[i])] for i in need_all})

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
try:
    pa.set_cpu_count(1)
except Exception:
    pass
data_dir = Path(X.DEFAULT_PAYLOAD).resolve() / "data"
out_lines = []
checks = {}
for d in dirs_needed:
    di = F.dirs.index(d)
    item_idx = np.where(F.item_dir == di)[0]
    start = int(item_idx.min())
    assert np.all(item_idx == np.arange(start, start + len(item_idx))), d
    base = data_dir / d
    subj_tbl = pq.read_table(base / "subjects.parquet").to_pandas()
    subj_tbl["subject_id"] = subj_tbl["subject_id"].astype(str)
    resp_path = base / "response.parquet"
    names = pq.ParquetFile(resp_path).schema_arrow.names
    rcol = X._first_col(names, X.RESPONSE_CANDIDATES)
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
    inames = pq.ParquetFile(base / "items.parquet").schema_arrow.names
    ccol, fcol = X._first_col(inames, X.CONTENT_CANDIDATES), X._first_col(inames, X.FEATURE_CANDIDATES)
    hcol = "content_hash" if "content_hash" in inames else None
    it = pq.read_table(base / "items.parquet", columns=["item_id"] + [c for c in (ccol, fcol, hcol) if c])
    ids = pc.cast(it.column("item_id"), pa.string()).to_pylist()
    cont = pc.cast(it.column(ccol), pa.string()).to_pylist() if ccol else [None] * len(ids)
    lengths = [len(c) if c else 0 for c in cont]
    feats = pc.fill_null(pc.cast(it.column(fcol), pa.string()), "").to_pylist() if fcol else [""] * len(ids)
    hashes = pc.fill_null(pc.cast(it.column(hcol), pa.string()), "").to_pylist() if hcol else [""] * len(ids)
    if ccol and any(h == "" for h in hashes):
        hashes = [h or hashlib.blake2b((c or "").encode("utf-8"), digest_size=8).hexdigest() for h, c in zip(hashes, cont)]
    info = {i_: (int(L or 0), f_ or "", h_ or ("id:" + i_), c_ or "") for i_, L, f_, h_, c_ in zip(ids, lengths, feats, hashes, cont)}
    keys = {}
    order_keys = []
    zi_replay = []
    for (s, c), grp in agg.groupby(["sid", "cond"], sort=False):
        if len(grp) < X.MIN_PAIR_ITEMS:
            continue
        iids = grp["iid"].tolist()
        ranks = [X._rank(d, i_) for i_ in iids]
        order = sorted(range(len(iids)), key=lambda j: ranks[j])[:X.MAX_ITEMS_PER_PAIR]
        for j in order:
            L, f_, h_, c_ = info.get(iids[j], (0, "", "id:" + iids[j], ""))
            if h_ not in keys:
                keys[h_] = (iids[j], f_, c_, L)
                order_keys.append(h_)
                zi_replay.append(X._hand_item_logit(f_, L))
    n_ok = len(order_keys) == len(item_idx)
    zi_ok = n_ok and bool(np.allclose(np.array(zi_replay), F.item_zi[item_idx], atol=1e-12))
    checks[d] = {"n_items_frame": int(len(item_idx)), "n_items_replay": len(order_keys), "zi_match": zi_ok}
    if not (n_ok and zi_ok):
        log(f"  MISMATCH {d}: {checks[d]}")
        continue
    needed_here = set(int(i) for i in need_all if int(F.item_dir[i]) == di)
    for k, h_ in enumerate(order_keys):
        idx = start + k
        if idx in needed_here:
            iid0, f_, c_, L = keys[h_]
            out_lines.append({"idx": idx, "dir": d, "content_hash": h_, "item_id": iid0,
                              "content": " ".join(str(c_).split())[:1500], "features": " ".join(str(f_).split())[:300],
                              "content_len": int(L)})
    log(f"  {d}: {len(order_keys)} items verified, {len(needed_here)} needed ({time.time()-t0:.0f}s)")
with open(CACHE / "deep_dive_items.jsonl", "w", encoding="utf-8") as fh:
    for r in out_lines:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
man = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "inv_dirs": sorted(inv), "as_dirs_seeded_order": as_dirs,
       "n_needed": {k: int(len(v)) for k, v in need.items()}, "n_needed_all": int(len(need_all)), "n_written": len(out_lines),
       "n_empty_content": int(sum(1 for r in out_lines if not r["content"])), "checks": checks,
       "all_checks_ok": all(c["n_items_frame"] == c["n_items_replay"] and c["zi_match"] for c in checks.values())}
(OUT / "items" / "items_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
log(f"wrote {len(out_lines)} items; all checks ok: {man['all_checks_ok']}; empty content {man['n_empty_content']} ({time.time()-t0:.0f}s)")
