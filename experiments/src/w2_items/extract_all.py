"""W2 step 1: recover item_features and item text for every frame item that a config-A test set can use.

    python -I extract_all.py

Same replay as deep_dive/items_extract.py (xpair_sim.cmd_build's item-index assignment, verified against the frame:
item count and hand-set item logit per benchmark), but for ALL benchmarks with eligible pairs.  Needed items = items
of rows with rank < 100 (config A caps pairs at 100 items) of pairs with >= 80 items.  Writes, outside the repository
(derived from the gated release, like deep_dive_items.jsonl):
  _downloads/measurement_db_payload/round4_cache/w2_items_all.jsonl  (idx, dir, features[:1000], text = normalised
      content head 1500 + tail 500 characters, content_len)
and experiments/results/w2_items/extract_manifest.json (counts and replay checks).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_common import *  # noqa: F401,F403
import hashlib

t0 = time.time()
F, folds = load()
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
need_mask = np.zeros(len(F.item_dir), bool)
m = (F.pair_n[row_pair] >= 80) & (F.row_rank < 100)
need_mask[F.row_item[m]] = True
dirs_needed = sorted({F.dirs[int(b)] for b in np.unique(F.item_dir[need_mask])})
log(f"needed items {int(need_mask.sum())} in {len(dirs_needed)} benchmarks")

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
try:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
except Exception:
    pass
data_dir = Path(X.DEFAULT_PAYLOAD).resolve() / "data"
checks = {}
n_written = 0
out_path = CACHE / "w2_items_all.jsonl"
fh = open(out_path, "w", encoding="utf-8")
for d in dirs_needed:
    di = F.dirs.index(d)
    item_idx = np.where(F.item_dir == di)[0]
    start = int(item_idx.min())
    assert np.all(item_idx == np.arange(start, start + len(item_idx))), d
    base = data_dir / d
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
    del resp, col, sid, iid, cond, y
    agg = df.groupby(["sid", "cond", "iid"], sort=False)["y"].mean().reset_index()
    del df
    inames = pq.ParquetFile(base / "items.parquet").schema_arrow.names
    ccol, fcol = X._first_col(inames, X.CONTENT_CANDIDATES), X._first_col(inames, X.FEATURE_CANDIDATES)
    hcol = "content_hash" if "content_hash" in inames else None
    it = pq.read_table(base / "items.parquet", columns=["item_id"] + [c for c in (ccol, fcol, hcol) if c])
    ids = pc.cast(it.column("item_id"), pa.string()).to_pylist()
    cont = pc.cast(it.column(ccol), pa.string()).to_pylist() if ccol else [None] * len(ids)
    lengths = [len(c) if c else 0 for c in cont]
    feats = pc.fill_null(pc.cast(it.column(fcol), pa.string()), "").to_pylist() if fcol else [""] * len(ids)
    hashes = pc.fill_null(pc.cast(it.column(hcol), pa.string()), "").to_pylist() if hcol else [""] * len(ids)
    del it
    if ccol and any(h == "" for h in hashes):
        hashes = [h or hashlib.blake2b((c or "").encode("utf-8"), digest_size=8).hexdigest() for h, c in zip(hashes, cont)]
    info = {i_: (int(L or 0), f_ or "", h_ or ("id:" + i_), j) for j, (i_, L, f_, h_) in enumerate(zip(ids, lengths, feats, hashes))}
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
            L, f_, h_, jj = info.get(iids[j], (0, "", "id:" + iids[j], -1))
            if h_ not in keys:
                keys[h_] = (f_, jj, L)
                order_keys.append(h_)
                zi_replay.append(X._hand_item_logit(f_, L))
    n_ok = len(order_keys) == len(item_idx)
    zi_ok = n_ok and bool(np.allclose(np.array(zi_replay), F.item_zi[item_idx], atol=1e-12))
    checks[d] = {"n_items_frame": int(len(item_idx)), "n_items_replay": len(order_keys), "zi_match": zi_ok}
    if not (n_ok and zi_ok):
        log(f"  MISMATCH {d}: {checks[d]}")
        continue
    nw = 0
    for k, h_ in enumerate(order_keys):
        idx = start + k
        if need_mask[idx]:
            f_, jj, L = keys[h_]
            text = WF.norm_text(cont[jj] if jj >= 0 else "")
            fh.write(json.dumps({"idx": idx, "dir": d, "features": str(f_)[:WF.FEAT_MAX_CHARS], "text": text,
                                 "content_len": int(L)}, ensure_ascii=False) + "\n")
            nw += 1
    n_written += nw
    checks[d]["n_written"] = nw
    del agg, info, keys, cont, feats, hashes
    log(f"  {d}: {len(order_keys)} items verified, {nw} written ({time.time() - t0:.0f}s)")
fh.close()
man = {"n_needed": int(need_mask.sum()), "n_written": n_written, "n_dirs": len(dirs_needed), "checks": checks,
       "all_checks_ok": all(c["n_items_frame"] == c["n_items_replay"] and c["zi_match"] for c in checks.values()),
       "out_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}
(RES / "extract_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
log(f"wrote {n_written} items; all checks ok: {man['all_checks_ok']} ({time.time() - t0:.0f}s)")
