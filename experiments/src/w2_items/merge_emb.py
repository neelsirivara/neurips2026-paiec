"""Merge the shards of w2_embed.py MODEL all --shard i/n into round4_cache/w2_emb_<MODEL>_all.npz.

    python -I merge_emb.py MODEL N
"""
import sys
from pathlib import Path
import numpy as np

tag, n = sys.argv[1], int(sys.argv[2])
CACHE = Path(__file__).resolve().parents[4] / "_downloads" / "measurement_db_payload" / "round4_cache"
parts = [np.load(CACHE / f"w2_emb_{tag}_all_shard{i}of{n}.npz") for i in range(n)]
idx = np.concatenate([p["idx"] for p in parts])
vecs = np.concatenate([p["vecs"] for p in parts])
empty = np.concatenate([p["empty"] for p in parts])
o = np.argsort(idx)
assert len(np.unique(idx)) == len(idx)
np.savez(CACHE / f"w2_emb_{tag}_all.npz", idx=idx[o], vecs=vecs[o], empty=empty[o])
nz = np.linalg.norm(vecs.astype(np.float32), axis=1)
print(f"merged {len(idx)} items; empty {int(empty.sum())}; norms of non-empty in [{nz[~empty].min():.4f}, {nz[~empty].max():.4f}]")
