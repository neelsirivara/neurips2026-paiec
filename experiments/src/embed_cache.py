"""Precompute sentence embeddings of the real-testbed items (experiments only).

The shipped predictor uses TF-IDF and never loads a neural model.  This script embeds every
item text of the local real testbed once with ``sentence-transformers/all-MiniLM-L6-v2``
(read from the local Hugging Face cache, offline) so that the harness can evaluate an
embedding-based similarity component under an otherwise identical predictor
(``model.EMBED_LOOKUP``).  Keys are SHA-1 digests of ``model._item_text(item)``.

    python embed_cache.py            # writes experiments/results/embeddings_real_testbed.npz
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402

OUT = HERE.parent / "results" / "embeddings_real_testbed.npz"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_embed_lookup(path=OUT):
    """Return a callable text -> list[float] (unit norm) or None, backed by the npz file."""
    data = np.load(path)
    keys = list(data["keys"])
    mat = data["vectors"].astype(float)
    index = {k: i for i, k in enumerate(keys)}

    def lookup(text: str):
        i = index.get(_key(text))
        return None if i is None else mat[i].tolist()

    lookup.n = len(keys)  # type: ignore[attr-defined]
    return lookup


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import real_testbed
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(int(os.environ.get("PAIEC_EMBED_THREADS", "2")))
    texts = {}
    for subject, sid, bid, items, meta in real_testbed.load_pairs():
        for it, _ in items:
            t = model._item_text(dict(it, benchmark_id=bid))
            texts[_key(t)] = t
    keys = sorted(texts)
    print(f"{len(keys)} unique item texts")
    t0 = time.time()
    st = SentenceTransformer(MODEL_ID, device="cpu")
    vecs = st.encode([texts[k] for k in keys], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    np.savez_compressed(OUT, keys=np.array(keys), vectors=vecs.astype(np.float32))
    print(f"wrote {OUT} {vecs.shape} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
