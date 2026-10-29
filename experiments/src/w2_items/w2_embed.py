"""W2-sem: sentence embeddings of the frame items (training release) -> round4_cache/w2_emb_<tag>_<subset>.npz

    python -I w2_embed.py MODEL SUBSET [--threads N] [--batch B]
MODEL: minilm (sentence-transformers/all-MiniLM-L6-v2, mean pooling) or bge (BAAI/bge-small-en-v1.5, CLS pooling);
SUBSET: diag (the INV + AS items of deep_dive/items/items_manifest.json) or all (every item of w2_items_all.jsonl).
Text: w2_feat.embed_text of the stored normalised text; max 256 tokens; vectors L2-normalised, stored float16 with the
frame item index.  The same embed_batch is used at scoring time (w2_block).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w2_feat as WF  # noqa: E402

args = sys.argv[1:]
tag, subset = args[0], args[1]
threads = int(args[args.index("--threads") + 1]) if "--threads" in args else 2
batch = int(args[args.index("--batch") + 1]) if "--batch" in args else 32
shard, nshard = (int(x) for x in args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(threads)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

torch.set_num_threads(threads)
DL = HERE.parents[3] / "_downloads"
CACHE = DL / "measurement_db_payload" / "round4_cache"
MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", None),
          "bge": ("BAAI/bge-small-en-v1.5", "cls", DL / "hf_bge_small_en_v1.5")}
MAX_LEN = 256


def embed_batch(tok, model, texts, pooling, device="cpu"):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    if pooling == "cls":
        v = out[:, 0]
    else:
        m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
    v = torch.nn.functional.normalize(v.float(), dim=-1)
    return v.cpu().numpy()


repo, pooling, local = MODELS[tag]
src = str(local) if local else repo
tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
model = AutoModel.from_pretrained(src, local_files_only=True).eval()
recs = [json.loads(l) for l in open(CACHE / "w2_items_all.jsonl", encoding="utf-8")]
if subset == "diag":
    man = json.loads((HERE.parents[1] / "results" / "deep_dive" / "items" / "items_manifest.json").read_text(encoding="utf-8"))
    keep = set(man["inv_dirs"]) | set(man["as_dirs_seeded_order"])
    recs = [r for r in recs if r["dir"] in keep]
texts = [WF.embed_text(r["text"]) for r in recs]
idx = np.array([r["idx"] for r in recs], dtype=np.int64)
del recs
order = np.argsort([len(t) for t in texts], kind="stable")
order = order[shard::nshard]          # shards interleave the length order (balanced work)
vecs = np.zeros((len(texts), 384), dtype=np.float32)
t0 = time.time()
for s in range(0, len(order), batch):
    j = order[s:s + batch]
    vecs[j] = embed_batch(tok, model, [texts[k] or " " for k in j], pooling)
    if (s // batch) % 50 == 0:
        print(time.strftime("%H:%M:%S"), f"{s + len(j)}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
empty = np.array([not t for t in texts])
vecs[empty] = 0.0
out = CACHE / (f"w2_emb_{tag}_{subset}.npz" if nshard == 1 else f"w2_emb_{tag}_{subset}_shard{shard}of{nshard}.npz")
sel = np.sort(order)
np.savez(out, idx=idx[sel], vecs=vecs[sel].astype(np.float16), empty=empty[sel])
print(time.strftime("%H:%M:%S"), f"wrote {out}: {len(idx)} items, {int(empty.sum())} empty ({time.time() - t0:.0f}s)", flush=True)
