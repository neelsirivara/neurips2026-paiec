"""W2-sem functional test of a built candidate with the REAL embedding model (laptop CPU, local HF cache).

    python -I w2s_functional.py CANDIDATE_DIR [--out JSON]

1. The deployed embedding (_w2s_compute) equals w2_embed's training-cache embedding (embed_batch) on 300 release
   texts after float16 rounding (max |diff|), and equals the cached training vectors in round4_cache.
2. On a synthetic global list (8 subjects x 4 benchmarks x 60 items, budgets 0, 1, 7, 31, release texts): the budget-0
   calls queue texts, the semantic tables are complete at budgets >= 1, predictions differ from the W2 char kernel,
   every probability is valid, timings (first call, mean / max per call) and that a fresh process with the disk cache
   from the previous one reproduces the predictions (cache reuse; max |diff|).
3. Fallbacks: with the model unavailable (PAIEC_W2S_DISABLE=1 in a subprocess, and with a model id that does not
   exist) predictions equal the W2 char-kernel candidate's exactly.
"""
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
cdir = Path(args[0]).resolve()
out = Path(args[args.index("--out") + 1]) if "--out" in args else None
ROOT = HERE.parents[2]
CACHE = HERE.parents[3] / "_downloads" / "measurement_db_payload" / "round4_cache"
W2_DIR = ROOT / "submission" / "candidates" / "w2_f05s1_k1q2"


def load(path, tag, env=None):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(texts, seed=0, n_subj=8, n_bench=4, n_items=60):
    rng = random.Random(seed)
    subj = [{"normalized_name": f"m{i}", "provider": "p", "release_date": "2025-06-01", "access_date": "", "harness": "",
             "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""} for i in range(n_subj)]
    items = {b: [{"benchmark_id": f"bench_{b}", "item_content": texts[(b * n_items + k) % len(texts)],
                  "item_features": f"task=t{k % 5}", "interactors": ""} for k in range(n_items)] for b in range(n_bench)}
    pairs = []
    for i, s in enumerate(subj):
        b = i % n_bench
        idx = list(range(n_items))
        rng.shuffle(idx)
        pairs.append((s, b, idx[:30], idx[30:]))
    lists = {}
    for B in (1, 7, 31):
        lab = []
        for s, b, acq, ev in pairs:
            lab += [[[s, items[b][k]], int(rng.random() < 0.5)] for k in acq[:B]]
        lists[B] = lab
    targets = [[s, items[b][k]] for s, b, acq, ev in pairs for k in ev]
    return lists, targets


if __name__ == "__main__" and "--child" in args:
    # child process: predictions of the candidate at budgets 0, 1, 7, 31 -> stdout JSON
    texts = json.loads(Path(args[args.index("--texts") + 1]).read_text(encoding="utf-8"))
    m = load(cdir / "model.py", "cand")
    lists, targets = synthetic(texts)
    res = {"preds": {}, "t_first": {}, "t_mean_ms": {}, "t_max_ms": {}}
    t0 = time.perf_counter()
    res["preds"]["0"] = [m.predict(x, []) for x in targets]
    res["t_mean_ms"]["0"] = 1000 * (time.perf_counter() - t0) / len(targets)
    for B, lab in lists.items():
        m.xp_clear_cache()
        ts = []
        ps = []
        for x in targets:
            t1 = time.perf_counter()
            ps.append(m.predict(x, lab))
            ts.append(time.perf_counter() - t1)
        res["preds"][str(B)] = ps
        res["t_first"][str(B)] = ts[0]
        res["t_mean_ms"][str(B)] = 1000 * sum(ts[1:]) / max(len(ts) - 1, 1)
        res["t_max_ms"][str(B)] = 1000 * max(ts[1:])
        st = m._W2_CACHE["entry"][1]
        res.setdefault("sem_tables", {})[str(B)] = (sum(1 for tb in st["w2_kern"].values() if "E" in tb), len(st["w2_kern"]))
    res["state"] = m._W2S["state"] if hasattr(m, "_W2S") else None
    res["sync_s"] = m._W2S["sync_s"] if hasattr(m, "_W2S") else None
    print("@@JSON@@" + json.dumps(res))
    raise SystemExit(0)

import numpy as np  # noqa: E402
sys.path.insert(0, str(HERE))
recs = [json.loads(l) for i, l in enumerate(open(CACHE / "w2_items_all.jsonl", encoding="utf-8")) if i % 97 == 0][:600]
texts = [r["text"] for r in recs if r["text"]][:400]
tmp = Path(tempfile.mkdtemp(prefix="w2s_func_"))
(tmp / "texts.json").write_text(json.dumps(texts), encoding="utf-8")
report = {"candidate": str(cdir)}
# 1. embedding identity
cand = load(cdir / "model.py", "cand_main")
assert cand._w2s_ready(wait=True), "model did not load"
import w2_feat as WF  # noqa: E402
etexts = [WF.embed_text(t) for t in texts[:300]]
dep = np.stack(cand._w2s_compute(etexts))
os.environ["W2S_DUMMY"] = "1"
sys.argv = ["w2_embed.py", "minilm", "diag"]
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True).eval()
ref = []
order = np.argsort([len(t) for t in etexts], kind="stable")
refm = np.zeros((len(etexts), 384), np.float32)
for s in range(0, len(order), 32):
    j = order[s:s + 32]
    enc = tok([etexts[k] or " " for k in j], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = mdl(**enc).last_hidden_state
    msk = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    v = torch.nn.functional.normalize(((h * msk).sum(1) / msk.sum(1).clamp(min=1e-9)).float(), dim=-1).numpy()
    refm[j] = v
refm = refm.astype(np.float16).astype(np.float32)
report["embed_max_abs_diff_vs_training_code"] = float(np.max(np.abs(dep - refm)))
# vs the cached training vectors (if the all-items cache exists)
allp = CACHE / "w2_emb_minilm_all.npz"
if allp.exists():
    z = np.load(allp)
    pos = {int(i): k for k, i in enumerate(z["idx"])}
    ids = [r["idx"] for r in recs if r["text"]][:300]
    cv = np.stack([z["vecs"][pos[i]].astype(np.float32) for i in ids])
    report["embed_max_abs_diff_vs_cache"] = float(np.max(np.abs(dep - cv)))
    report["cos_min_vs_cache"] = float(np.min(np.sum(dep * cv, 1)))
# 2. synthetic lists in fresh processes (twice: the second reuses the disk cache)
env = dict(os.environ)
env.pop("PAIEC_W2S_DISABLE", None)
cache_dir = cdir / ".paiec_w2s"
shutil.rmtree(cache_dir, ignore_errors=True)


def child(path, extra_env=None):
    e = dict(env)
    if extra_env:
        e.update(extra_env)
    p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), str(path), "--child", "--texts", str(tmp / "texts.json")],
                       capture_output=True, text=True, env=e)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return json.loads(line[0][8:])


r1 = child(cdir)
r2 = child(cdir)
report["run1"] = {k: r1[k] for k in ("t_first", "t_mean_ms", "t_max_ms", "sem_tables", "state", "sync_s")}
report["run2_cache_reuse"] = {k: r2[k] for k in ("t_first", "t_mean_ms", "t_max_ms", "sem_tables", "state", "sync_s")}
report["run2_max_abs_diff"] = max(abs(a - b) for B in r1["preds"] for a, b in zip(r1["preds"][B], r2["preds"][B]))
report["all_valid"] = all(0.0 <= p <= 1.0 and p == p for B in r1["preds"] for p in r1["preds"][B])
# 3. fallbacks
w2 = child(W2_DIR)
off = child(cdir, {"PAIEC_W2S_DISABLE": "1"})
report["disabled_equals_w2"] = all(a == b for B in w2["preds"] for a, b in zip(w2["preds"][B], off["preds"][B]))
report["sem_differs_from_w2_n"] = sum(1 for B in w2["preds"] for a, b in zip(w2["preds"][B], r1["preds"][B]) if abs(a - b) > 1e-9)
report["w2_timings"] = {k: w2[k] for k in ("t_first", "t_mean_ms", "t_max_ms")}
bad = tmp / "badmodel"
shutil.copytree(cdir, bad, ignore=shutil.ignore_patterns(".paiec_w2s"))
src = (bad / "model.py").read_text(encoding="utf-8")
(bad / "model.py").write_text(src.replace('W2S_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"', 'W2S_MODEL_ID = "nonexistent-org/no-such-model"'), encoding="utf-8")
nb = child(bad)
report["missing_model_equals_w2"] = all(a == b for B in w2["preds"] for a, b in zip(w2["preds"][B], nb["preds"][B]))
report["missing_model_state"] = nb["state"]
shutil.rmtree(cache_dir, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(report, indent=1))
if out:
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
