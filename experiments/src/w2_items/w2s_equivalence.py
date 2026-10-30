"""W2-sem: the candidate's stdlib/numpy model.py vs the simulator (w2s_sim.eval_grid) with the same constants.

    python -I w2s_equivalence.py NAME [--cdir DIR] [--n 24]

Like w2_equivalence.py (24 INV-tune test sets, binarised labels, item_content = "item <idx>", hand-set item logit and
char grams looked up by idx), and the embedding of an item is the training-cache vector of that idx (the candidate's
_w2s_embed_text / _w2s_vectors are redirected to the cache; the real model path is tested by w2s_functional.py).
Writes experiments/results/w2_items/candidates/NAME/equivalence.json
"""
import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2s_sim import *  # noqa: F401,F403

args = sys.argv[1:]
name = args[0]
NTS = int(args[args.index("--n") + 1]) if "--n" in args else 24
ROOT = Path(__file__).resolve().parents[3]
cdir = (Path(args[args.index("--cdir") + 1]) if "--cdir" in args else ROOT / "submission" / "candidates") / name
spec = importlib.util.spec_from_file_location("cand", cdir / "model.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
t0 = time.time()
W = World(with_grams=True)
attach_embeddings(W, "minilm")
F = W.F
recs = {}
for l in open(CACHE / "w2_items_all.jsonl", encoding="utf-8"):
    r = json.loads(l)
    recs[r["idx"]] = r
zi_of = {f"item {i}": float(z) for i, z in enumerate(F.item_zi)}
cand._item_logit = lambda it: zi_of.get(it.get("item_content"), 0.0)
cand._w2_text_grams = lambda content: WF.text_grams(recs[int(str(content)[5:])]["text"]) if str(content).startswith("item ") else {}
cand._w2s_embed_text = lambda content: str(content)


def _vecs(texts, allow_compute):
    out = []
    for t in texts:
        i = int(t[5:])
        r = W.row_of[i]
        out.append(W.Emb[r].astype(np.float32) if r >= 0 else None)
    return out


cand._w2s_vectors = _vecs
cand._w2s_ready = lambda wait: True
cand._W2S["np"] = np
cand._W2S_SKIP = False
setting = {"tau_f": cand.W2_TAU_FEAT, "share": cand.W2_FEAT_SHARE, "kern": cand.W2S_KIND, "q_c": cand.W2_KERN_Q,
           "q_s": cand.W2S_Q_S, "tau_k": cand.W2S_TAU_KERN, "topk": cand.W2S_TOPK or None, "m_low": cand.W2S_M_LOW,
           "label": "cand"}
S_all, ts_all = build_scenario(W, "INV", (0, 1))
pick = sorted(int(x) for x in np.random.RandomState(5).choice(len(ts_all), NTS, replace=False))
sub = [ts_all[i] for i in pick]
S = X.Streams(F, sub)
for b in X.BUDGETS[1:]:
    S.acq[b]["y"] = (S.acq[b]["y"] >= 0.5).astype(float)
preds = {}
eval_grid(W, S, sub, [setting], preds_out=preds)
sc = {}


def subj(p):
    if p not in sc:
        sc[p] = json.loads(F.subj_keys[int(F.pair_subj[p])])
    return sc[p]


def item(i):
    return {"benchmark_id": F.dirs[int(F.item_dir[i])], "item_content": f"item {int(i)}",
            "item_features": recs[int(i)]["features"] if int(i) in recs else "", "interactors": ""}


max_diff, n, n_sem = 0.0, 0, 0
for b in X.BUDGETS[1:]:
    E = S.acq[b]
    for t in range(len(sub)):
        cand.xp_clear_cache()
        me, mv = E["ts"] == t, S.ev["ts"] == t
        labeled = [[[subj(int(p)), item(int(i))], 1 if y >= 0.5 else 0] for p, i, y in zip(E["pair"][me], E["item"][me], E["y"][me])]
        ps = np.array([cand.predict_cross_pair([subj(int(p)), item(int(i))], labeled) for p, i in zip(S.ev["pair"][mv], S.ev["item"][mv])])
        st = cand._W2_CACHE["entry"][1]
        n_sem += sum(1 for tb in st["w2_kern"].values() if "E" in tb)
        idx = np.where(mv)[0]
        max_diff = max(max_diff, float(np.max(np.abs(ps - preds["cand"][b][idx]))))
        n += len(idx)
res = {"candidate": name, "setting": {k: v for k, v in setting.items() if k != "label"}, "n_testsets": len(sub),
       "n_predictions": n, "n_semantic_tables": n_sem, "max_abs_diff_vs_numpy": max_diff, "passes_1e-3": bool(max_diff < 1e-3),
       "seconds": time.time() - t0}
o = RES / "candidates" / name if "--cdir" not in args else cdir.parent / "records" / name
o.mkdir(parents=True, exist_ok=True)
(o / "equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
