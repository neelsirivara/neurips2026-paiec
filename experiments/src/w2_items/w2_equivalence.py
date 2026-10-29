"""Check that a W2 candidate's stdlib model.py reproduces the numpy simulator (w2_sim.eval_w2) with its constants.

    python -I w2_equivalence.py NAME [--n 24] [--scen INV]

Loads submission/candidates/NAME/model.py as its own module and reads W2_TAU_FEAT, W2_FEAT_SHARE, W2_TAU_KERN,
W2_KERN_Q.  On NTS test sets of the scenario (tune folds 0 and 1; labels binarised as the evaluator's are) it compares
predict_cross_pair on dict inputs (benchmark_id = the benchmark's directory name, item_features = the release
string, item_content = "item <idx>"; the hand-set item logit and the character grams are looked up by that
content, so both sides use the release text) with eval_w2's predictions at budgets 1..31.  Also checks that the
candidate's _w2_text_grams / _w2_parse_features equal w2_feat's on 2,000 release items, and that with both
strengths set to 0 the candidate reproduces xp_level_m050's model.py exactly.
Writes experiments/results/w2_items/candidates/NAME/equivalence.json
"""
import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_sim import *  # noqa: F401,F403

args = sys.argv[1:]
name = args[0]
NTS = int(args[args.index("--n") + 1]) if "--n" in args else 24
scen = args[args.index("--scen") + 1] if "--scen" in args else "INV"
ROOT = Path(r"C:\Users\neels\Downloads\gen_papers_2\neurips_challenges\paiec")
cdir = (Path(args[args.index("--cdir") + 1]) if "--cdir" in args else ROOT / "submission" / "candidates") / name


def load_mod(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cand = load_mod(cdir / "model.py", f"cand_{name}")
base = load_mod(ROOT / "submission" / "candidates" / "xp_level_m050" / "model.py", "m050")
t0 = time.time()
W = World(with_grams=cand.W2_TAU_KERN > 0)
F = W.F
recs = {}
for l in open(CACHE / "w2_items_all.jsonl", encoding="utf-8"):
    r = json.loads(l)
    recs[r["idx"]] = r
# featurisation identity on a sample
rng = np.random.RandomState(11)
keys_all = sorted(recs)
sample = [keys_all[i] for i in rng.choice(len(keys_all), 2000, replace=False)]
feat_ok = all(cand._w2_parse_features(recs[i]["features"]) == WF.parse_features(recs[i]["features"]) for i in sample)
gram_ok = all(cand._w2_text_grams(recs[i]["text"]) == WF.text_grams(recs[i]["text"]) for i in sample)
cand._W2_GRAM_CACHE.clear()
zi_of = {f"item {i}": float(z) for i, z in enumerate(F.item_zi)}
cand._item_logit = lambda it: zi_of.get(it.get("item_content"), 0.0)
base._item_logit = cand._item_logit
_orig_grams = cand._w2_text_grams
cand._w2_text_grams = lambda content: WF.text_grams(recs[int(str(content)[5:])]["text"]) if str(content).startswith("item ") else {}
tau_f, share, tau_k, q = cand.W2_TAU_FEAT, cand.W2_FEAT_SHARE, cand.W2_TAU_KERN, cand.W2_KERN_Q
S_all, ts_all = build_scenario(W, scen, (0, 1))
pick = sorted(int(x) for x in np.random.RandomState(5).choice(len(ts_all), NTS, replace=False))
sub = [ts_all[i] for i in pick]
S = X.Streams(F, sub)
for b in X.BUDGETS[1:]:
    S.acq[b]["y"] = (S.acq[b]["y"] >= 0.5).astype(float)
_, preds = eval_w2(W, S, sub, tau_f=tau_f, share=share, tau_k=tau_k, q=q, return_preds=True)
_, preds0 = eval_w2(W, S, sub, return_preds=True)
max_diff, max_diff_m050, n = 0.0, 0.0, 0
n_changed = 0
sc = {}


def subj(p):
    if p not in sc:
        sc[p] = json.loads(F.subj_keys[int(F.pair_subj[p])])
    return sc[p]


def item(i):
    return {"benchmark_id": F.dirs[int(F.item_dir[i])], "item_content": f"item {int(i)}",
            "item_features": recs[int(i)]["features"] if int(i) in recs else "", "interactors": ""}


for b in X.BUDGETS[1:]:
    E = S.acq[b]
    for t in range(len(sub)):
        cand.xp_clear_cache()
        base.xp_clear_cache()
        me, mv = E["ts"] == t, S.ev["ts"] == t
        labeled = [[[subj(int(p)), item(int(i))], 1 if y >= 0.5 else 0] for p, i, y in zip(E["pair"][me], E["item"][me], E["y"][me])]
        xs = [[subj(int(p)), item(int(i))] for p, i in zip(S.ev["pair"][mv], S.ev["item"][mv])]
        ps = np.array([cand.predict_cross_pair(x, labeled) for x in xs], float)
        p0 = np.array([base.predict_cross_pair(x, labeled) for x in xs], float)
        idx = np.where(mv)[0]
        max_diff = max(max_diff, float(np.max(np.abs(ps - preds[b][idx]))))
        max_diff_m050 = max(max_diff_m050, float(np.max(np.abs(p0 - preds0[b][idx]))))
        n_changed += int(np.sum(np.abs(ps - p0) > 1e-12))
        n += len(idx)
# strengths 0 -> exactly m050
cand.W2_TAU_FEAT, cand.W2_TAU_KERN = 0.0, 0.0
exact = True
E = S.acq[7]
for t in range(min(len(sub), 6)):
    cand.xp_clear_cache()
    base.xp_clear_cache()
    me, mv = E["ts"] == t, S.ev["ts"] == t
    labeled = [[[subj(int(p)), item(int(i))], 1 if y >= 0.5 else 0] for p, i, y in zip(E["pair"][me], E["item"][me], E["y"][me])]
    for p, i in zip(S.ev["pair"][mv], S.ev["item"][mv]):
        x = [subj(int(p)), item(int(i))]
        exact &= cand.predict(x, labeled) == base.predict(x, labeled)
res = {"candidate": name, "scenario": scen, "constants": {"W2_TAU_FEAT": tau_f, "W2_FEAT_SHARE": share, "W2_TAU_KERN": tau_k,
                                                         "W2_KERN_Q": q},
       "n_testsets": len(sub), "n_predictions": n, "n_changed_vs_m050": n_changed, "max_abs_diff_vs_numpy": max_diff,
       "m050_max_abs_diff_vs_numpy": max_diff_m050, "passes_1e-3": bool(max_diff < 1e-3),
       "featurisation_identical": bool(feat_ok), "grams_identical": bool(gram_ok), "zero_strength_equals_m050": bool(exact),
       "seconds": time.time() - t0}
out = RES / "candidates" / name
out.mkdir(parents=True, exist_ok=True)
(out / "equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
