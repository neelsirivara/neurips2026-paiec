"""Check that a candidate's stdlib model.py reproduces the numpy simulator with the same constants.

    python -I candidate_equivalence.py NAME

Loads submission/candidates/NAME/model.py as its own module, reads its SUBJECT_SHRINK, PRIOR_INTERCEPT and
XP_TAUS, and compares predict_cross_pair on dict inputs with xpair_sim's xp_fit_np / xp_predict_np (subject
logit = SUBJECT_SHRINK * shipped subject terms + PRIOR_INTERCEPT) on 24 config-A test sets at budgets 1..31,
as round 4's pre-registered equivalence check did for the shipped file (threshold 1e-3).
Writes experiments/results/deep_dive/candidates/NAME/equivalence.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import numpy as np  # noqa: E402
import xpair_sim as X  # noqa: E402
import model as shipped  # noqa: E402

name = sys.argv[1]
cdir = SRC.parents[1] / "submission" / "candidates" / name
spec = importlib.util.spec_from_file_location(f"cand_{name}", cdir / "model.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
t0 = time.time()
F = X.Frame(X.DEFAULT_PAYLOAD)
folds = X.load_folds()
taus = dict(cand.XP_TAUS)
zi_of = {f"item {i}": float(z) for i, z in enumerate(F.item_zi)}
cand._item_logit = lambda it: zi_of.get(it.get("item_content"), 0.0)
# shipped subject terms: F.pair_zs = 0.7 * z_terms (+ 0); the candidate uses SUBJECT_SHRINK * z_terms + PRIOR_INTERCEPT
z_terms = F.pair_zs / shipped.SUBJECT_SHRINK
subj_logit = cand.SUBJECT_SHRINK * z_terms + cand.PRIOR_INTERCEPT
tsets = X.make_testsets(F, folds, "A", R=300)
rng = np.random.RandomState(5)
pick = sorted(int(x) for x in rng.choice(len(tsets), 24, replace=False))
sub = [tsets[i] for i in pick]
S = X.Streams(F, sub)
for b in X.BUDGETS[1:]:
    S.acq[b]["y"] = (S.acq[b]["y"] >= 0.5).astype(float)


def with_o(T):
    T = dict(T)
    T["o"] = np.clip(subj_logit[T["pair"]] + T["zi"], -X.CAP, X.CAP)
    return T


max_diff, n = 0.0, 0
V = with_o(S.ev)
for b in X.BUDGETS[1:]:
    E = with_o(S.acq[b])
    fit = X.xp_fit_np(E, V, taus)
    p_np = X.xp_predict_np(E, V, fit, cand.XP_TAU_ITEM, cand.XP_PROBIT)
    me_all, mv_all = E["ts"], V["ts"]
    for t in range(len(sub)):
        cand.xp_clear_cache()
        me, mv = me_all == t, mv_all == t
        sc = {}
        subj = lambda p: sc.setdefault(p, json.loads(F.subj_keys[int(F.pair_subj[p])]))
        item = lambda i: {"benchmark_id": F.dirs[int(F.item_dir[i])], "item_content": f"item {int(i)}", "item_features": "",
                          "interactors": ""}
        labeled = [[[subj(int(p)), item(int(i))], 1 if y >= 0.5 else 0] for p, i, y in zip(E["pair"][me], E["item"][me], E["y"][me])]
        ps = np.array([cand.predict_cross_pair([subj(int(p)), item(int(i))], labeled) for p, i in zip(V["pair"][mv], V["item"][mv])], float)
        idx = np.where(mv)[0]
        max_diff = max(max_diff, float(np.max(np.abs(ps - p_np[idx]))))
        n += len(idx)
res = {"candidate": name, "taus": taus, "SUBJECT_SHRINK": cand.SUBJECT_SHRINK, "PRIOR_INTERCEPT": cand.PRIOR_INTERCEPT,
       "n_testsets": len(sub), "n_predictions": n, "max_abs_diff": max_diff, "passes_1e-3": bool(max_diff < 1e-3),
       "seconds": time.time() - t0}
out = SRC.parent / "results" / "deep_dive" / "candidates" / name
out.mkdir(parents=True, exist_ok=True)
(out / "equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
