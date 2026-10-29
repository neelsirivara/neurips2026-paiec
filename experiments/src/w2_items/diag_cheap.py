"""W2 step 0b (diagnostic, exploratory): within-benchmark predictability of the empirical item effect from cheap
text features (log length, digit fraction, option markers, newlines, question marks), infinite-label ceiling.
LOO ridge regression within each benchmark of the INV + AS items (deep_dive_items.jsonl).  Same effect
definition as diag_ceiling.py.  Writes experiments/results/w2_items/diag_cheap.json
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_common import *  # noqa: F401,F403

F, folds = load()
rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
NI = len(F.item_dir)
S_i = np.bincount(F.row_item, weights=F.row_y - rate[row_pair], minlength=NI)
C_i = np.bincount(F.row_item, minlength=NI).astype(float)
eff = np.where(C_i >= 5, S_i / np.maximum(C_i, 1), np.nan)
opt = re.compile(r"(?m)(^|\s)\(?[A-Ja-j][.):]\s")
by_dir = {}
for l in open(CACHE / "deep_dive_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if not np.isnan(eff[it["idx"]]):
        by_dir.setdefault(it["dir"], []).append(it)
out = {}
for d, its in sorted(by_dir.items()):
    if len(its) < 30:
        continue
    e = np.array([eff[it["idx"]] for it in its])
    e = e - e.mean()
    ss = float(np.sum(e ** 2))
    Z = np.array([[np.log1p(it["content_len"]), sum(c.isdigit() for c in it["content"]) / (len(it["content"]) + 1),
                   np.log1p(len(opt.findall(it["content"]))), np.log1p(it["content"].count("\n")),
                   np.log1p(it["content"].count("?"))] for it in its])
    sd = Z.std(0)
    Z = (Z[:, sd > 1e-9] - Z[:, sd > 1e-9].mean(0)) / sd[sd > 1e-9]
    rec = {"n": len(its)}
    for lam in (1.0, 10.0, 100.0):
        if Z.shape[1] == 0:
            rec[f"r2_l{lam:g}"] = 0.0
            continue
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        Hm = Z @ np.linalg.solve(A, Z.T)
        pred = Hm @ e
        loo = (pred - np.diag(Hm) * e) / (1 - np.diag(Hm))
        rec[f"r2_l{lam:g}"] = float(1 - np.sum((e - loo) ** 2) / ss)
    out[d] = rec
agg = {k: {"bench_mean": float(np.mean([r[k] for r in out.values()])), "median": float(np.median([r[k] for r in out.values()]))}
       for k in ("r2_l1", "r2_l10", "r2_l100")}
log(json.dumps(agg))
(RES / "diag_cheap.json").write_text(json.dumps({"per_bench": out, "aggregate": agg}, indent=1), encoding="utf-8")
