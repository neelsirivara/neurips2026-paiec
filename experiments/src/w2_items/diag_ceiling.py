"""W2 step 0 (diagnostic, exploratory, not part of any gate): how much of the within-benchmark variance of the
empirical item effect is predictable, with infinite labels, from (a) item_features categories and (b) text
similarity of other items of the same benchmark?  Uses the already-extracted INV + AS items
(_downloads/measurement_db_payload/round4_cache/deep_dive_items.jsonl).

    python -I diag_ceiling.py

Empirical item effect: eff_i = mean over the release rows of item i of (y - pair rate), items with >= 5 rows.
(a) LOO prediction of eff_i by the shrunk mean of the other items with the same (key, value), summed over keys
    by a per-benchmark least squares on the LOO category means (in-sample over 1-3 coefficients; tiny bias).
(b) LOO kernel smoother over char-4-gram TF-IDF cosine (idf within benchmark), weights cos^q.
Prints the per-benchmark R^2 (1 - SSE / SS around the benchmark mean) and the item-weighted average.
Writes experiments/results/w2_items/diag_ceiling.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_common import *  # noqa: F401,F403

t0 = time.time()
F, folds = load()
rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
NI = len(F.item_dir)
res_ = F.row_y - rate[row_pair]
S_i = np.bincount(F.row_item, weights=res_, minlength=NI)
C_i = np.bincount(F.row_item, minlength=NI).astype(float)
eff = np.where(C_i >= 5, S_i / np.maximum(C_i, 1), np.nan)
items = [json.loads(l) for l in open(CACHE / "deep_dive_items.jsonl", encoding="utf-8")]
by_dir = {}
for it in items:
    if not np.isnan(eff[it["idx"]]):
        by_dir.setdefault(it["dir"], []).append(it)
out = {}
for d, its in sorted(by_dir.items()):
    if len(its) < 30:
        continue
    e = np.array([eff[it["idx"]] for it in its])
    e = e - e.mean()
    ss = float(np.sum(e ** 2))
    if ss <= 0:
        continue
    rec = {"n": len(its), "sd_eff": float(e.std())}
    # (a) categories
    kv = [parse_features(it["features"]) for it in its]
    keys = sorted({k for p in kv for k, _ in p})
    preds = []
    for k in keys:
        vals = [dict(p).get(k) for p in kv]
        grp = {}
        for j, v in enumerate(vals):
            if v is not None:
                grp.setdefault(v, []).append(j)
        if len(grp) < 2:
            continue
        pk = np.zeros(len(its))
        for v, js in grp.items():
            s = e[js].sum()
            for j in js:
                pk[j] = (s - e[j]) / (len(js) - 1 + 3.0)
        preds.append(pk)
    if preds:
        P = np.column_stack(preds)
        beta, *_ = np.linalg.lstsq(P, e, rcond=None)
        rec["r2_feat"] = float(1 - np.sum((e - P @ beta) ** 2) / ss)
        rec["r2_feat_sum"] = float(1 - np.sum((e - P.sum(1)) ** 2) / ss)
    else:
        rec["r2_feat"] = rec["r2_feat_sum"] = 0.0
    # (b) text kernel
    texts = [it["content"] for it in its]
    Xm = char_tfidf_matrix(texts)
    Sm = (Xm @ Xm.T).toarray()
    np.fill_diagonal(Sm, 0.0)
    for q in (2, 4, 8, 16):
        W = Sm ** q
        for lam in (0.5, 2.0):
            pr = (W @ e) / (W.sum(1) + lam)
            rec[f"r2_text_q{q}_l{lam:g}"] = float(1 - np.sum((e - pr) ** 2) / ss)
    rec["mean_cos"] = float(Sm.sum() / max(len(its) * (len(its) - 1), 1))
    out[d] = rec
    log(f"{d:32s} n {len(its):5d} sd {rec['sd_eff']:.3f} feat {rec['r2_feat']:+.3f} text(q8,l2) {rec['r2_text_q8_l2']:+.3f} "
        f"meancos {rec['mean_cos']:.2f}")
agg = {}
ks = [k for k in next(iter(out.values())) if k.startswith("r2")]
for k in ks:
    w = np.array([out[d]["n"] for d in out], float)
    v = np.array([out[d][k] for d in out])
    agg[k] = {"item_weighted": float(np.sum(w * v) / w.sum()), "median": float(np.median(v)),
              "bench_mean": float(v.mean())}
log("aggregate: " + json.dumps(agg))
(RES / "diag_ceiling.json").write_text(json.dumps({"per_bench": out, "aggregate": agg}, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
