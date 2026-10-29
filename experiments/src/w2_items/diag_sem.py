"""W2-sem step 0 (diagnostic, infinite labels; INV + AS items): within-benchmark R^2 of the empirical item effect for
semantic kernels (sentence embeddings), the char-4-gram kernel and the categories, alone and combined.

    python -I diag_sem.py TAG [TAG ...]       (TAG = minilm: round4_cache/w2_emb_<TAG>_all.npz, INV + AS items)

Same effect definition and LOO smoothers as diag_ceiling.py.  Semantic kernels: weights cos^q (cos clipped at 0) over
all other items of the benchmark, or over the k nearest only.  Combination: per benchmark, least squares of the effect
on the LOO predictions of (categories, char kernel q2, semantic kernel) - in-sample over 3 coefficients (small
optimistic bias, the same for every row of the table).  Writes experiments/results/w2_items/diag_sem.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_common import *  # noqa: F401,F403

tags = sys.argv[1:] or ["minilm"]
t0 = time.time()
F, folds = load()
rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
NI = len(F.item_dir)
S_i = np.bincount(F.row_item, weights=F.row_y - rate[row_pair], minlength=NI)
C_i = np.bincount(F.row_item, minlength=NI).astype(float)
eff = np.where(C_i >= 5, S_i / np.maximum(C_i, 1), np.nan)
recs = {}
for l in open(CACHE / "w2_items_all.jsonl", encoding="utf-8"):
    r = json.loads(l)
    recs[r["idx"]] = r
emb = {}
man = json.loads((RES.parent / "deep_dive" / "items" / "items_manifest.json").read_text(encoding="utf-8"))
diag_dirs = set(man["inv_dirs"]) | set(man["as_dirs_seeded_order"])
for tag in tags:
    z = np.load(CACHE / f"w2_emb_{tag}_all.npz")
    emb[tag] = {int(i): v for i, v in zip(z["idx"], z["vecs"].astype(np.float32)) if recs[int(i)]["dir"] in diag_dirs}
by_dir = {}
for idx in emb[tags[0]]:
    if not np.isnan(eff[idx]):
        by_dir.setdefault(recs[idx]["dir"], []).append(idx)


def loo_kernel(Sm, e, q, lam, k=None):
    Sm = np.clip(Sm, 0, 1)
    np.fill_diagonal(Sm, 0.0)
    W = Sm ** q
    if k is not None and k < W.shape[1] - 1:
        thr = -np.sort(-W, axis=1)[:, k - 1:k]
        W = np.where(W >= thr, W, 0.0)
    return (W @ e) / (W.sum(1) + lam)


def r2(e, p, ss):
    return float(1 - np.sum((e - p) ** 2) / ss)


out = {}
for d, its in sorted(by_dir.items()):
    if len(its) < 30:
        continue
    e = np.array([eff[i] for i in its])
    e = e - e.mean()
    ss = float(np.sum(e ** 2))
    if ss <= 0:
        continue
    rec = {"n": len(its)}
    # categories (as diag_ceiling)
    kv = [parse_features(recs[i]["features"]) for i in its]
    keys = sorted({k for p in kv for k, _ in p})
    cat = np.zeros(len(its))
    preds = []
    for kk in keys:
        vals = [dict(p).get(kk) for p in kv]
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
        cat = P @ beta
    rec["r2_cat"] = r2(e, cat, ss)
    Xc = char_tfidf_matrix([recs[i]["text"] for i in its])
    Sc = (Xc @ Xc.T).toarray()
    chr_p = loo_kernel(Sc.copy(), e, 2, 2.0)
    rec["r2_char_q2"] = r2(e, chr_p, ss)
    for tag in tags:
        V = np.stack([emb[tag][i] for i in its])
        Ss = V @ V.T
        rec[f"{tag}_mean_cos"] = float((Ss.sum() - np.trace(Ss)) / max(len(its) * (len(its) - 1), 1))
        best = None
        for q in (2, 8, 16, 32, 64):
            for lam in (0.5, 2.0):
                p = loo_kernel(Ss.copy(), e, q, lam)
                rec[f"r2_{tag}_q{q}_l{lam:g}"] = r2(e, p, ss)
        for k in (5, 10, 20):
            p = loo_kernel(Ss.copy(), e, 8, 0.5, k=k)
            rec[f"r2_{tag}_q8_l0.5_top{k}"] = r2(e, p, ss)
        sem_p = loo_kernel(Ss.copy(), e, 16, 0.5)
        for name, cols in (("cat+char", [cat, chr_p]), (f"cat+{tag}", [cat, sem_p]), (f"char+{tag}", [chr_p, sem_p]),
                           (f"cat+char+{tag}", [cat, chr_p, sem_p])):
            P = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(P, e, rcond=None)
            rec[f"r2_comb_{name}"] = r2(e, P @ beta, ss)
    out[d] = rec
    log(f"{d:30s} n {len(its):5d} cat {rec['r2_cat']:+.3f} char {rec['r2_char_q2']:+.3f} " +
        " ".join(f"{t}(q16) {rec[f'r2_{t}_q16_l0.5']:+.3f}" for t in tags))
agg = {}
for k in next(iter(out.values())):
    if not k.startswith("r2"):
        continue
    v = np.array([out[d][k] for d in out if k in out[d]])
    w = np.array([out[d]["n"] for d in out if k in out[d]], float)
    agg[k] = {"bench_mean": float(v.mean()), "median": float(np.median(v)), "item_weighted": float((w * v).sum() / w.sum())}
for k in sorted(agg, key=lambda k_: -agg[k_]["bench_mean"])[:40]:
    log(f"  {k:36s} bench-mean {agg[k]['bench_mean']:+.4f} median {agg[k]['median']:+.4f} item-w {agg[k]['item_weighted']:+.4f}")
(RES / "diag_sem.json").write_text(json.dumps({"tags": tags, "per_bench": out, "aggregate": agg}, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
