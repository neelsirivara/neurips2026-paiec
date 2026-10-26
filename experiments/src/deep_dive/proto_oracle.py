"""Deep dive step 4: upper bounds -- how much would perfect side information be worth inside the shipped XP model?

  bench_oracle   offset uses the true benchmark mean logit (computed on ALL eligible pairs of the benchmark,
                 test pairs included): the ceiling for any benchmark-difficulty prior (e.g. LLM-rated items)
  item_oracle_c  offset adds c * leave-pair-out item effect estimated from every OTHER pair's responses to the
                 same item in the full release: the ceiling for an item-difficulty estimator as good as the
                 empirical results of all other release subjects
  both           both of the above
Scenarios A and HARD (see proto_level.py).  Writes results/deep_dive/proto_oracle.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
NB = int(F.pair_dir.max()) + 1
n = F.pair_n.astype(float)
Lp = np.log((rate * n + 0.5) / (n - rate * n + 0.5))
bL = np.bincount(F.pair_dir[elig], weights=Lp[elig], minlength=NB) / np.maximum(np.bincount(F.pair_dir[elig], minlength=NB), 1)
bm = np.bincount(F.pair_dir[elig], weights=rate[elig], minlength=NB) / np.maximum(np.bincount(F.pair_dir[elig], minlength=NB), 1)
hard = {F.dirs[b] for b in range(len(F.dirs)) if np.bincount(F.pair_dir[elig], minlength=NB)[b] > 0 and bm[b] <= 0.40}
# leave-pair-out item effect on the logit scale: residual (y - rate_pair) summed over other pairs, scaled by 1/(r(1-r))
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
rp = rate[row_pair]
w = 1.0 / np.clip(rp * (1 - rp), 0.05, None)
res_ = (F.row_y - rp)
NI = len(F.item_dir)
S_i = np.bincount(F.row_item, weights=res_ * w, minlength=NI)
W_i = np.bincount(F.row_item, weights=w * 0 + 1.0, minlength=NI)


def item_eff(T, k=2.0):
    # T rows are frame rows; leave out the row's own pair (each pair answers an item at most once in the frame)
    r = T["row"]
    s = S_i[T["item"]] - (res_[r] * w[r])
    c = W_i[T["item"]] - 1.0
    return np.clip(s / (c + k), -4, 4)

def sub_folds(keep):
    return [[d for d in f if d in keep] for f in folds]

res = {"scenarios": {}}
for sname, fl in (("A", folds), ("HARD", sub_folds(hard))):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    cl = S.unit_bench
    out = {}
    out["shipped"], _ = eval_xp(S, ts)
    # bench oracle: subject logit = true benchmark mean logit (- mean item logit of the pair so the offset is centred)
    zi_pair = np.add.reduceat(F.item_zi[F.row_item], F.pair_start[:-1]) / F.pair_n
    zb = bL[F.pair_dir] - zi_pair
    out["bench_oracle"], _ = eval_xp(S, ts, subj_logit_by_fold=[zb] * 5, cap=6.0)
    for c in (0.5, 1.0):
        out[f"item_oracle_{c:g}"], _ = eval_xp(S, ts, extra_offset=lambda T, c=c: c * item_eff(T))
    out["both_1"], _ = eval_xp(S, ts, subj_logit_by_fold=[zb] * 5, cap=6.0, extra_offset=lambda T: item_eff(T))
    orc = np.bincount(S.ev["unit"], weights=(S.unit_rate[S.ev["unit"]] - S.ev["y"]) ** 2, minlength=S.n_units) / S.unit_count
    rec = {"n_units": int(S.n_units), "oracle_rate": float(orc.mean()), "methods": {}}
    a0 = alc(out["shipped"])
    for k, br in out.items():
        r = {"alc": float(alc(br).mean()), "brier": [float(x) for x in br.mean(0)]}
        if k != "shipped":
            r["minus_shipped"] = paired(alc(br), a0, cl)
        rec["methods"][k] = r
        log(f"[{sname}] {k:16s} ALC {r['alc']:.4f} B {np.round(br.mean(0),4)}" + (f" diff {fmt(r['minus_shipped'])}" if k != 'shipped' else f" pair-rate oracle {orc.mean():.4f}"))
    res["scenarios"][sname] = rec
(OUT / "proto_oracle.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time()-t0:.0f}s)")
