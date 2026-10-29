"""Item-rating study, Amendment 6: the gate for the Qwen3-14B signals s1, s3 and smc.

    python -I eval_items_a6.py RATINGS.jsonl [--tag NAME] [--complete-only]

Same machinery as eval_items.py, with Amendment 6's rules:
  * unrated items get the fitted intercept b (mean imputation) for EVERY signal and scenario;
  * (a) and (a') use 98.33% intervals (Bonferroni over 3 signals); (b), (b') and (c) use 95%;
  * a signal passes if (a), (b), (a'), (b') and (c) all hold AND its INV gain vs the intercept-only control is
    >= 0.003 ALC;
  * --complete-only applies Amendment 5's partial-run rule (fully rated benchmarks only; >= 15 INV and >= 5 AS).
Writes experiments/results/deep_dive/items/eval_items_a6_<tag>.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *  # noqa: F401,F403
import hashlib
import rate_items as RI
import npeb as NP

args = sys.argv[1:]
rpath = Path(args[0])
tag = args[args.index("--tag") + 1] if "--tag" in args else rpath.stem
CACHE = Path(X.DEFAULT_PAYLOAD).resolve() / "round4_cache"
t0 = time.time()
F, folds = load()
man = json.loads((OUT / "items" / "items_manifest.json").read_text(encoding="utf-8"))
inv_dirs, as_dirs = set(man["inv_dirs"]), set(man["as_dirs_seeded_order"])
rat = {}
for l in open(rpath, encoding="utf-8"):
    r = json.loads(l)
    if r.get("s1") is not None and not r.get("error"):
        rat[r["key"]] = r
NI = len(F.item_dir)
SIG = {"s1": np.full(NI, np.nan), "s3": np.full(NI, np.nan), "smc": np.full(NI, np.nan)}
need_by_dir = {}
for l in open(CACHE / "deep_dive_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    need_by_dir.setdefault(it["dir"], []).append(it["idx"])
    r = rat.get(RI.prompt_key(RI.build_prompt(it["content"], it["features"])))
    if r is not None:
        for s in SIG:
            if r.get(s) is not None:
                SIG[s][it["idx"]] = r[s]
rated = ~np.isnan(SIG["s1"])
info = {"n_rated_items": int(rated.sum()), "n_mc_items": int((~np.isnan(SIG["smc"])).sum())}
if "--complete-only" in args:
    complete = {d for d, ix in need_by_dir.items() if all(rated[i] for i in ix)}
    inv_dirs, as_dirs = inv_dirs & complete, as_dirs & complete
    info.update(inv_complete=len(inv_dirs), as_complete=len(as_dirs), evaluable=len(inv_dirs) >= 15 and len(as_dirs) >= 5)
    if not info["evaluable"]:
        (OUT / "items" / f"eval_items_a6_{tag}.json").write_text(json.dumps({"info": info, "decision": "not evaluable"}, indent=1))
        raise SystemExit(f"not evaluable: {info}")
log(f"ratings {len(rat)}; {info}")
fod = fold_of_dir(F, folds)
item_fold = np.array([fod.get(int(d), -1) for d in F.item_dir])
dir_name = np.array(F.dirs)[F.item_dir]
in_study = np.isin(dir_name, list(inv_dirs | as_dirs))
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
row_ok = (F.pair_n[row_pair] >= 80) & in_study[F.row_item]
base_o_row = np.clip(-0.5 + F.item_zi[F.row_item], -X.CAP, X.CAP)


def fit_offset(Z, y, off, w, iters=30):
    Xm = np.column_stack([Z, np.ones(len(y))])
    beta = np.zeros(Xm.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + Xm @ beta)))
        H = (Xm * (w * p * (1 - p))[:, None]).T @ Xm + 1e-9 * np.eye(Xm.shape[1])
        step = np.linalg.solve(H, Xm.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


fits = {}
for name in ("b_only", "s1", "s3", "smc"):
    s = SIG.get(name)
    fits[name] = []
    for f in range(5):
        tr_items = (item_fold != f) & (item_fold >= 0) & in_study
        rows = row_ok & tr_items[F.row_item]
        it = F.row_item[rows]
        if s is None:
            Z = np.zeros((len(it), 0))
            mu, sd = 0.0, 1.0
        else:
            ok = ~np.isnan(s)
            mu, sd = float(np.mean(s[tr_items & ok])), float(np.std(s[tr_items & ok]) or 1.0)
            z = (s[it] - mu) / sd
            z[np.isnan(z)] = 0.0                      # mean imputation inside the fit, as at prediction time
            Z = z[:, None]
        b_of = F.item_dir[it]
        cnt = np.bincount(b_of, minlength=len(F.dirs)).astype(float)
        w = 1.0 / cnt[b_of]
        w *= len(w) / w.sum()
        fits[name].append({"mu": mu, "sd": sd, "beta": [float(x) for x in fit_offset(Z, F.row_y[rows], base_o_row[rows], w)]})
    log(f"fit {name}: " + "; ".join(str(np.round(fr["beta"], 3)) for fr in fits[name]))


def make_offset(name, ts_fold):
    s = SIG.get(name)

    def off(T):
        f = ts_fold[T["ts"]]
        out = np.zeros(len(T["ts"]))
        for k in range(5):
            m = f == k
            if not np.any(m):
                continue
            fr = fits[name][k]
            if s is None:
                out[m] = fr["beta"][-1]
            else:
                z = (s[T["item"][m]] - fr["mu"]) / fr["sd"]
                z[np.isnan(z)] = 0.0                  # unrated -> b
                out[m] = fr["beta"][0] * z + fr["beta"][-1]
        return out
    return off


def boot(diff, clusters, level, seed=0, n_boot=2000):
    u, inv = np.unique(clusters, return_inverse=True)
    S_, N = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(u), size=(n_boot, len(u)))
    st = np.sort(S_[idx].sum(1) / N[idx].sum(1))
    a = (1 - level) / 2
    return {"mean": float(diff.mean()), "lo": float(st[int(a * n_boot)]), "hi": float(st[int((1 - a) * n_boot) - 1]), "level": level}


sub = lambda keep: [[d for d in f if d in keep] for f in folds]
taus_c = dict(SHIPPED_TAUS); taus_c["mu"] = 2.0
base_sl = [np.full(F.n_pairs, -0.5)] * 5
res = {"prereg_sha256": hashlib.sha256((OUT / "items" / "PREREG_items.md").read_bytes()).hexdigest(),
       "amendment6_sha256": hashlib.sha256((OUT / "items" / "PREREG_items_AMENDMENT_6.md").read_bytes()).hexdigest(),
       "ratings": str(rpath), "info": info, "fits": fits, "scenarios": {}}
for sname, ts in (("INV", X.make_testsets(F, sub(inv_dirs), "A", R=300)), ("AS", X.make_testsets(F, sub(as_dirs), "A", R=300)),
                  ("H7INV", NP.make_testsets_H7(F, sub(inv_dirs), R=300, seed=0))):
    S = X.Streams(F, ts)
    tf = np.array([t["fold"] for t in ts])
    cl = S.unit_bench
    its = np.unique(np.concatenate([S.ev["item"], S.acq[31]["item"]]))
    base, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c)
    bo, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c, extra_offset=make_offset("b_only", tf))
    rec = {"coverage_s1": float(np.mean(rated[its])), "base_alc": float(alc(base).mean()), "b_only_alc": float(alc(bo).mean()),
           "signals": {}}
    for name in ("s1", "s3", "smc"):
        br, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c, extra_offset=make_offset(name, tf))
        dv, dc = alc(br) - alc(base), alc(br) - alc(bo)
        rec["signals"][name] = {"alc": float(alc(br).mean()), "vs_base_95": boot(dv, cl, 0.95), "vs_base_9833": boot(dv, cl, 1 - 0.05 / 3),
                                "vs_ctrl_95": boot(dc, cl, 0.95), "vs_ctrl_9833": boot(dc, cl, 1 - 0.05 / 3),
                                "brier_vs_ctrl": [float(x) for x in (br - bo).mean(0)]}
        r = rec["signals"][name]
        log(f"[{sname}] {name}: vs base {r['vs_base_95']['mean']:+.4f} [{r['vs_base_95']['lo']:+.4f}, {r['vs_base_95']['hi']:+.4f}]  "
            f"vs ctrl {r['vs_ctrl_95']['mean']:+.4f} [{r['vs_ctrl_9833']['lo']:+.4f}, {r['vs_ctrl_9833']['hi']:+.4f}] (98.3%)")
    res["scenarios"][sname] = rec
dec = {}
for name in ("s1", "s3", "smc"):
    I, A, H = (res["scenarios"][s]["signals"][name] for s in ("INV", "AS", "H7INV"))
    d = {"a": I["vs_base_9833"]["hi"] < 0, "b": A["vs_base_95"]["lo"] <= 1e-12, "a_prime": I["vs_ctrl_9833"]["hi"] < 0,
         "b_prime": A["vs_ctrl_95"]["lo"] <= 1e-12, "c": H["vs_ctrl_95"]["lo"] <= 1e-12, "bar_0.003": -I["vs_ctrl_95"]["mean"] >= 0.003}
    d["pass"] = all(d.values())
    dec[name] = d
res["decision"] = dec
res["any_pass"] = any(d["pass"] for d in dec.values())
(OUT / "items" / f"eval_items_a6_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"DECISION: {dec}; any pass: {res['any_pass']} ({time.time() - t0:.0f}s)")
