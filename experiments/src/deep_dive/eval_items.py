"""Item-rating study, step 3: fit the rating -> item-offset map leave-benchmark-out and apply the pre-registered gate.

    python -I eval_items.py RATINGS.jsonl [--tag NAME]

PREREG_items.md (+ AMENDMENT_1): base = xp_level_m050 (subject terms off, intercept -0.5, tau_mu 2); candidate = base +
w * z(s1) + b, with z standardised on the training folds' rated items, and (w, b) fitted per outer fold by logistic
regression on the training folds' benchmarks (INV and AS items), using the base budget-0 logit as an offset, every
benchmark weighted equally.  Scenarios INV and AS (config A, R = 300).
Gate: (a) INV ALC difference CI entirely < 0; (b) AS ALC difference not significantly > 0.
Exploratory: s3 and s1 + s3 variants; within-benchmark correlation of s1 with a leave-pair-out empirical item effect.
Writes experiments/results/deep_dive/items/eval_items[_TAG].json
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *  # noqa: F401,F403
import hashlib
import rate_items as RI

PREREG = OUT / "items" / "PREREG_items.md"
AMEND = OUT / "items" / "PREREG_items_AMENDMENT_1.md"
CACHE = Path(X.DEFAULT_PAYLOAD).resolve() / "round4_cache"
args = sys.argv[1:]
ratings_path = Path(args[0])
tag = args[args.index("--tag") + 1] if "--tag" in args else ratings_path.stem
t0 = time.time()
F, folds = load()
man = json.loads((OUT / "items" / "items_manifest.json").read_text(encoding="utf-8"))
inv_dirs, as_dirs = set(man["inv_dirs"]), set(man["as_dirs_seeded_order"])
# ratings -> frame item index (several items can share one prompt key)
rat = {}
for l in open(ratings_path, encoding="utf-8"):
    r = json.loads(l)
    if r.get("s1") is not None and not r.get("error"):
        rat[r["key"]] = r
API_MODEL = next((r.get("model") for r in rat.values() if r.get("backend") == "api"), None)
if API_MODEL:
    import rate_items_claude as RC
NI = len(F.item_dir)
s1 = np.full(NI, np.nan)
s3 = np.full(NI, np.nan)
n_items = 0
for l in open(CACHE / "deep_dive_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    n_items += 1
    if API_MODEL:
        k = RC.key_of(RC.build_prompt(it["content"], it["features"]), API_MODEL)
    else:
        k = RI.prompt_key(RI.build_prompt(it["content"], it["features"]))
    r = rat.get(k)
    if r is not None:
        s1[it["idx"]] = r["s1"]
        if r.get("s3") is not None:
            s3[it["idx"]] = r["s3"]
rated = ~np.isnan(s1)
log(f"ratings: {len(rat)} keys; items needed {n_items}, rated {int(rated.sum())}")
fod = fold_of_dir(F, folds)
item_fold = np.array([fod.get(int(d), -1) for d in F.item_dir])
dir_name = np.array(F.dirs)[F.item_dir]
in_study = np.isin(dir_name, list(inv_dirs | as_dirs))
row_pair = np.repeat(np.arange(F.n_pairs), F.pair_n)
row_ok = (F.pair_n[row_pair] >= 80) & in_study[F.row_item] & rated[F.row_item]
base_zs = -0.5
base_o_row = np.clip(base_zs + F.item_zi[F.row_item], -X.CAP, X.CAP)


def fit_offset(feats_rows, y, off, wts, iters=25):
    """Weighted logistic regression with offset; returns coefficient vector (intercept last)."""
    Xm = np.column_stack([feats_rows, np.ones(len(y))])
    beta = np.zeros(Xm.shape[1])
    for _ in range(iters):
        eta = off + Xm @ beta
        p = 1 / (1 + np.exp(-eta))
        g = Xm.T @ (wts * (y - p))
        H = (Xm * (wts * p * (1 - p))[:, None]).T @ Xm + 1e-9 * np.eye(Xm.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


signals = {"b_only": [], "s1": [s1], "s3": [s3], "s1+s3": [s1, s3]}
fits = {}
for name, sig in signals.items():
    ok_sig = np.all([~np.isnan(s) for s in sig], axis=0) if sig else rated.copy()
    fits[name] = []
    for f in range(5):
        tr_items = ok_sig & (item_fold != f) & (item_fold >= 0) & in_study
        mu = [float(np.mean(s[tr_items])) for s in sig]
        sd = [float(np.std(s[tr_items])) or 1.0 for s in sig]
        rows = row_ok & tr_items[F.row_item]
        it = F.row_item[rows]
        Z = np.column_stack([(s[it] - m) / d for s, m, d in zip(sig, mu, sd)]) if sig else np.zeros((len(it), 0))
        b_of = F.item_dir[it]
        cnt = np.bincount(b_of, minlength=len(F.dirs)).astype(float)
        wts = 1.0 / cnt[b_of]
        wts *= len(wts) / wts.sum()
        beta = fit_offset(Z, F.row_y[rows], base_o_row[rows], wts)
        fits[name].append({"mu": mu, "sd": sd, "beta": [float(x) for x in beta], "n_rows": int(rows.sum()),
                           "n_bench": int(len(np.unique(b_of)))})
    log(f"fit {name}: per-fold beta " + "; ".join(str(np.round(fr['beta'], 3)) for fr in fits[name]))
# all-fold fit of the primary signal (deployment constants if the gate passes; not used in any evaluation)
_ok = ~np.isnan(s1) & in_study
_mu, _sd = float(np.mean(s1[_ok])), float(np.std(s1[_ok]))
_rows = row_ok & _ok[F.row_item]
_it = F.row_item[_rows]
_b_of = F.item_dir[_it]
_cnt = np.bincount(_b_of, minlength=len(F.dirs)).astype(float)
_w = 1.0 / _cnt[_b_of]
_w *= len(_w) / _w.sum()
_beta = fit_offset(((s1[_it] - _mu) / _sd)[:, None], F.row_y[_rows], base_o_row[_rows], _w)
deploy_fit = {"signal": "s1", "mu": _mu, "sd": _sd, "w": float(_beta[0]), "b": float(_beta[1]), "n_rows": int(_rows.sum()),
              "n_bench": int(len(np.unique(_b_of)))}
log(f"all-fold fit (deployment constants): {deploy_fit}")


def make_offset(name, ts_fold):
    sig = signals[name]

    def off(T):
        f = ts_fold[T["ts"]]
        out = np.zeros(len(T["ts"]))
        for k in range(5):
            m = f == k
            if not np.any(m):
                continue
            fr = fits[name][k]
            items = T["item"][m]
            Z = (np.column_stack([(s[items] - mu) / sd for s, mu, sd in zip(sig, fr["mu"], fr["sd"])]) if sig
                 else np.zeros((len(items), 0)))
            if sig is not None and len(sig) == 0:
                Z = np.zeros((len(items), 0))
            val = Z @ np.array(fr["beta"][:-1]) + fr["beta"][-1]
            val[np.isnan(val)] = 0.0
            if sig:
                val[~np.all([~np.isnan(s[items]) for s in sig], axis=0)] = 0.0
            out[m] = val
        return out
    return off


sub = lambda keep: [[d for d in f if d in keep] for f in folds]
res = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
       "amendment_sha256": hashlib.sha256(AMEND.read_bytes()).hexdigest() if AMEND.exists() else None,
       "amendment2_sha256": (hashlib.sha256((OUT / "items" / "PREREG_items_AMENDMENT_2.md").read_bytes()).hexdigest()
                             if (OUT / "items" / "PREREG_items_AMENDMENT_2.md").exists() else None),
       "ratings_file": str(ratings_path), "n_rating_keys": len(rat), "n_items_rated": int(rated.sum()), "n_items_needed": n_items,
       "fits": fits, "deploy_fit": deploy_fit, "scenarios": {}}
taus_c = dict(SHIPPED_TAUS); taus_c["mu"] = 2.0
base_sl = [np.full(F.n_pairs, base_zs)] * 5
for sname, keep in (("INV", inv_dirs), ("AS", as_dirs)):
    ts = X.make_testsets(F, sub(keep), "A", R=300)
    S = X.Streams(F, ts)
    ts_fold = np.array([t["fold"] for t in ts])
    its = np.unique(np.concatenate([S.ev["item"], S.acq[31]["item"]]))
    cov = float(np.mean(rated[its]))
    br0, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c)
    alc_by = {}
    rec = {"n_units": int(S.n_units), "n_bench": int(len(np.unique(S.unit_bench))), "rating_coverage": cov,
           "base_alc": float(alc(br0).mean()), "base_brier": [float(x) for x in br0.mean(0)], "variants": {}}
    for name in signals:
        br, _ = eval_xp(S, ts, subj_logit_by_fold=base_sl, taus=taus_c, extra_offset=make_offset(name, ts_fold))
        d = paired(alc(br), alc(br0), S.unit_bench)
        alc_by[name] = alc(br)
        rec["variants"][name] = {"alc": float(alc(br).mean()), "brier": [float(x) for x in br.mean(0)], "minus_base": d,
                                 "brier_minus_base": [float(x) for x in (br - br0).mean(0)]}
        log(f"[{sname}] coverage {cov:.3f} base {alc(br0).mean():.4f} {name}: {alc(br).mean():.4f} diff {fmt(d)} per budget {np.round((br - br0).mean(0), 4)}")
    rec["ctrl_diff"] = paired(alc_by["s1"], alc_by["b_only"], S.unit_bench)
    log(f"[{sname}] s1 minus b_only control: {fmt(rec['ctrl_diff'])}")
    res["scenarios"][sname] = rec
g_inv = res["scenarios"]["INV"]["variants"]["s1"]["minus_base"]
g_as = res["scenarios"]["AS"]["variants"]["s1"]["minus_base"]
res["gate"] = {"a_inv_ci_below_0": bool(g_inv["ci_hi"] < 0), "b_as_not_sig_worse": bool(g_as["ci_lo"] <= 0)}
res["gate"]["pass"] = res["gate"]["a_inv_ci_below_0"] and res["gate"]["b_as_not_sig_worse"]
# amendment 2: the same gate against the intercept-only control (base + b fitted the same way with w = 0)
ctrl = {}
for sname in ("INV", "AS"):
    sc_ = res["scenarios"][sname]
    ctrl[sname] = sc_["ctrl_diff"]
res["gate_vs_b_only"] = {"a_inv_ci_below_0": bool(ctrl["INV"]["ci_hi"] < 0), "b_as_not_sig_worse": bool(ctrl["AS"]["ci_lo"] <= 0)}
res["gate_vs_b_only"]["pass"] = res["gate_vs_b_only"]["a_inv_ci_below_0"] and res["gate_vs_b_only"]["b_as_not_sig_worse"]
# exploratory: within-benchmark correlation of s1 with the leave-pair-out empirical item effect (logit-ish residual)
rate = np.add.reduceat(F.row_y, F.pair_start[:-1]) / F.pair_n
rp = rate[row_pair]
resid = F.row_y - rp
S_i = np.bincount(F.row_item, weights=resid, minlength=NI)
C_i = np.bincount(F.row_item, minlength=NI).astype(float)
eff = np.where(C_i >= 3, S_i / np.maximum(C_i, 1), np.nan)
corr = {}
for name, sel in (("INV", np.isin(dir_name, list(inv_dirs))), ("AS", np.isin(dir_name, list(as_dirs)))):
    cs = []
    for b in np.unique(F.item_dir[sel & rated & ~np.isnan(eff)]):
        m = (F.item_dir == b) & rated & ~np.isnan(eff)
        if m.sum() >= 20 and np.std(s1[m]) > 0 and np.std(eff[m]) > 0:
            cs.append(float(np.corrcoef(s1[m], eff[m])[0, 1]))
    corr[name] = {"n_bench": len(cs), "median": float(np.median(cs)) if cs else None, "mean": float(np.mean(cs)) if cs else None,
                  "frac_positive": float(np.mean(np.array(cs) > 0)) if cs else None}
res["within_benchmark_corr_s1_vs_item_effect"] = corr
log(f"within-benchmark corr(s1, empirical item effect): {corr}")
log(f"GATE vs base: {res['gate']}; GATE vs b-only control: {res['gate_vs_b_only']}")
(OUT / "items" / f"eval_items_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time() - t0:.0f}s)")
