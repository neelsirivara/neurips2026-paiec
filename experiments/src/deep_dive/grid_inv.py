"""Deep dive step 6: re-tune the XP prior sds on the inventory-benchmark scenario (closest rules-clean proxy:
its pair-rate oracle 0.161 matches the hidden-run oracle 0.160 reported by a participant), with nested
selection over the 5 benchmark folds; also report the selected setting on scenario A.
Writes results/deep_dive/grid_inv.json
"""
import sys, os, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *

t0 = time.time()
F, folds = load()
inv = inventory_dirs(F)
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
grid = {"bench": (1.0, 1.5, 2.0, 3.0), "pair": (0.6, 0.9), "item": (1.5, 2.25, 3.0), "mu": (0.5, 2.0)}
settings = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
lab = lambda s: "|".join(f"{k}{v:g}" for k, v in s.items())
res = {"grid": grid, "scenarios": {}}
alcs = {}
for sname, fl in (("INV", sub(inv)), ("A", folds)):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    out = {}
    for s in settings:
        taus = dict(SHIPPED_TAUS); taus["bench"] = s["bench"]; taus["pair"] = s["pair"]; taus["mu"] = s["mu"]
        br, _ = eval_xp(S, ts, taus=taus, ti=s["item"])
        out[lab(s)] = alc(br)
    ship = alc(eval_xp(S, ts)[0])
    alcs[sname] = (S, out, ship)
    log(f"{sname}: grid done ({time.time()-t0:.0f}s); shipped {ship.mean():.4f}; best {min(out, key=lambda k: out[k].mean())} {min(v.mean() for v in out.values()):.4f}")
# nested selection on INV
S, out, ship = alcs["INV"]
fold = S.unit_fold
sel = np.zeros(S.n_units)
picks = []
for f in range(5):
    tr = fold != f
    best = min(out, key=lambda k: out[k][tr].mean())
    picks.append(best)
    sel[fold == f] = out[best][fold == f]
d = paired(sel, ship, S.unit_bench)
allbest = min(out, key=lambda k: out[k].mean())
res["scenarios"]["INV"] = {"shipped": float(ship.mean()), "nested_selected": float(sel.mean()), "picks": picks,
                           "nested_minus_shipped": d, "allfold_best": allbest, "allfold_best_alc": float(out[allbest].mean())}
log(f"INV nested: {sel.mean():.4f} vs shipped {ship.mean():.4f}: {fmt(d)}; picks {picks}")
SA, outA, shipA = alcs["A"]
dA = paired(outA[allbest], shipA, SA.unit_bench)
res["scenarios"]["A"] = {"shipped": float(shipA.mean()), "inv_allfold_best_on_A": float(outA[allbest].mean()), "minus_shipped": dA}
log(f"A with INV's all-fold pick {allbest}: {outA[allbest].mean():.4f} vs {shipA.mean():.4f}: {fmt(dA)}")
(OUT / "grid_inv.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"done ({time.time()-t0:.0f}s)")
