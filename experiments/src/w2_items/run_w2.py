"""W2 tuning and confirmation (pre-registered in experiments/results/w2_items/PREREG.md).

    python -I run_w2.py tune F    # tune folds 0, 1: grid F on A, INV, MID, HARD   (tune K: grid K; parallel)
    python -I run_w2.py select    # FK = best F + best K on the tune folds; the pre-registered selection
    python -I run_w2.py confirm   # confirm folds 2, 3, 4: the selected variant (and, exploratory, the rest of the
                                  # grid) on A, INV, MID, HARD, SPARSE; the gate
Outputs experiments/results/w2_items/{tune,confirm}.json (+ per-unit ALC .npz).
"""
import hashlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_sim import *  # noqa: F401,F403

TUNE_FOLDS = (0, 1)
CONFIRM_FOLDS = (2, 3, 4)
F_GRID = [{"tau_f": tf, "share": sh} for tf in (0.25, 0.35, 0.5, 0.7) for sh in (0.5, 1.0)]
K_GRID = [{"tau_k": tk, "q": q} for tk in (0.5, 1.0, 2.0) for q in (2, 4, 8)]
TUNE_SCEN = ("A", "INV", "MID", "HARD")
CONFIRM_SCEN = ("INV", "A", "SPARSE", "MID", "HARD")
PREREG = RES / "PREREG.md"


def label(s):
    parts = []
    if s.get("tau_f", 0) > 0:
        parts.append(f"F{s['tau_f']:g}s{s['share']:g}")
    if s.get("tau_k", 0) > 0:
        parts.append(f"K{s['tau_k']:g}q{s['q']:g}")
    return "+".join(parts) or "m050"


def run(W, fold_ids, scen, settings, tag):
    """Per-unit ALC of m050 and each setting; paired differences with cluster-bootstrap CIs over benchmarks."""
    t0 = time.time()
    S, ts = build_scenario(W, scen, fold_ids)
    cl = S.unit_bench
    br0 = eval_w2(W, S, ts)
    a0 = alc(br0)
    rec = {"n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))), "base_alc": float(a0.mean()),
           "base_brier": [float(x) for x in br0.mean(0)], "variants": {}}
    store = {"base": a0, "unit_bench": cl}
    log(f"[{tag} {scen}] units {S.n_units} benches {rec['n_bench']} m050 ALC {a0.mean():.4f} ({time.time() - t0:.0f}s)")
    for s in settings:
        t1 = time.time()
        br = eval_w2(W, S, ts, tau_f=s.get("tau_f", 0.0), share=s.get("share", 0.5), tau_k=s.get("tau_k", 0.0), q=s.get("q", 8))
        a = alc(br)
        d = paired(a, a0, cl)
        rec["variants"][label(s)] = {"setting": s, "alc": float(a.mean()), "minus_m050": d,
                                     "brier_minus_m050": [float(x) for x in (br - br0).mean(0)],
                                     "brier_minus_m050_ci": [paired(br[:, i], br0[:, i], cl) for i in range(6)]}
        store[label(s)] = a
        log(f"  {label(s):16s} ALC {a.mean():.4f} diff {fmt(d)} per budget {np.round((br - br0).mean(0), 4)} ({time.time() - t1:.0f}s)")
    np.savez(RES / f"{tag}_{scen}_unit_alc.npz", **{k: v for k, v in store.items()})
    return rec


def prereg_sha():
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None


def cmd_tune_part(part):
    """Grid part 'F' or 'K' on the tune folds (two processes, one thread each) -> tune_<part>.json."""
    W = World(with_grams=(part == "K"))
    grid = F_GRID if part == "F" else K_GRID
    res = {"prereg_sha256": prereg_sha(), "folds": TUNE_FOLDS, "part": part, "scenarios": {}}
    for scen in TUNE_SCEN:
        res["scenarios"][scen] = run(W, TUNE_FOLDS, scen, grid, f"tune{part}")
        (RES / f"tune_{part}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("done")


def cmd_select():
    """FK = best F + best K on the tune folds; selection by the pre-registered criterion -> tune.json."""
    parts = {p: json.loads((RES / f"tune_{p}.json").read_text(encoding="utf-8")) for p in ("F", "K")}
    res = {"prereg_sha256": prereg_sha(), "folds": TUNE_FOLDS, "scenarios": {}}
    for scen in TUNE_SCEN:
        r = dict(parts["F"]["scenarios"][scen])
        r["variants"] = dict(parts["F"]["scenarios"][scen]["variants"])
        r["variants"].update(parts["K"]["scenarios"][scen]["variants"])
        res["scenarios"][scen] = r

    def crit(lbl):
        return float(np.mean([res["scenarios"][s]["variants"][lbl]["minus_m050"]["mean"] for s in TUNE_SCEN]))
    fl = [label(s) for s in F_GRID]
    kl = [label(s) for s in K_GRID]
    bf = min(fl, key=crit)
    bk = min(kl, key=crit)
    sf = next(s for s in F_GRID if label(s) == bf)
    sk = next(s for s in K_GRID if label(s) == bk)
    fk = dict(sf, **sk)
    W = World(with_grams=True)
    for scen in TUNE_SCEN:
        r = run(W, TUNE_FOLDS, scen, [fk], "tune_fk")
        res["scenarios"][scen]["variants"].update(r["variants"])
    cands = fl + kl + [label(fk)]
    allset = {label(s): s for s in F_GRID + K_GRID + [fk]}
    res["criterion"] = {lbl: crit(lbl) for lbl in cands}
    best = min(cands, key=lambda l_: res["criterion"][l_])
    res["best_F"], res["best_K"], res["FK"] = bf, bk, label(fk)
    res["FK_setting"] = fk
    res["selected"] = {"label": best, "setting": allset[best], "criterion": res["criterion"][best]}
    (RES / "tune.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"best F {bf} ({crit(bf):+.5f}), best K {bk} ({crit(bk):+.5f}), FK {label(fk)} ({crit(label(fk)):+.5f}); SELECTED {best}")


def cmd_confirm():
    tune = json.loads((RES / "tune.json").read_text(encoding="utf-8"))
    sel = tune["selected"]["setting"]
    others = [s for s in F_GRID + K_GRID if label(s) != label(sel)]
    W = World(with_grams=True)
    res = {"prereg_sha256": prereg_sha(), "tune_selected": tune["selected"], "folds": CONFIRM_FOLDS, "scenarios": {}}
    for scen in CONFIRM_SCEN:
        res["scenarios"][scen] = run(W, CONFIRM_FOLDS, scen, [sel], "confirm")
        (RES / "confirm.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    g = {s: res["scenarios"][s]["variants"][label(sel)]["minus_m050"] for s in ("INV", "A", "SPARSE")}
    res["gate"] = {"a_INV_ci_below_0": bool(g["INV"]["ci_hi"] < 0), "b_A_not_sig_worse": bool(g["A"]["ci_lo"] <= 0),
                   "c_SPARSE_not_sig_worse": bool(g["SPARSE"]["ci_lo"] <= 0)}
    res["gate"]["pass"] = all(res["gate"].values())
    (RES / "confirm.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"GATE {res['gate']}")
    # exploratory (after the gate is recorded): the other tuned settings on the confirm folds, INV and A only
    expl = {}
    for scen in ("INV", "A"):
        expl[scen] = run(W, CONFIRM_FOLDS, scen, others, "confirm_expl")
    res["exploratory_other_settings"] = expl
    (RES / "confirm.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune":
        cmd_tune_part(sys.argv[2])
    elif cmd == "select":
        cmd_select()
    elif cmd == "confirm":
        cmd_confirm()
    else:
        raise SystemExit("usage: run_w2.py tune F|K | select | confirm")
