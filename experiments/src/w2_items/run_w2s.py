"""W2-sem tuning and confirmation (pre-registered in experiments/results/w2_items/PREREG_SEM.md).

    python -I run_w2s.py tune SCEN        # tune folds 0, 1: the whole grid on one scenario (A, INV, MID or HARD)
    python -I run_w2s.py select           # criterion over the four tune scenarios -> tune_sem.json
    python -I run_w2s.py confirm SCEN     # confirm folds 2, 3, 4: m050, W2 (current candidate) and the selected variant
    python -I run_w2s.py gate             # gate from the five confirm files -> confirm_sem.json
"""
import hashlib
import itertools
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2s_sim import *  # noqa: F401,F403

TUNE_FOLDS = (0, 1)
CONFIRM_FOLDS = (2, 3, 4)
TUNE_SCEN = ("A", "INV", "MID", "HARD")
CONFIRM_SCEN = ("INV", "A", "SPARSE", "MID", "HARD")
PREREG = RES / "PREREG_SEM.md"
EMB = "minilm"
W2_SETTING = {"tau_f": 0.5, "share": 1.0, "kern": "char", "q_c": 2, "tau_k": 1.0}


def grid():
    out = []
    for tf, tk, topk, ml in itertools.product((0.35, 0.5, 0.7), (0.7, 1.0, 1.5, 2.0), (None, 10, 30), (1.0, 0.6)):
        for kern, qs in [("char", None)] + [("sem", q) for q in (8, 16, 32)] + [("both", q) for q in (8, 16, 32)]:
            s = {"tau_f": tf, "share": 1.0, "kern": kern, "q_c": 2, "q_s": qs, "tau_k": tk, "topk": topk, "m_low": ml}
            s["label"] = slabel(s)
            out.append(s)
    return out


def prereg_sha():
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None


def world():
    W = World(with_grams=True)
    attach_embeddings(W, EMB)
    return W


def summarize(S, out, ref):
    cl = S.unit_bench
    a0 = alc(out[ref])
    rec = {}
    for k, br in out.items():
        a = alc(br)
        rec[k] = {"alc": float(a.mean()), "minus_" + ref: paired(a, a0, cl) if k != ref else None,
                  "brier": [float(x) for x in br.mean(0)]}
    return rec


def cmd_tune(scen):
    W = world()
    S, ts = build_scenario(W, scen, TUNE_FOLDS)
    t0 = time.time()
    settings = grid()
    base = {"tau_f": 0.0, "share": 1.0, "kern": "none", "label": "m050"}
    out = eval_grid(W, S, ts, [base] + settings)
    log(f"[tune {scen}] {len(settings)} settings ({time.time() - t0:.0f}s)")
    a0 = alc(out["m050"])
    cl = S.unit_bench
    res = {"prereg_sha256": prereg_sha(), "scenario": scen, "n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))),
           "m050_alc": float(a0.mean()), "settings": {}}
    for s in settings:
        a = alc(out[s["label"]])
        res["settings"][s["label"]] = {"setting": {k: v for k, v in s.items() if k != "label"}, "alc": float(a.mean()),
                                       "minus_m050_mean": float(a.mean() - a0.mean()),
                                       "brier_minus_m050": [float(x) for x in (out[s["label"]] - out["m050"]).mean(0)]}
    np.savez(RES / f"tune_sem_{scen}_unit_alc.npz", unit_bench=cl, labels=np.array(["m050"] + [s["label"] for s in settings]),
             alc=np.stack([alc(out["m050"])] + [alc(out[s["label"]]) for s in settings]).astype(np.float32))
    (RES / f"tune_sem_{scen}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("done")


def cmd_select():
    parts = {s: json.loads((RES / f"tune_sem_{s}.json").read_text(encoding="utf-8")) for s in TUNE_SCEN}
    labels = list(parts["A"]["settings"])
    crit = {l_: float(np.mean([parts[s]["settings"][l_]["minus_m050_mean"] for s in TUNE_SCEN])) for l_ in labels}
    sem = [l_ for l_ in labels if parts["A"]["settings"][l_]["setting"]["kern"] in ("sem", "both")]
    best = min(sem, key=lambda l_: crit[l_])
    w2 = slabel(dict(W2_SETTING, topk=None, m_low=1.0))
    res = {"prereg_sha256": prereg_sha(), "criterion": crit, "w2_label": w2, "w2_criterion": crit[w2],
           "best_overall": min(labels, key=lambda l_: crit[l_]), "best_char_only": min([l_ for l_ in labels if l_ not in sem], key=lambda l_: crit[l_]),
           "selected": {"label": best, "setting": parts["A"]["settings"][best]["setting"], "criterion": crit[best]},
           "selected_per_scenario": {s: parts[s]["settings"][best]["minus_m050_mean"] for s in TUNE_SCEN},
           "w2_per_scenario": {s: parts[s]["settings"][w2]["minus_m050_mean"] for s in TUNE_SCEN},
           "top20": sorted(([l_, crit[l_]] for l_ in labels), key=lambda kv: kv[1])[:20]}
    (RES / "tune_sem.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"W2 {w2} {crit[w2]:+.5f}; best overall {res['best_overall']} {crit[res['best_overall']]:+.5f}; "
        f"best char-only {res['best_char_only']} {crit[res['best_char_only']]:+.5f}; SELECTED {best} {crit[best]:+.5f}")


def cmd_confirm(scen):
    sel = json.loads((RES / "tune_sem.json").read_text(encoding="utf-8"))["selected"]["setting"]
    sel = dict(sel)
    sel["label"] = "selected"
    w2 = dict(W2_SETTING, topk=None, m_low=1.0, label="W2")
    base = {"tau_f": 0.0, "share": 1.0, "kern": "none", "label": "m050"}
    W = world()
    S, ts = build_scenario(W, scen, CONFIRM_FOLDS)
    out = eval_grid(W, S, ts, [base, w2, sel])
    cl = S.unit_bench
    a = {k: alc(v) for k, v in out.items()}
    res = {"prereg_sha256": prereg_sha(), "scenario": scen, "n_units": int(S.n_units), "n_bench": int(len(np.unique(cl))),
           "alc": {k: float(v.mean()) for k, v in a.items()},
           "selected_minus_m050": paired(a["selected"], a["m050"], cl), "W2_minus_m050": paired(a["W2"], a["m050"], cl),
           "selected_minus_W2": paired(a["selected"], a["W2"], cl),
           "brier_selected_minus_m050": [paired(out["selected"][:, i], out["m050"][:, i], cl) for i in range(6)],
           "brier_selected_minus_W2": [paired(out["selected"][:, i], out["W2"][:, i], cl) for i in range(6)]}
    np.savez(RES / f"confirm_sem_{scen}_unit_alc.npz", unit_bench=cl, **a)
    (RES / f"confirm_sem_{scen}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"[confirm {scen}] sel-m050 {fmt(res['selected_minus_m050'])} W2-m050 {fmt(res['W2_minus_m050'])} "
        f"sel-W2 {fmt(res['selected_minus_W2'])}")


def cmd_gate():
    r = {s: json.loads((RES / f"confirm_sem_{s}.json").read_text(encoding="utf-8")) for s in CONFIRM_SCEN}
    g = {"a_INV_vs_m050_ci_below_0": r["INV"]["selected_minus_m050"]["ci_hi"] < 0,
         "b_A_vs_m050_not_sig_worse": r["A"]["selected_minus_m050"]["ci_lo"] <= 0,
         "c_SPARSE_vs_m050_not_sig_worse": r["SPARSE"]["selected_minus_m050"]["ci_lo"] <= 0,
         "d_INV_vs_W2_ci_below_0": r["INV"]["selected_minus_W2"]["ci_hi"] < 0,
         "e_A_vs_W2_not_sig_worse": r["A"]["selected_minus_W2"]["ci_lo"] <= 0}
    g["pass"] = all(g.values())
    res = {"prereg_sha256": prereg_sha(), "tune_selected": json.loads((RES / "tune_sem.json").read_text(encoding="utf-8"))["selected"],
           "gate": g, "scenarios": r}
    (RES / "confirm_sem.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"GATE {g}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "tune":
        cmd_tune(sys.argv[2])
    elif c == "select":
        cmd_select()
    elif c == "confirm":
        cmd_confirm(sys.argv[2])
    elif c == "gate":
        cmd_gate()
