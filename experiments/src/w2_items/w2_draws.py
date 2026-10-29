"""Formative draws of w2_f05s1_k1q2 vs xp_level_m050: aggregates only -> results/w2_items/draws.json

    python -I w2_draws.py

Parses the saved Codabench "Scoring output" logs with deep_dive/parse_feedback.parse (the same parser as
formative_summary.py).  Writes, per draw: number of pairs / distinct subject and benchmark IDs, mean Brier per budget
and the ALC; per arm: mean and SE over draws; arm difference (Welch SE); and the benchmark-matched difference
(per arm, pairs of all its draws pooled per benchmark ID; W2 minus m050 averaged with equal weight over the shared
benchmarks; SE over benchmarks).  Anonymous IDs are used only for this matching inside our own runs and are not
stored; no pair-level value is stored.
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deep_dive"))
from parse_feedback import parse  # noqa: E402

DL = HERE.parents[3] / "_downloads"
RES = HERE.parents[1] / "results" / "w2_items"
ARMS = {"xp_level_m050": [1010717, 1010835, 1010837, 1019450, 1019451, 1019452, 1019453],
        "w2_f05s1_k1q2": [1019682, 1019683, 1019685, 1019688, 1019903]}
# 1019686: platform status Failed ([PAIEC-INFRA-ARCHIVE-001] infrastructure archival error; complete scoring log, no
# recorded score); retried once as 1019903 (SWITCH_RULE.md update).  Reported separately, not in the W2 arm.
INFRA_FAILED = {1019686: "w2_f05s1_k1q2"}
NOTE_M050 = ("1010835 / 1010837 are the deployment probes A / B, whose predictions are xp_level_m050's "
             "(DEEP_DIVE.md section 11)")
BUDGETS = [0, 1, 3, 7, 15, 31]
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}


def log_path(sid):
    for p in (DL / f"cb_logs_{sid}" / "scoring_stdout.txt", DL / f"paiec_logs_{sid}" / "scoring_stdout.txt"):
        if p.exists():
            return p
    return None


def mean_se(v):
    return {"mean": st.mean(v), "se": (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else None, "sd": st.stdev(v) if len(v) > 1 else None,
            "n": len(v)}


out = {"note": "aggregates only; anonymous IDs are not stored. " + NOTE_M050, "draws": {}, "arms": {}, "missing": []}
tables = {}
for arm, ids in ARMS.items():
    for sid in ids:
        p = log_path(sid)
        t = parse(p.read_text(encoding="utf-8", errors="replace")) if p else None
        if not t or "ALC" not in t or not all(b in t for b in BUDGETS):
            out["missing"].append(sid)
            continue
        tables[sid] = t
        out["draws"][str(sid)] = {"arm": arm, "n_pairs": len(t["ALC"]), "n_subject_ids": len({r[0] for r in t["ALC"]}),
                                  "n_benchmark_ids": len({r[1] for r in t["ALC"]}),
                                  "n_eval_subject_items": sum(r[2] for r in t["ALC"]),
                                  "alc": st.mean(r[3] for r in t["ALC"]),
                                  "brier_by_budget": {str(b): st.mean(r[3] for r in t[b]) for b in BUDGETS},
                                  "ece_b0": st.mean(r[4] for r in t[0] if r[4] is not None)}
out["infra_failed_log_only"] = {}
for sid, arm in INFRA_FAILED.items():
    p = log_path(sid)
    t = parse(p.read_text(encoding="utf-8", errors="replace")) if p else None
    if t and "ALC" in t and all(b in t for b in BUDGETS):
        out["infra_failed_log_only"][str(sid)] = {
            "arm": arm, "note": "platform status Failed (infrastructure archival error); log complete; not counted",
            "n_pairs": len(t["ALC"]), "alc": st.mean(r[3] for r in t["ALC"]),
            "brier_by_budget": {str(b): st.mean(r[3] for r in t[b]) for b in BUDGETS}}
for arm, ids in ARMS.items():
    got = [sid for sid in ids if sid in tables]
    if not got:
        continue
    rec = {"draws": got, "alc": mean_se([out["draws"][str(s)]["alc"] for s in got]),
           "brier_by_budget": {str(b): mean_se([out["draws"][str(s)]["brier_by_budget"][str(b)] for s in got]) for b in BUDGETS}}
    out["arms"][arm] = rec
a, b = "xp_level_m050", "w2_f05s1_k1q2"
if a in out["arms"] and b in out["arms"]:
    def diff(ra, rb):
        se = None
        if ra["se"] is not None and rb["se"] is not None:
            se = (ra["se"] ** 2 + rb["se"] ** 2) ** 0.5
        return {"mean": rb["mean"] - ra["mean"], "se_welch": se}
    out["arm_difference_w2_minus_m050"] = {
        "alc": diff(out["arms"][a]["alc"], out["arms"][b]["alc"]),
        "brier_by_budget": {str(k): diff(out["arms"][a]["brier_by_budget"][str(k)], out["arms"][b]["brier_by_budget"][str(k)])
                            for k in BUDGETS}}
    bm = {}
    for sec in ["ALC"] + BUDGETS:
        pooled = {}
        for arm in (a, b):
            acc = {}
            for sid in out["arms"][arm]["draws"]:
                for r in tables[sid][sec]:
                    acc.setdefault(r[1], []).append(r[3])
            pooled[arm] = {k: sum(v) / len(v) for k, v in acc.items()}
        shared = sorted(set(pooled[a]) & set(pooled[b]))
        if len(shared) >= 2:
            d = [pooled[b][k] - pooled[a][k] for k in shared]
            bm[str(sec)] = {"n_benchmarks": len(shared), "mean_diff": st.mean(d), "sd": st.stdev(d),
                            "se": st.stdev(d) / len(d) ** 0.5, "n_benchmarks_improved": sum(1 for x in d if x < 0)}
    out["benchmark_matched_w2_minus_m050"] = bm
    # pair-matched (same subject ID and benchmark ID in both arms; aggregate over the shared pairs only, as in
    # DEEP_DIVE.md section 2: matching IDs denote the same subject / benchmark; no per-pair value is stored)
    pm = {}
    for sec in ["ALC"] + BUDGETS:
        pooled = {}
        for arm in (a, b):
            acc = {}
            for sid in out["arms"][arm]["draws"]:
                for r in tables[sid][sec]:
                    acc.setdefault((r[0], r[1]), []).append(r[3])
            pooled[arm] = {k: sum(v) / len(v) for k, v in acc.items()}
        shared = sorted(set(pooled[a]) & set(pooled[b]))
        if len(shared) >= 2:
            d = [pooled[b][k] - pooled[a][k] for k in shared]
            pm[str(sec)] = {"n_pairs": len(shared), "mean_diff": st.mean(d), "se": st.stdev(d) / len(d) ** 0.5,
                            "n_pairs_improved": sum(1 for x in d if x < 0)}
    out["pair_matched_w2_minus_m050"] = pm
    # budget-0 anchored arm difference: B0 predictions of the two zips are identical, so the B0 arm difference is pure
    # draw composition; differences of (B_k - B0) remove that common part
    ad = {}
    for k in BUDGETS[1:]:
        va = [out["draws"][str(s_)]["brier_by_budget"][str(k)] - out["draws"][str(s_)]["brier_by_budget"]["0"] for s_ in out["arms"][a]["draws"]]
        vb = [out["draws"][str(s_)]["brier_by_budget"][str(k)] - out["draws"][str(s_)]["brier_by_budget"]["0"] for s_ in out["arms"][b]["draws"]]
        se = ((st.stdev(va) ** 2 / len(va) + st.stdev(vb) ** 2 / len(vb)) ** 0.5) if len(va) > 1 and len(vb) > 1 else None
        ad[str(k)] = {"mean": st.mean(vb) - st.mean(va), "se_welch": se}
    out["b0_anchored_difference_w2_minus_m050"] = ad
    ids_a = set().union(*[{r[1] for r in tables[s]["ALC"]} for s in out["arms"][a]["draws"]])
    ids_b = set().union(*[{r[1] for r in tables[s]["ALC"]} for s in out["arms"][b]["draws"]])
    out["structure"] = {"distinct_benchmark_ids_m050": len(ids_a), "distinct_benchmark_ids_w2": len(ids_b),
                        "shared_benchmark_ids": len(ids_a & ids_b),
                        "shared_subject_ids": len(set().union(*[{r[0] for r in tables[s]["ALC"]} for s in out["arms"][a]["draws"]])
                                                  & set().union(*[{r[0] for r in tables[s]["ALC"]} for s in out["arms"][b]["draws"]]))}
(RES / "draws.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in out if k not in ("draws",)}, indent=1))
for sid, r in out["draws"].items():
    print(sid, r["arm"], r["n_pairs"], f"{r['alc']:.4f}", " ".join(f"{r['brier_by_budget'][str(b)]:.4f}" for b in BUDGETS))
