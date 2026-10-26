"""Aggregate summary of our formative Codabench runs + the simulator's scenario table -> results/deep_dive/formative_runs.json

    python -I formative_summary.py

Inputs: the saved "Scoring output" logs (private to the submitter) and results/deep_dive/proto_level3.json.
Only aggregates are written: per-budget means of Brier and ECE per run, structure counts (pairs, distinct subject and
benchmark IDs, overlaps between runs), and benchmark-matched run-to-run differences (equal weight per shared benchmark).
No pair-level values, success rates or anonymous IDs are stored.
"""
import itertools
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_feedback import parse  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "deep_dive"
DL = HERE.parents[3] / "_downloads"
RUNS = {  # name -> (submission id, zip, log path)
    "shipped": (1010370, "submission/paiec_submission.zip", DL / "paiec_logs_1010370" / "scoring_stdout.txt"),
    "xp_nosubj_mu2": (1010715, "submission/candidates/xp_nosubj_mu2.zip", DL / "cb_logs_1010715" / "scoring_stdout.txt"),
    "xp_level_m050": (1010717, "submission/candidates/xp_level_m050.zip", DL / "cb_logs_1010717" / "scoring_stdout.txt"),
    "xp_level_m100": (1010735, "submission/candidates/xp_level_m100.zip", DL / "cb_logs_1010735" / "scoring_stdout.txt"),
}
BUDGETS = [0, 1, 3, 7, 15, 31]
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
out = {"note": "aggregates only; anonymous IDs are not stored", "runs": {}, "benchmark_matched": {}, "overlap": {}}
tables = {}
for name, (sid, z, path) in RUNS.items():
    if not path.exists():
        continue
    t = parse(path.read_text(encoding="utf-8", errors="replace"))
    tables[name] = t
    rec = {"submission_id": sid, "zip": z, "n_pairs": len(t["ALC"]), "n_subject_ids": len({r[0] for r in t["ALC"]}),
           "n_benchmark_ids": len({r[1] for r in t["ALC"]}), "n_eval_subject_items": sum(r[2] for r in t["ALC"]),
           "alc_mean": st.mean(r[3] for r in t["ALC"]), "ece_summary_mean": st.mean(r[4] for r in t["ALC"]),
           "brier_by_budget": {str(b): st.mean(r[3] for r in t[b]) for b in BUDGETS},
           "ece_by_budget": {str(b): st.mean(r[4] for r in t[b]) for b in BUDGETS}}
    rec["alc_from_budget_means"] = sum(W[b] * rec["brier_by_budget"][str(b)] for b in BUDGETS)
    rec["D_B0_minus_B31"] = rec["brier_by_budget"]["0"] - rec["brier_by_budget"]["31"]
    out["runs"][name] = rec
names = list(tables)
for a, b in itertools.combinations(names, 2):
    pa = {(r[0], r[1]) for r in tables[a]["ALC"]}
    pb = {(r[0], r[1]) for r in tables[b]["ALC"]}
    out["overlap"][f"{a}|{b}"] = {"shared_pairs": len(pa & pb), "shared_subject_ids": len({x[0] for x in pa} & {x[0] for x in pb}),
                                   "shared_benchmark_ids": len({x[1] for x in pa} & {x[1] for x in pb})}
    rec = {}
    for sec in ["ALC"] + BUDGETS:
        def bm(t):
            acc = {}
            for r in t[sec]:
                acc.setdefault(r[1], []).append(r[3])
            return {k: sum(v) / len(v) for k, v in acc.items()}
        ma, mb = bm(tables[a]), bm(tables[b])
        sh = sorted(set(ma) & set(mb))
        if len(sh) >= 2:
            d = [mb[k] - ma[k] for k in sh]
            rec[str(sec)] = {"n_benchmarks": len(sh), "mean_diff": st.mean(d), "sd": st.stdev(d), "se": st.stdev(d) / len(d) ** 0.5}
    out["benchmark_matched"][f"{b}_minus_{a}"] = rec
out["distinct_benchmark_ids_all_runs"] = len(set().union(*[{r[1] for r in t["ALC"]} for t in tables.values()]))
out["distinct_subject_ids_all_runs"] = len(set().union(*[{r[0] for r in t["ALC"]} for t in tables.values()]))
# simulator scenario table (ALC differences vs shipped, 95% cluster-bootstrap CIs over benchmarks)
p3 = json.loads((RES / "proto_level3.json").read_text(encoding="utf-8"))
sc = {}
for s, rec in p3["scenarios"].items():
    row = {"shipped_alc": rec["methods"]["shipped"]["alc"], "oracle": rec["oracle"], "median_pair_rate": rec["unit_rate_median"]}
    for v, lab in (("nosubj_d0_mu2", "xp_nosubj_mu2"), ("nosubj_d-0.5_mu2", "xp_level_m050"), ("nosubj_d-1_mu2", "xp_level_m100")):
        m = rec["methods"][v]
        row[lab] = {"alc": m["alc"], "minus_shipped": {k: m["minus_shipped"][k] for k in ("mean", "ci_lo", "ci_hi")}}
    row["m050_minus_nosubj"] = row["xp_level_m050"]["alc"] - row["xp_nosubj_mu2"]["alc"]
    sc[s] = row
out["simulator_scenarios"] = sc
out["selected_submission"] = {"id": 1010717, "zip": "submission/candidates/xp_level_m050.zip",
                              "sha256": "cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715"}
(RES / "formative_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("runs", "overlap")}, indent=1)[:3000])
