"""Aggregate paired comparison of formative runs on the (subject_id, benchmark_id) pairs they share.

    python -I paired_feedback.py NAME=path/to/scoring_stdout.txt NAME2=path2 ...

The competition page states that anonymous IDs are permanent and that matching IDs denote the same subject or
benchmark, so rows of two of our submissions can be compared on shared pairs.  Only aggregates are printed: per
pair of runs, the number of shared pairs and the mean (and sd) of the per-pair differences in ALC and in each
budget's Brier.  No pair-level values, success rates or IDs are printed or stored.
"""
import sys
import statistics as st
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_feedback import parse  # noqa: E402

runs = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    t = parse(open(path, encoding="utf-8", errors="replace").read())
    runs[name] = {sec: {(r[0], r[1]): r[3] for r in rows} for sec, rows in t.items()}
names = list(runs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        shared = sorted(set(runs[a]["ALC"]) & set(runs[b]["ALC"]))
        print(f"{b} - {a}: shared pairs {len(shared)}")
        if not shared:
            continue
        for sec in ["ALC", 0, 1, 3, 7, 15, 31]:
            d = [runs[b][sec][k] - runs[a][sec][k] for k in shared if k in runs[a].get(sec, {}) and k in runs[b].get(sec, {})]
            sd = st.stdev(d) if len(d) > 1 else float("nan")
            print(f"   {str(sec):>4}: mean diff {st.mean(d):+.4f}  sd {sd:.4f}  se {sd / len(d) ** 0.5:.4f}  n {len(d)}")


# ---- benchmark-matched comparison (aggregate): per run, mean over its pairs on each benchmark; then the mean and sd
# over shared benchmarks of the run-to-run difference.  Subjects differ between runs, so this removes benchmark
# difficulty but not subject variation.
print("\nbenchmark-matched (equal weight per shared benchmark):")
def bench_means(run, sec):
    acc = {}
    for (s, b), v in run.get(sec, {}).items():
        acc.setdefault(b, []).append(v)
    return {b: sum(v) / len(v) for b, v in acc.items()}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        for sec in ["ALC", 0, 1, 3, 7, 15, 31]:
            ma, mb = bench_means(runs[a], sec), bench_means(runs[b], sec)
            sh = sorted(set(ma) & set(mb))
            if len(sh) < 2:
                continue
            d = [mb[k] - ma[k] for k in sh]
            sd = st.stdev(d)
            print(f"   {b} - {a} {str(sec):>4}: shared benchmarks {len(sh)} mean diff {st.mean(d):+.4f} sd {sd:.4f} se {sd / len(d) ** 0.5:.4f}")
