"""Summarise a saved Codabench "Scoring output" (formative feedback) of a PAIEC submission.

    python -I parse_feedback.py path/to/1010370_scoring_stdout.txt [more files ...]

Save the text of Codabench -> My submissions -> <id> -> Logs -> Scoring output as a .txt file first
(it is private to the submitter).  The parser is tolerant: it looks for section headers that mention
"ALC" or "Budget <n>" and, below them, rows containing a subject id, a benchmark id, an integer count
and one or two floats (Brier [, ECE]).  It prints, per budget, the pair-equal mean Brier and ECE,
the number of pairs and benchmarks, and the ALC recomputed from the per-budget means
(0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31).

Only aggregate numbers are printed.  It deliberately does NOT try to recover per-pair success
rates (the competition rules forbid extracting private test data from the feedback).
"""
import re
import sys
from collections import defaultdict

W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
ROW = re.compile(r"(subject_\w+)\W+(benchmark_\w+)\W+(\d+)\W+([0-9]*\.[0-9]+)(?:\W+([0-9]*\.[0-9]+))?")
HDR_BUDGET = re.compile(r"budget\s*[:=#]?\s*(\d+)", re.I)


def parse(text):
    section = None
    tables = defaultdict(list)
    for line in text.splitlines():
        m = ROW.search(line)
        if m and section is not None:
            tables[section].append((m.group(1), m.group(2), int(m.group(3)), float(m.group(4)),
                                    float(m.group(5)) if m.group(5) else None))
            continue
        h = HDR_BUDGET.search(line)
        if h and not m:
            section = int(h.group(1))
        elif "alc" in line.lower() and not m:
            section = "ALC"
    return tables


def main(paths):
    for p in paths:
        text = open(p, encoding="utf-8", errors="replace").read()
        t = parse(text)
        print(f"== {p}")
        if not t:
            print("   no tables recognised; please check the file format")
            continue
        means = {}
        for sec in sorted(t, key=lambda s: (-1 if s == "ALC" else s)):
            rows = t[sec]
            br = [r[3] for r in rows]
            ece = [r[4] for r in rows if r[4] is not None]
            n_items = sum(r[2] for r in rows)
            benches = {r[1] for r in rows}
            subjects = {r[0] for r in rows}
            mb = sum(br) / len(br)
            if sec != "ALC":
                means[sec] = mb
            print(f"   {str(sec):>4}: pairs {len(rows):3d} subjects {len(subjects):3d} benchmarks {len(benches):3d} "
                  f"eval subject-items {n_items:5d}  mean Brier {mb:.4f}" + (f"  mean ECE {sum(ece)/len(ece):.4f}" if ece else ""))
        if all(b in means for b in W):
            print(f"   ALC from per-budget means: {sum(W[b] * means[b] for b in W):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
