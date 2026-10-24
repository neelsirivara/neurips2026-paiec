"""Where and how often does the shipped novelty filter fire?  (experiments/results/novelty_filter_stats.json)

    python novelty_stats.py

Replays the acquisition stream of every pair (seed 0, the trajectory used in the main results) under
plain stream order ("first") and under the shipped novelty policy, without any predictor call, and
records per pair the number of skipped candidates, the maximum cosine similarity among the skipped
candidates, and whether the two acquired sets differ.  Needs no labels; cheap.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import labeling  # noqa: E402
import model  # noqa: E402
import run_experiments as R  # noqa: E402
import simulate  # noqa: E402

RESULTS = HERE.parent / "results"


def replay(raw_pairs: list, seed: int) -> dict:
    rows = []
    for raw in raw_pairs:
        subject, sid, bid, items = raw[0], raw[1], raw[2], raw[3]
        meta = raw[4] if len(raw) > 4 else {}
        rng = random.Random(f"{seed}|{sid}|{bid}")
        pair = harness.make_pair(dict(subject), sid, bid, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
        if pair is None:
            continue
        labeling.POLICY = "first"
        first = harness.run_acquisition(pair, labeling.acquisition_policy, pair["order"], needs_prediction=False)
        labeling.POLICY = "novelty"
        nov = harness.run_acquisition(pair, labeling.acquisition_policy, pair["order"], needs_prediction=False)
        considered = pair["order"][: max(len(first), pair["order"].index(nov[-1]) + 1 if nov else 0)]
        skipped = [i for i in considered if i not in nov]
        max_sim = 0.0
        for i in skipped:
            acquired_before = [j for j in nov if pair["order"].index(j) < pair["order"].index(i)]
            if acquired_before:
                sims = model.similarities(pair["acq"][i][0], [pair["acq"][j][0] for j in acquired_before])
                max_sim = max(max_sim, max(sims) if sims else 0.0)
        rows.append({"subject_id": sid, "benchmark_id": bid, "bench": meta.get("bench", bid), "model": meta.get("model", sid),
                     "harness": meta.get("harness", ""), "n_skipped": len(skipped), "differs": first != nov,
                     "max_sim_skipped": round(max_sim, 3), "n_candidates_seen": len(considered)})
    fired = [r for r in rows if r["n_skipped"] > 0]
    summary = {"n_pairs": len(rows), "n_pairs_fired": len(fired), "total_skipped": sum(r["n_skipped"] for r in rows),
               "max_skipped_per_pair": max((r["n_skipped"] for r in rows), default=0),
               "max_sim_skipped": max((r["max_sim_skipped"] for r in rows), default=0.0),
               "benches_fired": sorted({r["bench"] for r in fired}), "threshold": labeling.DEDUP_SIM, "pairs": rows}
    labeling.POLICY = "novelty"
    return summary


def main() -> None:
    out = {}
    import real_testbed
    raw = real_testbed.load_pairs()
    if raw:
        out["real"] = replay(raw, 0)
    syn = simulate.generate_pairs(0, R.N_SUBJECTS, R.N_BENCHMARKS, R.MAIN_SIGMA, R.MAIN_GAMMA)
    out["synthetic"] = replay(syn, 0)
    for k, v in out.items():
        print(f"{k}: filter fired on {v['n_pairs_fired']}/{v['n_pairs']} pairs, {v['total_skipped']} candidates skipped "
              f"(max {v['max_skipped_per_pair']} per pair; max cosine {v['max_sim_skipped']}); benches {v['benches_fired']}")
    (RESULTS / "novelty_filter_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", RESULTS / "novelty_filter_stats.json")


if __name__ == "__main__":
    main()
