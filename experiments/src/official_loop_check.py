"""Round 4: run the shipped model.py / labeling.py inside the organisers' streaming client.

Drives ``run_streaming`` of the verbatim copy of ``tools/streaming_ingestion.py`` (validator/tools/) with a
local mock coordinator that follows the documented protocol (fixed candidate order per pair, a streaming
``acquire`` event per candidate with ``context``, an ``evaluate`` event per budget 0, 1, 3, 7, 15, 31 whose
``labeled`` is the GLOBAL list of all pairs' acquired labels, then ``finished``).  Data: the 17-pair real
Ollama testbed (splits and candidate orders as in run_experiments.py, seed 0).  Checks that the hook is
called in streaming mode without predictions, that every probability is valid, and that the ALC equals the
one computed by xpair_sim.real_testbed_acq for the same policy.

    python official_loop_check.py [--dir ../../submission]
"""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_DIR = Path(sys.argv[sys.argv.index("--dir") + 1]).resolve() if "--dir" in sys.argv else HERE
sys.path.insert(0, str(_DIR))
sys.path.insert(1, str(HERE))
sys.path.insert(2, str(HERE / "validator" / "tools"))
import labeling  # noqa: E402
import model  # noqa: E402
import harness  # noqa: E402
import real_testbed  # noqa: E402
import streaming_ingestion as SI  # noqa: E402

BUDGETS = harness.BUDGETS


class MockCoordinator:
    def __init__(self, pairs):
        self.pairs = pairs
        self.events = self._script()
        self.pending = None
        self.labeled = []
        self.got = [[] for _ in pairs]
        self.predictions = {}
        self.n_acquire_events = 0
        self.eid = 0

    def _script(self):
        pos = [0] * len(self.pairs)
        for b in BUDGETS:
            if b > 0:
                for j, (pr, _) in enumerate(self.pairs):
                    order = pr["order"]
                    while len(self.got[j]) < b and pos[j] < len(order):
                        idx = order[pos[j]]
                        ctx = {"subject_id": pr["subject_id"], "benchmark_id": pr["benchmark_id"],
                               "labels_acquired": len(self.got[j]), "labels_remaining": harness.MAX_LABELS - len(self.got[j]),
                               "max_labels": harness.MAX_LABELS, "items_remaining": len(order) - pos[j]}
                        pos[j] += 1
                        reply = yield {"type": "acquire", "input": [pr["subject"], pr["acq"][idx][0]], "context": ctx}
                        if reply.get("query") is True:
                            self.got[j].append(idx)
                            self.labeled.append([[pr["subject"], pr["acq"][idx][0]], pr["acq"][idx][1]])
            targets = [{"input": [pr["subject"], it]} for pr, _ in self.pairs for it, _ in pr["eval"]]
            reply = yield {"type": "evaluate", "budget": b, "targets": targets}
            self.predictions[b] = reply["probabilities"]
        yield {"type": "finished"}

    def exchange(self, payload):
        if self.pending is None:
            ev = next(self.events)
        else:
            ev = self.events.send(payload)
        self.pending = ev
        self.eid += 1
        ev = dict(ev, protocol="streaming_alc_v1", event_id=self.eid, labeled=copy.deepcopy(self.labeled))
        if ev["type"] == "acquire":
            self.n_acquire_events += 1
        if ev["type"] == "finished":
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["budget", "row", "probability"])
            for b, ps in self.predictions.items():
                for i, p in enumerate(ps):
                    w.writerow([b, i, p])
            ev["predictions_csv"] = out.getvalue()
        return ev


def main() -> int:
    seed = 0
    raw = real_testbed.load_pairs()
    pairs = []
    for subject, sid, bid, items, meta in raw:
        rng = random.Random(f"{seed}|{sid}|{bid}")
        pair = harness.make_pair(dict(subject), sid, bid, [(dict(t[0]),) + tuple(t[1:]) for t in items], rng)
        if pair is not None:
            pairs.append((pair, meta))
    coord = MockCoordinator(pairs)
    calls = {"predict": 0}

    def predict(x, labeled):
        calls["predict"] += 1
        return model.predict(x, labeled)

    def prediction_stream(model_path, unique, labeled, concurrency, deadline, per_call_timeout):
        frozen = copy.deepcopy(labeled)
        for i, x in enumerate(unique):
            yield i, predict(copy.deepcopy(x), frozen)

    def group_inputs(inputs, n, deadline):
        unique, groups, index = [], [], {}
        for row, x in enumerate(inputs):
            key = json.dumps(x, sort_keys=True)
            if key not in index:
                index[key] = len(unique)
                unique.append(x)
                groups.append([])
            groups[index[key]].append(row)
        return unique, groups

    def validate_score(v):
        f = float(v)
        assert 0.0 <= f <= 1.0 and f == f
        return f

    @contextlib.contextmanager
    def call_window(seconds, name):
        yield

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.json"
        cfg.write_text(json.dumps({"protocol": "streaming_alc_v1"}))
        stats = SI.run_streaming(config_path=cfg, predict=predict, acquisition_function=labeling.acquisition_function,
                                 model_path=str(_DIR), output_dir=Path(tmp) / "out", concurrency=1,
                                 deadline=time.monotonic() + 3600, per_call_timeout=60, acquisition_timeout=60,
                                 call_window=call_window, validate_score=validate_score,
                                 prediction_stream=prediction_stream, group_inputs=group_inputs,
                                 log=lambda m: None, exchange=coord.exchange)
    # ALC per pair from the official loop's predictions
    alcs = []
    k = 0
    for pr, _ in pairs:
        n = len(pr["eval"])
        ys = [y for _, y in pr["eval"]]
        br = {b: harness.brier(coord.predictions[b][k:k + n], ys) for b in BUDGETS}
        alcs.append(harness.alc(br))
        k += n
    res = {"tested_dir": str(_DIR), "policy": labeling.POLICY, "xp_enabled": bool(model.XP_ENABLED),
           "acquisition_mode": stats["acquisition_mode"], "acquisition_calls": stats["acquisition_calls"],
           "acquisition_prediction_calls": stats["acquisition_prediction_calls"],
           "prediction_calls_count": stats["prediction_calls_count"], "labels_final": len(coord.labeled),
           "labels_per_pair": [len(g) for g in coord.got], "n_pairs": len(pairs),
           "alc_mean": sum(alcs) / len(alcs), "seconds": time.time() - t0}
    ref = HERE.parent / "results" / "round4" / "xpair_real_acq.json"
    if ref.exists():
        r = json.loads(ref.read_text(encoding="utf-8"))["runs"][0]
        key = "alc_fresh" if labeling.POLICY == "fresh" else "alc_first"
        res["alc_xpair_sim_same_policy"] = r[key]["mean"]
        res["matches_xpair_sim"] = abs(r[key]["mean"] - res["alc_mean"]) < 1e-9
    out = HERE.parent / "results" / "round4" / "official_loop_check.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    assert res["acquisition_mode"] == "stream" and res["acquisition_prediction_calls"] == 0
    assert all(n == harness.MAX_LABELS for n in res["labels_per_pair"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
