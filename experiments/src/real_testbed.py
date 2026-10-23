"""Load the local real item-level response testbed as PAIEC-style pairs.

Sources (all local, evaluation-only, out of the PAIEC inventory):
  * experiments/results/real_testbed/<bench>__<model>.jsonl  -- produced by
    real_testbed_generate.py (direct-answer harness, four Ollama models, three benchmarks)
  * ../../agents/agent_01/experiments/results/real/{gsm8k,svamp}_ollama-*.jsonl -- earlier
    8-shot chain-of-thought runs of the same Qwen2.5 models (greedy_correct field)

Each (model, harness, benchmark) with >= 80 items becomes one pair.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OWN_DIR = HERE.parent / "results" / "real_testbed"
AGENT01_DIR = HERE.parent / "results" / "real_testbed_external"  # frozen snapshot of agents/agent_01/experiments/results/real
SVAMP_PARQUET = (Path.home() / ".cache" / "huggingface" / "hub" / "datasets--ChilleD--SVAMP" / "snapshots" /
                 "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef" / "data" / "test-00000-of-00001.parquet")

MODEL_META = {
    "qwen2.5:1.5b-instruct": ("Qwen2.5-1.5B-Instruct", "Alibaba Cloud", "2024-09-19"),
    "qwen2.5:3b-instruct": ("Qwen2.5-3B-Instruct", "Alibaba Cloud", "2024-09-19"),
    "qwen2.5:7b-instruct": ("Qwen2.5-7B-Instruct", "Alibaba Cloud", "2024-09-19"),
    "llama3.2:3b": ("Llama-3.2-3B-Instruct", "Meta", "2024-09-25"),
}
SLUG_TO_MODEL = {re.sub(r"[^a-z0-9.]+", "-", m.lower()): m for m in MODEL_META}
AGENT01_SLUGS = {"ollama-qwen2-5-1-5b-instruct": "qwen2.5:1.5b-instruct",
                 "ollama-qwen2-5-3b-instruct": "qwen2.5:3b-instruct",
                 "ollama-qwen2-5-7b-instruct": "qwen2.5:7b-instruct"}


def _subject(model: str, harness: str, access_date: str) -> dict:
    name, provider, release = MODEL_META[model]
    return {"normalized_name": name, "provider": provider, "release_date": release,
            "access_date": access_date, "harness": harness, "harness_version": "",
            "reasoning_effort": "", "subject_features_extra": "quantization=Q4_K_M"}


def _bench_id(name: str) -> str:
    return "bench_" + hashlib.sha1(name.encode()).hexdigest()[:8]


def _svamp_types() -> dict:
    try:
        df = pd.read_parquet(SVAMP_PARQUET)
    except Exception:
        return {}
    return {(str(r["Body"]).strip() + " " + str(r["Question"]).strip()).strip(): f"type={r['Type']}"
            for r in df.to_dict("records")}


def load_raw() -> list:
    """Return list of dicts {model, harness, bench, items: [(item, label)], source}."""
    out = []
    for path in sorted(OWN_DIR.glob("*.jsonl")) if OWN_DIR.exists() else []:
        bench, slug = path.stem.split("__", 1)
        model = SLUG_TO_MODEL.get(slug)
        if model is None:
            continue
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(({"item_content": r["item_content"], "item_features": r.get("item_features", ""),
                           "interactors": ""}, int(bool(r["correct"]))))
        out.append({"model": model, "harness": "ollama-direct-answer", "bench": bench, "items": items,
                    "source": "own"})
    svamp_types = _svamp_types()
    if AGENT01_DIR.exists():
        for path in sorted(AGENT01_DIR.glob("*_ollama-*.jsonl")):
            m = re.match(r"(gsm8k|svamp)_(ollama-.*)$", path.stem)
            if not m or m.group(2) not in AGENT01_SLUGS:
                continue
            items = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                q = r["question"]
                feats = svamp_types.get(q, "") if m.group(1) == "svamp" else ""
                items.append(({"item_content": q, "item_features": feats, "interactors": ""},
                              int(bool(r["greedy_correct"]))))
            if items:
                out.append({"model": AGENT01_SLUGS[m.group(2)], "harness": "ollama-cot-8shot",
                            "bench": m.group(1), "items": items, "source": "agent_01"})
    return out


def load_pairs(min_items: int = 80) -> list:
    """Return raw pairs (subject, subject_id, benchmark_id, items) for the harness."""
    pairs = []
    for rec in load_raw():
        if len(rec["items"]) < min_items:
            continue
        subject = _subject(rec["model"], rec["harness"], "2026-10-21" if rec["source"] == "own" else "2026-06-15")
        sid = f"{rec['model']}|{rec['harness']}"
        pairs.append((subject, sid, _bench_id(rec["bench"]), [(dict(it), y) for it, y in rec["items"]],
                      {"bench": rec["bench"], "model": rec["model"], "harness": rec["harness"],
                       "n_items": len(rec["items"]), "accuracy": sum(y for _, y in rec["items"]) / len(rec["items"])}))
    return pairs


if __name__ == "__main__":
    for p in load_pairs():
        print(p[4])
