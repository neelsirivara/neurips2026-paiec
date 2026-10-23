"""Generate a small real item-level response testbed with local Ollama models.

This is an *evaluation-only* testbed (out of the PAIEC inventory; never used to fit
the shipped predictor).  It produces, for each (model, benchmark) pair, one binary
correctness label per item under a fixed direct-answer prompt (greedy decoding).

Benchmarks (all read from local copies; nothing is fetched at run time):
  * gsm8k  : first 200 test items of openai/gsm8k (HF cache)
  * svamp  : first 200 test items of ChilleD/SVAMP (HF cache); item_features=type=<Type>
  * bbh_mix: 30 items from each of 10 BIG-Bench-Hard subtasks (lukaemon/bbh parquet
             files under _downloads/paiec_data/bbh); item_features=subtask=<name>

Models (Ollama, CPU): qwen2.5:1.5b-instruct, qwen2.5:3b-instruct, qwen2.5:7b-instruct,
llama3.2:3b.

Output: experiments/results/real_testbed/<bench>__<model>.jsonl, one record per item:
  {"idx", "bench", "item_content", "item_features", "gold", "pred_raw", "pred", "correct"}
The script is resume-safe (already-written idx are skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "real_testbed"
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
BBH_DIR = HERE.parents[2] / "_downloads" / "paiec_data" / "bbh"

GSM8K_PARQUET = (HF_HUB / "datasets--openai--gsm8k" / "snapshots" /
                 "740312add88f781978c0658806c59bc2815b9866" / "main" / "test-00000-of-00001.parquet")
SVAMP_PARQUET = (HF_HUB / "datasets--ChilleD--SVAMP" / "snapshots" /
                 "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef" / "data" / "test-00000-of-00001.parquet")

BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "formal_fallacies", "hyperbaton", "navigate", "sports_understanding",
    "temporal_sequences", "web_of_lies",
]
MODELS = ["qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct", "llama3.2:3b", "qwen2.5:7b-instruct"]

MATH_SYSTEM = ("You are a precise math solver. Reply with ONLY the final numeric answer, "
               "no words, no units, no explanation.")
BBH_SYSTEM = ("Answer the question. Reply with ONLY the answer: the option in parentheses "
              "such as (A), or a single word such as Yes, No, True, False, valid, invalid.")

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s: str) -> str | None:
    s = str(s).split("####")[-1].strip().replace(",", "").replace("$", "").rstrip(".")
    m = _NUM.search(s)
    if not m:
        return None
    try:
        f = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return str(int(round(f))) if abs(f - round(f)) < 1e-6 else f"{f:.4f}".rstrip("0").rstrip(".")


def norm_bbh(s: str) -> str:
    s = str(s).strip().lower()
    m = re.search(r"\(([a-z])\)", s)
    if m:
        return f"({m.group(1)})"
    s = re.sub(r"[^a-z]", " ", s).strip()
    return s.split()[0] if s else ""


def load_items(bench: str) -> list[dict]:
    if bench == "gsm8k":
        df = pd.read_parquet(GSM8K_PARQUET).head(200)
        return [{"idx": i, "item_content": r["question"], "item_features": "",
                 "gold": norm_num(r["answer"])} for i, r in enumerate(df.to_dict("records"))]
    if bench == "svamp":
        df = pd.read_parquet(SVAMP_PARQUET).head(200)
        return [{"idx": i, "item_content": (str(r["Body"]).strip() + " " + str(r["Question"]).strip()).strip(),
                 "item_features": f"type={r['Type']}", "gold": norm_num(r["Answer"])}
                for i, r in enumerate(df.to_dict("records"))]
    if bench == "bbh_mix":
        out = []
        for sub in BBH_SUBTASKS:
            df = pd.read_parquet(BBH_DIR / f"{sub}.parquet").head(30)
            for r in df.to_dict("records"):
                out.append({"idx": len(out), "item_content": str(r["input"]).strip(),
                            "item_features": f"subtask={sub}", "gold": norm_bbh(r["target"])})
        return out
    raise ValueError(bench)


def ollama_chat(host: str, model: str, system: str, user: str, num_predict: int) -> str:
    body = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.0, "seed": 0, "num_predict": num_predict, "num_ctx": 2048,
                    "num_thread": int(os.environ.get("PAIEC_NUM_THREAD", "6"))},
    }
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]


def pick_host() -> str:
    hosts = [h for h in (os.environ.get("OLLAMA_HOST", "").replace("http://", ""), "127.0.0.1:11434", "127.0.0.1:11435") if h]
    for timeout in (3, 30, 120):  # a busy server (other local jobs) can take a while to answer
        for h in hosts:
            try:
                urllib.request.urlopen(f"http://{h}/api/tags", timeout=timeout).read(64)
                return h
            except Exception:
                continue
    print("warning: no Ollama host answered the probe; using 127.0.0.1:11434 anyway", flush=True)
    return "127.0.0.1:11434"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="svamp,gsm8k,bbh_mix")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many items per pair")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    host = pick_host()
    print(f"ollama host {host}", flush=True)
    for model in args.models.split(","):
        for bench in args.benches.split(","):
            items = load_items(bench)
            if args.limit:
                items = items[: args.limit]
            out_path = RESULTS / f"{bench}__{re.sub(r'[^a-z0-9.]+', '-', model.lower())}.jsonl"
            done = set()
            if out_path.exists():
                for line in out_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        done.add(json.loads(line)["idx"])
            todo = [it for it in items if it["idx"] not in done]
            print(f"[{model} | {bench}] {len(done)} done, {len(todo)} to do", flush=True)
            t0 = time.time()
            n_correct = 0
            with out_path.open("a", encoding="utf-8") as fh:
                for k, it in enumerate(todo, 1):
                    system = BBH_SYSTEM if bench == "bbh_mix" else MATH_SYSTEM
                    npred = 16 if bench == "bbh_mix" else 24
                    for attempt in range(3):
                        try:
                            raw = ollama_chat(host, model, system, it["item_content"], npred)
                            break
                        except Exception as exc:  # network hiccup: retry
                            print(f"  retry {attempt}: {exc}", flush=True)
                            time.sleep(5)
                    else:
                        raw = ""
                    pred = norm_bbh(raw) if bench == "bbh_mix" else norm_num(raw)
                    correct = (pred is not None) and (pred == it["gold"])
                    n_correct += int(correct)
                    rec = dict(it, bench=bench, model=model, pred_raw=raw[:200], pred=pred, correct=bool(correct))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if k % 25 == 0 or k == len(todo):
                        el = time.time() - t0
                        print(f"  {k}/{len(todo)} acc_so_far={n_correct / k:.3f} {el / k:.1f}s/item", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
