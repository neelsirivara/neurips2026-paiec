"""Item-rating study, API route: rate items with Claude through the Message Batches API (PREREG_items_AMENDMENT_3.md).

The client comes ONLY from neurips_challenges/anthropic_key.py (`from anthropic_key import client`). This script
never reads, prints, logs or stores the key. Run with plain `python` (the SDK lives in user site-packages), from this
directory, which holds only our own code. The items file is passed by path.

    python rate_items_claude.py pilot  --items ITEMS.jsonl --model claude-opus-5-5 [--n 50]
        count_tokens on n items (free) + n real requests (small cost), then a projected cost for all items
    python rate_items_claude.py submit --items ITEMS.jsonl --model M --state STATE.json [--order a,b,...] [--limit N]
        one batch (or several of <= 10,000 requests) of every unrated prompt; batch ids are saved to STATE.json
    python rate_items_claude.py collect --state STATE.json --out ratings_claude_<model>.jsonl
        poll until ended, then write one JSON line per item: {"key", "idx", "dir", "s1", "p", "backend", "model", ...}

Signal (fixed in the amendment): the model states a probability 0-100 that a 2025-26 frontier system completes the
task correctly. s1 = logit(clip(p / 100, 0.005, 0.995)). The Claude API has no token log-probabilities, so the probability
is verbalised. Claude Opus 5.5 cannot run with thinking off, so effort is set to "low". Claude Haiku 5.5 runs adaptive
thinking at effort "low". The cache key is the SHA-1 of the exact prompt; results are keyed by custom_id = that key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))          # neurips_challenges/ (anthropic_key.py)

PROMPT_API = ("You are estimating how hard a benchmark task is for state-of-the-art AI systems released in 2025-2026 (for "
              "example GPT-5, Claude Opus 4, Gemini 2.5 Pro), possibly running inside an agent harness. Read the task and "
              "estimate the probability, from 0 to 100, that such a system completes it correctly.\n\n"
              "Task:\n<<<\n{content}\n>>>\n\nItem attributes: {features}\n\n"
              "Reply with the integer probability only (0-100), nothing else.")
CONTENT_CHARS, FEATURE_CHARS = 1500, 300
MAX_TOKENS = 2048
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}   # $/MTok in, out (batch = 50%)


def build_prompt(content, features):
    c = " ".join(str(content or "").split())[:CONTENT_CHARS]
    f = " ".join(str(features or "").split())[:FEATURE_CHARS] or "none"
    return PROMPT_API.format(content=c, features=f)


def key_of(prompt, model):
    return hashlib.sha1((model + "\n" + prompt).encode("utf-8")).hexdigest()


def params(prompt, model):
    p = {"model": model, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}],
         "output_config": {"effort": "low"}}
    return p


def parse_p(msg):
    if getattr(msg, "stop_reason", None) in ("refusal", "max_tokens"):
        return None, msg.stop_reason
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    m = re.search(r"\d{1,3}(?:\.\d+)?", text)
    if not m:
        return None, "unparsed"
    v = float(m.group(0))
    if not 0 <= v <= 100:
        return None, "range"
    return v, None


def s_of(p):
    q = min(max(p / 100.0, 0.005), 0.995)
    return math.log(q / (1 - q))


def load_items(path, order=""):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    if order:
        pri = {d: i for i, d in enumerate(order.split(","))}
        items.sort(key=lambda r: (pri.get(r["dir"], len(pri)), r["idx"]))
    return items


def unique_prompts(items, model):
    seen, out = set(), []
    for r in items:
        p = build_prompt(r["content"], r["features"])
        k = key_of(p, model)
        if k not in seen:
            seen.add(k)
            out.append((k, p, r))
    return out


def cmd_pilot(a):
    from anthropic_key import client
    c = client()
    todo = unique_prompts(load_items(a.items, a.order), a.model)
    allp = len(todo)
    pilot = todo[:: max(1, allp // a.n)][: a.n]       # spread over the ordered list
    tin, tout, ok, fails, rec = 0, 0, 0, {}, []
    for k, p, r in pilot:
        n_in = c.messages.count_tokens(model=a.model, messages=[{"role": "user", "content": p}]).input_tokens
        msg = c.messages.create(**params(p, a.model))
        v, err = parse_p(msg)
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        ok += v is not None
        if err:
            fails[err] = fails.get(err, 0) + 1
        rec.append({"key": k, "count_tokens": n_in, "in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                    "p": v, "err": err})
    pin, pout = PRICES.get(a.model, (None, None))
    n = len(pilot)
    proj = None
    if pin is not None:
        per = (tin / n * pin + tout / n * pout) / 1e6
        proj = {"per_item_standard_usd": per, "all_items_batch_usd": per * allp * 0.5, "all_items_standard_usd": per * allp}
    res = {"model": a.model, "n_pilot": n, "n_unique_prompts": allp, "mean_in": tin / n, "mean_out": tout / n,
           "parsed": ok, "failures": fails, "projection": proj, "records": rec}
    out = HERE.parent.parent / "results" / "deep_dive" / "items" / f"api_pilot_{a.model}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))


def cmd_submit(a):
    from anthropic_key import client
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    c = client()
    state = json.loads(Path(a.state).read_text()) if Path(a.state).exists() else {"model": a.model, "batches": []}
    if state["model"] != a.model:
        raise SystemExit("state file belongs to another model")
    todo = unique_prompts(load_items(a.items, a.order), a.model)
    if a.limit:
        todo = todo[: a.limit]
    done = set(state.get("submitted_keys", []))
    todo = [t for t in todo if t[0] not in done]
    for i in range(0, len(todo), 10000):
        chunk = todo[i:i + 10000]
        b = c.messages.batches.create(requests=[Request(custom_id=k, params=MessageCreateParamsNonStreaming(**params(p, a.model)))
                                                for k, p, _ in chunk])
        state["batches"].append({"id": b.id, "n": len(chunk), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        state.setdefault("submitted_keys", []).extend(k for k, _, _ in chunk)
        Path(a.state).write_text(json.dumps(state, indent=1))
        print(f"batch {b.id}: {len(chunk)} requests")


def cmd_collect(a):
    from anthropic_key import client
    c = client()
    state = json.loads(Path(a.state).read_text())
    items = load_items(a.items) if a.items else []
    by_key = {}
    for r in items:
        by_key.setdefault(key_of(build_prompt(r["content"], r["features"]), state["model"]), r)
    with open(a.out, "a", encoding="utf-8") as fh:
        for b in state["batches"]:
            while True:
                st = c.messages.batches.retrieve(b["id"])
                if st.processing_status == "ended":
                    break
                print(f"{b['id']}: {st.processing_status} {st.request_counts}", flush=True)
                time.sleep(60)
            for res in c.messages.batches.results(b["id"]):
                r = by_key.get(res.custom_id, {})
                rec = {"key": res.custom_id, "idx": r.get("idx"), "dir": r.get("dir"), "backend": "api", "model": state["model"]}
                if res.result.type == "succeeded":
                    msg = res.result.message
                    v, err = parse_p(msg)
                    rec.update(p=v, s1=(s_of(v) if v is not None else None), in_tok=msg.usage.input_tokens,
                               out_tok=msg.usage.output_tokens)
                    if err:
                        rec["error"] = err
                else:
                    rec.update(p=None, s1=None, error=res.result.type)
                fh.write(json.dumps(rec) + "\n")
    print("collected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pilot", "submit", "collect"))
    ap.add_argument("--items", default="")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--order", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--state", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
