"""Item-rating study, Gemini route (PREREG_items_AMENDMENT_4.md). Standard library only.

The key handling stays inside neurips_challenges/gemini_key.py: this script calls that helper's own REST function
(key in the x-goog-api-key header, never in a URL). It never reads the key into a variable of its own, and never
prints, logs or stores it. gemini_key.generate() returns only the text, which is why the helper's _request is called:
the response's usageMetadata is needed for the cost accounting.

HARD GUARD: competition data may only go through a PAID-tier key. Free-tier inputs may be used for training, with
human review. Every mode that sends items therefore refuses to run without --paid-tier-confirmed. Pass it only after
the coordinator confirms that the user has linked billing.

    python -I rate_items_gemini.py dryrun --items ITEMS.jsonl          # no network: builds the requests, prints sizes
    python -I rate_items_gemini.py pilot  --items ITEMS.jsonl --paid-tier-confirmed [--n 50]
    python -I rate_items_gemini.py full   --items ITEMS.jsonl --paid-tier-confirmed --out ratings_gemini_2_5_flash.jsonl

Output lines: {"key", "idx", "dir", "s1", "p", "backend": "gemini", "model", "config", "in_tok", "out_tok",
"thought_tok", "sec", "error"?}. Keys are sha1(model + "\n" + prompt), the same convention as rate_items_claude.py, so
eval_items.py maps them back. Resumable: keys already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))          # neurips_challenges/ (gemini_key.py)
sys.path.insert(0, str(HERE))
import rate_items_claude as RC                    # noqa: E402  (same verbalised prompt, parser and key convention)

MODEL = "gemini-2.5-flash"
CONFIGS = {   # tried in order; the first that the API accepts AND that parses is used for the whole run
    "nothink": {"temperature": 0.0, "maxOutputTokens": 16, "thinkingConfig": {"thinkingBudget": 0}},
    "default": {"temperature": 0.0, "maxOutputTokens": 1024},
}
PRICE = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40)}   # $/MTok in, out incl. thinking (paid)
RES = HERE.parent.parent / "results" / "deep_dive" / "items"


class RateLimiter:
    def __init__(self, rpm):
        self.dt = 60.0 / max(rpm, 1)
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


def call(model, prompt, cfg, timeout=60.0, tries=6, limiter=None):
    import gemini_key
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    delay = 2.0
    for attempt in range(tries):
        if limiter:
            limiter.wait()
        try:
            return gemini_key._request("/models/%s:generateContent" % model, gemini_key.read_key(), body, timeout=timeout), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
                continue
            return None, "http_%d" % e.code
        except Exception as e:  # timeouts, connection errors
            if attempt < tries - 1:
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
                continue
            return None, type(e).__name__
    return None, "exhausted"


def parse(out):
    cand = (out.get("candidates") or [{}])[0]
    if cand.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
        return None, "finish_" + str(cand.get("finishReason"))
    text = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []) if not p.get("thought"))
    import re
    m = re.search(r"\d{1,3}(?:\.\d+)?", text or "")
    if not m:
        return None, "unparsed"
    v = float(m.group(0))
    return (v, None) if 0 <= v <= 100 else (None, "range")


def rate_one(k, p, r, cfg_name, limiter):
    t0 = time.time()
    out, err = call(MODEL, p, CONFIGS[cfg_name], limiter=limiter)
    rec = {"key": k, "idx": r["idx"], "dir": r["dir"], "backend": "gemini", "model": MODEL, "config": cfg_name}
    if out is None:
        rec.update(p=None, s1=None, error=err)
    else:
        v, perr = parse(out)
        um = out.get("usageMetadata", {})
        rec.update(p=v, s1=(RC.s_of(v) if v is not None else None), in_tok=um.get("promptTokenCount"),
                   out_tok=um.get("candidatesTokenCount"), thought_tok=um.get("thoughtsTokenCount", 0))
        if perr:
            rec["error"] = perr
    rec["sec"] = round(time.time() - t0, 3)
    return rec


def todo_list(items_path, order="", done=()):
    items = RC.load_items(items_path, order)
    seen, out = set(done), []
    for r in items:
        p = RC.build_prompt(r["content"], r["features"])
        k = RC.key_of(p, MODEL)
        if k not in seen:
            seen.add(k)
            out.append((k, p, r))
    return out


def pick_config(sample, limiter):
    """First config the API accepts and whose answers parse on 3 probe items."""
    for name in CONFIGS:
        recs = [rate_one(k, p, r, name, limiter) for k, p, r in sample[:3]]
        if sum(1 for x in recs if x.get("p") is not None) >= 2:
            return name, recs
    return None, []


def cmd_dryrun(a):
    t = todo_list(a.items, a.order)
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": t[0][1]}]}], "generationConfig": CONFIGS["nothink"]})
    print(json.dumps({"unique_prompts": len(t), "mean_prompt_chars": sum(len(x[1]) for x in t) / len(t),
                      "example_body_bytes": len(body), "configs": CONFIGS}, indent=1))


def run(a, todo, out_path, tag):
    limiter = RateLimiter(a.rpm)
    cfg, first = pick_config(todo, limiter)
    if cfg is None:
        raise SystemExit("no generation config both accepted and parsed on the probe items")
    lock = threading.Lock()
    recs = list(first)
    with open(out_path, "a", encoding="utf-8") as fh:
        for x in first:
            fh.write(json.dumps(x) + "\n")
        rest = todo[3:]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, x in enumerate(ex.map(lambda t: rate_one(t[0], t[1], t[2], cfg, limiter), rest), 1):
                with lock:
                    fh.write(json.dumps(x) + "\n")
                    recs.append(x)
                if i % 200 == 0:
                    fh.flush()
                    print(f"{tag}: {i + 3}/{len(todo)}", flush=True)
    return cfg, recs


def cmd_pilot(a):
    if not a.paid_tier_confirmed:
        raise SystemExit("refusing: pass --paid-tier-confirmed only after billing is linked (free-tier inputs may be used for training)")
    allt = todo_list(a.items, a.order)
    step = max(1, len(allt) // a.n)
    pilot = allt[::step][: a.n]
    t0 = time.time()
    cfg, recs = run(a, pilot, RES / "ratings_gemini_pilot.jsonl", "pilot")
    wall = time.time() - t0
    ok = [x for x in recs if x.get("p") is not None]
    n = len(recs)
    tin = sum(x.get("in_tok") or 0 for x in recs) / n
    tout = sum((x.get("out_tok") or 0) + (x.get("thought_tok") or 0) for x in recs) / n
    pin, pout = PRICE[MODEL]
    per = (tin * pin + tout * pout) / 1e6
    res = {"model": MODEL, "config": cfg, "generation_config": CONFIGS[cfg], "n": n, "parsed": len(ok),
           "errors": {e: sum(1 for x in recs if x.get("error") == e) for e in {x.get("error") for x in recs if x.get("error")}},
           "mean_in_tokens": tin, "mean_out_plus_thought_tokens": tout, "mean_latency_s": sum(x["sec"] for x in recs) / n,
           "wall_s": wall, "workers": a.workers, "rpm": a.rpm, "per_item_usd": per,
           "projected_full_usd": per * len(allt), "n_unique_prompts": len(allt),
           "projected_full_minutes_at_rpm": len(allt) / a.rpm,
           "projected_scoring_run_usd_per_1000_items": per * 1000,
           "p_distribution": sorted(x["p"] for x in ok)}
    (RES / f"api_pilot_{MODEL}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "p_distribution"}, indent=1))


def cmd_full(a):
    if not a.paid_tier_confirmed:
        raise SystemExit("refusing: pass --paid-tier-confirmed only after billing is linked")
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                x = json.loads(l)
                if x.get("p") is not None:
                    done.add(x["key"])
            except Exception:
                pass
    todo = todo_list(a.items, a.order, done)
    print(f"{len(todo)} prompts to rate ({len(done)} already done)", flush=True)
    if todo:
        cfg, recs = run(a, todo, a.out, "full")
        print(f"done: config {cfg}, parsed {sum(1 for x in recs if x.get('p') is not None)}/{len(recs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dryrun", "pilot", "full"))
    ap.add_argument("--items", required=True)
    ap.add_argument("--order", default=(RES / "rating_order.txt").read_text().strip() if (RES / "rating_order.txt").exists() else "")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(RES / "ratings_gemini_2_5_flash.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=600)
    ap.add_argument("--paid-tier-confirmed", action="store_true")
    a = ap.parse_args()
    {"dryrun": cmd_dryrun, "pilot": cmd_pilot, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
