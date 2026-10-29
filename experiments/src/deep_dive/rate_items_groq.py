"""Item-rating study, Groq route (PREREG_items_AMENDMENT_5.md). Standard library only.

The key stays inside neurips_challenges/groq_key.py. This script calls the helper's chat() and its _request() (for the
free model list); both put the key in the Authorization header. My code never stores, prints or logs it.

    python -I rate_items_groq.py models                                     # model ids the key can see (no tokens)
    python -I rate_items_groq.py pilot --items ITEMS.jsonl [--n 50]          # probe, pilot, rate-limit headers, feasibility
    python -I rate_items_groq.py full  --items ITEMS.jsonl --model M --deadline "2026-10-30 15:00" [--out ...]

Model rule (pre-registered): openai/gpt-oss-120b (reasoning_effort "low") if the key sees it AND its limits allow
rating every prompt before the deadline; else llama-3.3-70b-versatile (non-reasoning) under the same condition; else the
model whose limits allow the most prompts. The rating proceeds in the pre-registered order (inventory benchmarks first,
then the AS benchmarks in their seeded order). Whatever is complete at the deadline is what the gate evaluates
(benchmarks fully rated; see the amendment).
Output lines: {"key", "idx", "dir", "s1", "p", "backend": "groq", "model", "in_tok", "out_tok", "sec", "error"?}.
Keys are sha1(model + "\n" + prompt), so eval_items.py maps them back. Resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))          # neurips_challenges/ (groq_key.py)
sys.path.insert(0, str(HERE))
import rate_items_claude as RC                    # noqa: E402  (same verbalised prompt, s_of, key convention)

PREFERENCE = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile")
PARAMS = {"openai/gpt-oss-120b": {"temperature": 0.0, "max_tokens": 512, "reasoning_effort": "low", "include_reasoning": False},
          "llama-3.3-70b-versatile": {"temperature": 0.0, "max_tokens": 8}}
RES = HERE.parent.parent / "results" / "deep_dive" / "items"
HDR = ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-requests",
       "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")


def _dur(s):
    """Groq reset strings like '2m59.56s', '7.66s', '1h2m3s' -> seconds."""
    if not s:
        return None
    tot, num = 0.0, ""
    for part in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", str(s)):
        v, u = float(part[0]), part[1]
        tot += v * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[u]
    return tot


class Pacer:
    """Spaces requests to stay under the observed RPM/TPM; honours retry-after on 429."""

    def __init__(self, rpm=30, tpm=8000, tok_per_req=600):
        self.lock = threading.Lock()
        self.set(rpm, tpm, tok_per_req)
        self.next = 0.0

    def set(self, rpm, tpm, tok_per_req):
        per_req = max(60.0 / max(rpm, 1), 60.0 * tok_per_req / max(tpm, 1))
        self.dt = per_req * 1.05

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)

    def penalise(self, seconds):
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)


def call(model, prompt, pacer, timeout=60.0, tries=6):
    import groq_key
    delay = 2.0
    for attempt in range(tries):
        pacer.wait()
        try:
            out, hdr = groq_key.chat(model, [{"role": "user", "content": prompt}], **PARAMS[model])
            return out, {k.lower(): v for k, v in hdr.items()}, None
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                ra = e.headers.get("retry-after") if e.headers else None
                pacer.penalise(float(ra) if ra else delay)
                delay = min(delay * 2, 120.0)
                continue
            if e.code in (500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 120.0)
                continue
            return None, {}, "http_%d" % e.code
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            return None, {}, type(e).__name__
    return None, {}, "exhausted"


def parse(out):
    ch = (out.get("choices") or [{}])[0]
    text = ((ch.get("message") or {}).get("content") or "").strip()
    m = re.search(r"\d{1,3}(?:\.\d+)?", text)
    if not m:
        return None, "unparsed"
    v = float(m.group(0))
    return (v, None) if 0 <= v <= 100 else (None, "range")


def rate_one(model, k, p, r, pacer):
    t0 = time.time()
    out, hdr, err = call(model, p, pacer)
    rec = {"key": k, "idx": r["idx"], "dir": r["dir"], "backend": "groq", "model": model}
    if out is None:
        rec.update(p=None, s1=None, error=err)
    else:
        v, perr = parse(out)
        u = out.get("usage") or {}
        rec.update(p=v, s1=(RC.s_of(v) if v is not None else None), in_tok=u.get("prompt_tokens"),
                   out_tok=u.get("completion_tokens"))
        if perr:
            rec["error"] = perr
    rec["sec"] = round(time.time() - t0, 3)
    return rec, hdr


def todo_list(items_path, model, order="", done=()):
    items = RC.load_items(items_path, order)
    seen, out = set(done), []
    for r in items:
        p = RC.build_prompt(r["content"], r["features"])
        k = RC.key_of(p, model)
        if k not in seen:
            seen.add(k)
            out.append((k, p, r))
    return out


def visible_models():
    import groq_key
    out, _ = groq_key._request("/models", groq_key.read_key())
    return sorted(m.get("id", "") for m in out.get("data", []))


def limits_from(hdr):
    g = lambda k: hdr.get(k)
    num = lambda k: float(g(k)) if g(k) not in (None, "") else None
    return {"rpd": num("x-ratelimit-limit-requests"), "tpm": num("x-ratelimit-limit-tokens"),
            "remaining_requests": num("x-ratelimit-remaining-requests"), "remaining_tokens": num("x-ratelimit-remaining-tokens"),
            "reset_requests_s": _dur(g("x-ratelimit-reset-requests")), "reset_tokens_s": _dur(g("x-ratelimit-reset-tokens"))}


def feasibility(lim, n_prompts, tok_per_req, hours, rpm_guess=30):
    """Prompts ratable in `hours` under the observed daily-request and per-minute-token caps (RPM is not in the headers)."""
    per_min = min(rpm_guess, (lim["tpm"] or 1e9) / max(tok_per_req, 1))
    by_minute = per_min * 60 * hours
    by_day = (lim["rpd"] or 1e12) * max(1.0, hours / 24.0)
    cap = min(by_minute, by_day, n_prompts)
    return {"prompts_per_minute": per_min, "ratable_in_window": int(cap), "all_feasible": cap >= n_prompts,
            "hours_for_all": n_prompts / max(per_min, 1e-9) / 60.0}


def cmd_models(a):
    print(json.dumps(visible_models(), indent=1))


def cmd_pilot(a):
    vis = visible_models()
    cands = [m for m in PREFERENCE if m in vis]
    if not cands:
        raise SystemExit("none of the pre-registered models is visible to this key: %s" % ", ".join(vis[:20]))
    report = {"visible": vis, "candidates": {}}
    hours = max((datetime.strptime(a.deadline, "%Y-%m-%d %H:%M") - datetime.now()).total_seconds() / 3600.0, 0.0)
    for model in cands:
        allt = todo_list(a.items, model, a.order)
        step = max(1, len(allt) // a.n)
        pilot = allt[::step][: a.n if model == cands[0] else 10]
        pacer = Pacer()
        recs, last = [], {}
        t0 = time.time()
        for i, (k, p, r) in enumerate(pilot):
            rec, hdr = rate_one(model, k, p, r, pacer)
            recs.append(rec)
            if hdr:
                last = hdr
                lim = limits_from(hdr)
                if lim["tpm"]:
                    tpr = max(1, sum((x.get("in_tok") or 0) + (x.get("out_tok") or 0) for x in recs) / len(recs))
                    pacer.set(30, lim["tpm"], tpr)
        ok = [x for x in recs if x.get("p") is not None]
        n = len(recs)
        tin = sum(x.get("in_tok") or 0 for x in recs) / n
        tout = sum(x.get("out_tok") or 0 for x in recs) / n
        lim = limits_from(last)
        feas = feasibility(lim, len(allt), tin + tout, hours)
        report["candidates"][model] = {"n": n, "parsed": len(ok), "mean_in_tokens": tin, "mean_out_tokens": tout,
                                       "mean_latency_s": sum(x["sec"] for x in recs) / n, "wall_s": time.time() - t0,
                                       "limits": lim, "n_unique_prompts": len(allt), "window_hours": hours,
                                       "feasibility": feas,
                                       "errors": {e: sum(1 for x in recs if x.get("error") == e)
                                                  for e in {x.get("error") for x in recs if x.get("error")}}}
        with open(RES / "ratings_groq_pilot.jsonl", "a", encoding="utf-8") as fh:
            for x in recs:
                fh.write(json.dumps(x) + "\n")
    # pre-registered model rule
    pick = None
    for m in cands:
        c = report["candidates"][m]
        if c["parsed"] >= 0.9 * c["n"] and c["feasibility"]["all_feasible"]:
            pick = m
            break
    if pick is None:
        pick = max(cands, key=lambda m: report["candidates"][m]["feasibility"]["ratable_in_window"]
                   if report["candidates"][m]["parsed"] >= 0.9 * report["candidates"][m]["n"] else -1)
    report["chosen_model"] = pick
    (RES / "api_pilot_groq.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({m: {k: v for k, v in c.items()} for m, c in report["candidates"].items()}, indent=1))
    print("chosen:", pick)


def cmd_full(a):
    deadline = datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
    out_path = a.out or str(RES / ("ratings_groq_%s.jsonl" % a.model.replace("/", "_")))
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                x = json.loads(l)
                if x.get("p") is not None:
                    done.add(x["key"])
            except Exception:
                pass
    todo = todo_list(a.items, a.model, a.order, done)
    print(f"{len(todo)} prompts to rate ({len(done)} done); deadline {deadline}", flush=True)
    pacer = Pacer(a.rpm, a.tpm, a.tok_per_req)
    lock = threading.Lock()
    stop = threading.Event()

    def work(t):
        if stop.is_set() or datetime.now() >= deadline:
            stop.set()
            return None
        rec, hdr = rate_one(a.model, t[0], t[1], t[2], pacer)
        lim = limits_from(hdr) if hdr else {}
        if lim.get("remaining_requests") is not None and lim["remaining_requests"] < 5:
            pacer.penalise((lim.get("reset_requests_s") or 60.0) + 1.0)
        return rec

    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            if rec is None:
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                fh.flush()
                print(f"{i}/{len(todo)}", flush=True)
    print("stopped at deadline" if stop.is_set() else "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("models", "pilot", "full"))
    ap.add_argument("--items", default="")
    ap.add_argument("--order", default=(RES / "rating_order.txt").read_text().strip() if (RES / "rating_order.txt").exists() else "")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="")
    ap.add_argument("--deadline", default="2026-10-30 15:00")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=30)
    ap.add_argument("--tpm", type=float, default=8000)
    ap.add_argument("--tok-per-req", type=float, default=600)
    a = ap.parse_args()
    {"models": cmd_models, "pilot": cmd_pilot, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
