"""Generate deep_dive/rater_gemini_appendix.py.txt from the generic parts of rater_appendix.py.txt."""
from pathlib import Path

D = Path(r"C:\Users\neels\Downloads\gen_papers_2\neurips_challenges\paiec\experiments\src\deep_dive")
src = (D / "rater_appendix.py.txt").read_text(encoding="utf-8")


def block(start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


generic = block("def _rater_dir():", "def _pid_alive(pid):")
hooks = src[src.index("def rater_epoch():"):]

head = r'''

# ============================================================================= Gemini item-rating offset (deep dive 2026-10-29)
# Appended to the xp_level_m050 model.py by deep_dive/build_rater_gemini.py, ONLY if the pre-registered Gemini gate passed
# (PREREG_items_AMENDMENT_4.md).  Adds  w * (s1 - mu) / sd + b  to the prior offset of every item whose rating is
# available, where s1 = logit(p/100) and p is GEMINI_MODEL's stated probability (pre-registered prompt).  Everything else
# is m050.  The API key is read ONLY from submission_config.json next to this file ({"gemini_api_key": "..."}); the
# shipped file carries an empty key, which keeps the rater off (plain m050).  The key goes in the x-goog-api-key header,
# never in a URL, and is never printed or logged.  Safety:
#   * stdlib HTTPS, per-call timeout RATER_TIMEOUT, at most RATER_RETRIES retries on 429/5xx with backoff;
#   * per-process rate limit RATER_RPM_PER_PROCESS; ratings shared through a disk cache across workers and checkpoints;
#   * labelled entries are rated in a background thread by ONE worker per labelled list (O_EXCL lock), targets on demand
#     (at most RATER_MAX_PER_CALL synchronous ratings per call);
#   * no new calls after RATER_WALL_BUDGET seconds since the run started; the rater switches off in a process after
#     RATER_MAX_ERRORS consecutive failures;
#   * until the first successful rating every offset is 0 (plain m050, also when the API fails throughout); after that
#     an unrated item gets offset b (one level for labelled entries and targets);
#   * skipped entirely under the organisers' local smoke check (PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1) or PAIEC_RATER_DISABLE=1.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
GEMINI_GEN_CONFIG = __GEN_CONFIG__
RATER_W = __RATER_W__
RATER_B = __RATER_B__
RATER_MU = __RATER_MU__
RATER_SD = __RATER_SD__
RATER_TIMEOUT = 20.0
RATER_RETRIES = 2
RATER_RPM_PER_PROCESS = 40
RATER_THREADS = 4
RATER_MAX_PER_CALL = 2
RATER_MAX_ERRORS = 20
RATER_WALL_BUDGET = 5.0 * 3600.0
RATER_CONFIG_FILE = "submission_config.json"
RATER_CONTENT_CHARS, RATER_FEATURE_CHARS = 1500, 300
RATER_PROMPT = ("You are estimating how hard a benchmark task is for state-of-the-art AI systems released in 2025-2026 (for "
                "example GPT-5, Claude Opus 4, Gemini 2.5 Pro), possibly running inside an agent harness. Read the task and "
                "estimate the probability, from 0 to 100, that such a system completes it correctly.\n\n"
                "Task:\n<<<\n{content}\n>>>\n\nItem attributes: {features}\n\n"
                "Reply with the integer probability only (0-100), nothing else.")
_RATER_SKIP = (os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST") == "1" or os.environ.get("PAIEC_RATER_DISABLE") == "1")
_R = {"state": "init", "cache": {}, "cache_mtime": {}, "dir": None, "t0": None, "key": None, "errors": 0,
      "next_t": 0.0, "eleader": set()}
_R_LOCK = _threading.RLock()
_R_RATE = _threading.Lock()


'''

gem = r'''def _rater_config_key():
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), RATER_CONFIG_FILE)
        with open(p, encoding="utf-8") as fh:
            k = json.load(fh).get("gemini_api_key", "")
        return k.strip() if isinstance(k, str) else ""
    except Exception:
        return ""


def _rater_start():
    if _R["state"] != "init":
        return
    _rater_run_start()
    k = _rater_config_key()
    if not k:
        _R["state"] = "off"
        return
    _R["key"] = k
    _R["state"] = "ready"


def _gemini_http(model, body, timeout):
    """One generateContent call; returns the parsed JSON (raises on HTTP / network errors)."""
    import urllib.request
    req = urllib.request.Request(GEMINI_URL % model, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"x-goog-api-key": _R["key"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _rater_wait_rate():
    import time as _t
    with _R_RATE:
        now = _t.monotonic()
        t = max(now, _R["next_t"])
        _R["next_t"] = t + 60.0 / RATER_RPM_PER_PROCESS
    d = t - _t.monotonic()
    if d > 0:
        _t.sleep(min(d, 5.0))


def _rater_score(prompt):
    """s1 for one prompt, or None.  Never raises."""
    import time as _t
    import urllib.error
    if _R["state"] != "ready" or _R["errors"] >= RATER_MAX_ERRORS or _t.time() - _rater_run_start() > RATER_WALL_BUDGET:
        return None
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": GEMINI_GEN_CONFIG}
    delay = 1.0
    for attempt in range(RATER_RETRIES + 1):
        try:
            _rater_wait_rate()
            out = _gemini_http(GEMINI_MODEL, body, RATER_TIMEOUT)
            cand = (out.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []) if not p.get("thought"))
            m = re.search(r"\d{1,3}(?:\.\d+)?", text or "")
            if not m:
                _R["errors"] += 1
                return None
            v = float(m.group(0))
            if not 0 <= v <= 100:
                _R["errors"] += 1
                return None
            _R["errors"] = 0
            q = min(max(v / 100.0, 0.005), 0.995)
            return math.log(q / (1 - q))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < RATER_RETRIES:
                _t.sleep(delay)
                delay *= 2
                continue
            _R["errors"] += 1
            return None
        except Exception:
            if attempt < RATER_RETRIES:
                _t.sleep(delay)
                delay *= 2
                continue
            _R["errors"] += 1
            return None
    return None


def _rater_store(pairs):
    d = _rater_dir()
    with _R_LOCK:
        for k, v in pairs:
            _R["cache"][k] = v
    if pairs and d:
        try:
            with open(os.path.join(d, "ratings_%d.jsonl" % os.getpid()), "a", encoding="utf-8") as fh:
                for k, v in pairs:
                    fh.write("%s\t%r\n" % (k, v))
        except Exception:
            pass


def _rater_rate_many(prompts_keys):
    try:
        from concurrent.futures import ThreadPoolExecutor
        todo = [(p, k) for p, k in prompts_keys if k and k not in _R["cache"]]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=RATER_THREADS) as ex:
            res = list(ex.map(lambda pk: (pk[1], _rater_score(pk[0])), todo))
        _rater_store([(k, v) for k, v in res if v is not None])
    except Exception:
        pass


def _rater_lead_labelled(fp, prompts, keys):
    """One worker per labelled list rates its entries in the background (O_EXCL lock file per fingerprint)."""
    if fp in _R["eleader"]:
        return
    _R["eleader"].add(fp)
    d = _rater_dir()
    if not d:
        return
    try:
        fd = os.open(os.path.join(d, "elead_%s" % _rater_key(str(fp))[:16]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except Exception:
        return
    th = _threading.Thread(target=_rater_rate_many, args=(list(zip(prompts, keys)),), daemon=True)
    th.start()


def rater_offsets(items, rate_missing=True):
    """Item offsets (logits) for a list of item dicts; never raises."""
    if _RATER_SKIP or not items:
        return [0.0] * len(items)
    try:
        prompts = [_rater_prompt(it) if isinstance(it, dict) else "" for it in items]
        keys = [_rater_key(p) if p else "" for p in prompts]
        with _R_LOCK:
            _rater_start()
            if _R["state"] != "ready":
                return [0.0] * len(items)
            _rater_reload_cache()
        if rate_missing:
            missing = [(p, k) for p, k in zip(prompts, keys) if k and k not in _R["cache"]][:RATER_MAX_PER_CALL]
            if missing:
                _rater_store([(k, v) for k, v in ((k, _rater_score(p)) for p, k in missing) if v is not None])
        if not _R["cache"]:            # nothing rated yet (or the API is failing): plain m050
            return [0.0] * len(items)
        out = []
        for k in keys:
            v = _R["cache"].get(k) if k else None
            out.append(RATER_W * (v - RATER_MU) / RATER_SD + RATER_B if v is not None else RATER_B)
        return out
    except Exception:
        return [0.0] * len(items)


'''


def rep(s, old, new):
    assert s.count(old) == 1, old[:60]
    return s.replace(old, new)


hooks = rep(hooks, '''            active = _R["state"] == "ready" or bool(cache)
            can_rate = _R["state"] == "ready" and n_rated < sum(1 for k in keys if k)''',
            '''            active = _R["state"] == "ready" and bool(cache)
            can_rate = False
        if _R["state"] == "ready" and n_rated < sum(1 for k in keys if k):
            _rater_lead_labelled(fp, [_rater_prompt(it) if it else "" for it in items], keys)''')
hooks = rep(hooks, '''    offs = rater_offsets(items) if active or can_rate else [0.0] * len(items)''',
            '''    offs = rater_offsets(items, rate_missing=False) if active else [0.0] * len(items)''')
hooks = rep(hooks, '''                act_after = _R["state"] == "ready" or bool(_R["cache"])''',
            '''                act_after = _R["state"] == "ready" and bool(_R["cache"])''')
out = head + generic + gem + hooks
(D / "rater_gemini_appendix.py.txt").write_text(out, encoding="utf-8")
print(len(out), "chars written")
