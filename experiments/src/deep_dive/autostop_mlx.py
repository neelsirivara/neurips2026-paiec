"""Mac-side auto-stop for the Qwen3-14B rating (coordinator's rule of 2026-10-29).
Once every needed prompt of the inventory benchmarks plus the first N_MIN AS benchmarks (seeded order) is rated, the
script projects when the FULL set would finish. The rate is measured over the last 30 minutes, with mem_guard pauses
included. If the projection is later than DEADLINE (HH:MM, same day), it stops the rating process (SIGTERM; the wrapper's
trap releases ~/neurips/GPU_LOCK). Otherwise it lets the run finish the full set by itself. Polls every 5 minutes.
    ~/neurips/mlx_venv/bin/python -I autostop_mlx.py N_MIN DEADLINE      e.g.  10 23:00
"""
import json, os, signal, subprocess, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rate_items as RI
W = os.path.expanduser("~/neurips/paiec")
N_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEADLINE = sys.argv[2] if len(sys.argv) > 2 else "23:00"
order = open(os.path.join(W, "rating_order.txt")).read().strip().split(",")
target = set(order[:33]) | set(order[33:33 + N_MIN])
need_t, need_all = set(), set()
for l in open(os.path.join(W, "deep_dive_items.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    k = RI.prompt_key(RI.build_prompt(r["content"], r["features"]))
    need_all.add(k)
    if r["dir"] in target:
        need_t.add(k)
print(time.strftime("%F %T"), f"target (inv + {N_MIN} AS) prompts {len(need_t)}, full {len(need_all)}, deadline {DEADLINE}", flush=True)
hist = []
while True:
    done = set()
    try:
        for l in open(os.path.join(W, "ratings_qwen3_14b.jsonl"), encoding="utf-8"):
            try:
                x = json.loads(l)
                if x.get("s1") is not None:
                    done.add(x["key"])
            except Exception:
                pass
    except FileNotFoundError:
        pass
    now = time.time()
    hist.append((now, len(done & need_all)))
    hist = [h for h in hist if now - h[0] <= 1800] or hist[-1:]
    rate = (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], 1.0)   # prompts / s, pauses included
    left_t, left_all = len(need_t - done), len(need_all - done)
    eta_full = datetime.fromtimestamp(now + left_all / rate) if rate > 0 else None
    print(time.strftime("%F %T"), f"left target {left_t}, left full {left_all}, rate {rate * 60:.1f}/min, eta full {eta_full}", flush=True)
    if left_t == 0 and left_all > 0:
        dl = datetime.now().replace(hour=int(DEADLINE[:2]), minute=int(DEADLINE[3:]), second=0)
        if eta_full is None or eta_full > dl:
            for pid in subprocess.run(["pgrep", "-f", "rate_items_mlx.py"], capture_output=True, text=True).stdout.split():
                try:
                    os.kill(int(pid), signal.SIGCONT)   # in case mem_guard paused it, so SIGTERM is handled
                    os.kill(int(pid), signal.SIGTERM)
                    print(time.strftime("%F %T"), "target set complete and full-set eta past deadline: SIGTERM", pid, flush=True)
                except Exception as e:
                    print("kill failed", e, flush=True)
            break
    if left_all == 0 or not subprocess.run(["pgrep", "-f", "rate_items_mlx.py"], capture_output=True).stdout:
        print(time.strftime("%F %T"), "rating finished or ended by itself", flush=True)
        break
    time.sleep(300)
