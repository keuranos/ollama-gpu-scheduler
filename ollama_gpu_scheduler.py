#!/usr/bin/env python3
"""ollama_gpu_scheduler.py — admission controller for multiple Ollama instances.

THE PROBLEM
  Running several Ollama instances on shared GPUs (multi-user boxes, one box
  serving several agents/pipelines) produces a classic failure: a model load
  starts on a GPU without enough free VRAM and Ollama silently CPU-offloads
  part of the weights. The request then burns RAM, swap fills, and the call
  dies minutes later with a 500. Nothing in stock Ollama coordinates loads
  across instances.

THE FIX
  One proxy in front of all instances. Every Ollama-API request (native +v1)
  goes to the scheduler; it admits a model load only when the target GPU
  provably has room, evicts idle models when it doesn't, and FIFO-queues the
  request otherwise. A spill guard unloads any CPU-offloaded model that
  appears anyway (e.g. from processes that bypass the proxy), and a rogue
  watchdog flags Ollama daemons that ignore the proxy entirely.

PLACEMENT LAW
  1. Requests are admitted in FIFO order; a load happens only when a GPU fits.
  2. One large model per GPU, or several small ones within the per-GPU budget.
  3. Designated RESIDENT models live on their own instance and are never
     scheduled, queued, or evicted — their requests bypass placement entirely.
  4. Admission = free VRAM (from nvidia-smi, authoritative across processes)
     >= weights * safety + KV estimate derived from the request's num_ctx.
  5. No spill, ever: if nothing fits, the request waits; on timeout it gets an
     honest 503 instead of a doomed load.

QUICK START
  pip install aiohttp
  cp config.example.json config.json   # edit: instances, resident, affinity
  python3 ollama_gpu_scheduler.py --config config.json
  # point every caller (OLLAMA_HOST / base_url) at this proxy instead of the
  # raw instances — same API, no call-site code changes.
"""
import argparse
import asyncio
import json
import os
import time
from collections import deque

from aiohttp import web, ClientSession, ClientTimeout

DEFAULT_CONFIG_PATH = os.environ.get("GPUSCHED_CONFIG", "config.json")

# ----------------------------------------------------------------- config ---
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)

    listen = cfg.get("listen", {})
    cfg["listen_host"] = listen.get("host", "127.0.0.1")
    cfg["listen_port"] = int(listen.get("port", 11500))

    inst = {}
    for e in cfg.get("instances", []):
        gpus = e.get("gpus") or ([e["gpu_uuid"]] if e.get("gpu_uuid") else [])
        if not gpus:
            raise SystemExit("config: instance %r needs gpu_uuid or gpus[]" % e.get("name"))
        inst[e["name"]] = {"url": e["url"].rstrip("/"), "gpus": gpus}
    if not inst:
        raise SystemExit("config: at least one instance is required")
    cfg["instances"] = inst

    res = cfg.get("resident", {})
    cfg["resident_model"] = res.get("model", "")
    cfg["resident_url"] = res.get("url", "").rstrip("/")
    cfg["resident_gpu"] = res.get("gpu_uuid", "")

    mg = cfg.get("multi_gpu", {})
    cfg["multi_models"] = set(mg.get("models", []))
    cfg["multi_split_slack"] = float(mg.get("split_slack", 0.9))

    # Safety: a multi-GPU instance must never claim the resident model's GPU.
    if cfg["resident_gpu"]:
        for name, icfg in cfg["instances"].items():
            if len(icfg["gpus"]) > 1 and cfg["resident_gpu"] in icfg["gpus"]:
                raise SystemExit(
                    "config: multi-GPU instance %r includes the resident GPU" % name)

    # gpu -> instances sharing it (for cross-instance eviction before a
    # multi-GPU placement, and for multi-lock acquisition)
    cfg["gpu_shared_by"] = {}
    for name, icfg in cfg["instances"].items():
        for g in icfg["gpus"]:
            cfg["gpu_shared_by"].setdefault(g, []).append(name)

    cfg["affinity"] = cfg.get("affinity", {})
    cfg["gpu_budget_gb"] = float(cfg.get("gpu_budget_gb", 30.0))
    cfg["weight_safety"] = float(cfg.get("weight_safety", 1.12))
    cfg["kv_gb_per_8k_ctx"] = float(cfg.get("kv_gb_per_8k_ctx", 1.0))
    cfg["kv_cap_gb"] = float(cfg.get("kv_cap_gb", 6.0))

    q = cfg.get("queue", {})
    cfg["queue_max"] = int(q.get("max", 64))
    cfg["queue_wait_s"] = float(q.get("wait_s", 900))

    rw = cfg.get("rogue_watch", {})
    cfg["rogue_enabled"] = bool(rw.get("enabled", True))
    cfg["rogue_interval_s"] = float(rw.get("interval_s", 60))
    cfg["rogue_sanctioned"] = {u: int(n) for u, n in
                               (rw.get("sanctioned_users", {})).items()}

    sg = cfg.get("spill_guard", {})
    cfg["spill_enabled"] = bool(sg.get("enabled", True))
    cfg["spill_interval_s"] = float(sg.get("interval_s", 60))
    cfg["spill_threshold_gb"] = float(sg.get("threshold_gb", 0.5))

    cfg["size_cache_s"] = float(cfg.get("size_cache_s", 600))
    return cfg


CFG = None  # set in main()

API_PASSTHROUGH_GET = ("/api/tags", "/api/version", "/api/ps")

# ----------------------------------------------------------------- state ----
size_cache = {}            # model -> (ts, gb); gb <= 0.05 treated as cloud
inflight = {}              # (instance, model) -> active request count
inst_locks = {}
queue = []                 # FIFO of waiting model names
queue_event = asyncio.Event()
decisions = deque(maxlen=200)
stats = {"admitted": 0, "queued": 0, "evicted": 0,
         "rejected": 0, "queued_now_max": 0}
state = {}
started = time.time()


def note(msg):
    line = "[%s %s] %s" % (CFG.get("log_tag", "sched"),
                           time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    decisions.append(line)


# --------------------------------------------------------------- helpers ----
async def model_size_gb(sess, model):
    """Weight size in GB from any instance /api/tags (instances share a dir)."""
    now = time.time()
    hit = size_cache.get(model)
    if hit and now - hit[0] < CFG["size_cache_s"]:
        return hit[1]
    probe_url = next(iter(CFG["instances"].values()))["url"]
    try:
        async with sess.get(probe_url + "/api/tags",
                            timeout=ClientTimeout(total=8)) as r:
            for m in (await r.json()).get("models", []):
                gb = m.get("size", 0) / 1e9
                size_cache[m["name"]] = (now, gb)
    except Exception as e:
        note("size lookup failed for %s: %s" % (model, e))
    gb = size_cache.get(model, (0, None))[1]
    # Unknown model: assume a small one (conservative admission, never a
    # spill risk); a cached 0.0 (cloud model) is preserved as-is.
    return gb if gb is not None else 0.5


async def gpu_free_gb(sess, gpu_uuid):
    """Authoritative free VRAM on a GPU (accounts every process, not just ours)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        for line in out.decode().strip().splitlines():
            u, free = [x.strip() for x in line.split(",")]
            if u.startswith(gpu_uuid):
                return float(free) / 1024.0
    except Exception as e:
        note("nvidia-smi failed: %s" % e)
    return 0.0


async def ps_models(sess, base_url):
    """Models resident on an instance: {name: size_vram_gb}."""
    try:
        async with sess.get(base_url + "/api/ps",
                            timeout=ClientTimeout(total=6)) as r:
            return {m["name"]: m.get("size_vram", 0) / 1e9
                    for m in (await r.json()).get("models", [])}
    except Exception:
        return {}


def required_gb(weights_gb, body):
    """Weights (with safety margin) + rough KV allowance from num_ctx."""
    opts = body.get("options") or {}
    try:
        ctx = int(opts.get("num_ctx") or 8192)
    except (TypeError, ValueError):
        ctx = 8192
    kv = min(CFG["kv_cap_gb"], ctx / 8192.0 * CFG["kv_gb_per_8k_ctx"])
    return weights_gb * CFG["weight_safety"] + kv


def fits(free_gb, need_gb):
    return free_gb >= need_gb and (CFG["gpu_budget_gb"] - (32.0 - free_gb)) > 0


async def evict_idle(sess, inst_name, base_url, keep):
    """Unload non-resident models with no in-flight requests."""
    residents = await ps_models(sess, base_url)
    for name in list(residents):
        if name == keep or name == CFG["resident_model"]:
            continue
        if inflight.get((inst_name, name), 0) > 0:
            continue
        try:
            async with sess.post(base_url + "/api/generate",
                                 json={"model": name, "keep_alive": 0},
                                 timeout=ClientTimeout(total=30)) as r:
                await r.read()
            note("evicted %s from %s" % (name, inst_name))
            stats["evicted"] += 1
        except Exception as e:
            note("evict %s/%s failed: %s" % (inst_name, name, e))


async def try_admit(sess, inst_name, model, body):
    """Pure placement check. CALLER MUST HOLD inst_locks[inst_name]."""
    cfg = CFG["instances"][inst_name]
    gpus = cfg["gpus"]
    residents = await ps_models(sess, cfg["url"])
    if model in residents:
        return True
    wgb = await model_size_gb(sess, model)
    if wgb <= 0.05:          # cloud / remote models do not occupy VRAM
        return True
    need = required_gb(wgb, body)
    if len(gpus) == 1:
        free = await gpu_free_gb(sess, gpus[0])
        if fits(free, need):
            return True
        note("no fit on %s: need %.1fGB free %.1fGB (residents %s)"
             % (inst_name, need, free, list(residents)))
        if inflight.get((inst_name, model), 0) == 0:
            await evict_idle(sess, inst_name, cfg["url"], keep=model)
            free = await gpu_free_gb(sess, gpus[0])
            if fits(free, need):
                note("fit on %s after eviction (free %.1fGB)" % (inst_name, free))
                return True
        return False

    # ---- multi-GPU placement (tensor split across N cards) ----
    frees = [await gpu_free_gb(sess, g) for g in gpus]
    total, n = sum(frees), len(gpus)
    # Ollama splits layers roughly evenly across visible devices; require the
    # sum to fit AND every card to hold its equal share (with slack) — a
    # 2GB/55GB pair must not pass on sum alone.
    per_need = need / n
    slack = CFG["multi_split_slack"]
    if total >= need and all(f >= per_need * slack for f in frees):
        return True
    note("no multi-GPU fit on %s (%d GPUs): need %.1fGB, free %s (residents %s)"
         % (inst_name, n, need, ["%.1f" % f for f in frees], list(residents)))
    if inflight.get((inst_name, model), 0) > 0:
        return False
    # Evict idle models on EVERY instance sharing any of these GPUs (the
    # dual instance itself + single-GPU instances pinned to the same cards),
    # then re-check once.
    peers = sorted({p for g in gpus for p in CFG["gpu_shared_by"].get(g, [])})
    for peer in peers:
        if inflight.get((peer, "*"), 0):
            continue
        await evict_idle(sess, peer, CFG["instances"][peer]["url"], keep=model)
    frees = [await gpu_free_gb(sess, g) for g in gpus]
    total = sum(frees)
    if total >= need and all(f >= per_need * slack for f in frees):
        note("multi-GPU fit on %s after eviction (free %s)"
             % (inst_name, ["%.1f" % f for f in frees]))
        return True
    return False


def order_for(model):
    """Candidate instances for a model. Multi-GPU models route ONLY to
    multi-GPU instances (and vice versa) — a small model must not squat two
    cards, a big one cannot fit one."""
    multi = model in CFG["multi_models"]
    names = [n for n, c in CFG["instances"].items()
             if (len(c["gpus"]) > 1) == multi]
    pref = CFG["affinity"].get(model)
    if pref in names:
        names.remove(pref)
        names.insert(0, pref)
    return names


# --------------------------------------------------------------- dispatch ---
async def forward(app, base_url, body):
    stream = bool(body.get("stream"))
    url = base_url + app["request"].path
    sess = app["sess"]
    if stream:
        resp = web.StreamResponse(status=200)
        resp.content_type = "application/x-ndjson"
        await resp.prepare(app["request"])
        async with sess.post(url, json=body, timeout=ClientTimeout(
                total=None, sock_read=1800)) as up:
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
        await resp.write_eof()
        return resp
    async with sess.post(url, json=body,
                         timeout=ClientTimeout(total=None, sock_read=1800)) as up:
        data = await up.read()
        return web.Response(status=up.status, body=data,
                            content_type=up.content_type or "application/json")


async def dispatch(app, model, body):
    """Admit + forward one chat/generate request. Raises 503 on queue timeout."""
    stats["queued"] += 1
    deadline = time.time() + CFG["queue_wait_s"]
    while True:
        # 1. resident shortcut: never queued, never evicted, never placed
        if model == CFG["resident_model"] and CFG["resident_url"]:
            return await forward(app, CFG["resident_url"], body)
        # 2. try instances in affinity order. For single-GPU models the
        #    per-instance lock is enough. For multi-GPU models the
        #    RESERVATION acquires the instance lock plus every peer-instance
        #    lock sharing any of its GPUs, so no single-GPU placement can
        #    interleave on any card mid-load (locks ordered by id: deadlock-
        #    free; deduped via dict.fromkeys).
        for inst_name in order_for(model):
            cfg = CFG["instances"][inst_name]
            gpus = cfg["gpus"]
            if len(gpus) > 1:
                peer_names = {p for g in gpus
                              for p in CFG["gpu_shared_by"].get(g, [])}
                peer_names.discard(inst_name)
                lock_list = [inst_locks[inst_name]] + \
                    [inst_locks[p] for p in sorted(peer_names)]
                lock_list = list(dict.fromkeys(lock_list))
                lock_list.sort(key=lambda l: id(l))
                for l in lock_list:
                    await l.acquire()
                try:
                    if await try_admit(app["sess"], inst_name, model, body):
                        inflight[(inst_name, model)] = inflight.get((inst_name, model), 0) + 1
                        try:
                            out = await forward(app, cfg["url"], body)
                            stats["admitted"] += 1
                            return out
                        finally:
                            inflight[(inst_name, model)] -= 1
                finally:
                    for l in lock_list:
                        l.release()
            else:
                async with inst_locks[inst_name]:
                    if await try_admit(app["sess"], inst_name, model, body):
                        inflight[(inst_name, model)] = inflight.get((inst_name, model), 0) + 1
                        try:
                            out = await forward(app, cfg["url"], body)
                            stats["admitted"] += 1
                            return out
                        finally:
                            inflight[(inst_name, model)] -= 1
        # 3. queue
        if time.time() > deadline:
            stats["rejected"] += 1
            raise web.HTTPServiceUnavailable(
                text=json.dumps({"error": "scheduler: no GPU capacity for %s "
                                          "within %.0fs" % (model, CFG["queue_wait_s"])}),
                content_type="application/json")
        note("queueing %s (%d waiting)" % (model, len(queue)))
        queue.append(model)
        stats["queued_now_max"] = max(stats["queued_now_max"], len(queue))
        queue_event.clear()
        try:
            await asyncio.wait_for(queue_event.wait(),
                                   timeout=max(1.0, min(20.0, deadline - time.time())))
        except asyncio.TimeoutError:
            pass
        finally:
            try:
                queue.remove(model)
            except ValueError:
                pass
        await asyncio.sleep(0.5)


# ----------------------------------------------------------------- routes ---
async def handle_get(request):
    app = request.app
    app["request"] = request
    if request.path in API_PASSTHROUGH_GET:
        first = next(iter(CFG["instances"].values()))["url"]
        async with app["sess"].get(first + request.path,
                                   timeout=ClientTimeout(total=10)) as up:
            data = await up.read()
        return web.Response(status=up.status, body=data,
                            content_type="application/json")
    raise web.HTTPNotFound(text="scheduler: unknown path %s" % request.path)


async def handle_api(request):
    """POST /api/chat | /api/generate | /api/embed | /api/embeddings | /api/show"""
    app = request.app
    app["request"] = request
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="scheduler: invalid JSON")
    model = body.get("model", "")
    if not model:
        raise web.HTTPBadRequest(text="scheduler: missing model")
    if request.path in ("/api/chat", "/api/generate"):
        return await dispatch(app, model, body)
    # embeddings (both legacy /api/embeddings and batched /api/embed) / show:
    # forwarded raw to the affinity instance — embed models are small, no
    # placement needed; the body is passed through untouched so the caller's
    # response shape is preserved exactly.
    inst = CFG["instances"][order_for(model)[0]]
    async with app["sess"].post(inst["url"] + request.path, json=body,
                                timeout=ClientTimeout(total=None, sock_read=600)) as up:
        data = await up.read()
    return web.Response(status=up.status, body=data, content_type="application/json")


async def handle_v1(request):
    """OpenAI-compatible endpoint (for clients configured with OLLAMA_BASE_URL/v1)."""
    app = request.app
    app["request"] = request
    if request.path == "/v1/models":
        first = next(iter(CFG["instances"].values()))["url"]
        async with app["sess"].get(first + "/api/tags",
                                   timeout=ClientTimeout(total=10)) as up:
            tags = await up.json()
        return web.json_response({
            "object": "list",
            "data": [{"id": m["name"], "object": "model", "owned_by": "ollama"}
                     for m in tags.get("models", [])]})
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="scheduler: invalid JSON")
    model = body.get("model", "")
    if request.path == "/v1/embeddings":
        ollama_body = {"model": model, "prompt": (body.get("input") or [""])[0]}
        inst = CFG["instances"][order_for(model)[0]]
        async with app["sess"].post(inst["url"] + "/api/embeddings",
                                    json=ollama_body,
                                    timeout=ClientTimeout(total=None, sock_read=600)) as up:
            data = await up.read()
        return web.Response(status=up.status, body=data, content_type="application/json")
    ollama_body = {"model": model, "stream": bool(body.get("stream", False))}
    if "messages" in body:
        ollama_body["messages"] = body["messages"]
        path = "/api/chat"
    else:
        ollama_body["prompt"] = body.get("prompt", "")
        path = "/api/generate"
    if body.get("options"):
        ollama_body["options"] = body["options"]
    out = await dispatch(app, model, ollama_body)
    if isinstance(out, web.StreamResponse):
        return out  # NOTE: /v1 streaming returns ndjson, not SSE
    data = json.loads(out.body)
    msg = (data.get("message") or {}).get("content", "") if path == "/api/chat" \
        else data.get("response", "")
    return web.json_response({
        "id": "gpu-scheduler", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": msg}}],
        "usage": {"prompt_tokens": data.get("prompt_eval_count", 0),
                  "completion_tokens": data.get("eval_count", 0)}})


async def handle_health(request):
    return web.json_response({"ok": True, "uptime_s": int(time.time() - started)})


async def handle_status(request):
    instances = {}
    for name, cfg in CFG["instances"].items():
        try:
            async with request.app["sess"].get(cfg["url"] + "/api/ps",
                                              timeout=ClientTimeout(total=6)) as up:
                ms = (await up.json()).get("models", [])
            instances[name] = {"gpus": len(cfg["gpus"]),
                               "residents": [{"name": m["name"],
                                              "vram_gb": round(m.get("size_vram", 0) / 1e9, 1)}
                                             for m in ms]}
        except Exception:
            instances[name] = {"gpus": len(cfg["gpus"]), "residents": []}
    return web.json_response({
        "queue_depth": len(queue), "queue": list(queue),
        "inflight": {"%s/%s" % k: v for k, v in inflight.items() if v},
        "instances": instances,
        "resident": CFG["resident_model"],
        "multi_gpu_models": sorted(CFG["multi_models"]),
        "stats": stats,
        "recent_decisions": list(decisions)[-30:],
        "rogue_ollama": state.get("rogue", []),
    })


# --------------------------------------------------------------- watchdogs --
async def spill_guard(app):
    """Unload any model on a managed instance that is CPU-offloading
    (size_vram << size). Skips models with in-flight requests."""
    while CFG["spill_enabled"]:
        try:
            for inst_name, cfg in CFG["instances"].items():
                async with app["sess"].get(cfg["url"] + "/api/ps",
                                           timeout=ClientTimeout(total=6)) as r:
                    for m in (await r.json()).get("models", []):
                        spill = (m["size"] - m.get("size_vram", 0)) / 1e9
                        if spill <= CFG["spill_threshold_gb"]:
                            continue
                        if inflight.get((inst_name, m["name"]), 0) > 0:
                            continue
                        note("SPILL DETECTED %s/%s: %.1fGB in RAM — unloading"
                             % (inst_name, m["name"], spill))
                        try:
                            async with app["sess"].post(
                                    cfg["url"] + "/api/generate",
                                    json={"model": m["name"], "keep_alive": 0},
                                    timeout=ClientTimeout(total=30)) as r2:
                                await r2.read()
                            stats["spills_caught"] = stats.get("spills_caught", 0) + 1
                        except Exception as e:
                            note("spill unload failed: %s" % e)
        except Exception as e:
            note("spill_guard error: %s" % e)
        await asyncio.sleep(CFG["spill_interval_s"])


async def rogue_watch(app):
    """Flag `ollama serve` daemons outside the sanctioned set — evidence that
    someone is bypassing the proxy."""
    while CFG["rogue_enabled"]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ps", "-eo", "user,args",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            data, _ = await proc.communicate()
            per_user = {}
            for line in data.decode().splitlines():
                s = line.strip()
                parts = s.split(None, 1)
                if len(parts) < 2 or "ollama serve" not in s or "bash" in parts[1][:20]:
                    continue
                per_user.setdefault(parts[0], []).append(s)
            rogue = []
            for user, procs in per_user.items():
                allowed = CFG["rogue_sanctioned"].get(user, 0)
                if user not in CFG["rogue_sanctioned"]:
                    rogue.extend(procs)
                elif len(procs) > allowed:
                    rogue.extend(procs[allowed:])
            if rogue:
                note("ROGUE ollama serve: %s" % "; ".join(rogue))
            state["rogue"] = rogue
        except Exception as e:
            note("rogue scan failed: %s" % e)
        await asyncio.sleep(CFG["rogue_interval_s"])


# -------------------------------------------------------------------- app ---
async def on_start(app):
    app["sess"] = ClientSession()
    if CFG["rogue_enabled"]:
        app["watch"] = asyncio.create_task(rogue_watch(app))
    if CFG["spill_enabled"]:
        app["spill"] = asyncio.create_task(spill_guard(app))
    note("up on %s:%d (resident %r; pool %s)"
         % (CFG["listen_host"], CFG["listen_port"], CFG["resident_model"],
            list(CFG["instances"])))


async def on_cleanup(app):
    for key in ("watch", "spill"):
        if key in app:
            app[key].cancel()
    await app["sess"].close()


def main():
    global CFG
    ap = argparse.ArgumentParser(description="Ollama GPU scheduler")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    CFG = load_config(args.config)
    inst_locks.update({name: asyncio.Lock() for name in CFG["instances"]})

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/sched/status", handle_status)
    for p in API_PASSTHROUGH_GET:
        app.router.add_get(p, handle_get)
    app.router.add_post("/api/chat", handle_api)
    app.router.add_post("/api/generate", handle_api)
    app.router.add_post("/api/embeddings", handle_api)
    app.router.add_post("/api/embed", handle_api)
    app.router.add_post("/api/show", handle_api)
    app.router.add_post("/v1/chat/completions", handle_v1)
    app.router.add_post("/v1/completions", handle_v1)
    app.router.add_post("/v1/embeddings", handle_v1)
    app.router.add_get("/v1/models", handle_v1)
    web.run_app(app, host=CFG["listen_host"], port=CFG["listen_port"], print=None)


if __name__ == "__main__":
    main()