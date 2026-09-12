#!/usr/bin/env python3
"""test_gpu_scheduler.py — offline unit tests for the placement law.

Run: python3 test_gpu_scheduler.py
No live ollama needed; nvidia-smi and instance /api/ps are monkeypatched.
"""
import asyncio
import sys
import tempfile
import time

sys.path.insert(0, ".")
import ollama_gpu_scheduler as gs


def seed_size(model, gb):
    gs.size_cache[model] = (time.time(), gb)


def make_cfg(tmpdir, extra=""):
    conf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=tmpdir)
    conf.write("""
{
  "instances": [
    {"name": "gpu-a", "url": "http://127.0.0.1:11436", "gpu_uuid": "GPU-aaaa"},
    {"name": "gpu-b", "url": "http://127.0.0.1:11439", "gpu_uuid": "GPU-bbbb"}
  ],
  "resident": {"model": "muse:latest", "url": "http://127.0.0.1:11438", "gpu_uuid": "GPU-cccc"},
  "affinity": {"big:31b": "gpu-a", "code:27b": "gpu-b"},
  "rogue_watch": {"enabled": false},
  "spill_guard": {"enabled": false},
  "gpu_budget_gb": 30.0
%s}
""" % extra)
    conf.close()
    return conf.name


def set_gpu_free(fake):
    async def fake_free(sess, uuid):
        for k, v in fake.items():
            if uuid.startswith(k):
                return v
        return 0.0
    gs.gpu_free_gb = fake_free


async def run_all():
    # --- single-GPU law (existing) ---
    set_gpu_free({"GPU-aaaa": 29.0})
    async def ps_empty(sess, url): return {}
    gs.ps_models = ps_empty
    seed_size("tinyA", 1.0)
    async with gs.inst_locks["gpu-a"]:
        assert await gs.try_admit(None, "gpu-a", "tinyA", {"options": {"num_ctx": 8192}})
    print("PASS small model fits")

    set_gpu_free({"GPU-aaaa": 8.5})
    async def ps_big(sess, url): return {"big:31b": 20.2}
    gs.ps_models = ps_big
    async def ev_noop(sess, inst, url, **kw): pass
    gs.evict_idle = ev_noop
    seed_size("code:27b", 17.7)
    async with gs.inst_locks["gpu-a"]:
        assert not await gs.try_admit(None, "gpu-a", "code:27b", {"options": {"num_ctx": 8192}})
    print("PASS large blocked even after eviction attempt")

    set_gpu_free({"GPU-bbbb": 6.0})
    async def ps_small(sess, url): return {"embed:0.6b": 0.3}
    gs.ps_models = ps_small
    async def ev_frees(sess, inst, url, **kw):
        set_gpu_free({"GPU-bbbb": 26.0})
    gs.evict_idle = ev_frees
    async with gs.inst_locks["gpu-b"]:
        assert await gs.try_admit(None, "gpu-b", "code:27b", {"options": {"num_ctx": 8192}})
    print("PASS evict-then-fit")

    assert all("muse:latest" != n for n in gs.CFG["instances"])
    print("PASS resident instance not in pool")

    seed_size("cloudy:cloud", 0.0)
    async with gs.inst_locks["gpu-a"]:
        assert await gs.try_admit(None, "gpu-a", "cloudy:cloud", {})
    print("PASS cloud bypass")

    assert gs.order_for("code:27b")[0] == "gpu-b"
    assert gs.order_for("big:31b")[0] == "gpu-a"
    assert set(gs.order_for("unknown")) == {"gpu-a", "gpu-b"}
    print("PASS affinity ordering")

    async def ps_res(sess, url): return {"code:27b": 19.8}
    gs.ps_models = ps_res
    async with gs.inst_locks["gpu-b"]:
        assert await gs.try_admit(None, "gpu-b", "code:27b", {})
    print("PASS already-resident passes")

    n8 = gs.required_gb(17.7, {"options": {"num_ctx": 8192}})
    n64 = gs.required_gb(19.9, {"options": {"num_ctx": 65536}})
    assert 19 < n8 < 22, n8
    assert n64 > n8 + 4, n64
    assert gs.fits(29.5, n8) and not gs.fits(24.0, n64)
    print("PASS KV estimate scales; oversized ctx cannot fit")

    # --- multi-GPU law (new) ---
    # T-M1: multi model routes ONLY to multi instances (never single-GPU pool)
    assert all(n not in ("gpu-a", "gpu-b") for n in gs.order_for("huge:70b"))
    assert gs.order_for("tinyA") == ["gpu-a", "gpu-b"]
    print("PASS routing separation: multi model never to single-GPU pool")

    # T-M2: sum fits but one card too empty -> reject (equal-split rule)
    set_gpu_free({"GPU-aaaa": 2.0, "GPU-bbbb": 55.0})
    async def ps_none2(sess, url): return {}
    gs.ps_models = ps_none2
    seed_size("huge:70b", 40.0)
    async with gs.inst_locks["gpu-a"], gs.inst_locks["gpu-b"]:
        ok = await gs.try_admit(None, "dual", "huge:70b", {"options": {"num_ctx": 8192}})
    assert not ok, "split 2/55 must not pass on sum"
    print("PASS uneven split rejected (sum 57 but min card 2GB)")

    # T-M3: both cards have room -> admit
    set_gpu_free({"GPU-aaaa": 28.0, "GPU-bbbb": 28.0})
    async with gs.inst_locks["gpu-a"], gs.inst_locks["gpu-b"]:
        ok = await gs.try_admit(None, "dual", "huge:70b", {"options": {"num_ctx": 8192}})
    assert ok
    print("PASS even-split admit when both cards free")

    # T-M4: eviction on peer instances sharing the cards opens room
    set_gpu_free({"GPU-aaaa": 8.0, "GPU-bbbb": 8.0})
    async def ps_squatters(sess, url): return {"big:31b": 20.2, "code:27b": 19.8}
    gs.ps_models = ps_squatters
    async def ev_clears(sess, inst, url, **kw):
        set_gpu_free({"GPU-aaaa": 28.0, "GPU-bbbb": 28.0})
    gs.evict_idle = ev_clears
    async with gs.inst_locks["gpu-a"], gs.inst_locks["gpu-b"]:
        ok = await gs.try_admit(None, "dual", "huge:70b", {"options": {"num_ctx": 8192}})
    assert ok
    print("PASS multi-GPU evict-then-fit across shared cards")

    # T-M5: slack matters — one card just below its share fails
    # need 40GB over 2 cards = 20 each; slack 0.9 -> min 18.0
    set_gpu_free({"GPU-aaaa": 17.5, "GPU-bbbb": 40.0})
    async def ps_empty3(sess, url): return {}
    gs.ps_models = ps_empty3
    gs.evict_idle = ev_noop   # restore no-op (T-M4's fake freed the cards)
    async with gs.inst_locks["gpu-a"], gs.inst_locks["gpu-b"]:
        ok = await gs.try_admit(None, "dual", "huge:70b", {"options": {"num_ctx": 8192}})
    assert not ok, "17.5 < 18.0 share floor"
    print("PASS split_slack enforced (17.5GB card < 18.0GB share)")

    # T-M6: small model never routed to multi instance
    gs.CFG["instances"]["dual"] = {"url": "http://127.0.0.1:11499", "gpus": ["GPU-aaaa", "GPU-bbbb"]}
    gs.inst_locks["dual"] = asyncio.Lock()
    gs.CFG["gpu_shared_by"]["GPU-aaaa"] = ["gpu-a", "dual"]
    gs.CFG["gpu_shared_by"]["GPU-bbbb"] = ["gpu-b", "dual"]
    assert gs.order_for("tinyA") == ["gpu-a", "gpu-b"]
    assert gs.order_for("huge:70b") == ["dual"]
    print("PASS routing separation small<->multi")

    # T-M7: multi-GPU queue timeout -> 503 (no capacity anywhere)
    set_gpu_free({"GPU-aaaa": 1.0, "GPU-bbbb": 1.0})
    async def ev_stay(sess, inst, url, **kw): pass
    gs.evict_idle = ev_stay
    gs.CFG["queue_wait_s"] = 1.0
    try:
        await gs.dispatch({"sess": None, "request": None}, "huge:70b",
                          {"options": {"num_ctx": 8192}})
        raise AssertionError("expected 503")
    except Exception as e:
        assert getattr(e, "status", None) == 503, e
    print("PASS multi-GPU queue timeout -> 503")

    # T-M8: single-GPU 503 path still works
    gs.CFG["queue_wait_s"] = 1.0
    try:
        await gs.dispatch({"sess": None, "request": None}, "code:27b",
                          {"options": {"num_ctx": 8192}})
        raise AssertionError("expected 503")
    except Exception as e:
        assert getattr(e, "status", None) == 503, e
    print("PASS single-GPU queue timeout -> 503")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        extra = (', "multi_gpu": {"models": ["huge:70b"], "split_slack": 0.9}')
        path = make_cfg(tmpdir, extra)
        gs.CFG = gs.load_config(path)
        gs.CFG["instances"]["dual"] = {"url": "http://127.0.0.1:11499",
                                       "gpus": ["GPU-aaaa", "GPU-bbbb"]}
        gs.inst_locks.update({n: asyncio.Lock() for n in gs.CFG["instances"]})
        asyncio.run(run_all())
        print("ALL TESTS PASS")


if __name__ == "__main__":
    main()