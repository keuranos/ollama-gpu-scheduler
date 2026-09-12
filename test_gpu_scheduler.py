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


def make_cfg(tmpdir):
    conf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=tmpdir)
    conf.write("""
{
  "instances": [
    {"name": "gpu-a", "url": "http://127.0.0.1:11436", "gpu_uuid": "GPU-aaaa"},
    {"name": "gpu-b", "url": "http://127.0.0.1:11439", "gpu_uuid": "GPU-bbbb"}
  ],
  "resident": {"model": "muse:latest", "url": "http://127.0.0.1:11438"},
  "affinity": {"big:31b": "gpu-a", "code:27b": "gpu-b"},
  "rogue_watch": {"enabled": false},
  "spill_guard": {"enabled": false},
  "gpu_budget_gb": 30.0
}
""")
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
    # T1: small model fits on a mostly-free card
    set_gpu_free({"GPU-aaaa": 29.0})
    async def ps_empty(sess, url): return {}
    gs.ps_models = ps_empty
    seed_size("tinyA", 1.0)
    async with gs.inst_locks["gpu-a"]:
        assert await gs.try_admit(None, "gpu-a", "tinyA", {"options": {"num_ctx": 8192}})
    print("PASS small model fits")

    # T2: large blocked by resident large
    set_gpu_free({"GPU-aaaa": 8.5})
    async def ps_big(sess, url): return {"big:31b": 20.2}
    gs.ps_models = ps_big
    async def ev_noop(sess, inst, url, **kw): pass
    gs.evict_idle = ev_noop
    seed_size("code:27b", 17.7)
    async with gs.inst_locks["gpu-a"]:
        assert not await gs.try_admit(None, "gpu-a", "code:27b", {"options": {"num_ctx": 8192}})
    print("PASS large blocked even after eviction attempt")

    # T3: eviction frees room -> fits
    set_gpu_free({"GPU-bbbb": 6.0})
    async def ps_small(sess, url): return {"embed:0.6b": 0.3}
    gs.ps_models = ps_small
    async def ev_frees(sess, inst, url, **kw):
        set_gpu_free({"GPU-bbbb": 26.0})
    gs.evict_idle = ev_frees
    async with gs.inst_locks["gpu-b"]:
        assert await gs.try_admit(None, "gpu-b", "code:27b", {"options": {"num_ctx": 8192}})
    print("PASS evict-then-fit")

    # T4: resident model excluded from the schedulable pool
    assert all("muse:latest" != n for n in gs.CFG["instances"])
    print("PASS resident instance not in pool")

    # T5: cloud bypass
    seed_size("cloudy:cloud", 0.0)
    async with gs.inst_locks["gpu-a"]:
        assert await gs.try_admit(None, "gpu-a", "cloudy:cloud", {})
    print("PASS cloud bypass")

    # T6: affinity ordering
    assert gs.order_for("code:27b")[0] == "gpu-b"
    assert gs.order_for("big:31b")[0] == "gpu-a"
    assert set(gs.order_for("unknown")) == {"gpu-a", "gpu-b"}
    print("PASS affinity ordering")

    # T7: already-resident passes without check
    async def ps_res(sess, url): return {"code:27b": 19.8}
    gs.ps_models = ps_res
    async with gs.inst_locks["gpu-b"]:
        assert await gs.try_admit(None, "gpu-b", "code:27b", {})
    print("PASS already-resident passes")

    # T8: KV estimate scales with ctx and gates placement
    n8 = gs.required_gb(17.7, {"options": {"num_ctx": 8192}})
    n64 = gs.required_gb(19.9, {"options": {"num_ctx": 65536}})
    assert 19 < n8 < 22, n8
    assert n64 > n8 + 4, n64   # big ctx must cost real VRAM
    assert gs.fits(29.5, n8) and not gs.fits(24.0, n64)
    print("PASS KV estimate scales; oversized ctx cannot fit")

    # T9: queue timeout path raises 503 (restore eviction to no-op first)
    set_gpu_free({"GPU-aaaa": 1.0, "GPU-bbbb": 1.0})
    async def ps_none(sess, url): return {}
    gs.ps_models = ps_none
    async def ev_stay(sess, inst, url, **kw): pass
    gs.evict_idle = ev_stay
    seed_size("code:27b", 17.7)
    gs.CFG["queue_wait_s"] = 1.0
    try:
        await gs.dispatch({"sess": None, "request": None}, "code:27b", {"options": {"num_ctx": 8192}})
        raise AssertionError("expected 503")
    except Exception as e:
        assert getattr(e, "status", None) == 503, e
    print("PASS queue timeout -> 503")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_cfg(tmpdir)
        gs.CFG = gs.load_config(path)
        gs.inst_locks.update({n: asyncio.Lock() for n in gs.CFG["instances"]})
        asyncio.run(run_all())
        print("ALL TESTS PASS")


if __name__ == "__main__":
    main()