# ollama-gpu-scheduler

A single admission controller that sits in front of **multiple Ollama instances** sharing one or more GPUs.

**The problem it solves:** on a box with several Ollama daemons (multi-user server, several agents/pipelines on one machine), nothing coordinates model loads across instances. A load that starts on a GPU without enough free VRAM silently CPU-offloads part of the weights — RAM and swap fill up, and the request dies minutes later with a 500. This has to be experienced to be believed, and then it keeps happening.

**The fix:** point every caller at one proxy instead of the raw instances. Same Ollama API — no call-site code changes. The proxy admits a model load only when the target GPU provably has room.

## The placement law

1. Requests are admitted in **FIFO order**; a model loads only when a GPU fits.
2. **One large model per GPU**, or several small ones within the per-GPU budget.
3. Designated **resident** models (e.g. an always-on intuition/creative model) live on their own instance, bypass placement entirely, and are never queued or evicted.
4. Admission check uses **nvidia-smi free VRAM** (authoritative — it accounts *every* process on the card, not just Ollama's) versus `weights × safety + KV estimate` derived from the request's own `num_ctx`.
5. If nothing fits: **evict idle** residents on the target GPU → try the sibling instance → **FIFO queue** → honest `503` after the queue deadline. A doomed load is never started, so nothing ever spills to RAM.

Two background guardians enforce the law against reality:

- **spill guard** — any model on a managed instance showing the CPU-offload signature (`size_vram` far below `size`) is unloaded immediately. This catches spills from processes that bypassed the proxy.
- **rogue watchdog** — flags `ollama serve` daemons running outside the sanctioned set (per-user allowlist). Evidence that someone is bypassing the scheduler, surfaced in `/api/sched/status` instead of discovered during a RAM-fire postmortem.

## Quick start

```bash
pip install aiohttp
cp config.example.json config.json   # edit: your instances, GPUs, resident model
python3 ollama_gpu_scheduler.py --config config.json
```

Then point **all** callers at the proxy:

```bash
export OLLAMA_HOST=http://127.0.0.1:11500        # ollama CLI
# or base_url="http://127.0.0.1:11500" in Python / OpenAI-style clients
```

Run it as a systemd user service (template in `systemd/`).

## Configuration

Everything machine-specific lives in `config.json` (see `config.example.json`):

| Key | Meaning |
|---|---|
| `instances` | Backing Ollama daemons: `{name, url, gpu_uuid}`. The GPU UUID (from `nvidia-smi --query-gpu=uuid`) makes placement stable across reboots — identical cards can reorder as `cuda:N`. |
| `resident` | A model + instance that is always warm (never scheduled, never evicted). Its requests are forwarded untouched. |
| `affinity` | Optional model → preferred instance map. Unknown models first-fit across the pool. |
| `gpu_budget_gb` | Usable VRAM per card (a little under 32 GB to leave headroom). |
| `weight_safety` | Margin on weights to cover CUDA graph/workspace variance (1.12 ≈ +12%). |
| `kv_gb_per_8k_ctx`, `kv_cap_gb` | Rough KV-cache allowance per 8k of context, capped. |
| `queue` | FIFO depth and max wait before an honest `503`. Size this *below* your callers' HTTP timeouts — see below. |
| `rogue_watch.sanctioned_users` | Per-user counts of allowed `ollama serve` daemons. Unknown users = always rogue. |
| `spill_guard` | Interval and threshold (GB) for detecting CPU-offloaded models. |

### Sizing the queue against caller timeouts

The queue adds up to `queue.wait_s` of *waiting* before generation starts. The invariant to maintain:

```
caller inner HTTP timeout  >  queue.wait_s  +  expected generation time
```

For example, with 300–600 s caller timeouts, `wait_s: 240` leaves comfortable margin. Callers that can't tolerate waiting will fail their tick and retry — which is exactly the behaviour you want under contention (visible in stats, harmless to state), rather than a slow RAM leak.

## API

Ollama-compatible (so existing clients just work):

- `POST /api/chat`, `/api/generate` — admitted through the placement law (streaming passes through)
- `POST /api/embeddings`, `/api/show` — forwarded to the affinity instance
- `GET /api/tags`, `/api/version`, `/api/ps` — passthrough

OpenAI-compatible (for `OLLAMA_BASE_URL/v1` style clients):

- `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- `GET /v1/models`

Operations:

- `GET /health` — liveness
- `GET /api/sched/status` — queue depth, in-flight map, per-instance residency, decision log, rogue list

## Tests

```bash
python3 test_gpu_scheduler.py   # offline unit tests (monkeypatched nvidia-smi)
```

Covers: small models co-residing, large blocked by resident large, eviction-then-fit, resident model never scheduled, cloud models bypassing placement, affinity ordering, KV estimate scaling.

## Design notes

- **Why nvidia-smi and not Ollama's own numbers?** Ollama only knows about its own models. The card is shared — other users, other daemons, FLUX/comfy processes. Admission must see the whole truth, which is also why a model that fits *by Ollama's accounting* can still spill.
- **Why a proxy and not one big Ollama instance?** Different callers need different keep_alive, context defaults, and GPU pinning; and a single instance serializes all callers behind one slot queue with no placement intelligence. The scheduler keeps instances separate (each pinned to a GPU by UUID) and adds the missing cross-instance brain.
- **`num_ctx` honesty matters.** The KV estimate comes from the request's own `options.num_ctx`. If a client asks for a context that cannot fit the card (e.g. 64k on a 32 GB card), the request will queue until timeout rather than spill — fix the client's `num_ctx`. Model names that *mention* a context size ("...-65k") are just names; check what the request actually asks for.

## Limitations

- NVIDIA-only (nvidia-smi parsing). Patches welcome for ROCm/Intel.
- One node. Multi-node placement is a different (much harder) problem.
- `/v1` streaming returns Ollama ndjson, not OpenAI SSE — fine for Ollama-origin clients; SSE-shim wanted.
- Cloud/remote models (zero weights) bypass placement by design — nothing to place.
- The rogue watchdog detects bypass attempts; it does not (yet) prevent them. Prevention is either OS-level (per-user firewalls on Ollama ports) or social (your co-tenant adding the proxy URL).

## License

MIT