# KV Cache Transceiver V2 (Python Runtime)

In [disaggregated serving](../features/disagg-serving.md), the KV cache transceiver moves a
request's KV cache from a context (prefill) instance to a generation (decode) instance.

This guide covers the **V2 transceiver**: a Python control plane over a NIXL data plane,
implemented in `tensorrt_llm/_torch/disaggregation/` and selected with
`cache_transceiver_config.transceiver_runtime='PYTHON'`. It implements the runtime's
`KvCacheTransceiver` interface (`tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`), so the
executor drives it through the same handful of calls it uses for any transceiver.

```{note}
This document covers the design ideas, the configuration surface and the extension points. It
deliberately avoids the internal wire protocol and other implementation details, which change as
the code evolves; for those, read the code and its docstrings.
```

## What it offers

| Capability | How it is achieved |
|------------|--------------------|
| One transfer path for heterogeneous caches | Peers exchange a serialized description of their cache layout. A new cache manager, pool role or window class is *described*, not coded into the transfer path. |
| Non-attention state | Recurrent (Mamba / KDA) state travels in the same session, and the same write, as paged KV. |
| KV cache manager V2 | Its Python-owned pools are driven directly, so disaggregated serving works with that manager. |
| Generation-first disaggregation | The transceiver publishes its own connection metadata and holds context requests until the generation peer has registered. |
| Optional write coalescing | A request's scattered blocks can be gathered into one contiguous region and sent as a single large write. |
| Cheap extension | New transfer backends, cache layouts and state kinds plug into small Python interfaces. |

Requirements and limits: it needs the `NIXL` backend, context parallelism 1 and a finite transfer
timeout, and its control plane runs on Python threads.

## Usage

Selection is per server instance. Set `cache_transceiver_config.transceiver_runtime` to `PYTHON`
to use V2; it fails fast if the effective backend is not `NIXL`. The default, `auto`, uses V2 when
the model asks for it and the backend is `NIXL`.

```yaml
# context_config.yml / gen_config.yml
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON   # or omit it and let the model decide via 'auto'
```

Under `auto`, the model class decides through `get_preferred_transceiver_runtime()` on
`PretrainedModel` (`tensorrt_llm/_torch/models/modeling_utils.py`). Model authors override that
hook to opt an architecture in, so the list of models using V2 lives in the code, not here. The
hook is consulted only on the PyTorch backend's standard model-loading path, and only when
disaggregated serving is already enabled — it never turns on a transceiver the user did not
configure.

Some routes are available only here:

- A model that prefers KV cache manager V2 keeps that manager in disaggregated serving only on the
  `PYTHON` route with `NIXL`; on any other route its preference is not honored.
- Hybrid Mamba caches on manager V2, and models whose recurrent state only this runtime can move
  (such as KDA linear attention), raise an explicit error elsewhere rather than degrading quietly.
- Generation-first scheduling is implemented here only.

Transfer deadlines, polling intervals and write coalescing are tuned through the remaining fields
of `CacheTransceiverConfig` (`tensorrt_llm/llmapi/llm_args.py`), which carry their own descriptions
and defaults.

## Design

Moving a request's KV between two instances sounds like a copy. What makes it hard is that the two
instances rarely agree on what the cache looks like, some of the bytes exist on more than one rank,
a copy can fail in the middle, the cost is dominated by fragment count rather than payload size,
and all of it runs on the executor's critical loop. Each section below takes one of those problems
and the decision that answers it.

### Cache layout is exchanged as data

A KV cache stopped being one key pool and one value pool per layer. Sliding windows split layers
into life cycles, MLA collapses K and V into one, sparse attention adds index pools, some pools
hold bytes that are identical on every rank, and hybrid models carry recurrent state that is not
paged at all. If the transfer path has to recognize these shapes, every new attention variant
arrives as new transfer code, and every pairing of variants is a new case to test.

So the transfer path is not allowed to recognize anything. Each rank describes its own cache in a
page table — layer groups, the pools behind them, per-layer byte ranges — and peers exchange that
description once. Two rules keep the description from turning into a second model config: role
labels are opaque, compared for equality and never interpreted, so a cache manager can invent a
role without a change here; and remapping semantics are a closed set, a small `MapperKind`
(head-major, token-major, replicated) that says how bytes are re-cut when the two sides shard
differently.

The result is that a new pool role costs nothing here, and a genuinely new byte layout costs one
mapper kind. Heterogeneous TP/PP/DP, sliding windows, MLA, index pools and recurrent state stop
being separate features and become one operation: match descriptions, then move bytes.

### Ownership is derived from topology

When the two sides shard differently, one destination rank overlaps several source ranks, and some
of what they hold is the same data — pools replicated across ranks, or KV heads duplicated because
the receiver has fewer shards. If every overlapping rank sends its copy, the wire carries the same
bytes several times, and once writes are coalesced into one region those senders overlap each other
outright. If the rule is drawn too tightly instead, nobody sends and the request is silently short.

Ownership is therefore derived, not negotiated. From the two page tables and the two topologies,
V2 computes once per peer which source ranks feed this destination, which local pool corresponds to
which remote pool, and, for anything replicated, which single rank owns each copy. The result is
cached, so a request costs pointer arithmetic rather than a fresh negotiation.

### One direction: the context side writes

Any distributed copy can stop halfway — a peer dies, a request is cancelled, a deadline passes. If
both sides may initiate, then registration, completion and failure each have two code paths, and
after a mid-flight failure it is genuinely ambiguous who is responsible for the cleanup.

V2 keeps one direction: the generation instance states what it needs, the context instance pushes
the bytes with one-sided writes and reports the outcome. The control plane between instances
carries small metadata only, never KV bytes. That leaves one registration path, one completion
path, and no remote read that can fail halfway.

It also fixes the shape of cancellation, honestly rather than conveniently. A write already in
flight cannot be recalled, so cancelling a request completes only once its writes have drained, and
memory that an abandoned write may still land in is held aside instead of handed to the next
request.

### Coalescing scattered blocks into one write

A request's KV is spread over blocks, and remapping cuts each block into per-layer fragments. Sent
as they are, one transfer becomes a long list of small descriptors, and its cost tracks the length
of that list more than the size of the payload.

The answer is to trade two device-local copies for a much shorter list: gather the request into one
contiguous region, send a single large write, and scatter it back on arrival. This is opt-in per
instance through `kv_cache_bounce_size_mb`, and it stays an optimization rather than a mode.
Transfers below its size gates, and topologies where splitting one region between several senders
would be unsafe, fall back to the per-block path on their own, so correctness never depends on the
setting.

### A layered core behind a thin facade

The decisions above only hold if the code implementing them stays separable: a layout description
that quietly depends on NIXL, or a transport that peeks at a cache manager, would put every future
model change back on the transfer path. The packages are therefore ordered, and each one may only
depend on the ones above it.

| Package | Responsibility |
|---------|----------------|
| `disaggregation/base/` | Contracts: transfer agent, memory descriptors, KV slice, session types. |
| `disaggregation/nixl/` | NIXL agents implementing the transfer-agent contract (a compiled binding when available, a pure-Python fallback otherwise). |
| `disaggregation/resource/` | Adapters over the KV cache managers: page-table construction, region extraction, prefix reuse across the supported managers. |
| `disaggregation/native/` | The implementation: control-plane messaging, peer matching, sender and receiver sessions, mixer policies, coalescing, perf logging. |
| `disaggregation/transceiver.py` | `KvCacheTransceiverV2`: the executor-facing facade — bookkeeping, cross-rank agreement, request-state transitions. |

No code in the transfer path branches on a model architecture. Model specifics enter as page-table
data, a mapper kind, or a mixer policy — which is what keeps the extension points below cheap.

### Per-step consensus across ranks

Transfers finish at different moments on different ranks. If one rank calls a request done while
another does not, their request states diverge: the first releases the cache and moves on, the
second keeps waiting for a transfer that nobody will finish, and the two executors hang on each
other. Cancellations and failures create the same split from the other end.

Every rank therefore votes, once per step, in a single batched collective: a cancellation or a
failure anywhere is global, while completion requires all of them. Because that collective sits on
the executor loop, it is skipped entirely in the cases where ranks cannot disagree — a single rank,
or a topology where the decision is already local.

### How a transfer flows

1. The first time a generation instance talks to a context instance, the two exchange rank
   descriptions and validate that their layouts are compatible. An incompatible peer is remembered,
   so later requests to it fail fast — and only they fail.
2. When prefill finishes, the context instance opens a send session and returns its endpoint to the
   orchestrator with the context response.
3. The generation instance opens a receive session and asks the overlapping context ranks for the
   blocks it needs.
4. Each context rank matches the two descriptions, computes source and destination pointers, and
   submits the writes from a worker thread.
5. The transfer completes once every expected sender has reported — and, with coalescing enabled,
   once the scatter into the local pools has landed. Both executors then reconcile the outcome
   across their ranks and release the sessions.

Generation-first scheduling inverts the beginning: the generation instance registers first, the
context instance holds the request until every rank knows about it, and the context instance sends
the first generated token and any draft tokens alongside the KV.

## Extending

| Goal | Seam |
|------|------|
| Another transfer library | Implement `BaseTransferAgent` (`base/agent.py`). |
| Another control-plane transport | Implement `MessengerInterface` (`native/messenger.py`). |
| A new cache manager or pool layout | Describe it in the page table (`resource/kv_extractor.py`); for prefix reuse, add a `CacheReuseAdapter` (`resource/cache_reuse.py`). |
| A new byte layout or head remapping | Add a `MapperKind` and its mapper family (`resource/page.py`, `native/mixers/attention/`). |
| A new non-attention state | Add a mixer policy under `native/mixers/`. |
| A different staging strategy | Implement `BounceTransport` (`native/bounce/core.py`). |

## Testing

Unit and single-node integration tests live in `tests/unittest/disaggregated/`: peer and pool
matching, page-table extraction, rank info, messaging, coalescing, recurrent-state transfer,
bounded polling, and a two-instance harness driven over MPI.

`examples/disaggregated/slurm/cache_transceiver_test/` is a standalone two-node harness. It fills a
KV cache, transfers it, verifies the bytes and reports the achieved bandwidth, without running a
model.
