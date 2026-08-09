# Paged attention caches

Device-resident caching remains the default. Cache residency is configured
separately from `WeightResidency`; static weights, streamed weights,
activations, recurrent state, and KV or compressed-latent state have separate
budgets and lifetimes.

`CacheResidencyPolicy::Paged` divides attention state into immutable sealed
blocks and a mutable device tail. One `CacheResidencyManager` is shared by all
attention layers in a model cache, so finite device and host byte limits apply
globally instead of once per layer. Constructors reject zero device budgets,
zero recent-block protection, non-positive block sizes, and unbounded implicit
limits.

```rust
use safemlx_lm::{CacheResidencyPolicy, PagedCacheOptions};

let options = PagedCacheOptions::new(
    128,          // tokens per block
    512 << 20,    // finite logical device-cache bytes
    2 << 30,      // finite logical host-cache bytes
    1,            // recent device blocks protected per layer
)?
.with_full_attention(true);

let cache = model.new_cache_with_options(CacheResidencyPolicy::Paged(options))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`new_cache()` constructs device-resident state. Paged construction is exposed
through these current entry points:

- High-level `Model::new_cache_with_options`
  supports Llama/Mistral, DeepSeek, Kimi Linear, GPT-OSS, dense Qwen2/Qwen3,
  Inkling, and Nemotron-H.
- `PipelineModel::new_cache_with_options`
  supports every advertised pipeline stage adapter. It materializes the
  adapter's canonical per-layer cache descriptors, including heterogeneous
  fixed-state slots.
- `ExpertParallelModel::new_cache_with_options`
  supports DeepSeek, GPT-OSS, dense Qwen, Inkling, and Nemotron-H replicated
  attention state.

LFM2/LFM2-MoE and Qwen hybrid models can use paged attention through the
pipeline runtime. Their high-level and tensor-parallel model entry points reject
paged residency explicitly.

Recurrent linear-attention, Mamba, and convolution state are not block-paged;
they remain device resident while supported attention KV uses the paged block
manager. Inkling uses this split because its attention KV can be paged
independently from its four bounded convolution histories.
Kimi Linear likewise uses a heterogeneous cache: KDA layers retain three
bounded convolution histories plus an F32 recurrent state, while MLA layers
retain compressed no-RoPE latents. That cache supports resident and
weight-layerwise execution. Schema-v4 prompt snapshots persist both state kinds;
KDA itself is not represented as paged KV.
Nemotron-H uses the same split at a different layer granularity: Mamba layers
retain bounded convolution/SSM state, attention layers page ordinary KV with
their exact full or sliding policy, and MLP/MoE layers remain stateless.

## Sliding and full attention

Sliding-window paging is naturally bounded. Sealed blocks older than the
semantic attention window are discarded unless persistence retention was
explicitly enabled. Absolute token offsets continue to advance after discard.
A multi-token update keeps every new token plus the preceding visible window,
so each query can attend to the correct earlier tokens in the same update.

Full attention is explicit and experimental. It uses an FP32 online-softmax
reduction over ordered blocks: each block updates a running row maximum,
normalization sum, and weighted-value accumulator. Causal positions, rank-2
boolean or additive masks, grouped-query heads, multi-token queries, sliding
windows, pinned prefixes, and GPT-OSS learned sinks are preserved. Running
state is evaluated before the current block lease is released, and no
whole-history key/value array is assembled.

Every full-attention decode token still reads all historical blocks. Host
paging reduces logical device residency but can add large per-token transfers.
Live disk-backed full attention can approach rereading the complete context for
every token and is intended only for experimentation. DeepSeek compressed
latents reduce stored bytes, but attention still consumes every historical
latent block and reconstructs head-specific contributions one block at a time.
No throughput improvement is promised.

Apple unified memory means logical host/device placement does not create
additional physical capacity. Residency reports describe the runtime's logical
tiers and requested transfer bytes. They do not claim physical disk bytes:
mapped pages may be served by the operating-system cache. SafeMLX's immutable
weight residency manager, expert cache, and paged attention cache use MLX
completion events to retain transfer sources and order compatible execution
streams. Paged host-to-device promotion runs on a dedicated same-device
transfer stream. Blockwise key/value and compressed-MLA attention keep a fixed
two-block window (current plus next), submit each completed recurrence before
waiting on the next block, and thereby permit the next promotion to overlap
current-block attention. A one-block device budget falls back to demand
promotion rather than exceeding the configured limit. Device-to-host demotion
must establish independent CPU ownership, so it host-waits only for the exact
demotion completion before cloning storage; it no longer synchronizes the
whole transfer stream. This is cache-block pipelining, not general activation
double buffering. Disk reads and writes use a bounded manager-owned worker
queue. Duplicate block operations
join one shared completion, capacity waits occur without holding cache state,
and reset or truncation cancels queued work from the prior generation. The
worker is stopped and joined on drop. Host-to-disk demotion returns after queue
admission rather than waiting for the write: the block remains readable from
its host staging arrays until the worker publishes the disk backing.
`in_flight_write_blocks` and `in_flight_write_bytes` report this writeback state,
and those staging arrays remain included in `current_host_bytes`. A later
device-to-host demotion waits for an in-flight write when that write must finish
to keep the finite host-byte budget; queue-capacity backpressure is enforced
separately. Demand promotion does not release that host charge while the writer
still owns the staging allocation. Reset and truncation cancel obsolete work
but wait for each canceled task to release its arrays before returning, so a
new generation cannot reuse capacity that is still physically occupied.
Asynchronous write failures are returned by the next
residency-changing cache operation. Live shard names include a process-unique
namespace, write identity, representation, and pipeline/tensor/expert rank.
Publication uses an atomic no-replace hard link, so processes sharing a live
directory cannot overwrite one another. Temporary-file guards remove partial
shards on every ordinary error and worker panic.

Cache clearing is fallible when paging is active. Llama, GPT-OSS, Inkling,
pipeline, tensor-parallel, expert-parallel, and compressed-latent reset surfaces
propagate manager errors (including active leases) and leave the retained state
intact instead of reporting a successful reset.

Live disk backing is disabled by default and requires a dedicated directory,
finite logical disk budget, and finite queue capacity:

```rust
let options = PagedCacheOptions::new(128, 512 << 20, 2 << 30, 1)?
    .with_full_attention(true)
    .with_live_disk("/private/tmp/my-live-cache", 8 << 30, 2)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Mutable tails are never written. Sealed blocks are evaluated before a worker
receives them. Required blocks are protected by leases; queue or I/O failures
are returned to inference. Ephemeral live-cache files are removed when the last
manager is dropped. Persistent prompt-cache files are never treated as
ephemeral state.

## Reusable prompt caches

A prompt cache is a completed immutable prefix, not an arbitrary snapshot
of lazy in-flight arrays. `Model::save_prompt_cache` seals partial tails,
writes one bounded safetensors shard per cache block, and flushes data and the
manifest before publication. A new destination is published with one directory
rename. Replacement writes an immutable generation under the existing
destination and atomically switches its flushed `CURRENT` pointer, so the
canonical cache path always resolves to either the previous complete generation
or the new complete generation. Existing destinations are replaced only when
`PromptCacheOptions::replace_existing` is true. Unix publication synchronizes
the affected directories after renames; Windows publication uses
`MoveFileExW` with write-through semantics because ordinary file handles cannot
flush directory metadata there.

`inspect_prompt_cache` reads a bounded safetensors header and validates tensor
metadata and payload boundaries without reading payload bytes or creating MLX
arrays. `Model::load_prompt_cache` and `LoadedModel::load_prompt_cache` first
validate every owned layer's representation and exact model-derived cache
dimensions against the loaded model (including rank-local tensor-parallel KV
head counts), then memory-map the compatible immutable shards as read-only disk
blocks. Arrays are copied from those retained mappings only on demand, and
suffix tokens append new mutable and sealed blocks without modifying imported
files. Schema version 4 stores a SHA-256 digest of every exact safetensors
payload. The digest is checked once against the mapped bytes before the shard is
converted into MLX arrays, so opening remains mmap-lazy while same-length payload
corruption cannot be consumed.

Prompt-cache loading accepts schema version 4. Any other schema version is
rejected before cache tensors are opened.

The manifest records:

- schema version, model family, effective model type, caller-supplied checkpoint
  fingerprint, and a canonical architecture fingerprint derived from the loaded model;
- global layer range and one authoritative ordered `LayerCachePolicy` per owned
  decoder layer, including state kind, exact full/sliding policy, positive
  per-layer window, and tensor geometry;
- ordered fixed-state tensor contracts and shards for convolution, recurrent,
  multimodal prefix-embedding, and multimodal position-delta state;
- representation, block ranges, array names, shapes, dtypes, batch size, total
  prefix length, and per-block payload SHA-256;
- SHA-256 of the exact little-endian prefix token IDs plus a required
  caller-supplied processed-prefix fingerprint. Multimodal callers include the
  media bytes and processor configuration in that fingerprint;
- sink count and distributed topology;
- optional application namespace, which is never accepted as compatibility
  evidence.

Loading requires the exact prefix token IDs and rejects wrong fingerprints,
architecture settings, ordered layer policies, layer ranges, topology, block
size, or prefix identity. Inspection and import share validation for missing,
duplicate, reordered, or unexpected layer payloads; policy/state-kind
mismatches; tensor geometry; gaps; ranges; shapes and dtypes; extra arrays; and
paths or symlinks escaping the cache root. All checks run before arrays attach
to a live cache. A tensor-parallel cache is rank-local and cannot be reused
under a different topology. Pipeline stages persist only their global layer
range. Expert-parallel attention state is recorded as replicated, not
expert-sharded.

The architecture-neutral distributed scheduler owns one isolated program state
per active request. The decoder adapter stores one complete rank-local
`PipelineCache` in that state, so device caches add directly across requests. A
paged cache's limits apply to that request's `CacheResidencyManager`, not to the
scheduler as an aggregate pool; deployments bound aggregate exposure with
`max_active_requests` and choose per-request `PagedCacheOptions` accordingly.
No arrays are shared between request identities. EOS and cancellation drop the
state and cache, while `release_request_cache` hands an idle decoder cache back
to the caller for explicit schema-v4 persistence.

`AttentionPolicy::Sliding { window: N }` includes the current token in the `N`
positions. Ordinary live state therefore needs at most `N - 1` past positions
between calls. Paged backing is block-granular and may retain an older block
outside that logical range; absolute-position masks keep it invisible. With
`with_persistence_retention(true)`, discarded history is deliberately kept and
the persisted layer may contain the full prefix. Otherwise a sliding layer may
begin at a block boundary at or before the required `N - 1` history. Reload
never changes full attention to sliding attention or the reverse.

Llama/Mistral, dense Qwen2/Qwen2.5, all-full Qwen3, and GPT-OSS share this
layout. Arbitrary full/sliding ordering and distinct windows are preserved; no
architecture-specific manifest field or model-wide persistence window exists.
DeepSeek-V3/R1 uses the same layout with `CompressedLatentRotary` state. Kimi
Linear records KDA convolution/recurrent or MLA state per layer; Qwen3-Next and
Qwen3.5 record linear-attention convolution/recurrent state beside full-attention
KV; Gemma 4, Inkling, and Qwen3-VL persist the non-transient prefix state needed
for exact continuation. LFM2 records causal-convolution history beside its
full-attention KV, while Nemotron-H records Mamba convolution/recurrent state,
full-attention KV, and empty `StateSlots` entries for MLP/MoE-only layers.
Resident and bounded-weight execution use these same family helpers and layouts.
Pure LFM2/LFM2-MoE pipeline stages consume this layout directly: the paged
residency manager is stage-owned even when a stage has only fixed state, and
publication/reload validates every state slot's global owner, role, shape,
dtype, canonical order, and prefix offset.

`PipelineModel`, family-specific generalized tensor-parallel models, and
`ExpertParallelModel` expose matching `save_prompt_cache` and
`load_prompt_cache` workflows. Callers pass one shared root; each process
publishes `rank-NNNNN` beneath it. Pipeline manifests contain only the stage's
global layer interval, tensor-parallel manifests retain rank-local KV heads,
and expert-parallel manifests explicitly record replicated attention state.
LFM2 expert-parallel publication records replicated convolution state.
Nemotron-H expert-parallel publication atomically records resident Mamba state
and paged attention blocks under the same rank topology. Every load derives
family, effective type, exact ordered layer ownership and policies, and topology
from the loaded distributed model before opening its rank directory.

Checkpoint fingerprints are supplied by the application because hashing every
checkpoint byte can be expensive. They must be based on stable checkpoint
content or a trusted immutable model identifier, not solely on an absolute
path. Applications that require a stronger identity must compute and supply a
content hash. Architecture fingerprints are deterministic identities derived by
each supported loaded model from its normalized dimensions, ordered state and
attention layout, RoPE and scaling settings, quantization layout, and other
values that can change cached activations.
Model-aware distributed saves and every model-aware load require caller
descriptors to match this derived identity. Applications can obtain the exact
value from `prompt_cache_architecture_fingerprint` on loaded, pipeline,
tensor-parallel, and expert-parallel model surfaces.

Realtime Moshi/PersonaPlex temporal/depth state is isolated per request by the
canonical scheduler and can be released as one `RealtimeSession`. Persistent
publication remains deferred because a prompt-cache snapshot does not yet
encode continuous timing, delayed streams, sampler/PRNG state, and depth-decoder
state as one atomic identity. In-process handoff is nevertheless fail-closed:
the session records the selected checkpoint files' SHA-256 content identity and
the normalized execution identity, including effective quantization. Resident,
host-layerwise, and dense-streamed loads of the same artifact are compatible;
same-shaped models with different weight content are not.
A loaded prefix contains model state, not logits for an
empty suffix. Run at least one suffix token before sampling, or persist logits
separately in the application.

Run the deterministic text-prefix example with explicit compatibility
identities:

```sh
cargo run -p safemlx-lm --example paged_prompt_cache -- \
  /path/to/model /tmp/reusable-prefix \
  --prompt "Explain unified memory briefly." \
  --suffix-token 42 \
  --checkpoint-fingerprint sha256:CHECKPOINT_DIGEST
```

The example runs uninterrupted and restored suffix paths, prints logit parity,
and reports block, tier, transfer, attention-scan, mapping, queue, and
persistence counters. Use `--live-disk-dir` to demonstrate explicit live
backing. Compare that output with `--device-cache` for the ordinary cache and
with the model's configured ordered attention schedule for bounded sliding
residency. Do not flush privileged operating-system caches when measuring.

`CacheResidencyReport::per_layer` provides current token, representation,
tier, byte, protection, and in-flight-write observations plus cumulative
layer-attributable promotions, demotions, transfer bytes and waits, demand
hits and misses, failures, and prefill/decode attention scans. GPU promotion
completion is reported asynchronously through dependent execution;
`transfer_wait` measures host-visible disk and CPU ownership boundaries rather
than time behind a same-device event wait. The report contains at
most 128 identified entries and retains no per-block or per-call event
history. Once the identified-row limit is reached, activity for every later
layer is accumulated directly into `per_layer_overflow`; current observations
for additional active layers are folded there as well. The overflow row also
receives manager-level failures for which no single causal layer exists.
`per_layer_overflow_layers` counts those currently active omitted layers, and
the overflow row preserves their exact current aggregate even when it also
contains cumulative history. This keeps telemetry memory and report size
bounded while allowing consumers to reconcile layer-attributable cumulative
counters and every current block and byte with the global totals.

## Runtime validation

MLX execution tests remain opt-in so the default test suite can run on hosts
without a Metal device. On a Metal host, the ignored cache suite covers paged
append/truncate transactions, blockwise masks, tier budgets, asynchronous disk
writeback, event-ordered two-block promotion, and physical host/device storage
replacement. Inkling has an
uninterrupted-versus-paged parity test for both global and sliding attention.
The two-process Ring tests exercise DeepSeek compressed-latent prompt-cache
save, rank-local topology validation, reload, and restored decode for pipeline
and tensor parallelism:

```sh
cargo test -p safemlx-lm --lib 'cache::tests::' -- --ignored --test-threads=1
cargo test -p safemlx-lm --lib 'cache_residency::tests::host_' -- --ignored --test-threads=1
cargo test -p safemlx-lm --lib inkling_global_and_sliding_attention_paged_parity
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_deepseek_pipeline_persistence -- --ignored --exact --test-threads=1
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_pipeline -- --ignored --exact --test-threads=1
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_dense_stream_pipeline -- --ignored --exact --test-threads=1
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_moe_pipeline -- --ignored --exact --test-threads=1
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_deepseek_tensor_parallel_persistence -- --ignored --exact --test-threads=1
```
