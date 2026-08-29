# Model loading, quantization, and memory

`eredu` separates static weight placement from per-request runtime state.
Choose and budget them independently: moving model layers does not move the KV
cache, and paging a cache does not change where model weights live.

## Preflight

Applications run `eredu::api::inspect_local_model` with the same
`LocalInspectionOptions` they intend to load.
Inspection validates the artifact, architecture, requested quantization,
residency policy, topology, processor, and media companion files without
materializing weights. It is the cheapest place to reject an incompatible
backend request. Apply `api::inspect_text_model` to that report when admission
also requires tokenizer, chat-template, EOS, semantic-streaming, or native-tool
readiness.

The selected local backend's `eredu::api::LocalLoadOptions` contain portable
load policy:

- `quantization`: an optional transformation for eligible dense weights; and
- `weight_residency`: the static parameter placement policy; and
- `required_session_capabilities`: fail-closed requirements for the realized
  session.

The same requirement applies to realtime loading. MLX Moshi and PersonaPlex
routes report persistent request cache and portable output observation, but not
named activation inspection; an unsupported requirement fails before checkpoint
payload materialization. Loaded realtime models expose the admitted report with
`LocalRealtimeModel::session_capabilities`.

These facade-owned options are configured with builders and queried with
accessors. They do not contain an MLX device or parallel context. Applications
derive them from a portable `ExecutionPlan` with
`LocalInspectionOptions::for_execution_plan` when inspection must exactly
match a planned load. Backend-author tooling that binds native distributed
groups uses `eredu-backend-mlx` directly.

Non-replicated distributed topologies use the same architecture-erased loader
with an MLX parallel context.

## Weight residency

### Fully resident

`eredu_runtime::WeightResidency::fully_resident()` materializes the complete
selected model or rank-local shard before execution. This is the simplest and
usually fastest policy when it fits. Infrastructure clients configure this
policy through the owning `eredu-runtime` crate.

### Host-layerwise

Host-layerwise execution keeps repeated execution groups in host-transfer
storage and promotes a bounded window to the execution device. Static modules
such as embeddings, final normalization, and the output head remain pinned
when the architecture requires them.

The device-layer window and host/device byte limits are hard admission inputs.
A dedicated same-device transfer stream and completion leases allow the next
group to be prepared while the current group executes. A narrow window may
still stall when a needed transfer does not complete in time.

On Apple silicon, “host” and “device” are logical tiers in the same unified
physical memory. Host-layerwise placement can control executable working sets
and lifetimes, but it does not add RAM.

### Dense disk streaming

Disk streaming keeps ordinary execution groups checkpoint-backed and uses
bounded disk, host, and device caches. Host lookahead and the background queue
control how much future materialization may be in flight. This policy reduces
resident parameter bytes at the cost of checkpoint reads and conversion during
execution.

SafeTensors mappings and GGUF readers are bounded separately. Operating-system
page caching can make repeated reads faster, so logical read counters should
not be interpreted as physical storage traffic.

### Independent expert cache

Registered MoE families can combine any supported non-expert policy with a
separate routed-expert cache. The cache has its own device, host, scratch, and
prefill-bank budgets and deterministic eviction policy.

Routing has one unavoidable host boundary: Eredu reduces the route tensor to
a compact demand histogram, reads that metadata, and then selects the exact
checkpoint experts. Route rows and weights remain on-device. Missing experts
are acquired as one residency batch; the execution stream waits on an
aggregate completion event before constructing the compact bank.

The scratch limit is a hard bound for one compact bank. Multi-token prefill may
be split to stay within the configured prefill-bank target and scratch bound.
Decode remains one routed bank. Exact routing never substitutes a different
expert when the requested one is unavailable.

## Load-time quantization

Eredu can convert eligible dense F32/F16/BF16 weights to an affine or MXFP4
layout while loading. Conversion is driven by the architecture's parameter
plan, not by blindly quantizing every matrix. Norms, convolutions, routers,
position tables, and specialized modality parameters remain dense unless that
architecture explicitly registers a safe transformation.

Bounded policies transform admitted tensor or row selections rather than
materializing the entire dense checkpoint. Independent expert caches use the
same approach for selected expert banks.

Checkpoint-native packed weights remain native. GGUF K-quant, IQ, affine, and
MXFP4 tensors, or native block-FP8 SafeTensors, are never silently transcoded
into a requested format. An incompatible transformation fails during preflight.

## Runtime-state residency

`CacheResidencyPolicy::Device` keeps request state on the execution device and
is the default. `CacheResidencyPolicy::Paged` divides sealable attention state
into immutable blocks plus a mutable device tail.

A paged cache has finite per-request limits. Multiple caches can share a
`CacheResidencyPool`, which adds process-wide device, host, transfer-in-flight,
and live-disk limits. Per-cache limits remain active so one request cannot
consume the entire shared pool.

Runtime state has explicit behavioral classes:

| Class | Typical state | Behavior |
| --- | --- | --- |
| Sealable paged | KV or compressed MLA blocks | immutable blocks can move between device, host, and optional disk tiers |
| Always-device mutable | rolling convolution histories and small prefix metadata | stays resident because it changes in place |
| Layer-scoped offloadable | large recurrent matrices used by one layer at a time | can be moved at layer boundaries |
| Stateless | MLP- or MoE-only layers | no request-state payload |

Only sealable state enters the paged block store. A cache representation cannot
misclassify mutable recurrent state as immutable KV.

Sliding-window attention discards blocks older than the semantic window unless
persistence retention is requested. Full-attention paging retains history and
uses blockwise online softmax; every decode token still visits all retained
blocks, so host or disk paging can be very expensive.

Optional live-disk backing requires a dedicated directory, finite byte budget,
and finite worker queue. Mutable tails are never written. Temporary live-cache
files are removed when their manager is dropped and are separate from reusable
prompt caches.

## Reusable prompt caches

A prompt cache is a completed immutable prefix. Saving seals partial tails and
publishes a manifest plus bounded SafeTensors shards atomically. Loading first
validates the entire manifest and owned layer layout, then attaches compatible
shards as read-only disk blocks and promotes them only on demand.

Schema version 8 records:

- model family, effective type, checkpoint identity, and an architecture
  fingerprint;
- the exact ordered per-layer state kind, attention policy, window, tensor
  geometry, and distributed ownership;
- convolution, recurrent, multimodal prefix, and position-delta state where
  applicable;
- block ranges, shapes, dtypes, and payload digests; and
- exact prefix token identity plus the caller's processed-prefix fingerprint.

Load accepts schema version 8 only. Model family, architecture fingerprint,
prefix identity, layer ownership, topology, state representation, tensor
geometry, and payload digests must match before arrays become live. Multimodal
applications should include media bytes and processor settings in the
processed-prefix fingerprint.

Distributed prompt caches are rank-local beneath one shared root. A cache
cannot be reopened under a different tensor, pipeline, or expert topology.
Realtime Moshi and PersonaPlex sessions remain outside this decoder prompt-cache
format.

## Accounting and observability

Residency reports separate logical device parameter bytes, charged host-buffer
capacity, disk payload bytes, transfers, cache hits and misses, evictions, and
in-flight ownership. Expert, dense-stream, and request-cache reports are
separate because they have different budgets and lifetimes.

Static parameter estimates do not include activations, KV or recurrent state,
kernel memory, allocator caches, checkpoint mappings, all temporary workspaces,
or opaque backend-driver allocations. Use MLX allocator statistics and process
memory observation alongside Eredu reports when sizing an application.
