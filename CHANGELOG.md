# Changelog

This file records consumer-facing compatibility changes and Git lineage notices.
Workspace crates have independent release versions; see the
[release guide](doc/releasing.md) for package release records.

## 2026-09-23 — Recalibrated input-score memory forecasts

Forecasts now account for shared K/V layouts, bounded live tile graphs and retained
outputs using backend-declared `FullKeyAttentionTiles` facts. MLX's score allowance
is calibrated at 32 working bytes per element. Longer-key and legacy records keep
conservative per-tile allowances; missing mechanism facts remain unbounded.
Existing JSON without `full_key_tiles` remains readable. Rust consumers constructing
`InputScoreAttentionMechanism` literals must supply the new optional field (`None`
preserves legacy sizing). Numerical and peak-memory validation is recorded in
[generation memory](doc/generation-memory.md).

## 2026-09-23 — Bounded input-score tile graphs

MLX explicit input-score attention evaluates groups of at most 32 query tiles for
large full-key calls, releasing temporary graphs while retaining completed outputs
and shared K/V. Smaller calls remain lazy. Native numerical and allocation tests
cover the change; memory and timing measurements are recorded in
[generation memory](doc/generation-memory.md). Forecast allowances remain
conservative pending recalibration.

## 2026-09-23 — Shared input-score attention layouts

MLX explicit input-score attention now reuses expanded K/V and contiguous BF16
projection layouts across query tiles for complete key rows up to 8,192 positions.
The optimization applies to the shared mechanism, preserves arithmetic and lazy
execution, and leaves longer-row blockwise execution unchanged. Forecast copy
allowances remain conservative pending recalibration.

## 2026-09-23 — Reduced-precision load-time quantization

Exact selected-task quantization now accepts F16/BF16 checkpoint recipes with
unloaded Float32 source slots. Recipe precision controls the transform and affine
companions; source shapes, floating slot categories and admitted output metadata
remain validated. This fixes affine 4-bit loading of the official BF16 LFM2.5
checkpoint. Native short/long generation and continuation validation now passes.

## 2026-09-23 — Dense LFM2/LFM2.5 memory forecasts

Cold, loaded and continuation forecasts now bound dense LFM2 hybrid workspace,
including gated convolution and MLX's repeated input-score attention buffers.
Geometry comes from the normalized schedule; native tile and temporary-buffer
facts are retained during selection. Full-pass prefill remains explicit. Missing
backend facts, routed variants and previously excluded paths remain unbounded.
`WorkspaceGeometry` adds optional `gated_convolution`, `input_score_attention`
and `mixed_precision_parameter_bytes` fields: older JSON remains readable; Rust
literals need the new fields.
See [calibration and limits](doc/generation-memory.md#dense-lfm2lfm25-hybrid-workspace).

## 2026-09-23 — Installed continuation state lower bounds

Continuation plans and phase reports now include the known required logical
payload of installed state instead of a zero lower end. Native capacity upper
bounds remain unchanged. The calculation honors sliding windows, layer offsets,
fixed tensor dtypes and conditional state, excluding optional tensors and capacity
rounding. Zero-token forecasts and restored continuations use the same accounting;
no allocator observation, reservation or resident-memory credit is introduced.

## 2026-09-23 — Capture forecasts from selection geometry

Loaded ordinary MLX logits captures now forecast admitted transform sizes and
scheduled occurrences, capped by capture limits. Large top-k capture quotas no
longer inflate a small selection to the whole quota. Output-horizon alternatives
recompute retained records; continuations include actual charged history without
refunding it after restore. Unknown sources, partitioned captures, and interventions
retain a labeled limit fallback. Independent trace and allocator allowances remain.
Forecasting performs no native submissions or budget reservations.

## 2026-09-23 — Tighter continuation host allowances

Continuation forecasts now bound future semantic history by unspent trace bytes,
replacing the previous event-header-per-byte multiplier. Existing history is
counted directly; a retained encoded trace remains a separate allowance. Branch
reservations use the same bound with the fresh child's trace budget. Restore
preserves consumed trace accounting. A 64 MiB trace budget now contributes at most
64 MiB of future semantic history instead of several GiB.

## 2026-09-23 — Mid-session memory forecasts

Ordinary token iterators and controlled sessions now expose
`forecast_remaining_generation`. Settled MLX continuations project the actual
cache frontier, native growth/capacity, decode workspace, sampling and retained
capture/control state without repeating completed prefill or loading. Iterator
`synchronize` establishes a read-only observation boundary. Forecasting does not
advance or extend generation, copy state, or consume observation budgets.
Unsupported or unknown storage stays explicit; speculative continuations remain
outside this ordinary API. See [continuation accounting](doc/generation-memory.md#mid-session-continuation-forecasts).

## 2026-09-23 — Device-aware cold memory forecasts

`GenerationMemoryOptions::for_local_device(input, plan.device())` now derives
placement, host/device availability and allocator overhead from the selected
local device. Application budgets and reserves remain explicit. The portable
`for_hardware_device` constructor performs the capacity mapping from supplied
observations and a backend host-execution fact. Invalid device selections are
rejected; missing capacity stays unknown. The CLI uses the new constructor;
`for_local_backend` remains available for explicit placement.

## 2026-09-23 — Speculative generation memory forecasts

Loaded independent autoregressive drafting now has a bounded planning envelope
when both models have ordinary workspace coverage. Prepared-request and
raw-token facade calls account for both models, rollback/replay, verification,
sampling and configured lookahead. Forecasts retain a serializable speculative
plan; output-limit alternatives recompute both models. The CLI uses this path.
Embedded heads, feature-conditioned assistants and cold reports without a
realized drafter retain explicit unknowns. See
[speculative forecast coverage](doc/generation-memory.md#speculative-requests).

## 2026-09-23 — Managed MLX allocator-cache default

Native model realization now caps an untouched MLX allocator-cache default at
256 MiB, preserving smaller defaults. This process-global policy reduces the
default forecast allowance and can trade some throughput for lower retention.
Explicit settings, including direct native setter calls, remain authoritative.
Applications that need the native policy should configure
`LocalAllocatorCachePolicy::PreserveNative` before the first model is realized.
`with_allocator_cache_limit(bytes)` continues to select an explicit limit.

`local_allocator_cache_policy()` reports the native limit and its provenance.
Forecasts and getters remain observational: a pure cold forecast before runtime
initialization still reports the policy in force. See
[generation-memory policy and calibration](doc/generation-memory.md) for details.

## 2026-09-22 — Experimental memory-accounting lineage archived

`main` was reset and force-pushed to
[`c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3`](https://github.com/jbg/eredu/commit/c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3)
before generation-memory forecasting was implemented afresh. This was a Git
history rewrite, not a normal API change on top of the previous `main`.
Pins from the experimental lineage, including
[`b530d4bb48cc9165b17c269aef8673c36e18d626`](https://github.com/jbg/eredu/commit/b530d4bb48cc9165b17c269aef8673c36e18d626)
through
[`5e14c3868885bf91e7a90569e1e03f7bd0761ff1`](https://github.com/jbg/eredu/commit/5e14c3868885bf91e7a90569e1e03f7bd0761ff1),
are no longer ancestors of current `main`. This notice documents that earlier
rewrite retroactively.

The archived experiment introduced **enforced memory admission and managed
prepared-chat/plain-text execution**, including `eredu_runtime::working_memory`,
`eredu_core::HostMetadataFunding`, and facade APIs such as `PreparedChatRequest`,
`PreparedChatSession` and `ManagedPlainTextRequest`. Those APIs belong to the
archived lineage. The current [generation-memory API](doc/generation-memory.md)
provides descriptive forecasts; it is not a rename or compatibility layer for
the experiment's allocation accounting and execution authorization.

| Purpose | Immutable revision |
|---|---|
| Return to the last experimental revision pushed to `main` | [`5e14c3868885bf91e7a90569e1e03f7bd0761ff1`](https://github.com/jbg/eredu/commit/5e14c3868885bf91e7a90569e1e03f7bd0761ff1) |
| Inspect the complete later archive snapshot | [`c4883758e7b5441a65e8125e1b6da043260059df`](https://github.com/jbg/eredu/commit/c4883758e7b5441a65e8125e1b6da043260059df) |
| Return to the pre-experiment base from which current `main` resumed | [`c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3`](https://github.com/jbg/eredu/commit/c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3) |

The old commits remain reachable through
[`archive/memory-accounting-experiment`](https://github.com/jbg/eredu/tree/archive/memory-accounting-experiment).
For an application already pinned to another archived revision, retain that
exact SHA to reproduce its API; do not replace it with the archive branch tip.
For example, a Cargo Git dependency can retain the former `main` API with
`rev = "5e14c3868885bf91e7a90569e1e03f7bd0761ff1"`. Keep first-party Git dependencies
on the same revision when they share public types.

To inspect the old revision without changing an existing checkout:

```sh
git fetch origin refs/heads/archive/memory-accounting-experiment:refs/remotes/origin/archive/memory-accounting-experiment
git worktree add --detach ../eredu-archived-api 5e14c3868885bf91e7a90569e1e03f7bd0761ff1
```

The archive preserves work for reference, not as a validated release. Its final
checkpoint explicitly records incomplete native integration and packaged-consumer
verification. The experiment was stopped after accounting and authorization grew
beyond the memory-estimation goal. Current development resumes from the base
above; archived implementation work is not a supported upgrade path to current
forecasting.
