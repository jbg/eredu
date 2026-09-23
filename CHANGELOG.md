# Changelog

This file records consumer-facing compatibility changes and Git lineage notices.
Workspace crates have independent release versions; see the
[release guide](doc/releasing.md) for package release records.

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
