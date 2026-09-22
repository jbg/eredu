# Changelog

This file records consumer-facing compatibility changes and Git lineage notices.
Workspace crates have independent release versions; see the
[release guide](doc/releasing.md) for package release records.

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
