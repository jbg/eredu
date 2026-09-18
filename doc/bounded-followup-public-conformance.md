# Public prepared-chat conformance

The portable facade fixtures compile the exact caller-supplied tokenizer and
chat-template configuration into the runtime's retained sources, prepare a chat,
and execute `PreparedChatRequest`. Their neutral backend has ordinary generation
support; ordinary semantic tests do not require a speculative implementation.
Prompt, step and sequence storage use the fixture's real runtime pool and
execution identity. Explicit capacities are caller ceilings, not estimated
allocation costs. Source and shared-shell owners retain their original payers.

## Current public result

The complete portable integration target passes **38 tests, zero failures, two
ignored** on 2026-09-18. The ignored cases require external LFM2 and pinned GGUF
checkpoint paths. The final combined build took 55.61 seconds; this suite ran in
15.40 seconds:

```sh
RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu --no-default-features \
  --test backend_conformance --test portable_facade
```

Log: `/private/tmp/tokenizer-canonical-observation-public-tests.log`.
This run covers canonical source compilation, ordinary/manual semantic parity,
Qwen tool termination, Nanbeige XML whitespace and replay semantics, reasoning
termination, sparse vocabulary holes, resolved sampling policy, source/receipt
custody, cancellation and neutral provider error preservation. The fixture's
workspace quote moves its funded diagnostics with `finish_report`, matching the
production producer; it does not use the cloning report API or increase capacity.
This run includes the canonical candidate quote producer, reservation-owned
admission reports, shared text/media execution-context loans and the cold
grammar inspection/schema/JSON/Lark frame corrections. The final rerun also
includes canonical neural error ownership, destination-aware architecture hooks,
paid observation rebinding and shared routing declarations. The earlier cold
compiler checkpoint remains in `/private/tmp/eredu-final-cold-portable-suites2.log`.

The speculative cases use the same invocation preparation. They verify that an
exact supplied prefix reaches the backend unchanged, invalid sparse IDs fail
before prompt construction, single/batch failures retain provider classification,
and Mirostat settings survive mixed batches. Passing these cases establishes
speculative text behavior only.

The separate ordinary-only semantic library tests cover Required and Auto tools,
arguments split across tokens, forbidden/text policy, reasoning, committed-event
order, once-only terminal delivery, control forcing, source refusal, record
ownership and manual/uninterrupted parity. Scoped prior results and current
remaining work are indexed in [the consolidation overview](bounded-followup.md).

The complete `backend_conformance` target now passes **126 tests, zero failures**
(3.37 seconds). Its capture path constructs the actual finite host schedule and
runtime capture bank; host-summary producers use actual values and retain their
bank through output or failure. The suite includes capture, continuation,
media-attribution, provider-error and snapshot/branch behavior, including terminal
restore across all four capture/intervention modes. Command:

```sh
cargo test --offline -p eredu --no-default-features --test backend_conformance
```

Log: `/private/tmp/tokenizer-canonical-observation-public-tests.log`. The four record-construction
and refusal tests also pass, including the public closed output checkpoint's
retained payer. A prepared-session parity test passes all four policy modes and
keeps an escaped checkpoint charged after session, model and other records retire.
These are neutral behavioral results, independent of native device validation.

The final compatibility cleanup removes two test-only plain-startup forwarding
helpers from production and requires an explicit serialized `CapturedStep`
outcome, preserving valid `Untracked` records. Focused checks pass **27 distinct
tests**: six core capture tests, eleven runtime checkpoint tests and ten facade
startup tests; one external-checkpoint test is ignored. Exact commands,
environment, log hashes and source hashes are recorded in
`/private/tmp/tokenizer-final-cleanup-checks.json`. This is portable behavioral
evidence, not a native completion claim.

The earlier exact-coordinate indexing checkpoint passed all 211 NN library tests and all
1,085 runtime working-memory tests, including the migrated alias consumers.
Logs: `/private/tmp/contracts-index-coordinates-nn-all.log` and
`/private/tmp/contracts-index-coordinates-runtime-tests.log`.

The final cold-compiler/controller check also passes 47 facade constraint tests
(two external-tokenizer probes ignored), plus the retained schema-row owner
regression. Logs: `/private/tmp/eredu-final-cold-facade-constraints.log` and
`/private/tmp/eredu-final-cold-schema-row-owner-test2.log`. The matching dependency
compiler suite passes all 90 llguidance tests; see
[cold grammar compilation](bounded-followup-grammar.md).

The native-feature Rust metadata check passes for all facade tests and examples
with `mlx,metal,image,audio` enabled (50.75 seconds, `DOCS_RS=1`):
`cargo check --offline -p eredu --no-default-features --features mlx,metal,image,audio --tests --examples`.
Log: `/private/tmp/eredu-canonical-native-all-metadata3.log`. It establishes caller
and feature compatibility, without executing native operations.

Incremental/residual admission now returns only the reservation and accepted
quote; callers borrow diagnostics from the reservation. It constructs reports
through the funded metadata worker and retains allocating failures through their
existing error owner. Residual-admission tests pass after this ownership change. The final sweep
passes **244 tests, zero failures** in 0.36 seconds, including terminal zero-span
identity/backing preservation, terminal refusal, funded quote aliases and paid
copy-on-write. Command:
`cargo test --offline -p eredu-runtime --lib working_memory::residual::tests`
with the same compiler environment above. Final binary-run log:
`/private/tmp/tokenizer-shared-quote-residual-complete.log`; the preceding Cargo
run is `/private/tmp/eredu-terminal-owned-diagnostics-residual-all.log`.

The subsequent terminal control correction also passes 75 `bounded_prefill`
tests and the same 244 residual tests (243 under `residual::tests`, plus the
adjacent original-sequence layout test). A terminal placement has no source,
span or score-indexing role; its actual control-bank metadata remains priced.
The behavioral fixture verifies no executor calls or output and unchanged
nonzero state. Logs: `/private/tmp/eredu-terminal-prefill-controls-tests.log`,
`/private/tmp/eredu-terminal-prefill-residual-tests.log` and
`/private/tmp/eredu-terminal-prefill-residual-adjacent-test.log`.

The terminal-copy follow-up permits sealing an exact validated empty inference
schedule while retaining actual copied state, source pins and host controls. The
three focused terminal cases pass, including exact-capacity admission,
one-byte-short refusal and unhealthy source rejection. The same current binary
passes 243 residual cases and 33 sampling-copy cases. Logs are
`/private/tmp/contracts-terminal-copy-tests3.log`,
`/private/tmp/contracts-terminal-seal-residual-tests.log` and
`/private/tmp/contracts-terminal-seal-sampling-copy-tests.log`. This is neutral
contract coverage; native terminal restoration is validated separately.

## Portable feature checks

The portable feature milestone passed these offline checks with Rust 1.98.0 and
incremental compilation disabled:

| Packages | Feature policy | Result |
| --- | --- | --- |
| collections, GGUF, NN macros, core, checkpoint, runtime, architectures | Default | Passed |
| text, media, facade | No default features | Passed |
| NN, codec | All features | Passed |
| media | All features | Passed |

Exact commands are retained in
`/private/tmp/tokenizer-final-portable-defaults-check.log`,
`/private/tmp/tokenizer-final-portable-no-defaults-check.log`,
`/private/tmp/tokenizer-final-nn-codec-all-features-check.log`, and
`/private/tmp/tokenizer-final-media-all-features-check.log`. These builds establish
portable compilation and optional-feature boundaries. After the typed LayerNorm
and shared media-transaction changes, NN with all features and architectures
passed again in 9.92 and 22.05 seconds; logs are
`/private/tmp/tokenizer-final-neutral-nn-all-features.log` and
`/private/tmp/tokenizer-final-neutral-architectures.log`. The public suites above
were also rerun after the shared execution-context and cold compiler frame changes. The no-default native
backend library/test metadata check also passes after its test-only feature gates
were corrected; native device behavior is validated separately.


## Tokenizer semantics and independent references

The WordLevel and Whitespace fixtures preserve their original semantics. WordLevel
uses the pinned hashbrown table facts to pay forward/reverse tables and spellings
before construction. Sparse IDs do not create a dense table or a serialization
loop through holes. Token and ID output use the same whole-span lookup, including
unknown-token failure. Whitespace retains the exact Unicode language
`\w+|[^\w\s]+` as an immutable source used by the shared regex engine.

The focused WordLevel suite passes 11 tests, including reached reservation
refusals, maximum sparse IDs, Unicode spans, added tokens and missing unknown-token
behavior. The complete tokenizer checkpoint passes 291 tests with three existing
ignored benchmarks. A trait-object source-alias test compares the shared data
address rather than vtable identity.

An independent release-mode program builds pristine crates.io tokenizers 0.23.2
alongside the selected fork. Across **4,913** combinations of Unicode words,
combining marks, punctuation, join controls and whitespace, ordinary/source token
IDs, spellings and byte offsets match exactly; ID-only output matches those IDs.
For 250 repetitions of a mixed input, pristine ordinary encoding took 80.70 ms,
selected ordinary encoding 88.87 ms (1.10x), and selected ID-only encoding 29.01 ms.

The standalone manifest is `/private/tmp/eredu-wordlevel-reference/Cargo.toml`:

```sh
cargo run --release --offline --manifest-path /private/tmp/eredu-wordlevel-reference/Cargo.toml
```

Output: `/private/tmp/contracts-wordlevel-reference.log`. This is a particular
reference workload, not a universal throughput guarantee. Added-token adverse
workloads and construction storage are recorded separately in
[tokenizer consolidation](bounded-followup-tokenizer.md).

The BPE byte-fallback worker uses a fixed six-byte ASCII spelling and a borrowed
all-or-unknown check. It preserves unknown fusion order without temporary
formatted strings or fallback vectors. The independent pristine executable
compared **864** complete/partial byte-vocabulary and fusion cases: ordinary IDs,
spellings and offsets, and original ID-only output all match. Source:
`/private/tmp/eredu-wordlevel-reference/src/bin/byte_fallback.rs`; result:
`/private/tmp/contracts-bpe-fallback-reference.log`.

These comparisons validate the tokenizer mechanisms they exercise. Native
multimodal execution, released tool admission and application memory/cache policy
have separate acceptance records and remain explicitly scoped.

## Qualified native public checkpoint

The selected-executor optimized checkpoint passes **all seven public native
cases** at the unchanged **8 GiB** capacity. TP2 now quotes the actual executor's
group-submission mechanism while retaining the resident registry. This closes
the runtime-completion artifact's TP2 admission regression; that failed attempt
remains recorded separately.

The executable is
`/private/tmp/eredu-current-prepared-chat-native-selected-executor-opt`, SHA-256
`06fdf15f943ffb3dcfbb7a1e26ba0c2bbc4cf25caf624d58c9ca1500c902e50a`.
Build provenance is `/private/tmp/eredu-public-selected-executor-opt-build-record.json`
and `/private/tmp/eredu-selected-executor-opt-cli-build-command.json`. Six Rust
packages use development-profile optimization level 2 with debug assertions and
overflow checks enabled; native development guards remain unchanged. All runs
use an explicit **64 MiB Rust test-thread stack**. These are functional checks,
not performance benchmarks or default-stack qualification.

Exact commands, full log paths and hashes are in
`/private/tmp/eredu-prepared-selected-executor-opt-checks.json`. Times below are
process wall time, including startup, rather than test-harness time.

| Case | Process wall seconds | Log SHA-256 |
| --- | ---: | --- |
| Combined image/audio tools | 4.433 | `cde179c959d3f82b3e64e06be2a28ea37d0d5776b9e0216025ca15680a1f3e58` |
| TP2 image/tools | 3.677 | `a90d913be393438180b627255d0ddadb9c3a6ff44f96d5bfe05d535558aaa147` |
| PP2 image/tools | 5.426 | `c4b84615ce1c3871e07a9c6844d40df1d198ea1d3071c59f297705b6490d1d8a` |
| TP2/PP2 image/tools | 9.802 | `fb9fa7a4b1a404cfa75a40501e5ff3b4788944e54638f5786ffbb070ced8f739` |
| Image/manual/speculative parity | 3.434 | `ef08036f540b039b9bd25b851706b9a411779dd7cdfd6d83c712007090260bc1` |
| Image snapshots | 2.679 | `c88a077d63fefb1f25e1bd2a82ec5ff3214572389c1f849c83a054aeb4c9662b` |
| Ordinary tools/capture/child sampling | 1.533 | `c716cb3a48c817ecdb3f1afcbd7be24e4abf8afc62e434264bb84109f8d5cadc` |

The matching optimized CLI artifact also passes all six Dense/routed
Resident/Host/Disk cases. Its exact records are
`/private/tmp/eredu-native-selected-executor-opt-cli-dense.json`,
`/private/tmp/eredu-native-selected-executor-opt-cli-mova_resident.json`,
`/private/tmp/eredu-native-selected-executor-opt-cli-mova_host.json`, and
`/private/tmp/eredu-native-selected-executor-opt-cli-mova_disk.json`; see
[CLI validation](bounded-followup-cli.md) for that distinct matrix.
The six paged-control passes remain scoped to the prior runtime-completion
artifact, and the four released tool runs to the recorded CPU routing-source
artifact. These counts overlap earlier checkpoints and are not a combined total.
Both corrected readiness fixtures also pass native execution. They use the
backend test target at optimization level 0, selected neutral dependencies at
level 2, and unchanged production code. Exact runs are recorded in
`/private/tmp/eredu-readiness-fixture3-backend-checks.json`; artifact and build
provenance are in `/private/tmp/eredu-readiness-fixture3-backend-build-record.json`
and `/private/tmp/eredu-readiness-fixture3-retry-backend-build-command.json`.

### Earlier public checkpoints

The earlier CPU routing-source artifact includes canonical typed-zero and
Transpose traces and physical publication receipts across lazy Host sources and
persistent canonical CPU arrays. All **seven prepared-chat cases pass**:
ordinary tool/capture/child-sampling
control, combined image/audio tools, three TP/PP image/tool placements, independent
speculative image parity, and media snapshots. The separate paged Device/Host/Disk
control target passes **six cases together** (4.97 seconds reported by the test
harness; 10.62 seconds process wall time). Limits are unchanged. These passes use an explicit **64 MiB Rust
test-thread stack**; they do not establish default-stack execution or completion
of the distributed CLI matrix. Dense resident, Host and Disk control cases now
complete. Source retention and local bank composition pass the separate
29-case source-contract checkpoint; subsequent CPU source qualification is in the
[native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json).
The later six-case CLI pass is recorded above and in
[CLI validation](bounded-followup-cli.md).

The prepared-chat executable is
`/private/tmp/eredu-current-prepared-chat-native-cpu-routing-source`, SHA-256
`3a57f042f0c86eb9587b2d1c20f54f6ed61dcc654cfdea000a2ecfd5582aef2a`.
The control executable is
`/private/tmp/eredu-current-control-native-cpu-routing-source`, SHA-256
`363e5ff11728163e74f45a59b6682d7291751a2160f1798c229b5c77c0e0a922`.
Build provenance is `/private/tmp/eredu-public-cpu-routing-source-build-record.json`
and `/private/tmp/eredu-cpu-routing-source-build-command.json`;
exact commands, per-process wall times and logs are in
`/private/tmp/eredu-prepared-cpu-routing-source-checks.json` and
`/private/tmp/eredu-native-cpu-routing-source-control.json`.
Durations below are test-harness times, distinct from process startup/wall time.
These debug runs are functional validation, not performance benchmarks.

The preceding seven/six-case checkpoint remains in
`/private/tmp/eredu-public-source-receipt-build-record.json`,
`/private/tmp/eredu-prepared-source-receipt-checks.json` and
`/private/tmp/eredu-native-source-receipt-control.json`.

The recorded ordinary-filter runs without `RUST_MIN_STACK` abort on the default
2 MiB test stack. After the first fixture split, initial startup succeeds and the
failure moves to the first checkpoint's `fork_snapshot`. Symbolication identifies finite
large debug frames in the fixture and shared resume quotation chain. The second
test-only split separates checkpoint creation from serial fork/exchange/restore;
independent review preserves assertion order and exact owner lifetimes. Its
`DOCS_RS=1` metadata check passes in 48.17 seconds, but native validation of that
split still fails on the default test stack. The diagnostic rerun proceeds
through the earlier capture/restore lifecycle and aborts during the later
`sampling::check_recorded_child` call to `LoadedModel::start_controlled_chat`
(exit -6, 9.279 seconds process wall time). The external probe resolves
`sampling::emit` within a finite unoptimized startup chain. No further
default-stack repair is planned; the full passing native result remains
qualified to the explicit 64 MiB debug test stack. Records:
`/private/tmp/eredu-native-source-receipt-default-stack.json`,
`/private/tmp/eredu-native-disk-custody-probe-default-stack.json`,
`/private/tmp/native-default-stack-diagnosis.json`, and
`/private/tmp/native-prepared-chat-fork-split-metadata.log`.

## Child resume-time sampling: native execution

The public native filter
`ordinary_tool_capture_snapshot_restore_and_fork_preserve_committed_semantics`
passed **one test in 2.84 seconds** at the historical canonical-cell receipt
checkpoint, including Required and Auto and the resume-time child sampling trial.
Log:
`/private/tmp/eredu-native-canonical-cell-receipt-ordinary.log`. Its prepared-chat
artifact is `/private/tmp/eredu-current-prepared-chat-native-canonical-cell-receipt`,
SHA-256 `f395cd1607e81665bbd5f43035469a9d155269c423d31244a483a927e63b0041`,
recorded in `/private/tmp/eredu-public-canonical-cell-receipt-build-record.json`.
It uses the same 8 GiB capacity and explicit 64 MiB test stack, and includes both
test-only phase splits. The earlier 2.89-second source
receipt pass remains in `/private/tmp/eredu-native-source-receipt-ordinary.log`,
and the 2.81-second pass in
`/private/tmp/eredu-native-existing-physical-ordinary.log`. The separate
second-split diagnostic run reaches child startup but overflows on the default
stack, as described above; it does not establish default-stack execution.

The trial supplies temperature 0.5 and seed 711 through
`GenerationBranchOptions::sampling` at the actual recorded fork endpoint. It
checks the requested change and before/after sampler facts in the branch record,
the installed child facts, and the unchanged parent's identity, committed prefix
and pending forced token. A snapshot of the changed child restores those facts
and matches manual advancement with uninterrupted delivery. Returning to the
parent preserves its original continuation. Both paths match the deterministic
fixture's token and semantic-event oracle; one snapshot and one branch suffice,
and copy spending increases without refund when the owners retire.

The existing direct default-restore, terminal snapshot and capture assertions
remain covered. Direct `PreparedChatResumeSettings` preserve saved sampling;
explicit child changes use the recorded branch options on the same underlying
prepared fork worker. This result establishes the exercised sampler lifecycle
and tool semantics, not a statistical randomness or throughput benchmark.

## Combined image/audio tool fixture: native execution

The public native filter
`audio::authenticated_image_audio_tools_preserve_ordinary_manual_and_recorded_semantics`
uses Gemma Unified's actual nonzero image patches and audio frames, with the
existing encoder geometry and masks. It checks Required and Auto through ordinary
`run`, manual `advance` and recorded sessions, without a drafter. Assertions cover
authenticated source refusal, exact rendered/decoder coordinates and attribution,
uneven prefill followed by multiple cached predictions, split tool arguments,
ordered once-only events, and matching token IDs and `GrammarComplete`
termination without visible text or reasoning.

Compiler metadata passes with `mlx,metal,image,audio` and `DOCS_RS=1`:
`cargo check --offline -p eredu --no-default-features --features mlx,metal,image,audio --test prepared_chat_native`.
The corrected fixture's metadata log is
`/private/tmp/tokenizer-audio-child-sampling-metadata2.log`. Its initial native
attempt correctly rejected fixture-specific marker spellings at authenticated
association. The fixture now uses Gemma's declared `<|image|>` and `<|audio|>`
markers; no production association check or capacity changed.

At the historical canonical-cell receipt checkpoint, actual Metal execution
**passed one test in 3.82 seconds**, covering all six
Required/Auto × ordinary/manual/recorded combinations at the unchanged 8 GiB
capacity. Log: `/private/tmp/eredu-native-canonical-cell-receipt-audio-tools.log`.
The executable is the canonical-cell receipt binary identified in the child
sampling section, with provenance in
`/private/tmp/eredu-public-canonical-cell-receipt-build-record.json`. The
earlier 3.97-second pass remains in
`/private/tmp/eredu-native-source-receipt-audio-tools.log`; the 3.89-second pass is in
`/private/tmp/eredu-native-existing-physical-audio-tools.log`, with provenance in
`/private/tmp/eredu-public-existing-physical-build-record.json`. These runs use an
explicit 64 MiB Rust test-thread stack and are functional debug validation,
not performance benchmarks.

The decoder uses deterministic scripted token transitions to isolate the
semantic join. This result establishes actual nonzero image/audio encoder
execution and canonical tool delivery for the fixture, not released-model audio
understanding or independent audio-logit accuracy.
