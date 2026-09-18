# Prepared-chat conformance

Portable fixtures compile the caller's exact tokenizer/template configuration,
prepare a chat and execute `PreparedChatRequest` through a neutral ordinary-only
backend. Prompt, sequence and step storage use the real fixture pool and execution
identity. Ordinary semantic tests require no speculative implementation.

## Portable behavior

The portable facade suite has 38 passing tests and two ignored external fixtures;
backend conformance has 126 passing tests. Coverage includes disabled, forbidden,
Required and Auto tool policies, reasoning, fragmented arguments, event ordering,
cancellation, source refusal and manual/uninterrupted parity.

Capture records require explicit outcomes, retain prompt attribution and round
trip diagnostic ownership without recreating source authority. Controller tests
cover source/copy retirement, named templates, invalid EOS, caller stops and
logical token domains with overlapping added-token IDs.

```sh
cargo test -p eredu --no-default-features --test portable_facade
cargo test -p eredu --no-default-features --test backend_conformance
```

## Native public behavior

Seven prepared-chat cases pass: ordinary tools with capture and child sampling;
image tools; image snapshot/restore; image speculation parity; TP2, PP2 and
combined TP2/PP2 image tools; and combined audio/image tools. The image-tools
case includes independent speculation parity, so these categories overlap.
The exact seven test filters are in the
[native result record](validation/bounded-native-results.json).

Required and Auto run through ordinary, manual and recorded sessions. Tests use
uneven prefill chunks, cached decode, semantic coordinates and fragmented tool
arguments. Snapshot/fork/restore retains ownership and cumulative budgets.
Parallel media cases use an 8 GiB capacity. The combined audio/image fixture runs
real nonzero encoders with a deterministic decoder; it proves ingress, tool
delivery and source attribution, not released-model media understanding.

Six paged-control cases exercise Device/Host/Disk parity, snapshots, forks,
resets, escaped outputs and fresh loading. Six distributed CLI cases exercise
Dense and Mova in Resident/Host/Disk modes. Mova uses TP2/PP2/EP2 with eight local
CPU ranks, both samplers, eight-token parity, repeated unfinished parent/child
exchange, paid Hybrid reset and cumulative capture refusal.

## Qualification

The recorded public and CLI builds use optimization level 2 for six Rust
packages, retaining debug assertions, overflow checks and native development
guards. Tests use a 64 MiB Rust test-thread stack. Default 2 MiB debug-stack
execution is not established. CLI cases retain a 120-second deadline and
64 GiB capacity. These functional runs are not throughput benchmarks.

Apple CPU/Metal and Ring are the native hardware scope. An unrun configuration
is not established by a portable test or metadata build. Released text and image
requests have separate [checkpoint validation](prepared-chat-validation.md).
