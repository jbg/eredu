# CLI source and session migration

The initial inventory found one preparation branch and three generation
consumers: semantic/literal prepared chat, prepared speculative chat, and raw
text. The first two used superseded convenience request/driver APIs; the raw
branch owned another incremental-decoder/EOS loop around `generate_tokens`.

The main binary now compiles tokenizer and template sources from the loaded
model's retained configuration, preserving GGUF metadata and explicit overrides.
Chat uses `PreparedChatRequest` with explicit semantic or literal output policy;
speculation uses `PreparedChatSpeculativeRequest` with
`PreparedChatPrompt::Rendered` and `generate_prepared_chat_speculative(request)`.
The prepared chat owns the tokenizer source, so no separate tokenizer argument
is passed at generation startup. Raw text uses the canonical managed plain-text source and decoder driver. The CLI
retains checkpoint overrides, Mirostat, seed, speculation scheduler/assistant,
tool and reasoning configuration, incremental semantic output, cancellation on
output failure, TTFT, telemetry, and numerical expert-cache benchmarks. Terminal
finish reasons come from the shared driver, without an EOS-length heuristic.

`--managed-memory-capacity-bytes` is positive, defaults to 1 GiB, and is shared
by preparation and execution. Isolated automatic-benchmark children inherit it.
This capacity is separate from weight residency. Caller-owned terminal output
and reports remain application storage; this migration does not claim to bound
all CLI I/O or model loading.

Default generation performs no ordinary prompt encoding. Ordinary chat
telemetry reads the exact prompt count from the prepared session's canonical
attribution. Explicit expert-cache benchmarks retain their exact token-ID
witness. Raw/speculative telemetry still uses optional inspection because those
public outputs do not expose a prompt count. A backend omitting chat attribution
also uses inspection only when verbose/telemetry output is requested. None of
these inspection vectors grants generation authority.

`tests/k2_distributed_control.rs` is a separate complex caller: native
multi-rank resident/host/disk execution, exact token baseline, observed records,
snapshots, restore/fork/exchange, cancellation and intervention. It now uses the
canonical source/record/snapshot composition while preserving those assertions.
Its native execution is recorded separately from compiler checks in the
[CLI validation record](validation/bounded-followup-native-cli-control-2026-09-18.json).
Portable and native metadata checks pass. All three directly changed argument and
finish-reason unit tests pass on the rebuilt native CLI artifact.

## Current native execution scope

The selected-executor optimized artifact passes **all six residency cases**.
Dense and Mova each pass Resident, Host and Disk, with both greedy and stochastic
sampling, eight-token parity, snapshots, restore, repeated unfinished parent/child
exchanges, paid Hybrid reset and the intentionally exhausted cumulative Encoded
capture limit. Mova uses TP2/PP2/EP2 across eight local CPU ranks. Each case retains
its 120-second fixture deadline and 64 GiB request ceiling.

| Case | Process wall time | Result |
| --- | ---: | --- |
| Mova Resident | 65.01 s | Passed |
| Mova Host | 64.26 s | Passed |
| Mova Disk | 63.49 s | Passed |
| Dense Resident/Host/Disk together | 41.03 s | 3 passed |

The executable is
`/private/tmp/eredu-current-cli-distributed-control-selected-executor-opt`, SHA-256
`bbcb3257c20aec02840b41e31db5462c0d0b026c1e59c9e34a118fc04ec5afa6`.
Exact commands and log hashes are in
`/private/tmp/eredu-native-selected-executor-opt-cli-{mova_resident,mova_host,mova_disk,dense}.json`;
build provenance is
`/private/tmp/eredu-selected-executor-opt-cli-build-command.json` and
`/private/tmp/eredu-public-selected-executor-opt-build-record.json`.
The [validation record](validation/bounded-followup-native-cli-control-2026-09-18.json)
retains prior failures and scoped passes.

The build optimizes six Rust packages at level 2 while preserving development
debug assertions, overflow checks and the native development profile and guards.
Test processes use an explicit 64 MiB Rust test-thread stack. These are functional
runs, not throughput benchmarks or proof of default-stack behavior. No Cargo
build or competing native matrix ran during the distributed cases.

The partitioned executor lends its actual resident policy. Its explicit group
submission mechanism distinguishes shared graph traversal from manual traversal,
so source accounting retains the addressable parameter registry while pricing
only submissions that execute. CPU indexed execution includes all four parent
operands in its completion frontier. Native role retirement follows the actual
enclosing scope, with one bounded deadline and custody retained through timeout.
These changes close the prior resident identity/root-capacity refusals and the
premature readiness retirement error. Earlier unoptimized Host/Disk timeouts
remain recorded; the accepted optimized runs use the same deadline.

`ExistingPhysical` publication preserves the canonical ordinary, copy-funded or
prepaid charge. Loaded Array and Host inventories share a payload-free receipt;
lazy Host sources and canonical parameter Array cells retain actual publication
proof. The separately recorded persistent Disk fixture completes four admitted
requests with nonzero parity, source-baseline retirement, stable cached parameter
identities and warm payload reuse witnessed by its real reader.

The source3 backend checkpoint passes 66 focused cases, including actual typed
CPU arithmetic, views, row movement, selectors, complete constructor inventories,
funded exclusions and exact empty Slice sources. The selected-executor artifact
passes seven focused backend cases and all seven public tool/media
cases, including TP2 at 8 GiB. Both corrected native readiness fixtures also pass;
six paged-control cases and four released text/image requests retain
their separate prior artifact scope.
