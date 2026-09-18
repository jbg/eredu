# Bounded inference consolidation

This work consolidates the execution paths introduced around `77e08045`, audited
against `c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3`. Ordinary prepared chat now
supports admitted semantic tools and authenticated media without speculation.
This guide describes the consolidated design and its evidence. The final public,
distributed and readiness matrices pass. Linked records retain
source checkpoints, historical failures and measurement limits.

## Canonical execution

`LoadedModel::prepare_chat` borrows a request and compiles it against the loaded
model's retained tokenizer and template. Its closed `PreparedChat` owns the
render, selected policy, compilation receipt and authenticated capacity.
`PreparedChatRequest` selects rendered text, an exact token prefix, or an
`OriginalModelInput` produced by `prepare_chat_input`.

Ordinary generation starts with `start_prepared_chat`. `PreparedChatSession`
uses `ControlledTextGeneration`, the committed-token cursor, the source-funded
controller and the shared semantic publisher. Manual `advance`, uninterrupted
`run`, pause/resume and controlled records advance that same session. Ordinary
tool calling requires an ordinary generation backend only.

`prepare_chat_invocation` validates the retained source, exact input, capacity
and resolved generation settings, then prepares the controller, semantic channels
and stops. Ordinary and speculative requests call this worker. Speculation adds
an actual drafting strategy, proposal/verification state and an event window;
`PreparedChatSpeculativeRequest` consumes the same prepared chat and input enum.
Its single, batch, controlled and observed consumers retain prompt and error
custody. An exact token prefix is validated directly and is never decoded and
re-encoded. Native speculative media derives each role's binding from the same
authenticated host source and that role's actual copied cache. Native image/tool
parity passes with this retained binding.

The request's optional raw capture and intervention declarations are admitted
once actual text or retained media geometry is available. Existing original
sources can instead be supplied through its options. Record, observation,
transport and snapshot quotas remain distinct from host allocation admission.

See [the public integration guide](prepared-chat.md) and
`eredu/examples/prepared_chat_generate.rs` for an ordinary tool request and
source-authenticated host-media input. The example publishes semantic events;
it does not execute requested tools.

## Consolidation inventory

| Former duplication | Current owner and distinction |
| --- | --- |
| DAAC and packed added-token matchers, cache-dependent model representations | One tokenizer matcher and shared source/encoding workers; explicit cache capacity. Semantic and performance evidence is in [tokenizer consolidation](bounded-followup-tokenizer.md). |
| Allocating chat/text startup and source-funded semantic startup | `PreparedChat`, common invocation preparation and `PreparedChatSession`. The former allocating public generation/observation front doors and their request/controller/decoder representations have been removed. |
| Speculative-only semantic composition | Shared controller, channel, stop, semantic-state and callback preparation. The old speculative request family and independent single/batch/control startup implementations have been removed. |
| Ordinary and enforced layer acquisition | One retirement, window, transfer, lease and population worker with explicit finite-attempt/recovery policy. Foreground and background I/O retain their actual completion witnesses. |
| Reconstructed ordinary Host snapshots and retained-owner validation | One locked manager, destination and immutable-owner comparison for ordinary and paid validation/pinning; temporary parameter constructors use one source census for CPU and Metal. |
| Ordinary and admitted FP8/native setup | Shared finite kernel definitions and fixed invocation setup. CPU, Metal and CUDA arithmetic remain distinct where required. |
| Capture drain and compatibility record families | One fallible custody-preserving delivery and controlled-record representation, with source-paid record and prompt-attribution construction. |
| Cancellation overload chain | Boundary-aware agreement after retirement, preserving cancellation sampling and distributed agreement. |
| Owned/borrowed parameter and graph adapters | Borrowed canonical contracts with explicit paid ownership at consuming boundaries. Contract consolidation is implemented; [contract evidence](bounded-followup-contracts.md) records the representation and ownership audit. |
| Eagerly formatted neural source errors and retained-error adapters | One typed retained-source owner, with prospective constructor funding and original-cause custody. Caller-owned diagnostic text remains a distinct message value. |
| All-KV outer-only and grouped Hybrid copies | One grouped worker for ordinary and prepared copies, preserving every actual child table, including empty metadata headers; obsolete selectors, publication variants and preparation-only forwarding methods are removed. |
| KV-only paged reset construction | Shared source-authenticated manager construction for KV and Hybrid state, with component roles and prepared context passed through the canonical neutral reset contract. |

The obsolete ordinary grammar/controller constructors and their allocating
runtime variants have been removed. Cold semantic inspection retains its
tokenizer compiler and supported reporting behavior. Distinct checkpoint
formats, protocol dialects, family equations, devices and residency policies are
supported mechanisms, not compatibility implementations.

Detailed inventories and results are recorded in
[native consolidation](bounded-followup-native.md),
[ordinary front doors](bounded-followup-ordinary-frontdoors.md),
[controlled composition](bounded-followup-control.md),
[record ownership](bounded-followup-records.md), and
[parameter contracts](bounded-followup-parameters.md).

## Ownership and enforcement

- Source compilation reserves before tokenizer, template, grammar, schema and
  semantic preparation. Compiler receipts authenticate the exact tokenizer,
  selected execution and original account; registration or equal-valued copies
  do not create authority.
- Immutable grammar, render, declarations and source bytes share their paid
  owners. Mutable parser, cursor, sampling and continuation state has independent
  prospective construction and copy admission.
- Media binding retains the actual completed host/native source and
  architecture-declared render and decoder coordinates. It compares ordered text
  tokens and exact framing; hashes or caller-provided ranges cannot authorize a
  substitution. Prefill chunks share the prepared media ingress.
- The native backend retains admission through completion, terminal failure or
  safe teardown. Polling failure does not establish completion. Escaped events,
  outputs, errors, input aliases and snapshots retain their original payers.
- Restoring, forking or exchanging a branch preserves sampling state and
  cumulative observation, transport, copy and attempt spending. A terminal branch
  uses authentic empty execution geometry and emits no extra prediction.
- Unknown required bounds are typed refusals. Increasing an arbitrary allowance,
  changing grammar semantics or retrying without enforcement is not a repair.

These are framework-managed allocation bounds. Application buffers, copied
callbacks/events, allocator caches and unrelated process/system memory need
separate accounting. A successful framework test does not validate application
memory or cache policy.

Core/runtime contracts and architecture equations remain backend-neutral.
Architectures declare family geometry and semantic projections; the backend
realizes tensors, transfers, collectives and completion. The facade owns
tokenizer, tool and generation composition, with backend-neutral public errors.
Portable feature builds and manifest/visibility boundaries enforce that division;
the [architecture guide](backend-architecture.md) records the native safety and
optional-feature boundaries.

## Behavioral evidence

The table summarizes coherent, scoped runs. Counts from different rows overlap
and must not be added into a whole-tree total. Exact commands, source and binary
hashes, ignored fixtures and failed attempts remain in the linked records.

| Contract | Recorded result and scope | Evidence |
| --- | --- | --- |
| Public ordinary tools without speculative capability | A neutral ordinary-only backend covers disabled, forbidden, Required and Auto policies, reasoning, split arguments, ordered committed events, cancellation, source refusal and manual/uninterrupted parity. Latest portable checkpoints pass 126 backend-conformance and 38 portable-facade tests, with two external fixtures ignored. | [Public conformance](bounded-followup-public-conformance.md) |
| Controller, admission and escaped ownership | 372 facade-library tests pass, six are ignored; the exhaustive schema source sweep passes separately. All 244 residual-admission tests pass. Reached allocation refusals retain their first cause and payer; unreached cuts are not counted as swallowed refusals. | [Controller](bounded-followup-constraint-controller.md), [admission diagnostics](bounded-followup-quote-diagnostics.md), [records](bounded-followup-records.md) |
| Canonical portable contracts | 772 architecture tests pass, one is ignored; 206 runtime integration tests, 216 neural unit tests and five neural doctests pass. Paid state, parameter, path, media, boundary, collective and observation producers use the actual destination. | [Contract evidence](bounded-followup-contracts.md) |
| Cold grammar and schema construction | All 90 dependency compiler tests and 47 facade controller tests pass. Inspection, schema construction/copy and JSON/Lark emission fund their actual recursive frame overlap while retaining cumulative heap charges and original failure custody. | [Grammar construction](bounded-followup-grammar.md), [schema compiler](bounded-followup-schema-compiler.md) |
| Native public chat and control | The selected-executor optimized artifact passes seven prepared-chat cases, including TP2 at the unchanged limit. The separately recorded runtime-completion artifact passes six paged-control cases. Required/Auto ordinary, manual and recorded runs preserve uneven prefill, cached decode, semantic coordinates, split arguments, snapshots, forcing, child sampling/reseed and cumulative spending. TP2, PP2 and TP2/PP2 image/tool cases use the unchanged 8 GiB capacity. | [Public conformance](bounded-followup-public-conformance.md), [native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json) |
| Actual image/audio ingress with tools | The combined image/audio fixture passes both tool policies through ordinary, manual and recorded sessions. Real nonzero encoders join the canonical semantic path; its scripted decoder tests tool delivery and source attribution, not released-model media understanding. Independent speculative image/tool parity is separate passing evidence. | [Public conformance](bounded-followup-public-conformance.md) |
| Native numerical mechanisms, residency and custody | The 25-case mechanism and 15-case custody/reset/copy matrices pass. They cover serial resident/Host/Disk execution, assistants, YaRN, InputProducts, grouped RMS, completion, source refusal, Hybrid copies and cleanup. CPU Host spill and repeated Metal Host reload preserve escaped aliases at the unchanged 256-byte two-page Device limit. | [Native design and scope](bounded-followup-native.md), [native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json) |
| Selected CPU/Metal bank sources and persistent Disk | All 29 focused cases pass, including exact equation/copy populations, grouped projection, scalar slices, row candidates and publication. Four admitted Disk requests preserve nonzero token/state parity, cached parameter identity and exact retirement; actual reader counters witness payload I/O and legitimate warm reuse. Final affected CPU-source checks are tracked below. | [Native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json) |
| Observation parity | 13 workspace cases, two nonzero numerical parity cases and three TP2/intervention cases pass for affected DeepSeek V3/V4 paths. Ordinary and observed entries use the same coordinate-preserving worker; disabled instrumentation only suppresses hooks. | [Native design and scope](bounded-followup-native.md) |
| Tokenizer semantics and source authority | One matcher serves ordinary and admitted execution. Independent tests cover added/special tokens, overlaps, Unicode, flags, source refusal, cache policy and model/template construction. Ordered composition and exact default-Onig behavior include 10,008,576 contextual pattern comparisons. | [Tokenizer consolidation](bounded-followup-tokenizer.md), [composition](bounded-followup-tokenizer-composition.md), [source custody](bounded-followup-tokenizer-source.md) |

The final capture compatibility cleanup removes production forwarding APIs used only by tests
and implicit legacy record-outcome deserialization. Records require an explicit
outcome; the real `Untracked` outcome remains supported. Its focused portable
record contains 27 distinct passing tests and one ignored fixture. The
[public conformance guide](bounded-followup-public-conformance.md) links that
record and the explicit round-trip/refusal coverage.

### Released-checkpoint evidence

The official `Qwen/Qwen3.5-0.8B` checkpoint is pinned to revision
`2fc06364715b967f1860aea9cf38778875588b17`. The retained independent numerical
comparison covers prefill and three cached decode steps using ordinary resident
execution; it does not establish bounded tool admission. Artifact hashes and
numerical tolerances are in the
[native numerical record](validation/bounded-followup-native-2026-09-18.json).

The ordinary public example separately passes Required and Auto for both text
and authenticated-image sensor requests at the unchanged **64 GiB** framework
capacity, prefill chunk size **128** and maximum **48** generated tokens. Each
of the four runs commits 25 tokens, publishes exactly one complete
`reading({"value":17})` call and finishes with `GrammarComplete`. The image
request crosses an uneven prefill boundary and executes the real encoder; its
value is supplied in text, so this is not visual-recognition evidence.
[Released tool validation](bounded-followup-released-tools.md) retains the
requests, events, executable and 13 freshly checked checkpoint-file hashes.

The separate generic-action Required request completes its 48-token allowance
without funding failure but emits prose and truncates without a tool call. It
remains an unsuccessful model request; successful admission is not proof of
useful tool behavior for arbitrary prompts.

## Performance and memory

The canonical added-token matcher has measured benefits and an explicit storage
tradeoff. On the recorded arm64 release build, chat-delimiter search improves
from 828.32 to 1,303.57 MiB/s; the adverse shared-prefix/no-match case decreases
from 620.61 to 370.26 MiB/s, about 34 additional microseconds per 32 KiB. The
source-derived retained buffer bound is `56*N + 29*B + 56` bytes, plus 11,844
bytes of fixed construction controls, compared with the former prefix
representation's `40*N + B` buffer bytes; `N` is declaration count and `B` is
total spelling bytes, including duplicates. Search uses a 56-byte iterator and
fixed locals without heap scratch. Arbitrary overlapping populations do not
have a claimed linear-time bound. Exact generators, all six comparison cases,
construction times and retained capacities are in
[tokenizer measurements](bounded-followup-tokenizer.md#storage-and-search-cost).

`ModelCachePolicy { capacity }` configures entry count on the same model
implementation; zero disables the cache. It is neither a byte limit nor native
allocator-cache policy. The pinned archive, licenses and current local changes
are recorded in [tokenizer provenance](../third-party/tokenizers-upstream.json);
parser-fork provenance is retained separately in
[the parser manifest](../third-party/parser-upstream.json).

The four released text/image runs take 24.35–31.25 seconds, including loading and
planning. Peak RSS is 3,127,853,056–3,186,294,784 bytes; peak process footprint is
5,613,948,216–5,679,861,048 bytes. These serial debug runs are functional
validation, not throughput benchmarks. RSS and footprint include process and
allocator activity outside admitted framework owners; the configured 64 GiB
capacity is a ceiling, not measured usage. Exact per-run counters are in
[released tool validation](bounded-followup-released-tools.md).

Native quotes and producers account for the same actual construction, numerical,
copy, validation and publication populations. Sequential temporary parser frames
reuse a paid live peak; persistent heap growth and copy/attempt spending remain
cumulative. Shared immutable sources and native backing receipts avoid duplicate
charges without refunding work or dropping escaped ownership.

## Acceptance status and qualifications

The selected-executor artifact passes **all seven public prepared-chat cases**
and **all six distributed CLI residency cases**. Mova Resident, Host and Disk
(TP2/PP2/EP2, eight CPU ranks) complete in 65.01, 64.26 and 63.49 seconds process
wall time. The three Dense modes pass together in 41.03 seconds. The CLI retains
its 120-second per-case deadline and 64 GiB ceiling; public media/tool cases
retain their 8 GiB ceiling. Both samplers, eight-token parity, snapshots,
unfinished parent/child exchanges, paid reset and cumulative capture refusal
remain exercised.

Resident partition execution borrows its actual policy owner. The selected
executor explicitly declares whether it submits graph groups or traverses them
manually, so the quote preserves the resident registry without inventing group
completions. CPU indexed execution includes its real four-operand completion
frontier. Native readiness retires authority only after the enclosing role is
terminal, with one bounded deadline and retained custody through timeout.

Seven focused backend cases pass on the optimized artifact: four communication
regressions, manual-executor completion accounting and two CPU/Metal source
population cases. Two native readiness fixtures also pass: a completed inner
event cannot retire a genuinely live child scope, and a 1 ms deadline preserves
the original source and producer account until terminal retirement. They verify
nonzero output through the matching retained observer without resubmission.
These two use the development backend test target at optimization level 0 with
the selected neutral dependencies at level 2; production code is identical.
The earlier source3 artifact's **66 passing
backend cases**, the runtime-completion artifact's **six paged-control cases**,
and **four released text/image requests** remain separate scoped evidence.
Counts overlap and are not a combined whole-tree total.

The public and CLI builds optimize six Rust packages at level 2 while retaining
development debug assertions, overflow checks and the same native development
profile and guards. These are functional runs, not throughput benchmarks. The
unoptimized deadline failures and earlier source refusals remain in the
[CLI record](validation/bounded-followup-native-cli-control-2026-09-18.json) and
[native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json).

The following qualification limits remain:


- Native results cover the recorded Apple CPU/Metal and Ring configurations.
  Other accelerator/platform runs are not inferred from them. Fresh non-Unix
  file-opening qualification remains unverified; recorded Unix behavior is
  preserved. CPU sources retain exact rank, dtype and physical-span guards;
  the rank-11/12 Broadcast case is a tested typed refusal, not supported execution.
- Native debug tests use a **64 MiB test-thread stack**. After two test-only
  helper splits, the default-stack trial passes the earlier capture phases but
  overflows at later recorded child-sampling startup. External frame capture
  identifies a finite unoptimized call chain; default-stack success is not claimed.
- Tokenizer/controller admission is profile-specific. Supported built-in
  profiles do not qualify arbitrary custom components, decoder compositions or
  opaque callbacks. Unknown required bounds remain typed refusals before the
  operation. Optional dependency logging/meta-validation outside the facade's
  selected features has separate limits in the [grammar guide](bounded-followup-grammar.md).
- No downstream checkout was available. Its actual call site, selected
  checkpoint/configuration, memory checks and allocator-cache changes remain
  unverified. The runnable public example and
  [migration handoff](prepared-chat.md) show ordinary admitted tools and
  authenticated media; applications must still account for their own buffers,
  copied events, results and cache policy through loading, prefill, decode, reset
  and completion-gated retirement.

## Reproduction

Use the actual qualified compiler executable and disable incremental compilation
for allocation-qualified checks on the recorded host:

```sh
RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
CARGO_INCREMENTAL=0 cargo test --offline -p eredu --no-default-features --test portable_facade
```

Unrecognized compiler/allocation models remain source refusals; this environment
setting does not relax admission. Native SDK/compiler/executable hashes and
commands are retained in the evidence records. Historical failures remain there
rather than being presented as current design or erased by a later passing run.

The optimized native matrix additionally uses these Cargo profile overrides,
with `RUST_MIN_STACK=67108864` for test processes:

```sh
--config 'profile.dev.package."eredu-backend-mlx".opt-level=2' \
--config 'profile.dev.package."eredu".opt-level=2' \
--config 'profile.dev.package."eredu-cli".opt-level=2' \
--config 'profile.dev.package."eredu-runtime".opt-level=2' \
--config 'profile.dev.package."eredu-architectures".opt-level=2' \
--config 'profile.dev.package."eredu-nn".opt-level=2'
```

The complete build commands and executable hashes are recorded in the native
validation records. These overrides do not change native guards, request
capacity or the distributed fixture deadline.
