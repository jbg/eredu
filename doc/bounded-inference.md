# Bounded inference implementation record

<a id="current-integration-status"></a>

## Current integration status (2026-09-18)

Shared selective readout chooses required hidden positions before vocabulary
projection. Ordinary, managed and controlled generation use the same bounded,
completion-gated, cancellable prefill driver; intermediate chunks normally update
state without producing scores, while sequence consumers retain their full demand.
Native token-only prefill defaults to at most 512 positions per chunk, subject to
explicit configuration and admitted capacity. The applicable implementation and
planned validation are complete at the recorded scopes below. Hardware gaps,
preexisting API boundaries and request-specific unknown-source refusals remain
explicit; no overall completion percentage is asserted.

Released Qwen validation covers 5, 128, 512, 1,500 and 3,000 prompt tokens with
ordinary/managed/controlled token parity. Independent full-score reference checks
cover prefill and three cached decodes at 5 and 3,000 positions. At 3,000 tokens,
the allocator-corrected managed 128-position run records
4,811,351,232 bytes of MLX peak allocation, compared with 22,433,750,408 bytes for
the ordinary full chunk and 4,813,825,208 bytes for equivalent ordinary chunking.
The managed measurement belongs to artifact SHA-256
`d0dcefcf2ff567ac0e78c9c39014382e7345bdb1596149f21ac252564dd4e5b2`,
recorded under `managed_3000_after_allocator_corrections`. The later
`graph_construction_class_index` timing run has its own artifact and measurements;
its timing is not combined with this peak. These are allocation measurements,
not process-memory guarantees. Revisions, tolerances, process observations and
chunk-selection/refusal results are in the
[released-checkpoint record](validation/bounded-public-qwen-2026-09-16.json).

Completed selected public coverage includes:

- Managed realtime Resident, Host and Disk four-frame ordinary parity, plus
  pure TP against the serial reference on both real Ring ranks. The complete
  Host and TP checks pass, including actual model-collective count parity;
  scheduler consensus is counted separately. Disk passes actual transfer-byte
  and count deltas, joined background completion, zero failures and queue bounds.
  Resident refusal before branching, discard and release/resume, and paid
  full-pager preparation/rollback remain covered at their recorded scopes.
- Independent expert-cache Resident/Host/Disk ordinary, managed and controlled
  execution, all three saved-state paths, and speculative parity at their recorded
  scopes. The transformed public path passes in milestone 228 after preserving
  the actual affine projection/router source and quantized-store publication;
  the focused ordinary-source publication check also passes.
- Resident, Host and Disk plain/media generation, all 18 recorded media
  mode/residency combinations, and controlled pending/postcommit saved-state
  paths. Resident/Host/Disk paged generation and saved lifecycles pass, including
  actual Host eviction/reload, Disk I/O and final owner/file retirement.
- Embedded Qwen, Nemotron, Inkling, V3, V4 and DSpark parity and complete selected
  snapshot/restore/fork/exchange replay. V4 and DSpark retire all 180 accounts per
  fixture under a finite 12 GiB limit; constrained replay preserves failure causes.
  Independent and Embedded speculative batches and selected background loading pass.
- External assistants on shared/split streams and both CPU/GPU placement
  directions, including saved-state replay and cumulative limits. Selected CPU
  half targets, CPU TP/combined and CPU TP saved-state execution pass.
- TP, PP and combined generation, saved-state replay and all six Host/Disk
  parallel comparisons. Composite Gemma passes TP/PP/combined; raw Gemma image/audio
  passes backend/facade generation and saved restore/fork at the default stack.
- Released chat-template parity, explicit/default behavioral profiles, and the
  shared Forbidden, Active and Auto controllers. Public Auto tool completion now
  passes ordinary/managed/controlled comparison with a real sampled tool token,
  registered full-schema callback, once-only events and retained output custody.
  Its terminal first commit correctly submits no speculative verification round;
  Active/Forbidden and component split-token cases cover the other transitions.
- Full-schema borrowed input/source checks for scalar/numeric/literal and nested
  object rules, URI/references, selected literal patterns, propertyNames,
  dependencies and small/large uniqueItems. Modern-draft string annotations and
  contains/minContains/maxContains also pass the affected source check. These
  checks preserve ordinary semantics, source accounting and the first funding
  failure with escaped input custody.
- Public Histogram TP/PP/combined, complete vocabulary delivery, selected CPU
  score/intervention capture and TP/PP/combined unit-output capture. Routed
  ordinary/managed/controlled generation now passes, along with committed
  snapshot/restore/fork spending and pending uneven-prefill saved-state checks.
  Projected TP, PP and combined prefill comparisons also pass. Separate projected
  Host ownership/association/refusal and paid contiguous/additive assembly,
  source votes and exact native transfer components retain their scoped evidence.

The latest completed public milestones close sparse routed TP, PP and combined
capture, GPT-OSS MXFP4 combined execution, and plain TP/EP and PP/EP parity.
Shared-write PP, additive combined capture and paged-media capture also pass. Exact source-program
admission, independent peak accounting, foreign-source refusal and account-fence
checks pass. Exact commands, immutable executable hashes, reached failures and
dependency gates are retained in the
[native milestone record](validation/bounded-media-and-submission-2026-09-16.json).

All six final feature configurations resolve to passing results. Portable
behavior retains 124 original backend-conformance passes, adds the corrected
exact rejection case, then passes the previously unrun facade suites: 30 portable
cases plus one realtime case. This totals 156 passing cases and one preexisting
checkpoint-dependent ignored case across those runs. The raw initial failed
command and each corrective follow-up are retained; this is not a claim that all
original commands passed. The no-MLX weak-native-flags cfg correction, CPU-only
optional-library/null-source fixes and their initial failures remain recorded.
CPU-only passes in 30.56176 s; default facade plus image/audio passes in 60.16949 s.
The final backend Disk build covers the later non-feature-specific itinerary fix;
unchanged passing feature configurations were not rerun.

The selected 24-case native readout fixture verifies cold/actual output dtypes,
F32/F16/BF16, dense/affine, tied/untied, strided/contiguous numerics and StateOnly
omission. Shared lifecycle and bounded scope reviews are complete. CUDA/NCCL
need a compatible builder; weak flags without MLX do not validate those native
realizations. Unknown workspace rejects before execution; strict admission
remains required.

Milestone 224 closes all six TP/PP/combined Preview and Summary evidence cases
and both focused neutral source/completion cases. Milestone 225 closes affine
projection dtype/geometry and the expanded 24-case native comparison. Milestone
226 quotes the affine router and exposes the later transformed source-inventory
gap. Milestone 227 passes the focused affine-router source, final-submission
bank and typed policy-cause checks; Host execution reaches its final telemetry
assertion and managed TP reaches its final collective-counter assertion. The three
new neutral distributed preparation/completion tests pass after integration of
source-funded diagnostics and the error-retention hook. Milestone 228 closes the
transformed expert-cache public comparison (1.608 s) and exact ordinary-source
publication check (1.544 s). Milestone 229 closes Host (0.59447 s) and pure TP on
both real ranks (0.72220 s), including actual model-collective count parity, using
artifact SHA-256
`48cea70f9d33f4585701331026d04dbf537630e858dceec62ae7f29ccc1ff9c8`.
Milestone 231 closes the focused Disk startup-source checks; 234 reaches all four
frames with numerical parity after the exact itinerary correction, then exposes
a stale source-read assertion. Milestone 235 closes the full Disk check
(0.44317 s) using actual transfer and joined-background witnesses. Its runtime
SHA-256 is `ad5ca3b5041d368e5ac2c24a66224e3375c57fff33cd3e6bfe1df1cc563775ca`;
production is unchanged from 234. Earlier selected results remain closed. Only
affected checks were rerun; no overall percentage is asserted.

The [lifecycle reconciliation](validation/bounded-lifecycle-reconciliation-2026-09-17.json)
closes the shared-mechanism review using current source and recorded behavioral
results. It covers cancellation at settled chunk boundaries, pending/failed
completion custody, atomic concurrent admission, and pending/committed saved
state with cumulative copy and observation spending. Restore shares its original
ledger; an independently admitted fork starts from saved usage. Physical owner
retirement does not refund cumulative attempts or copies. This was a source and
evidence review, with no new runs or concurrent released-checkpoint measurement.
The final feature and native checks have their own records above.

The native completion-lifetime audit reused the existing rollback and custody
evidence and found one concrete error-preservation bug: addressable/EP callback
failure could be replaced by a later successful-scope check. The integrated fix
retains the callback cause and the existing child-root recovery owner. No new
broad test run or realtime support claim follows from that audit.

Combined TP/PP/EP generation and its saved-state replay now pass the public
ordinary/managed/controlled checks (6.89 s and 9.216 s). Routed EP and TP/EP
capture also pass (3.18 s and 4.871 s). These close distinct required paths at
their recorded scopes, alongside the later realtime and feature results above.

Routed PP/EP and TP/PP/EP capture now pass (7.543 s and 10.436 s).
Focused compact and sequential addressable-source checks and initialized integer
publication also pass. Realtime account retention and distributed intervention
outcome receipt validation pass their neutral checks. That earlier milestone
covered the neutral mechanisms; later native realtime and distributed
intervention results are recorded separately above.

Paged-media capture saved state is now closed at its selected public scope:
both pending and committed restore/fork pass after counting source-validated
canonical tail entries, including valid zero-byte markers. The focused check
also rejects mismatched end coordinates and foreign tails. Expert saved-state
replay now keeps the request's 32 GiB shared domain ceiling while separately
limiting each copy to 8 GiB; constrained refusal checks remain intact.

Selected public Forbidden, Active and Auto/tool execution, including the default
annotation/contains follow-up, is closed at its recorded scope. Concrete schemas
using unqualified general regex/pattern-key execution, fractional multipleOf,
unevaluated evaluator graphs, contentSchema backing or legacy asserted
format/content helpers retain typed unknown-bound refusals before that work.
Custom callback bounds are not inferred. These are request-specific limitations,
not whole missing controller or model families; see the detailed
[controller and schema scope](execution-control.md#managed-prepared-chat-controllers-and-tool-completion).

The [native integration evidence](validation/bounded-media-and-submission-2026-09-16.json)
records successful checks and reached failures separately. Historical notes below
preserve intermediate evidence; their old pending/support statements do not
override this current summary.

## Historical integration notes (superseded status)

The following notes retain the sequence of component evidence. Their intermediate
blockers and support statements are superseded by the current status above and
the latest entries in the validation records.

Selected implicit/separable Metal convolution now passes six nonzero stride,
dilation, group, F16/BF16 and singleton-compaction cases using source-derived
construction bounds. Borrowed chat defaults/caller context, nested map/array lookup
and conditionals pass ordinary parity, failure retirement and public managed/manual
execution. Remaining template operations and controller profiles are tracked separately.

Embedded prediction has shared request/occurrence accounting, source-retained
fallible copies and an actual prepared native cache factory. Target quoting retains
hidden capture separately from vocabulary demand. Current native state projection
preserves declared geometry, frontiers and aliases; the prediction workspace
materializer now uses those actual source states. The materializer, completed-source
native leaf-copy and captured-prefill cancellation/rollback checks pass. A shared
prediction operation driver now calls the existing equations for seed, sequential,
fused and replay phases; sequential and fused production scheduler checks pass.

The native parameter binder, exact resident/streamed operation banks and native
scope/completion mechanism compile together. Full prediction equation quoting
now passes prefill, proposal and replay using the same operation driver and exact
retained target inspection/admission. V3 prediction construction reuses retained
declaration/task metadata; the shared inspection path no longer clones complete
artifact inspection data per quote. Typed error propagation is integrated.

The invocation-local context, shared scheduler provenance and per-branch
completion evidence compile together. Retained Qwen, Inkling and Nemotron
constructors now pass their exact state/companion/refusal checks; changed
Nemotron nonzero serial/parallel behavior also passes. The signed native token
source and exact physical module-call recorder pass focused checks.

Target source quoting, supplementary checkpoint-source inventory, prepaid
nested root-list submission, completed-logits transport and explicit completion
points now compile together. Exact variable completion-root populations pass
Graph/Record accounting checks. V4 and DSpark retained constructors pass their
geometry, companion and custody checks; owned outer-observer compatibility and
immutable fused-output source retention also pass.

The fresh-target public Embedded entry, heterogeneous module bank, both native
phase adapters, qualified per-chunk token views and proposal/verification/replay
token packets now compile together. Eager token upload and signed/unsigned
registered ranges pass native checks for numerical values and escaped custody.
Shared activation delivery preserves its frame through buffer retirement. Exact
observer invocation and escaped execution/overlay/session identity checks pass.

Captured-seed immutable packets, two-input concatenation and the exact loaded
activation factory now compile together; predecessor custody, source delivery
and exact catalog/scope/hook validation cases pass. The first public Qwen
Embedded run exposed incorrect layerwise workspace preparation for a resident
target. The selected-residency correction, populated-target startup and funded native
internal observation consumer now compile. Cold capture uses actual cumulative
source usage; stale usage refuses admission without spending a role. A subsequent
Qwen constructor failure exposed loss of the prepared-source loan during initial
embedded target construction. The shared handoff correction, generated capture
retention and controlled snapshot/continuation forwarding now compile together.
Qwen target/recurrent constructors and the generic fixed/compressed parameter
description use funded metadata construction; the public Qwen run has advanced
past constructor qualification to an original prefill-control identity mismatch.
New physical window geometry and Model-role intervention claim checks pass.
Retained routed materialization/parameter descriptions, native cross-window
capture and internal edits are staged or in progress. Six-family public parity
and controlled snapshot/resume validation remain pending. Scoped evidence is recorded in
doc/validation/bounded-media-and-submission-2026-09-16.json under
embedded_token_views_and_observation_workspace_passed and
embedded_prefill_view_callers_and_source_identity_passed. Shared bounded chat
length/count, default/d and plain scalar-array/Unicode join pass their focused
parity and retirement checks. Remaining template/controller profiles stay separate.

Native ungrouped block-FP8 projection uses finite prepared kernels and
source-derived host/native construction bounds. Whole and observed projection
fixtures pass with floating/E8M0 scales, including effective-input capture and
typed callback-failure retirement, alongside rank-12 strided rotary. Grouped
contiguous FP8 now passes four numerical cases across floating/E8M0 scales and
both selected grouped kernels. Its finite source families are split by actual
scale dtype within the existing native initialization limit. Packed grouped FP8
also passes independent 137-row components and a 65-token chunk tail with F32 and
E8M0 scales; this exercises source-derived scale-row origins and final retirement.
Grouped observation Units/Finish now passes dense and chunked FP8 intervention,
including typed shape refusal and complete source/error retirement.

Independent speculative execution now uses the shared PrefillDriver with a real
original whole-prompt copy, static span views and per-span native completion.
Actual multi-chunk prefill followed by categorical sampling passes, including
nonzero output, key advancement, refusal and final retirement. Seed, sequential
split, positional-key and uniform-draw execution also pass. Completed state
carries exact allocation, backing, native budget and request-account witnesses
into the existing isolated-copy machinery. A different request sharing the same
schedule/header is rejected before copying or charging; valid restore passes.

Independent RNG snapshots pass selected-backing and repeated-copy checks, draw
parity, exact request identity and final retirement. Native adaptive Mirostat
passes three actual sampling commits with penalties, mu/history snapshots, key
advancement and ordinary-worker comparison. Five focused neutral cases cover
adaptive history/failure/forced-choice behavior, trace-limit parity and prompt
materialization error retirement. Semantic forks and escaped text events preserve
their actual account through destruction.

Source-explicit public managed and controlled plain-speculation APIs now both
pass against a fresh-process ordinary generation baseline. The selected tiny
resident F32 independent pair processes a five-token prompt in 2/2/1 chunks and
produces four identical token IDs and visible text. Controlled steps expose the
same committed tokens; returned output remains valid after model/source teardown.
The selected resident single-request stochastic continuation now also passes:
ordinary/managed parity, controlled snapshot/restore/fork, RNG and text replay,
invalid/pending/clear/re-forced token choices, independent branch isolation, and
cumulative copy/trace limit refusal. Continuation extends paid future occurrence
storage without recycling spent roles; categorical keys retain their authenticated
sampling stream. Public ready-host independent speculation now also passes across
ordinary, managed and controlled execution: both three-layer target and drafter
use retained host sources, a five-token prompt in 2/2/1 chunks and four stochastic
output tokens. The numerical quote and source-loading populations remain separate;
the drafter now uses the same source preparation as the target. Foreground-disk
speculation now passes the same public parity case with real source loading for
both models. Immutable source publication retains its accepted source ticket
without entering funded-scope registration bookkeeping; success, failed attachment,
canonical aliases and poisoned quarantine preserve the exact charge. Broader family/residency combinations,
batching, split-device transport and prediction assistants remain unfinished.

Live raw capture now passes through the public managed plain-text entry point:
a five-token prompt in 2/2/1 chunks yields the full prompt logits and three cached
decode frames, which remain readable after model/source teardown. The native
prefill role comes from its actual admitted graph owner. Cold completion-frontier
and accepted-transfer accounting also pass. Funded capture checkpoint,
restoration and saved absolute-origin quote cases pass, preserving post-snapshot
cumulative spending and independent fork budgets. Native snapshots retain the
drained capture checkpoint and selected source. Fresh observed resume now installs
a new capture bank through the saved-copy transaction. Public pending and postcommit restore/fork now pass with shorter remaining output
allowances. The shared saved readout demand, observer continuation forwarding and
monotonic cursor limit preserve absolute capture coordinates, post-snapshot spending,
independent fork budgets and payloads after source teardown. Core raw capture matches
ordinary full-prompt and controlled execution numerically through cached decode. Token-score capture now shares its
numerical program with ordinary execution and uses terminal-only readout. Public
TokenScores and TopCandidates now match ordinary full-logit reference rows through
chunked prefill and cached decode, both alone and mixed with full-sequence capture.
Private scalar temporaries retire after exact completed-root validation; cleanup
requires ownership identity without asking for another future submission slot.
These public cases pass at vocabulary 64 and 4,097; the larger case crosses
the native sort tile and multiple normalization chunks. Original speculative raw
capture also matches ordinary/managed/controlled execution, and its saved
restore/fork case preserves spent allowance and escaped shared payload custody.
The source-copy and aborted-evidence checks retain exact charges through final
alias retirement. Speculative TokenScores/TopCandidates now pass their combined
ordinary/managed/controlled full-logit oracle, including exact unconstrained
candidate domains and ordered score IDs. Summary capture passes the public
vocabulary-4,097 prompt/decode case and native zero/nonfinite/mixed-chunk retirement
cases. Speculative Summary also matches the ordinary full-row oracle through
managed and controlled execution. Histogram now passes the same public
vocabulary-4,097 prompt/decode comparison and ordinary/managed/controlled
speculative full-row oracle. Shared edge/nonfinite/integer/Bool/empty Histogram
branches pass. Static logits interventions now pass public ordinary versus
source-funded execution, with continuous draining and controlled restore/fork.
The six applicable logits actions compose in admitted order; bounded Preview and
Summary evidence match independent host expectations at each operation boundary,
including strided regions, truncation and nonfinite outputs. Ordinary captures
retain pre-edit logits. Duplicate-role rejection preserves installed edits,
clearing restores ordinary sampling, and restoration recovers saved source aliases.

The shared selective state-space scan now has native memory, construction and
completion bounds derived from the actual recurrence. Prepared-Metal execution
passes ordinary MLX and independent host-reference comparison for uneven 5/2
prefill, BF16 source with nonzero state, and one-token cached execution. Both
state and output retain their admitted ownership through final retirement. This
closes the reusable primitive; complete family/request qualification is separate.
The shared pooled-mask worker also passes prepared-Metal checks against an integer
reference, including partial windows and coordinates beyond F32 exactness. Query-specific
mask gathering now also passes Bool/F32/F16/BF16, signed-index and strided-source
checks after compiling the existing native GatherAxis variants. Tensor-bound clip
passes scalar/row/column broadcasts, reduced precision and reversed bounds against
the ordinary worker. Pooled-position selection now passes masked, strided BF16
and 2,051-position multiblock-tail checks against native and host references,
including actual pooled-mask gathering. Indexed attention now passes its original
prepared-native four-contraction fixture, including masks, sink, strided BF16,
differing key/value widths and the empty-local branch. All seven refactored static
activation actions pass strided ordinary-native and exact cold-population checks;
borrowed half payloads preserve bits. Prepared F16/BF16 execution now also passes
all seven actions with exact storage bits, strided regions and scalar rounding.
The shared numerical intervention callback passes source-custody and failed-spending
checks, and the allocation-free neutral slice resolver passes strided and overflow
checks. Actual loaded declaration/session authentication, atomic installation,
source-owned evidence and shared budget preflight are integrated. The public
fixtures above exercise the resulting original numerical execution path.
Hyper collapse/expand/head now also pass a complete original native cycle against
former einsum and independent host references, using F32, strided BF16 and F16.
Cold inspection preserves constant-time counting for large Sinkhorn iteration
counts. These are shared primitive results; complete consuming-family qualification
remains a separate item.

Prepared-media execution passes all 18 selected combinations: conditional and
nonconditional fixtures, ordinary/managed/controlled routes, and resident,
host-layerwise/foreground-disk execution. Prefill and cached decode agree.
Pending restore/fork and postcommit restore/fork pass their complete selected
native cases, including resumed execution and final retirement. The shared
layerwise constructor accounting now includes actual logical parameter slots;
composite source handoff and saved-media source registration/publication are
integrated and exercised. These fixtures do not establish arbitrary format,
state, background-loading or distributed coverage.

See validation/bounded-media-and-submission-2026-09-16.json, especially
media_source_controller_joined_milestone, for exact artifacts and scope.
It supersedes the earlier host seal, pending identity and resumed publication
failures. The former 65% estimate is withdrawn; integrated code and demonstrated
execution are recorded separately.

Public released Qwen ordinary, managed and controlled generation now match at
5, 128, 512, 1,500 and 3,000 prompt positions, each producing four greedy tokens.
The managed/controlled measurements use the normal production planner, a 64 GiB
managed-domain ceiling and requested 128-position chunks. Each measurement ran in
a fresh process, serially, without concurrent compilation or native tests.

| Prompt positions | Ordinary MLX peak, chunk 128 (GB) | Managed MLX peak (GB) | Controlled MLX peak (GB) | Admitted incremental maximum (GB) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 3.769 | 3.776 | 3.776 | 9.701 |
| 128 | 4.544 | 4.538 | 4.552 | 14.019 |
| 512 | 4.634 | 4.628 | 4.627 | 16.998 |
| 1,500 | 4.700 | 4.694 | 4.694 | 23.502 |
| 3,000 | 4.814 | 4.811 | 4.808 | 34.592 |

GB here means decimal billions of bytes. The five-position request selects a
five-position chunk. MLX peaks include retained model allocations; the admitted
incremental maximum is a conservative managed-domain requirement, not the
observed MLX peak or a total process/system-memory guarantee. Independent process
high-water marks and exact byte counts are retained in
`validation/bounded-public-qwen-2026-09-16.json`.

At 3,000 positions, the ordinary full-chunk run peaks at 22.434 GB of MLX
allocation. Managed chunked execution peaks at 4.811 GB, a 78.55% reduction,
and ordinary execution with the same chunk policy has essentially the same peak.
All produce `[561,6511,314,9338]`. Those earlier debug measurements took
65.80 and 65.32 seconds for managed and controlled generation. Profiling identified
a linear scan of native graph allocation slots. Indexing the ten existing size
classes reduced the same managed 3,000-position run to 4.705 seconds, about 14 times
faster, with the same output IDs and chunk/budget settings. Its MLX peak is
4.821 GB and admitted incremental requirement is 34.669 GB. Whole-process peak
RSS is 3.298 GB and physical footprint is 6.336 GB. The earlier ordinary chunked
run took 2.607 seconds; controlled execution has not been remeasured. This change
preserves allocation order and budget authority and does not establish a tight
practical admission ceiling.

The independent MLX-LM full-score comparison covers prefill and three cached
decodes. At five positions the maximum absolute deviation is 0.15865; at 3,000 it
is 0.275862. Both pass the combined `atol=0.25`, `rtol=0.02` criterion and match all
four argmax IDs. This numerical evidence is distinct from public token parity.
The checkpoint revision, source hashes, commands, versions and hardware are
recorded with the validation evidence. Earlier failed runs and superseded bounds
remain historical records. A subsequent 128-position request under a 13 GiB
managed-domain ceiling automatically selects a 102-position chunk, admits an
11,730,745,220-byte incremental maximum and produces the same four tokens. An
8 GiB five-position request rejects during preparation with exact required and
available byte counts; it produces no output and leaves MLX active allocations
unchanged from the loaded baseline. Shared-pool concurrent admission has existing
neutral/native fixture evidence; no concurrent released-Qwen measurement is
claimed. The wider applicable family/path matrix remains open.

Native admission now retains complete opening/closing state, all potentially
escaped score backings and validation producers while reusing equation workspace
only after exact successful native Record retirement. Unknown carryover keeps the
previous complete generation sum. Both focused envelope checks and the independent
192-case storage census comparison pass. Exact mixed F32/BF16 source layout now
removes only a proven redundant weight compaction; its source/fallback regression
passes. Immutable kernel-source qualification is memoized without changing actual
initialization or allocation checks. Storage census lookup and borrowed exclusion
now use one prepaid identity index, built with binary-run merging through idle
DFS frames; its 11 focused report regressions and independent review pass.
The released managed and controlled 3,000-token results above include the subsequent allocator corrections: next-fit Graph allocation and capacity-preserving synchronization of concurrent physical allocations.
Native exact/approximate GELU, affine/MXFP4 supplied-stream projection and
Llama3/proportional rotary now pass their focused nonzero operation witnesses
under calculated native reservations. The rotary witness creates frequency
ancestors inside the admitted graph. These checks close those operation receipts;
they do not establish every surrounding family/state/request path.
See `validation/bounded-completed-equation-2026-09-16.json`.

Composite media preparation shares budgeted graph, state and identity
construction and immutable configuration ownership. Its native interval/source
case passes; the current admitted execution matrix, capture, continuation and
distributed qualification remain open. Independent speculation authenticates its
shared finite occurrence contract to exact target/draft source epochs and retains
account custody through native completion. Occurrence installation and copies
are connected; remaining sampling, scheduler containers and public request
composition are separate unfinished work.

End-to-end public managed-memory inference remains unfinished. The internal
admission paths below have validated components and selected execution fixtures;
they do not establish a complete bound from tokenizer construction through final
output retirement. Public managed entry points remain closed where required
native, host or facade storage lacks an original construction and lifetime bound.
Unknown required workspace must reject before the affected operation. Even a
complete Eredu-managed bound will not be a process/system-memory guarantee.

The public plain-text API joins the existing tokenizer/plain-text driver. The
first public native resident fixture now passes managed generation and controlled
advancement with automatic Graph/Record capacities and the normal top-k/top-p/min-p
filters. Both produce IDs `[8,38,26,1]` and text `iMAb`, matching the recorded
ordinary baseline, after 2/2/1 prefill and three cached decode steps producing
four output tokens. The fixture also
checks pre-start cancellation, a one-byte refusal and retained output after model
retirement. This establishes that selected tiny F32 dense public path; it does not
establish complete family/path coverage. The mixed recurrent/full-attention
hybrid also passes ordinary, managed and controlled execution with IDs
`[63,32,32,32]` and text `?GGG`. Its earlier 64-byte RMS copy exhausted the
underestimated allowance because original Metal rounds each positive allocation
to a page. The shared physical bound now uses that actual rule for every equation
and sampling population; the original debit guard is unchanged. Finite pointwise, row-kernel and recurrent-scan families are integrated;
the recurrent/grouped milestone passes four focused cases, including independent
scan numerics and long-sequence workspace bounds. Complete Graph arena sizing for the first
single-GPU resident profile is integrated and passes four focused Rust cases plus
one native fragmented-allocation case (124 assertions). Admitted GPU stream/encoder
construction is integrated; both focused native
checks pass, covering actual GPU/CPU execution, retained source custody and strict
refusal of ordinary stream wrappers. The selected resident Record arena includes every attempted
constructor extent and the allocator's required split tail, with selected capacity
kept distinct from the caller's ceiling. Neutral custody and native fragmented
retirement/failed-prefix checks pass (255 native assertions).
BF16 projection source ownership and numerics now pass their focused checks.
Router mixed CPU/GPU synchronization is integrated with native numerical
verification pending. Direct batched attention ID construction and the original
physical population query pass their native ownership checks. Public managed chat
now shares source-budgeted rendering and the ordinary/controlled cursor; its
focused portable parity, metadata-authentication and error-lifetime check passes.
Default filters, penalty recipes and direct explicit-attention recipes pass their
cold checks after genuine source initialization, and the direct numerical
softcap/mask/sink case passes. Positive-temperature Standard sampling now passes
public ordinary/managed/controlled parity at seed 827, producing IDs
`[60,41,15,12]` (`8Ppm`). The same completion retains both token and advanced key.
Public admitted resident KV and mixed recurrent/full-attention hybrid reset both
pass a second request while the first output remains alive, including an
incremental one-byte refusal before reset. Hybrid reset uses the same neutral
table-publication driver for the outer and fixed-role child tables.
Native chat passes ordinary/managed/controlled parity for the existing exact
released SmolLM compiler image, producing `[38,32,26,56]` (`MGA `) after 8-position
chunks. This native record covers the exact template image. Source-based
compilation is now integrated for text, message access, conditions and nonnested
message loops. Four focused portable checks pass, including ordinary-render
parity, selected metadata and source-account failure/retirement. Public native
source-compiled chat now also passes ordinary/managed/controlled parity, yielding
`[45,26,56,63]` (`TA 1`) after two 8-position chunks. Argument-free string `trim` now shares ordinary filter dispatch and the
bounded borrowed-slice worker; both focused Unicode/concatenation parity and
ordinary registration/argument/error checks pass. Other filters, calls, macros
and nested loops remain unfinished. Query/sliding attention uses fixed Slice views and prepaid output
banks; its new cold recipe and numerical query-tile/scalar-sliding checks pass.
True key-block completion and Mirostat now share an integrated bounded nested
completion mechanism. Mirostat now passes public ordinary/managed/controlled
parity at seed 827, yielding `[49,41,15,14]` (`XPpo`). The nested completion uses the
actual resident-bank tag and authenticates completed logits/key before the next
execution scope. Original key-block attention now passes independent full-score
reference comparison for 8,193 keys, a nonuniform mask, softcap and learned sink
(atol `2e-5`). All 66 nested completion attempts and final output custody retire
under automatically derived capacities. The publication-lifetime and router
fixtures have corrections integrated for their distinct construction and immutable
worker-loan boundaries. Publication lifetime and the mixed CPU/GPU router now
pass, including exact request-owner retirement after output release. The router
check uses its actual custody owner because unrelated pool storage can also
retire during the check. These results do not establish complete host/disk or
broader router coverage.

Ready-host loading now has the cold architecture destination projection, finite
residency-controller constructor, detached exact checkpoint reader, asynchronous
nested neural completion, native copy/aggregate completion and move-only provider
to policy handoff integrated. The detached-reader lifetime/failure case and shared
window boundary/unknown/overflow cases pass. Planning candidates reuse one admitted
host snapshot instead of rebuilding layout, names, shapes and materialization
estimates. The repeated layer-binding path now uses finite rows and direct parameter field
traversal with shared atomic validation. Its manager producer, immutable host/static
parameter births, canonical initial-tier aliases and faithful detached policy catalogs
are integrated under the same source/request accounts. The native C++/Rust source
producer builds, and four focused finite-binding/closure cases pass. The combined
backend/runtime and public builds pass. The selected public dense host-layerwise
fixture now passes ordinary, managed and controlled generation with IDs
`[8,38,26,1]` and text `iMAb`, matching the resident baseline. Two nonzero decoder
units execute through a one-unit device window, with 2/2/1 prefill and three
cached decodes producing four tokens. Original consumer waits submit their funded
receipt; each transfer owns its consumer observation until terminal retirement,
and completed unit teardown releases its exact application and lease-node pins.
Pre-start cancellation, one-byte refusal and output retention after model/source
retirement also pass. This establishes the selected ready-host dense path; broader
family/quantization/state combinations and disk/parallel admission remain open.

Public managed plain snapshot restore and fork now pass with both selected dense
and recurrent/full-attention hybrid resident fixtures. Each runs from pending
prefill and from the first committed token through the ordinary shared driver.
Restored and branched output matches uninterrupted generation (`[8,38,26,1]`
and `[63,32,32,32]` respectively). Cancellation and one-byte refusal preserve
saved state; copy use remains cumulative, and logical retention retires after
the existing explicit native cleanup boundary. Model drop queues native payload
retirement, so retained usage correctly persists until that cleanup occurs.
This qualifies these selected public plain resident cases, not the wider state,
chat/control, capture, media, speculative or distributed matrix.

Remaining wider work includes operation storage, transfers and publication bookkeeping;
broader reset/state/copy lifetimes; media encoder-once
ingress, request-wide speculative accounting and sequence-observer chunking; and the corresponding
family/residency/parallel cancellation and snapshot/restore/fork intersections.
Templates, constraints and retained output must share the same original admission.
Final public activation, capability reports and support documentation depend on
closing those paths. Pinned released-Qwen reference and memory measurements are
already recorded below. Historical milestones retain their stated scope; later
records supersede their individual unfinished lists.

| Path | Internal admission integration | Remaining limits |
| --- | --- | --- |
| Public managed plain text | Loaded-model source compilation authenticates actual tokenizer configuration. Public controlled start and uninterrupted generation share the original driver, borrowed events and retained output. Two focused portable cases pass without concrete backend features. Admitted native stream and finite pipeline-cache ownership checks pass. | The tiny dense public native managed/controlled fixture passes and matches ordinary output. The hybrid public managed/controlled fixture also passes after original per-birth page rounding was included. Graph/Record arenas are derived automatically from selected allocations when component ceilings are omitted. The public request installs its total ceiling before stop compilation and encoding; concurrent source preparation and early rejection pass focused checks. Startup currently compares the full immutable vocabulary/merge configuration; public managed chat has portable and exact-template native coverage; general template compilation, grammar and wider path coverage remain separate. |
| Resident token-ID prefill/decode | Shared ordinary/controlled chunk and readout drivers; direct prepared prompt upload; retained equation/sampling recipes; finite native graph/root reservations; prepared retained-storage collectors. | The tiny dense public path is admitted and passes. Original physical page rounding is integrated. Broader workers and remaining sampler policies still need closure. Unknown requirements reject. Axis storage, borrowed kernel invocation, validation capacity and sampler completion are implemented and passed the joined dense resident check. Recurrent scans also account for both outputs and pass the selected native numerical check. |
| Cached resident successor | Exact opening revision/frontier, new receipts and registered source-root credit; old physical charges remain pinned. | Arbitrary populated unquoted state cannot be promoted into a funded request. |
| Host-layerwise weights | Cold architecture destinations, finite controller/read workers, copy/aggregate/neural completion, cached source metadata/window owners and finite atomic parameter binding are integrated. Original quotes retain source custody; identity-only saved quotes do not retain host buffers. | Source constructor, immutable host/static parameter births and faithful detached policy catalogs are integrated. Public ordinary/managed/controlled parity now passes for the tiny F32 dense decoder with actual one-unit-window transfers and eviction, cancellation/refusal and retained output. Wider family/quantization/state combinations still require coverage. |
| Foreground direct disk weights | Retained exact encoded plans, priced windows/receipts, zero host prefetch and exact completion, with device decoder state. Original operation construction consumes the retained source workspace. | Detached sources, canonical host/device ownership, live source-window capacity, finite read attempts, native copy budgets and public activation are integrated. Selected tiny dense ordinary/managed/controlled generation agrees on `[8,32,26,1]` with foreground reads. Background prefetch, additional transforms and wider family/state combinations remain unfinished. |
| Funded native save/recopy | Immutable decoder/sampler/key/pending components retain protected source and destination custody. Selected public plain dense/hybrid, prepared-media, raw-capture and resident independent-speculation snapshots pass their native cases. | Additional family/state/residency intersections, grammar/controller policies and distributed copies remain unfinished. |
| Fresh native saved-source resume | Shared ordinary/controlled driver, fresh request authority and saved F/A/history/key. Resident KV, KV-only and grouped Hybrid fixed/recurrent/MLA state are validated. | Selected resident, ready host-layerwise and foreground direct-disk weights are supported with device decoder state; absent-table and paged resume remain unfinished. |
| Table-backed Pooling resume and decoder/key copy-source credit | Applied through the same source-bound driver and sealed admission; combined validation passed. | This is not minimal temporal peak accounting or whole-facade restore. |
| Paged/offloaded decoder | Functional mechanisms exist. | Complete state-manager/block/table and workspace admission is unfinished. |
| TP/PP/EP and combinations | Functional family/topology mechanisms exist. | Finite request-wide communication, materialization and auxiliary admission is unfinished. |
| Prepared media and realtime | Retained encoder sources and architecture-aligned decoder spans use the shared driver. All 18 selected native conditional/nonconditional × ordinary/managed/controlled × resident/host/disk cases pass, including cached decode; selected pending/postcommit restore/fork also passes. | Broader media formats and encoder/state/controller combinations, captured media, unfinished-source continuation, realtime, background loading and distributed admission remain unfinished. |
| Capture and interventions | Public original raw capture and saved restore/fork pass. TokenScores and TopCandidates share ordinary numerical workers and match full-logit reference rows through selected chunked prefill and cached decode, alone and mixed with sequence capture. Source custody, failure evidence and cumulative quotas have focused coverage. | Large vocabulary 4,097 and original speculative raw public/replay cases pass. Speculative score/candidate transforms, Summary/Histogram, interventions, additional state/residency intersections and parallel capture remain to validate or complete. |
| Speculation/prediction | Independent autoregressive target/drafter execution uses shared chunk scheduling and request accounting. Selected public resident/host/disk ordinary, managed and controlled runs agree; resident continuation, RNG, restore/fork, forced choices and cumulative spending pass. Original capture source, invocation and failed-frame delivery are integrated. | Raw public captured speculation and replay pass; score/candidate transforms are integrated for validation. Embedded/external prediction assistants, batching, split devices/media, background loading and distributed request-wide admission remain unfinished. |
| Whole-facade snapshot/restore/fork and mutable controls | Selected plain, prepared-media, raw-capture and resident independent-speculation cases preserve immutable saved state, native/host custody and nonrefunding copy limits. Plain capture forks have independent admitted budgets; speculative forks retain their existing shared request-wide capture ledger. | Wider family/state/residency combinations, managed chat/tool/grammar controls, additional speculative capture transforms and distributed admission remain unfinished. |

Portable materialized embedded and external-assistant tests exercise default
512-position chunking at a 513-token boundary, selective projection, explicit
uneven chunks, whole-callback compatibility, and cancellation before further work.
Those tests passed in the current portable integration. They establish shared-driver
behavior for assistants; the selected native independent-autoregressive paths are
covered separately above. Managed assistant admission remains unfinished.

The detailed [foreground disk contract](#foreground-direct-disk-admission-2026-09-13),
[host-layerwise contract](#host-layerwise-managed-text-admission-2026-09-13) and
[capture budget scope](bounded-capture.md) explain the relevant boundaries.
Native fresh-resume validation is recorded in
[`bounded-native-fresh-resume-2026-09-14.json`](validation/bounded-native-fresh-resume-2026-09-14.json).
The combined validation is recorded in `bounded-resume-capture-accounting-2026-09-14.json` below.

Admission reserves before prompt/sampler allocation, binds the original core run,
checks exact session/source/controller state and retains authority through native
completion or recovery. Surviving physical allocations keep independent charges
after request workspace retires. Cancellation, failure and snapshots cannot refund
cumulative logical allowances or declare unsettled native work complete.

Released-Qwen measurements below validate readout selection and chunking. They do
not establish released-checkpoint validation of every enforced-budget path. The
remaining work is the concrete integration listed above plus family/path numerical,
capacity and cancellation coverage; no unfinished path is an inherent family limit.

The current implementation milestone integrates finite storage collectors and
shared completion-root ownership with the actual decoder operation recipes.
The first actual resident request now completes through the original admission
path: a five-token prompt in 2/2/1 chunks followed by four generated tokens matches
the ordinary driver's tokens and nonzero KV state. The fixture verifies retirement
through the final tokenizer/output owners and returns the pool to zero. Initial
empty state uses the same projected-root binding as cached state; unregistered
nonempty state still refuses. This validates the selected tiny resident path,
not the full public facade or the complete family/path memory guarantee.

Fifteen earlier prompt/source/accounting/completion checks, the fixed collector
case, custom-kernel configuration lifetime case, five native view/reduction cases
and the borrowed Metal library-builder case passed. The native cases include
numerical results, refusal before output replacement, retained backing and cache
reuse after builder destruction. Completion and sampling now prepare independent
C shells under their paid owners; completion construction also requests safe
release on early refusal. The joined resident path passed. The existing host-only sequence path also passes ordinary/controlled parity, while explicitly requested native arenas retain their strict missing-arena rejection.
The recurrent scan now counts both independent outputs, omits the unused zero-state
allocation when state is supplied, and slices chunks through fixed native axes.
The focused three-case batch passed: scan bounds across real chunk boundaries,
selected resident/host/disk prefill obligations, and nonzero Qwen hybrid GPU
parity through 2/2/1 prefill plus cached decode. The latter uses the existing
ordinary/controlled path; it is not yet an original hybrid admission result.
These results are component evidence, not public budget activation.

The next selected native batch passed six cases: ordinary grouped projection,
strided depthwise convolution against a scalar reference, retained host-source
ownership and deferred unpinning, ordinary hybrid parity, exact-budget saved
layerwise resume, and preservation of unresolved bounded-operation obligations.
The new grouped original BF16 fixture now passes deferred validation, invalid-ID
masking and numerical parity after ordinary reference leaves are settled before
entering the original domain. Flat host-source snapshots also pass alias deduplication,
dispatch-order preservation and duplicate-unit refusal. The first original hybrid numerical path passes with
tracking capacity derived from all known native constructors: tokens and full KV,
recurrent and convolution state match the ordinary driver. Its earlier 1 MiB arena
ran out while records overlapped; a single-constructor minimum alone is not a
complete lifetime fit. The new constant-padding CPU case passed 190 assertions,
including strided and empty input geometry under prepared controls, and the native
Record allocation-extent case passed 12 assertions.

Three additional selected cases pass: packed-bank chunk/population accounting,
fixed chunk-output storage with detached custody, and typed tracking-capacity
retry through the shared planner. Two portable recipe tests also pass for borrowed
selection semantics and reentrant cache inspection. Known graph/payload facts stay
separate from custom-kernel host readiness: persistent kernel definitions, generated
source and caches still require ownership, including the shared helpers used by
dense/hybrid execution. The fixed pointwise invocation now removes temporary
configuration storage and explicitly keeps direct SiLU/Sigmoid/GatedProduct host
admission closed until those persistent owners are funded. Fixed invocation
multi-output/refusal, independent pointwise rounding and the explicit missing-owner
gate pass. The router implementation compiles, but its original numerical fixture
now refuses at the enclosing model's shared-kernel gate before routing executes.
The admitted pointwise definition constructor is integrated and uses one immutable
native block, retained through the existing shared initialization account. Its two
new ownership cases pass, including one-short refusal, shared-account lifetime and
deferred outputs surviving definition retirement. The latter exposed and fixed
negative template values producing invalid native function names; template values
and equations are unchanged. Generated source/cache ownership remains open.
Earlier numerical milestones do not establish that coverage or override the stricter gate.

Finite pointwise source/cache slots are now integrated. All five focused checks
pass, including exact/short admission, deferred numerical evaluation, source
copying and final native-alias retirement. The lifetime case exposed an ordinary
stream synchronization gap; synchronization now uses the existing completion
observation and retirement boundary outside scoped/original inference. Shared RMS,
row-sum and attention-softmax helpers also use finite source families and fixed
invocations, deriving dimensions from existing input metadata and the dispatch
grid. Their three focused numerical checks pass, including independent BF16
rounding. These changes do not establish whole-request Graph or built-in kernel
cache coverage.

The public plain-text preparation ceiling now uses the existing neutral pending
account and account ledger before stop compilation or tokenization. Source work
continues to reserve its own measured storage, and all concurrent domain work sees
the supplied ceiling. Success releases this preparation account only after the
ordinary request account is established. Owning startup failures retain it through
partial-source cleanup. The focused concurrent-source case and extended public
ordinary/controlled case pass; a one-byte request never enters stop compilation
or encoding. Full public native validation remains pending the built-in cache.

## Audited execution seams

`ReplicatedTextSession` owns ordinary, routed and partitioned text transactions,
prompt-cache identity, rollback, publication and native completion. Its prefill
path requests final-position readout before calling `index_text_output` to remove
the retained length-one sequence axis. Direct `forward`, `sequence_logits`, and
prediction-target methods keep their complete sequence requirements.

`LayerwiseRuntime` shares resident and bounded unit traversal. Architectures own
`finish_forward`, parallel readout and partition output, including tied embeddings,
packed projections, normalization and logit transforms. `LayeredForwardState`
and family forward contexts also retain prediction inputs; selecting scores
must not truncate those captures. Media graph ingress and prepared decoder
coordinates must be separated before chunking: slicing a prepared media input
or executing the encoder for each chunk is invalid.

Core `Completion` distinguishes successful completion from independent evidence
that failed work no longer uses resources. A polling error does not settle work.
Core `SessionAuthority` retains move-only submission ownership; the lease can
now retain neutral resource reservations alongside native completion/recovery.

## Contracts implemented

Core `OutputDemand` distinguishes state-only execution, the final position, and
complete sequence scores. Runtime derives chunk demand: intermediate chunks of
a final-position request are state-only. Architecture `execute_readout` selects
hidden positions along an explicitly declared axis before invoking the output
equation. It returns no scores and never invokes that equation for state-only
execution. Selection retains the sequence axis, including multi-stream hidden
geometry. Every existing layered architecture implements the explicit selection
contract. Resident, bounded-unit, routed, tensor-parallel, and pipeline traversal
carry demand through the existing execution loop. Pipeline boundaries keep full
hidden geometry; only the output owner selects readout positions. Publication
placeholders use the demanded score shape. Full prediction captures remain
separate, including V4's ordered concatenation of intermediate target captures.

Ordinary prepared text prefill/decode requests final-position readout. Custom
observers retain sequence readout by default; the no-op observer opts into final
position selection. Native tensor/error adapters forward that requirement.
Parallel sessions agree readout demand before state mutation: an observer on any
rank can require complete sequence geometry across all participants. Explicit
state-only traversal skips the final readout equation and retains completion
dependencies. `prefill_input_with_readout` and its result-bearing counterpart
execute one prepared span through the same session transaction, observation,
publication, rollback, and commit methods. They force exact completion even when
no scores exist. The mandatory backend completion contract accepts optional
scores; MLX waits on retained state and active token-validation arrays. A native
mechanism without exact completion is rejected before chunk execution. The
ordinary native token-only prefill now reaches this scheduler by default.

Readout selection includes Moshi/PersonaPlex temporal text and depth decision
projections. The shared sequential traversal merges public demand, each actual
sample/forced directive, diagnostics, and explicit required observation before
the corresponding projection. Canonical realtime execution retains optional
public scores: unobserved forced decisions need none, while sampling and
diagnostics still project their actual rows. Wider low-level final-position
calls select normalized hidden rows before vocabulary projection and retain the
complete temporal hidden value and mutable state. V3/V4 embedded draft modes
have their own prediction-unit demands; this change grants no managed request
admission or native graph/storage bound.

`RuntimeStateEstimate` now separates persistent-state coverage from optional
text execution workspace. `ExecutionWorkspaceEstimate` binds bounds to batch,
cached/input/output positions, chunk size and output demand. Its mandatory
components cover activations, attention/masks, vocabulary/sampling, state
growth/copies, materialization/transfers, and retained auxiliary resources.
Each component supplies an explicit bound and assumptions or an unknown reason.
The conservative sum excludes media workspace already present in the state
report. Estimates must cover the peak across both prefill and decode, including
the entire output allowance. Shared allocations must be assigned to one term.

Configuration-only capability reports and state-only estimates now report
`PersistentStateOnly`, including text Qwen hybrid and media configurations.
Conservative media accounting cannot fill an unknown text-workspace bound.
Strict admission and application memory budgets reject unknown required bounds.
A known lower bound already exceeding the budget can reject immediately.
Safety reserves and allocator measurements never upgrade coverage. Legacy
serialized estimates lacking workspace remain insufficient for strict admission.

Cold metadata execution in `eredu-nn::workspace` now runs the existing tensor and
neural equations against a selected mechanism's allocation facts. It stores no
tensor values and accesses no native device. Dense and packed projections retain
their exact parameter and companion geometry, including independent FP8 row
origins. Rotary, grouped normalization, gated-delta and selective state-space
operations retain their semantic options and output/state geometry. Declared
metadata capabilities describe traceable operations, not native support.

The trace sums all unique new output storage and all per-operation scratch until
a completed span boundary. Shared views and cloned handles charge the same
backing allocation once; explicitly retained roots are separated from transient
storage. It makes no assumptions about native donation, contiguity, numerical
values or early reclamation. Unpriced operations make the complete quote unknown.
The runtime adapter assigns the aggregate graph to the activation component with
explicit assumptions and preserves mandatory bounds for work outside the graph.
It rejects retained physical storage exceeding the persistent-state estimate.

Inference spans now explicitly seed the complete opening state before tracing
their updates. The report retains all unique closing backing allocations,
including unchanged buffers, and adds displaced opening storage to the transient
bound until completion. A cache replacement therefore includes old/new overlap;
a narrow view keeps its full allocation capacity. Primitive-only traces and
unknown retained capacity cannot authorize inference. New-allocation diagnostic
fields keep their previous meaning; inference admission uses the state-aware
transient result.

Existing MLX state can now be projected without evaluation, polling, waiting or
native allocation. The native query reports completed allocator or certified
host-transfer backing identity and capacity; unrecognized foreign/custom storage
and unfinished graphs stay unknown. The adapter deduplicates aliases before constructing neutral metadata
roots. It does not substitute logical tensor size for physical capacity.

Resident compressed-cache projection imports all four native arrays: latent and
rotary capacity buffers plus their logical views. Restored independent views
remain separate allocations, while compact snapshots retain their actual compact
capacity. Local geometry comes from architecture selection; native capacity and
growth facts come from the selected cache. Continuation then uses the existing
portable storage equations without prefix replay. Paged manager/block inventory,
other structured state projections and production preflight remain required.

State-overlap validation passed 83 neural tests, 28 focused runtime tests,
12 architecture workspace tests, two native allocation-query tests, one native
alias-projection test and one Metal compressed-cache projection regression
(127 tests). The runtime witness now quotes 192 bytes for a cache-replacement
peak previously understated as 120 bytes and rejects a 128-byte budget. The
Metal regression starts from live, independently copied and compact snapshot
state, checks actual backing identities/capacities without prefix replay, and
compares nine continuation appends across capacity growth with exact nonzero
F32 values. Native test targets and the backend without default features also
passed their build checks. These checks add no request/process peak measurements. Commands,
source hashes and remaining gaps are recorded in
[`bounded-state-overlap-2026-09-13.json`](validation/bounded-state-overlap-2026-09-13.json).

Ordinary and hybrid resident state now project their actual layers into the
metadata mechanism. The complete inventory shares backing identities across
layers, attention keys/values and fixed roles. It preserves absolute frontiers,
key-only storage, empty one-token-window history, absent fixed slots and unknown
lazy buffers. Architecture-declared symbolic shapes, dtypes and role inventories
are validated before metadata equations run. Native storage must match the exact
concatenation or causal-window mechanism; other cache representations still need
their own projections. Compressed layers use their actual capacity stores in
that same shared inventory. This closes an existing-state input gap in request
quotes; production admission before preparation remains unfinished.

Resident-state validation passed 31 focused runtime tests and four native
regressions (35 tests), plus native test-target and no-default-feature build
checks. Coverage includes shared physical buffers across layers, full and
sliding attention, key-only state, empty one-token-window history, fixed-role
absence, lazy backing, and exact native F32 validation before conservative
floating conversion. The existing compressed live/restored/snapshot continuation
regression also passes through the shared inventory. No new request peak or
released-checkpoint measurements were added. Reproducible commands, hashes and
remaining integration gaps are in
[`bounded-resident-state-projection-2026-09-13.json`](validation/bounded-resident-state-projection-2026-09-13.json).

Prepared execution now retains an architecture-owned `PreparedInferenceBlueprint`
with the exact selection and shared source graph. The replicated quote reuses the
production construction driver and typed state-profile dispatch, executes the
selected parameter formats, and advances a clone of the actual resident-state
projection through every prompt chunk and reserved decode. Opening and closing
frontiers must match the scheduled positions. Source readers, caches, leases and
provenance stay shared; quoting does not reopen artifacts or acquire payloads.
The loaded MLX executable retains both this blueprint and the Metal mechanism
facts captured from its realized target. State inspection is unavailable during
active, indeterminate or fenced transactions and preserves unknown lazy backing.

This is an equation quote, not complete request admission. It supports replicated,
routed and composite text with ordinary/fixed/compressed/pooling resident state. Selected weight-residency
policies remain intact, but their materialization/transfers still need explicit
outside bounds. Partitioned and prediction quote routes,
paged state, media, sampling/copy ownership and facade preflight remain
unfinished. The construction driver returns typed rejections for absent routes;
no unsupported path is advertised as a complete budget capability.

Prepared-quote validation passes 54 tests: three architecture tests (including
297 family/profile, weight-residency, chunk and readout combinations), 31 runtime
workspace tests, 19 bounded-prefill tests, and one native Metal regression.
The architecture cases preserve source-graph identity and unchanged payload-read
counters, selected quantization, unknown host/storage costs and original state.
The native Qwen hybrid fixture produces bounded equation quotes before and after
prefill, continues with three finite nonzero cached decodes, and quotes after its
original checkpoint directory has been moved. Both backend build configurations
also pass. No new native peak or released-checkpoint comparison is claimed.
Commands and source hashes are recorded in
[`bounded-prepared-workspace-2026-09-13.json`](validation/bounded-prepared-workspace-2026-09-13.json).

`quote_inference_workspace` now composes the entire request through a single
cold callback that advances its selected metadata equations. Prefill boundaries
and readout demands come from the ordinary `PrefillDriver`; every reserved
decode position is inspected as well, including later cache-growth boundaries.
This conservatively includes one decode per reserved output position, even when
the first generated token comes from prefill. A decode always includes its one
score row. The result keeps the peak of simultaneous tensor-plus-host transients
across completed spans, rather than adding independent domain peaks or summing
successive graphs. Any missing span bound leaves the full request unknown.

Reports retain exact geometry, completed-span count, the largest known span,
separate diagnostic domain peaks and the first coverage gap. Composition still
requires the selected persistent-state capacity and all untraced preparation,
materialization, sampling, observation and snapshot bounds. Candidate chunk
selection uses these complete-request reports with the existing shared pool.
Core's context policy and runtime's request identity check run before asking the
provider to inspect any equations, so a rejected output allowance cannot start
a large cold traversal. Native production reservation wiring is still unfinished.

Complete-request composition validation passed five core capability tests,
24 runtime memory tests, 14 bounded-prefill tests and 33 cold MLX workspace
tests. The Qwen cold matrix now uses this shared traversal for 36 dense/affine,
tied/untied, chunk-size and output-demand combinations: 240 request spans plus
36 cached-prefix warmup spans. It checks every readout shape and state frontier.
Both the Metal test-target check and the backend without default features passed.
These checks add no native allocation or process-memory measurements. Commands,
source hashes and remaining integration gaps are recorded in
[`bounded-inference-workspace-2026-09-13.json`](validation/bounded-inference-workspace-2026-09-13.json).

`workspace_estimation` exercises real Llama, Qwen2 and Qwen3 decoder equations
with dense/packed, tied/untied weights, batch two, an existing two-token prefix,
chunks 1/2/3/7, all output demands and three cached decodes. It also traces
Qwen3.5 and Qwen3-Next recurrent attention, convolution history and asymmetric
key/value state matrices. Quotes drive the ordinary adaptive planner and shared
reservation pool; an unknown attention primitive prevents strict admission even
with unlimited capacity. These tests use an explicitly specified portable
allocation mechanism. Its facts are never substituted for MLX facts. Full native
operator, cache, materialization and retained-resource quote integration remains
unfinished, as does metadata coverage of the full family/path matrix.

The metadata backend also implements distributed vocabulary operations. A
validated rank context and the architecture's vocabulary range determine local
weight/companion shapes, global token-validation policy, masked contributions,
sum reduction and exact uneven gather widths. Tied readout reuses the embedding's
local parameter storage. Missing transport facts remain unknown even when the
local projection is fully priced, and ownership mismatches are rejected before
projection or observation.

Equation traces distinguish tensor buffers from disjoint operation-owned host
workspace. `operation_bound` prices tensor outputs and scratch;
`host_workspace_bound` prices staging, conversion, sorting and transfer buffers
not sharing that backing storage. Host mappings of unified tensor storage are
counted once. An absent host fact remains unknown even for views, and a zero
requires an explicit derivation. Existing residency and untraced request
preparation still require separate estimates. Runtime/driver bookkeeping,
allocator caches, JIT programs and unrelated process memory remain outside
the Eredu-managed working-memory guarantee.

The report's `tensor_buffers` subreport supports comparisons with MLX active
allocation peaks. Its main total and transient bounds require both tensor and
host facts. Audited Metal operators have independently derived host bounds:
shared native payloads are counted once, while Rust rotary-frequency vectors and
long-row attention masks have explicit additional charges. Other operations and
formats remain unknown. Complete tensor coverage cannot silently turn into
complete managed coverage. Strict admission, application budgets and the
reservation pool reject the gap. A diagnostic admission without strict estimation cannot reserve it,
and neither a safety reserve nor smaller chunks substitutes for missing facts.
When both domains are bounded, their combined bytes select chunks and compete
for the same shared capacity. Host charges survive metadata completion and
handle release until the next explicit completed-span boundary.

Domain validation passed 70 neural tests, seven memory-policy/completion tests,
22 cold backend tests and eight architecture workspace tests. The portable facade
passed 19 tests with its existing external LFM fixture ignored; the Metal and
Accelerate test-target build check passed. These checks add no native allocation
measurement. Commands, source hashes and the concurrent-reservation example are
recorded in [`bounded-workspace-domains-2026-09-13.json`](validation/bounded-workspace-domains-2026-09-13.json).

Runtime now provides `WorkspaceConcatStateFactory` for a selected
exact-concatenation cache and architecture-declared fixed state. Full and sliding
KV, key-only sharing, recurrent/convolution slots and named-segment reset use the
ordinary state traits. A sliding tail keeps its complete backing allocation in
the report. State publication follows successful metadata operations; an error
preserves the prior frontier and roots without refunding earlier trace work.
Metadata clones preserve storage identity and are not estimates of native
isolated snapshot copies. Other native cache strategies still require their own
selected realization and costs.

The shared layered traversal now traces complete Qwen 3.5 dense, Qwen 3.5 MoE
and Qwen3-Next hybrid models: 108 trajectories and 720 reported spans, with batch
two, a prefix built through the equations, chunks 1/3/7, all three readout
demands, dense/affine-four-bit and tied/untied parameters, and three cached
decodes. Every span checks the attention and fixed-state frontiers, retained
component geometry, and the positions consumed by vocabulary projection.

The neural metadata backend implements the blockwise attention interface with
separate begin, accumulate and finish descriptors. They retain causal absolute
coordinates, mask origin, prefix/window policy, sinks and FP32 online state;
gapped prefix-plus-window block ranges remain valid. The logical K/V
reconstruction telemetry is not a native scratch bound. Every stage remains
charged through the containing completed span, and missing tensor or host facts
remain unknown. Kimi's actual latent-attention equations exercise the interface
across 48 trajectories and 320 reported spans, covering direct/low-rank queries,
fused/split K/V reconstruction, dense/packed parameters, explicit/no mask and
uneven chunks. Its fixture cache uses immutable metadata views and makes no claim
about native compressed-cache capacity, transfers or snapshot copies.

State/blockwise validation and reproducible commands are recorded in
[`bounded-state-blockwise-metadata-2026-09-13.json`](validation/bounded-state-blockwise-metadata-2026-09-13.json).
The checks pass 75 neural tests, 12 focused runtime tests, ten architecture
workspace tests, and 102 portable facade/conformance tests (one existing
checkpoint-dependent test ignored). Backend builds pass both without default
features and with Metal/Accelerate test targets.
This extension adds no native allocation measurements and does not complete the
remaining family state profiles, native workspace facts, full-request budget
composition or production reservation wiring.

Resident compressed state is now traced with an explicit selected capacity
increment. `WorkspaceResidentStateFactory` combines ordinary attention/fixed
state with that mechanism; architecture-owned typed profile dispatch exposes
all six ordinary access profiles. The default native compressed increment is a
cold backend fact (currently 256), rather than a family constant in the portable
factory. Initialization distinguishes exact-capacity input reuse from zero
padding, and continuation traces padding concatenation and full-capacity slice
updates. Retained logical views preserve backing storage. Checkpoint restore,
compact isolated snapshots and regrowth preserve their selected copy geometry
without refunding earlier work. New slice-update, contiguous and independent
copy descriptors keep tensor and host facts separate. Native admission remains
incomplete while either required domain is unknown.

Complete Kimi hybrid and DeepSeek-V3 traversals add 144 metadata trajectories
and 960 reported spans: batch two, a two-token warm prefix, chunks 1/3/7, all
readout demands, dense/affine-four-bit parameters, direct/low-rank queries,
capacity steps 4/256 and three cached decodes. The tests check padded capacity,
logical and fixed-state geometry, full destination sizes in slice updates and
preprojection vocabulary demand. Existing Qwen hybrid trajectories now use the
same resident state factory. This is state-mechanism inspection coverage; full
paged/quantized/offloaded state, native facts, request composition and production
budget wiring remain outstanding.

Native conformance exposed a deep-copy defect with two batches and spare cache
capacity: after restoring a 264-token prefix backed by 512-token buffers, the
second batch began with padding instead of its saved values. The native wrapper
now compacts strided input on MLX's reused default CPU stream before its
synchronous logical copy. Direct wrapper regressions cover padded, transposed
and broadcast layouts. Selected Metal storage facts include the independent
destination and possible compaction, as well as full slice-update destinations,
two possible update reshapes and update dtype conversion. Shared CPU/Metal
tensor allocations count once. That validation did not price separate host
workspace, so those tensor facts alone did not constitute complete native
admission bounds.

[`bounded-compressed-storage-workspace-2026-09-13.json`](validation/bounded-compressed-storage-workspace-2026-09-13.json)
records the source hashes and reproducible commands. The validation passes 79
neural tests, 19 focused runtime tests, 12 architecture workspace tests and 102
portable facade/conformance tests (one existing checkpoint-dependent test
ignored). Native checks pass the compressed-cache numerical regression, five
existing snapshot tests, three copy/stream wrapper tests and four storage-fact
tests. The latter include 280 copy and 884 update cases: all observed active
tensor peaks fit the source-derived bounds and all logical values match exactly.
The matrix includes F32/F16/BF16, I32/U32/U8/bool, padded/transposed/broadcast
layouts, empty tensors, mixed floating update types and capacity boundaries.
Metal test targets and the backend without default features compile. The native
linker emits its existing large-unwind-table warning. These results validate
the selected storage mechanisms; full family/path budget enforcement remains
unfinished, and they add no new released-model peak measurements.

Metal convolution quotes now follow the actual selected depthwise, implicit,
explicit-unfolding and Winograd mechanisms. They include possible input/weight
casts and contiguous copies, grouped weight transposition, direct Steel split-K
(including one-row outputs), Winograd padding and transforms, and final dtype
restoration. Convolution has no separate host payload staging: its scalar fill,
casts and temporary data use the shared Metal allocator already counted by the
tensor bound. Driver metadata remains outside the managed payload domain. This
zero host fact does not fill missing host bounds for any other operation.

The native audit also corrected transposed output cropping: `[2,7,3]` input,
`[7,3,3]` weights, stride two and padding four must produce `[2,7,7]`; MLX's
input-cropping path produced `[2,3,7]`. The adapter now crops the expanded result
and quotes its retained full backing. Large native F16 Winograd transforms also
exceeded the scalar-reference tolerance, so matching F16/BF16 operands now use
F32 computation in Metal builds and restore their output dtype. This precision
choice costs F32 temporaries and a final conversion buffer, all included in the
selected bound. Execution and pricing share the cold dispatch predicates.

[`bounded-metal-convolution-workspace-2026-09-13.json`](validation/bounded-metal-convolution-workspace-2026-09-13.json)
records 756 passing native cases across 21 convolution configurations, all nine
F32/F16/BF16 operand pairs and four contiguous/strided input-weight combinations.
Every observed tensor peak fits its bound and output dtype promotion is preserved.
Numerical checks use an independent scalar convolution equation with tolerance
`0.02 + 0.02 * abs(reference)`: all values for ordinary fixtures and 257 spread
positions for the large Winograd fixture, whose complete output is also checked
for finite values. The maximum absolute error is 0.9375 in the BF16
single-row, 4,257-term reduction; the Winograd maximum is 0.02734375. Both
satisfy the stated combined tolerance. The native crop regression, 27 cold
workspace tests, and Metal and
no-default-features backend builds pass. Other native operator matrices were not
rerun. NAX branches remain source-bounded but untested on this M3 Ultra. These
checks do not complete full-request quotes or production budget reservations.

The subsequent managed-host audit supplies explicit facts for audited basic,
dense-product, normalization, reduction, indexing, padding, storage, convolution,
gated-delta, rotary and attention operations. These facts recognize individual
supported names and formats after validating the same descriptor used by tensor
pricing. They do not give future or unknown operators a default zero. Borrowed
host inputs to tensor constructors remain part of caller-owned preparation.

Rotary construction charges its host F32 vectors separately: two native YaRN
vectors plus one for input-product arithmetic when selected, one proportional
vector, or one default/linear input-product vector. Its native first-use storage
also remains charged when the caller supplies cosine/sine tensors. Input-score
attention with key rows above 8,192 positions uses at most one 256-byte host mask
per completed key tile; successive calls release that local vector. Constant
padding needs one filled result and scalar conversion. Edge padding includes the
initial zeros, input copy and up to two full-result updates per padded axis,
without relying on donation. All these native buffers share the tensor domain.

The managed-host audit's Qwen hybrid cold fixture composed known tensor and host bounds
through its actual layered equations. It exercises 18 dense and 18 packed
configurations, including tied/untied readout, chunks 1/3/7, all three output
demands, a warm prefix and three cached decode steps: 276 spans in total. Dense
spans had complete equation bounds; packed spans retained explicit projection and
embedding gaps at that stage. This is a cold equation-coverage result, not a full-request quote
or production budget enforcement. Preparation, parameter residency, sampling,
snapshots, observations and transport still need their enclosing reservations.

[`bounded-managed-host-workspace-2026-09-13.json`](validation/bounded-managed-host-workspace-2026-09-13.json)
records 31 passing cold workspace tests, 154 native padding cases with 1,559,950
exact value comparisons, 1,374 native rotary allocation comparisons and the
compressed-cache snapshot/restore regression. Every measured padding and rotary
peak fits its selected tensor bound; separate host facts come from source
allocation/lifetime derivations. Padding's largest observed incremental peak is
1,376,260 bytes against a 2,097,172-byte bound. Metal test targets and the backend
without default features compile. This phase adds no released-model or process
memory measurements and does not close the remaining request-budget gaps.

Packed equation pricing now includes affine 2/3/4/5/6/8-bit groups of
16/32/64/128 values, MXFP4, all 14 native GGML block formats in both byte orders,
and contiguous E4M3 block-FP8 projections with 128-by-128 floating or UE8M0 scales.
It validates physical weight and companion geometry before deriving a bound.
Affine and MXFP4 lookup prices selected-row gathering and dequantization;
GGML kernels decode directly from their retained packed weights. Projection
includes applicable casts, input/weight compaction, shape copies, optional output
bias, split-K partials and the possible second reduction pass. FP8 adds activation
quantization, its scales and the UE8M0 scale lookup. The latter has a separate
1,024-byte Rust host table; its native copy belongs to the tensor domain.
The affine embedding adapter also promotes differently typed floating scale
and bias companions after gathering. MLX's native dequantizer reads both through
the scale dtype, so a later output conversion cannot correct mismatched input
reads. The selected-row promotion avoids decoding or copying the full table and
fits the quoted gather/conversion capacities.

The same Qwen hybrid fixture now has complete equation bounds for both its
18 dense and 18 affine configurations, including tied readout and all three
output demands. These facts still require separate enclosing bounds for request
preparation, residency, materialization, sampling, capture, snapshots and
transport before they can authorize production inference. Unsupported operation
descriptors retain explicit gaps; the ordinary embedding factory does not
select block FP8, whose separately declared grouped layouts require their own
mechanism facts.

[`bounded-packed-workspace-2026-09-13.json`](validation/bounded-packed-workspace-2026-09-13.json)
records 2,106 native cases and 3,101,416 value comparisons across all supported
formats, F32/F16/BF16, contiguous/strided operands, tied readout and selected-row
lookup. It includes all six mixed affine companion pairs, both GGML byte orders
with distinct rows, FP8 partial blocks, and quantized split-K's second reduction
pass. Every measured native peak fits its bound. The largest incremental peak
is 689,312 bytes against a 1,463,306-byte bound. Input preparation and parameter
residency precede these incremental measurements and require separate quotes.

All 608,888 F32 projection values match the independently decoded F64 dot
products exactly. Projection tolerance is `0.03 + 0.02 * abs(reference)`;
low-precision affine/MXFP4 QMM additionally accounts for independently calculated
tile/partial rounding and native low-precision reduction additions. GGML and FP8
receive no such allowance. Lookup tolerance is `0.002 + 0.01 * abs(reference)`.
The largest absolute projection error is 99.25 in the BF16 IQ4XS fixture and
satisfies the combined tolerance. The 33 cold workspace tests and Metal test
target/no-default-features builds pass. These checks add no released-model or
process-memory measurements and leave production request reservations unfinished.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --lib --features metal,accelerate \
  backend::nn::workspace::packed::tests::native \
  -- --include-ignored --nocapture --test-threads=1
```

Runtime's metadata submission adapter lets the existing parallel layered driver
record these equations. Forked inspection lanes share one ledger; synchronous
host completion and payload release preserve its charges until the caller
explicitly begins another completed span. This completion has no native resource
or inference authority. The Llama parallel fixture exercises 108 rank/configuration
trajectories and 720 spans: TP 1/2, uneven 37-row vocabulary, batch two,
dense/affine and tied/untied weights, an existing two-token prefix, chunks 1/3/7,
all three readout demands and three cached decodes. It checks local projection
rows, complete gathered output, both block reductions and exact retained KV bytes.
Its affine local dimensions preserve complete published quantization groups.
These are metadata and runtime-conformance results, not native transport or
production budget-enforcement measurements.

The metadata backend also implements grouped and hyper-connected neural
contracts. Router descriptors retain all learned parameters, arithmetic, grouped
top-k policy, supplied IDs and pre-dispatch interventions. Original and effective
decisions share one selector invocation. Grouped banks retain packed/independent
topology and all physical companions, output shards, gated/ReLU2 equations,
reduction order and tensor-parallel bias separation. The explicit-group linear
path prices the full multiplication shape before selecting diagonal groups.
Grouped-unit observation exposes all logical selected rows before down
projection; a native bound must include its own sorting, tiling and staging.
Metadata supplies shapes rather than numerical routes or native callback counts.
Replacement shape, dtype, trace identity and observer failures are checked before
the finish operation, without refunding preceding charges.

Hyper-connection metadata retains complete mixing parameters and Sinkhorn
iterations, all four collapse outputs, and sublayer injection/stream mixing.
Final coefficient hooks execute before the stream sum. Overflow and inconsistent
or cross-trace state fail before the affected operation. Native grouped and hyper
allocation facts remain unpriced; these additions cannot by themselves establish
a complete native quote.

The existing routed Qwen layered driver and resident expert provider now execute
36 metadata trajectories and 240 completed spans with an existing two-token
prefix, batch two, chunks 1/3/7, dense/affine and tied/untied weights, all readout
demands and three cached decodes. Another three trajectories (20 spans) preserve
unknown quotes when grouped native facts are absent. Six tensor-parallel expert
trajectories (12 spans) check the global router, local intermediate width and
single output reduction through the ordinary routed provider. These extend
inspection coverage; complete state-profile, source-residency and family/path
quote construction, and production request reservation, remain outstanding.

Grouped/hyper validation passed 66 neural tests, eight architecture workspace
tests, three metadata-submission tests and the backend check without default
features. The portable facade passed 19 tests with its existing external LFM
fixture ignored. Commands, source hashes and matrix details are in
[`bounded-grouped-hyper-metadata-2026-09-13.json`](validation/bounded-grouped-hyper-metadata-2026-09-13.json).

The preceding parallel metadata validation passed all 56 neural tests, five architecture workspace tests and
three metadata-submission tests, plus the backend check without default features.
The portable facade passed 19 tests; its existing LFM tokenizer fixture remains
ignored without the external checkpoint. Commands, exact coverage and source
hashes are recorded in
[`bounded-parallel-metadata-2026-09-13.json`](validation/bounded-parallel-metadata-2026-09-13.json).

Reproduce the metadata and admission conformance with:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-nn --all-features
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test workspace_estimation
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib working_memory
```

The first concrete MLX facts price Metal gated-delta scans using their actual
internal chunk policy (including the 64/65 and 256/257 transitions), temporary
F32 casts, input copies, state versions and output assembly. Native output and
scratch allocations include host-page rounding and the vendored allocator's
bounded oversized-cache reuse. `host_page_size` is an OS-only query; cold facts
create no device or stream. Unknown native operators still prevent a complete
quote. Allocator-cache residency, Metal heaps, driver/JIT storage and unrelated
process memory remain outside these tensor-buffer bounds.

Common Metal facts additionally cover initialization, pointwise arithmetic,
SiLU/sigmoid/softplus/GELU/ELU, policy-selected gated products, concatenation,
static indexing, views and causal masks. The bounds follow the vendored native
equations: each unary/binary/ternary operation includes possible input casts and
its output; custom activations include widening, contiguous input copies and
conversion back to the input type. All intermediates remain charged until
completion. Floating metadata conservatively represents up to four bytes per
element. Mixed signed/unsigned promotions that could require wider storage stay
unknown.

Range slices retain their existing allocation identity. Integer-axis selection
can trigger a native reshape copy for strided inputs, so the neutral descriptor
retains the number of removed axes and the native fact includes that copy.
Causal-mask coordinates are explicitly I32: the previous macro default was F32,
which could conflate adjacent positions above 2^24. None of these operator facts
fills the remaining attention, materialization or retained-resource gaps.

Reduction facts now cover sum and mean, including the native minimum output
allocation, an optional contiguous input copy, and at most one partial
accumulator: 4,096 scalars for the largest all-reduce branch or 128 partials per
output for the strided column branches. Mean additionally prices its divisor
scalar and division. These are implementation bounds rather than multipliers
fitted to observed peaks. Metadata preserves unsigned promotion for byte sums.

Normalization facts compose the actual native equations for LayerNorm, RMS,
L2, learned-offset scales, grouped scales, and both gated normalization orders.
They include native fast/custom kernels and fallback reductions, parameter casts,
F32 widening, reshape/contiguous copies and every arithmetic intermediate.
The cold quote takes the maximum of applicable precision branches. Empty
pointwise broadcasts additionally retain a scalar-buffer allowance.

The expanded isolated suite passed 300 normalization cases, 122 reduction cases
and 12 empty-broadcast activation cases, together with all 454 earlier native
cases. Normalization covered all 15 supported policies at widths 1/32/768/8193,
strided inputs and scale vectors, and five precision pairs. Reduction exercised
both axes and layout orders, including the large all-reduce accumulator above
64 MiB of F32 input; sum/mean results matched an independent host calculation
at absolute tolerance 0.002 plus relative tolerance 0.01. The bounds retain the
native dispatch thresholds, so short reductions pay no partial-array charge.
All 888 native peaks fit their bounds. The 15 workspace tests, 50 portable NN
tests and four architecture equation/admission tests passed. Exact rows,
derivations, hashes and the rerun command are recorded in
[`bounded-metal-normalization-workspace-2026-09-13.json`](validation/bounded-metal-normalization-workspace-2026-09-13.json).

Dense-product facts now price matmul, fused-bias tensor linear operations,
constructed dense projections and tied dense readout. Their bounds include
floating casts, stride-dependent copies, frontend flatten/unflatten and native
batch collapse. An unbatched weight stays unbatched when MLX flattens the input.
SIMD split-K uses at most 32 partials per result; the separate NAX branch can
exceed 32 for large reduction widths and can require three for small odd widths.
The bound follows both native formulas and takes their maximum. Constructed
projections also include the custom BF16 row kernel and separate bias addition.
These equations use the already selected hidden-position geometry, so tied
final-position readout is quoted without allocating full-sequence scores.
That dense-product validation left packed projections and collective storage
unpriced; the subsequent packed facts cover ordinary packed projections.
For a batched zero-width tensor linear input, the adapter supplies explicit
flattened dimensions to addmm. This preserves the zero dot product and broadcast
bias without asking MLX to infer a dimension from an empty array.

The isolated matrix suite passed 960 numerical and peak-allocation cases across
three layouts and five precision pairs, including zero-width bias, tied readout,
native batch collapse and reduction widths through 131,073. Every output matched
an independent scalar dot-product calculation at absolute tolerance 0.002 plus
relative tolerance 0.01. The largest incremental allocation was 2,130,672 bytes,
within its 4,391,994-byte bound. All 888 earlier native measurements also passed
in the same run (19 workspace tests). Input preparation and oracle conversions
are explicitly settled before each matrix measurement's residency baseline.
The M3 Ultra cannot execute NAX, so its split-K formula has source audit and cold
boundary coverage only; that native validation gap remains explicit. Exact rows,
source hashes, derivation and command are recorded in
[`bounded-metal-matrix-workspace-2026-09-13.json`](validation/bounded-metal-matrix-workspace-2026-09-13.json).

Native gather and dense embedding facts now include domain comparisons, safe
indices and the validation reductions retained by completion. Unsigned gather
indices include possible I64 comparison casts; metadata preserves unsigned argmin
results and the exact gather axis. Dense lookup additionally prices I32 token
normalization and optional zero-sentinel masking. Metal gathers read the strided
source directly, so no full-table copy or widening is charged. A cold conformance
check holds the lookup quote constant when vocabulary grows from 37 to 1,000,003
rows while the selected IDs and embedding width remain fixed.

Floating softmax facts distinguish the final-axis kernel, which uses a single
result/contiguous buffer even with precise accumulation, from the general-axis
max/subtract/exp/sum/divide/cast equation. The latter includes both reduction
workspaces and possible widening. Empty floating softmax aliases its input.
Integer softmax's axis-dependent conversion remains unpriced pending an explicit
output contract. The subsequent packed facts cover packed lookup; remaining
attention paths still need their own facts.

The isolated native suite passed 216 dense embedding, 378 gather and 168 softmax
cases, plus all 1,848 earlier measurements (26 tests). Gather and embedding checked
258 deferred validation failures after evaluating both output and retained
assertion arrays. Selected values and sentinel rows matched exact host indexing.
Floating probabilities matched an independent f64 softmax at absolute tolerance
0.001 plus relative tolerance 0.01. All peaks fit their bounds; the largest new
peak was 1,445,290 bytes for precise general-axis F16 softmax, within its
4,429,904-byte bound. Source hashes, detailed observations and the command are in
[`bounded-metal-index-softmax-workspace-2026-09-13.json`](validation/bounded-metal-index-softmax-workspace-2026-09-13.json).

Attention facts now select the native fused vector/full kernel or fallback
product equation from exact Q/K/V geometry. They include promotion and layout
copies, grouped-query replication, products, masks, sink concatenation, softmax,
optional score caps and input-score rounding. Sliding attention includes its
256-row query tiles, retained-key coordinates, mask construction, chunk outputs,
concatenation and the reshape to `[batch, queries, heads * value_width]`. Ordinary
attention keeps its four-dimensional result. Shared checked sliding geometry
rejects overflowing positions and missing current keys before native work.

Two-pass fused scratch has an explicit native configuration contract. The first
cold inspection or two-pass execution retains `MLX_SDPA_BLOCKS`; later environment
edits cannot invalidate its quote. An unset or nonpositive setting uses the
geometry-dependent union of device defaults. A positive override must fit a
native integer and be a multiple of 32. Validation caught the native final
reduction silently dropping incompatible block groups; those settings now return
an error. Valid larger overrides retain their actual scratch count.

The attention suite passed 468 nonzero cases across F32/F16/BF16, grouped and
ungrouped heads, asymmetric value widths, strided Q/K/V/masks/sinks, fused and
fallback dispatch, score caps, rounded score tiles and sliding cuts. All outputs
matched the independent f64 host equation at absolute tolerance 0.003 plus
relative tolerance 0.03; maximum absolute error was 0.00278397. Every attention
peak fit its bound. The largest was 11,266,041 bytes against a 61,203,535-byte
conservative bound. Separate processes verified overrides of 32, 256 and 2,048;
the last used 2,163,712 incremental bytes against a 4,495,350-byte bound.
Seventeen cold cases checked parsing, invalid-count rejection, changes after
capture and eight concurrent readers. Eight existing attention regression tests
also passed, including independent explicit-rounding fixtures. The full workspace
run passed 3,081 native cases and all 30 tests. The 52 neutral neural tests, four
architecture workspace tests and backend check without default features passed.
Source hashes, complete measurements, limits and reproducible commands are in
[`bounded-metal-attention-workspace-2026-09-13.json`](validation/bounded-metal-attention-workspace-2026-09-13.json).

Input-score attention with key rows above 8,192 now has allocation facts for its
two-pass accumulator. Its one-query/256-key policy is shared by execution and
inspection. Each accumulator step waits for successful native ownership
retirement as well as array readiness; on failure, existing recovery retains
unresolved native work and the enclosing request keeps its authority. Quotes
therefore take the maximum block workspace plus live normalization/value state
and finished query outputs, or the final cast/concatenation phase, whichever is
larger. Context growth does not accumulate already-retired block scratch.

The expanded attention suite passed 648 cases, including 96 large-row cases
that check native ownership retirement before final-output submission. Every
peak fit its bound. The largest large-row peak was 835,006 bytes against a
3,117,322-byte bound; maximum absolute numerical error was 0.00108182. The
complete workspace run passed 3,261 measurements and 31 tests. All 44 attention,
cache and recovery regression tests passed, as did the backend check without
default features. Source hashes, measurements, commands and the asynchronous
cache-cleanup test correction are recorded in
[`bounded-metal-blockwise-workspace-2026-09-13.json`](validation/bounded-metal-blockwise-workspace-2026-09-13.json).

Rotary facts cover native fused application, caller-supplied frequencies,
constructed default/linear/wavelength/proportional/YaRN policies, explicit
input-product rounding and broadcast sine/cosine inputs. The quote includes
per-batch outputs before concatenation, scalar offsets, possible frequency and
input casts, intermediate products, and the complete lazy first-use frequency
graph. It therefore also bounds an invocation whose operator was just
constructed. Frequency state already present in residency is conservatively
counted again by this per-operation upper bound; composing a tighter full request
quote will require retaining and deduplicating that state explicitly.

Wavelength-scaled partial rotation now limits its second half to the rotated
dimensions and preserves the tail. Explicit rotary positions are generated as
integers before conversion to F32, preserving absolute-coordinate rounding above
16,777,216 and rejecting an overflowing endpoint.

All 1,344 rotary and 30 explicit-embedding cases passed numerical and allocation
checks across F32/F16/BF16, strided inputs, partial widths, both pair orderings,
frequency policies and lengths through 3,000. The largest peak was 7,352,908 bytes
against a 12,570,577-byte bound. The independent f64 rotation equation used
absolute tolerance 0.006 plus relative tolerance 0.04; maximum absolute error was
0.00990904, or 0.00027268 for F32. A separate large-position fixture checked exact
integer-coordinate rounding and endpoint rejection. The full workspace suite
passed 4,635 measurements and 36 tests; the final five rotary tests and portable
backend check also passed after adding native grid-overflow rejection. All 52
neutral neural tests and four architecture workspace tests passed. Details and
source hashes are in
[`bounded-metal-rotary-workspace-2026-09-13.json`](validation/bounded-metal-rotary-workspace-2026-09-13.json).

Attention and rotary facts cover forward-inference tensor buffers; host scratch,
other mechanisms, complete request composition and production reservation wiring
remain outstanding.

The common-operation measurement passed 435 cases: 29 equations at three shapes
with F32, F16, BF16 and two mixed-precision input pairs. Inputs remain live and
use transposed nonzero data; the allocator cache remains populated between cases.
Every incremental native peak fits the cold bound. Three additional causal-mask
cases at offset 16,777,217 match an independent integer oracle, including a
zero-distance window, and stay within their bounds. The 16 recurrent cases also
passed again in the same process. Exact observations, source hashes, runtime
versions and derivations are recorded in
[`bounded-metal-basic-workspace-2026-09-13.json`](validation/bounded-metal-basic-workspace-2026-09-13.json).

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --lib --features metal,accelerate \
  backend::nn::workspace -- --include-ignored --nocapture --test-threads=1
```

The isolated native scan measurement passed all 16 combinations of F32/BF16,
scalar/vector decay and lengths 1/65/257/3000, with strided nonzero inputs. At
3,000 positions, observed incremental MLX peaks ranged from 1,073,124 to
1,430,580 bytes, below the derived bounds of 9,766,467 to 10,918,467 bytes. The
bound intentionally retains every possible temporary until completion; the
native scheduler can release many earlier. These are operator measurements,
not model-wide inference admission. The existing cached-continuation and
CPU/Metal numerical parity tests passed as well. Exact rows, versions,
hardware and commands are recorded in
[`bounded-metal-recurrent-workspace-2026-09-13.json`](validation/bounded-metal-recurrent-workspace-2026-09-13.json).

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --lib --features metal,accelerate \
  backend::nn::workspace::tests::metal_recurrent_observed_peak_fits_cold_workspace_bound \
  -- --ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --lib --features metal,accelerate \
  backend::nn::gated_delta::tests -- --include-ignored --nocapture --test-threads=1
```

Run the measurement alone in its process, with Metal device access. The ordinary
sandbox cannot initialize Metal; this is an environment restriction, not a
passed hardware check.

The metadata ledger also represents an output that may be either a view or an
independent copy. It conservatively charges the candidate copy and retains all
possible backing owners, deduplicating shared roots. It cannot infer that a small
cache view releases a larger buffer merely from the view's logical dimensions.

Runtime `WorkingMemoryPool` atomically charges existing unique physical residency
and incremental admitted requests in one shared domain. Cloned reservations
retain the same charge until their last owner is released; peak usage never
rewinds. Executable identity and exact geometry are checked before prefill.
Cloning a reservation cannot start a second prefill or restore consumed start
authority. Native completion/recovery must retain the reservation even after a
partially successful submission fails. Backend integration must share the pool
between sessions using the same physical allocations, and retain request state
reservations through decode and any snapshots referencing those allocations.

Runtime `plan_prefill` quotes candidate sizes before allocation and chooses the
largest admitted chunk at or below the requested size. It tests every smaller
size when necessary because native algorithm bounds need not be monotone.
Successful planning already holds an atomic reservation against concurrent work.

`PrefillDriver::step` owns at most one submission, waits for exact completion
before advancing, preserves prompt-relative and absolute coordinates, checks
cancellation before more work, and fences after failure. An observation failure
retains its pending completion. Independent release evidence can clear that
pending handle without making the driver resumable. `run` wraps the same `step`
driver for uninterrupted execution; each result is delivered once after
completion. Parallel executors agree cancellation before submission and after
settlement, so cancellation on one rank prevents every rank's next chunk.
Terminal calls do not start new collectives after peers have returned.
The driver stores output/completion types rather than an executor loan. A
persistent owner can borrow its session for one advance and drop that adapter
before the next advance; request identity and one-use authority remain retained
by the same driver. The nine-family session fixture exercises this lifetime.

`try_prefill_unbudgeted_source` is the native adapter's shared ordinary entry.
It derives each local retained frontier, agrees source availability across
participants, runs `PrefillDriver`, and restores ordinary public score axes.
Direct and composite native adapters supply architecture-owned sources. Ordered
native text parts are sliced and concatenated only within the current span.
This entry explicitly has no working-memory charge; it is not a fallback from
rejected strict admission. Media ingress and full-sequence observers currently
retain their existing whole-request protocol. The cancellable entry receives
the facade's live token; generic chunk configuration still needs to reach it.

`SessionPrefill` borrows the session exclusively and invokes its existing
transaction for each architecture-prepared span. It checks the exact reservation,
readout demand and completion capability before native input preparation. Required
backend reservation guards cover preparation, execution and cancellation
agreement, including the final completion vote after preparation settles; MLX
uses its existing native recovery scope. A failed or unobservable
scope retains its charge in recovery and fences the session. The adapter now
also attaches the charge to retained state before preparing a span. That owner
keeps it live through subsequent decode even after the caller drops the driver
and reservation. Checkpoints, native copies and branch slots inherit retention;
restoration merges newer and saved charges before native work. State replacement
cannot refund a charge still held by a checkpoint, descendant or native recovery.
This fixes retained-state lifetime for explicitly reserved requests; default
facade preflight and shared pricing of additional
snapshot/copy storage remain unfinished.

Reserved state now also retains the exact request and its logical decoder
frontier. Architecture input inspection exposes the batch and semantic decoder
span without native allocation. Runtime agrees the expected chunk length,
prefill/decode phase, output demand, batch and retained position before
checkpointing or model work. Every decode is one position within the reserved
output allowance. Smaller arbitrary chunks are also rejected: a quote for one
native shape cannot prove a bound for every smaller shape. Single-row sequence
and final-position readout are equivalent, while state-only demand remains
distinct. Auxiliary proposal inputs cannot use an ordinary decoder quote.

Execution verifies the resulting frontier before output observation. Snapshot
restoration restores the logical position and exact request while preserving all
backing charges. An older unadmitted snapshot cannot erase an installed admission.
Direct, prepared-input and sequence calls retain a native scope through
publication and force completion when reserved; failed settlement fences further
execution and retains native recovery ownership. These checks enforce supplied
admission geometry and lifetime. They do not fill the remaining native quote,
facade preflight, independent-copy and auxiliary-resource accounting gaps.

The admission conformance extension checks invalid batches and decoder lengths,
prefill/decode substitution, exhaustion, and cancellation followed by an attempted
decode. Rejections precede projection and preserve exact nonzero state. Each of
the 432 reserved family trajectories also rejects completed output transactionally,
retries three cached decodes, restores the prompt checkpoint and replays those
decodes through the sequence API. Rollback restores the logical frontier without
releasing its charge. Neutral scope-failure injection covers failed opening,
failed settlement, retry, fencing and independent release of quarantined ownership.
The validation record is
[`bounded-inference-spans-2026-09-13.json`](validation/bounded-inference-spans-2026-09-13.json).
The final checks pass all 639 runtime tests, the expanded family matrix test,
seven native retention/snapshot regressions and 102 portable facade/conformance
tests: 749 tests total. One existing external-checkpoint facade test is ignored.
The backend test targets with Metal/Accelerate and the backend without default
features both type-check. This extension adds no native peak-memory measurements.

The retention conformance matrix uses nine family configurations across resident,
host-layerwise and disk-streamed execution. Its 432 reserved trajectories drop
the caller's reservation before three cached decodes, then keep a checkpoint
alive across reset and restore. Cancellation and preparation failure likewise
retain the completed prefix's charge until state replacement. A separate mock
injects a failure after wholesale state replacement and checks that both the
saved and newer reservations remain charged. Native key/value, fixed hybrid and
pooling owners exercise checkpoints, independent copies, prediction forks and
restore with nonzero values; these are lifetime witnesses, not new native
allocation-bound measurements. Reproducible commands and results are recorded in
[`bounded-inference-retention-2026-09-13.json`](validation/bounded-inference-retention-2026-09-13.json).
Validation passed 16 bounded-prefill tests, 77 neutral session tests, the family
matrix test, two native retention/recovery tests, five native snapshot tests,
83 portable backend-conformance tests and 19 portable-facade tests (203 total).
The existing external LFM checkpoint facade test remains ignored. Both backend
feature checks passed.
Each span also compares its admitted start with the decoder frontier in retained
state. Participants agree this check before preparing tokens. A stale cached
prefix produces typed `StateFrontierMismatch`, preserving the existing state;
an explicit stateless partition contributes no local state frontier.

`PreparedTextPrefill` supplies pure-text spans with either cold host token IDs or
existing native tokens. Host input materializes only the current span. Explicit
query/key masks select prompt-relative queries and cached-prefix-relative keys,
preserving broadcast axes and batch rows. The complete prompt identity commits
only with the final span. This source deliberately does not slice media inputs;
architecture-owned retained encoder results and semantic media spans remain to
be connected. Parallel callers must coordinate successful construction of every
rank's adapter before entering its shared driver.

`PreparedCompositeTextPrefill` adapts token-only ingress to a composite family's
existing prepared-input admission. It creates each owned token span under the
same reservation guard, retains the exact resulting admission, and checks that
the admitted decoder coordinates equal the scheduled span. It accepts host or
native token IDs; media preparation still requires retained encoder ingress.
Nonzero conformance now covers dense and routed conditional Qwen, Qwen-VL,
Inkling and Muse-Glimmer through this source. Chunk sizes 1, 2, 3 and 5 preserve
prefill scores, two subsequent cached decodes and exact mutable state in both
`step` and `run`; each operation projects one vocabulary row. These additional
composite tests use resident weights and explicit no-budget request authority.
Dense Muse-Glimmer, Inkling, conditional Qwen and Qwen-VL additionally match
unchunked partitioned execution through two cached decodes under TP2, PP2 and
TP2×PP2. These checks use chunks of two and three positions through `run` and
`step`, respectively, and exercise state-only output across partition boundaries.

These contracts bound Eredu-managed storage only after integration with proved
native mechanism bounds. They do not guarantee total process or system memory,
and do not rely on MLX's advisory allocator limit.

## Remaining implementation and validation

The [current integration status](#current-integration-status) is the authoritative
remaining-work list. Selected successful paths below remain closed unless a
relevant change requires another check; neither component coverage nor a single
fixture establishes every family and execution profile.

| Area | Recorded coverage | Remaining obligation |
| --- | --- | --- |
| Readout and prefill | Shared pre-vocabulary demand, uneven bounded chunks, state-only intermediate spans and selected family/residency/parallel parity. | Reconcile all applicable tied/quantized output, direct scoring, speculative verification, prediction, observation and intervention demands. |
| Distributed execution | Selected TP/PP/combined generation, saved-state replay, CPU and Host/Disk comparisons pass. | Close EP ScatterSum and dependent EP combinations; complete IndependentCache admission. |
| Capture | Routed resident generation/saved state, projected parallel prefill and complete TP unit-output saved capture pass. | Close projected decode and sharded saved owners, routed Host/Disk facts, additive delivery and applicable nested/windowed/sparse ownership. |
| Media | Selected raw/prepared media generation and saved-state paths pass, including eighteen native mode/residency comparisons. | Complete prepared-media capture and paged decoder media capture through the integrated shared drivers. |
| Packed sources | Exact GPT-OSS MXFP4 companions and projection source check passes. | Validate the existing biased TP managed/controlled path and applicable lifecycle. |
| Speculation and policy | Selected Independent/Embedded/external assistants and saved replay, released templates, behavioral profiles, Forbidden/Active/Auto tool completion and default annotation/contains checks pass. | Reconcile applicable combinations and actual source-specific typed refusals; do not relabel closed controllers or existing speculative batches as unimplemented. |
| Lifetime and budgets | Selected concurrent capacity/refusal, cancellation, completion, escaped payload and nonrefunding snapshot/copy checks pass. | Finish the cross-path audit of retained native/Host owners, asynchronous transfers, incremental residency and cumulative observation/receiver costs. |
| Realtime decisions | Moshi and reduced PersonaPlex neutral demand, frame continuation, state and observation coverage remains recorded. Explicit-capacity frame preparation/submission now uses the shared scheduler. | Native managed token frames, existing bounded/paged/TP and saved-state validation remain open; see the current status above. Pre-existing ordinary PP/combined rejection is retained. |
| Released reference and final checks | Pinned Qwen independent prefill/decode, 5–3,000-position matrix, equivalent chunk policies and measured allocation/process results are recorded. | Consolidate evidence and feature checks after remaining joins; no repeated benchmark solely because another component changes. |

Background loading, resident/Host/Disk paged state and selected speculative batches
already have successful evidence. Their applicable combinations remain part of
final reconciliation; they are not blanket missing entry mechanisms. Historical
notes elsewhere in this record describe earlier integration states.

Representative existing evidence:
[sliding text and residency](validation/bounded-additional-dense-prefill-integration-2026-09-14.json),
[hybrid parallel boundaries](validation/bounded-parallel-prefill-boundary-integration-2026-09-14.json),
[Gemma4 state and partition boundaries](validation/bounded-gemma4-causal-prefill-2026-09-14.json),
[media and observation spans](validation/bounded-media-observation-prefill-integration-2026-09-14.json),
[Moshi demand](validation/bounded-moshi-decision-demand-integration-2026-09-14.json),
[PersonaPlex numeric continuation](validation/bounded-personaplex-numeric-integration-2026-09-14.json),
and the [latest native integration record](validation/bounded-media-and-submission-2026-09-16.json).
Each record describes its own scope; intermediate limitations in older records
must be checked against newer results before being carried forward.

The motivating score tensor has 3,000 × 248,320 × 2 = 1,489,920,000 bytes
(approximately 1.39 GiB) at batch one. A single row is 496,640 bytes. This is
geometry arithmetic, not a measured native peak-memory result.

## Focused verification

```sh
cargo test -p eredu-core --lib
cargo test -p eredu-runtime --test bounded_prefill
cargo test -p eredu-architectures --test reference_numeric bounded_readout
cargo test -p eredu-architectures --lib capability::tests
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance
cargo check -p eredu-backend-mlx --no-default-features
```

The nonzero scheduler fixture compares five chunk sizes and three output demands
under explicit and uninterrupted advancement through three cached decodes. It
also tests cancellation, unresolved completion failure, submission failure,
identity binding, one-use start authority, shared capacity and unknown bounds.
Readout fixtures check actual projection input shapes and values before the
projection call, including batched and multi-stream hidden states.

Initial contract verification on 2026-09-13 passed the 252-test core suite, then
all five focused core capability tests after adding two coverage/arithmetic
regressions; all six architecture capability tests; 11 bounded-prefill tests;
and both numerical readout tests. Portable facade/conformance passed 101 tests
with one pre-existing checkpoint-dependent test ignored. MLX compiled with
`--no-default-features`. These results establish the new contracts and existing
build compatibility, not production bounded-prefill or native memory behavior.


The production readout migration adds a nonzero matrix for Llama, Qwen2/Qwen3
(tied and untied), LFM2, Kimi Linear, Nemotron-H, Qwen3.5 and Qwen3-Next. It compares
all three demands at chunk sizes 1, 2, 3 and 5 through resident and rebuilt-unit
traversal, then checks three cached decodes and exact retained state. Prepared
session tests additionally include Gemma 2 and Nanbeige and assert the actual
vocabulary operator receives one position. Affine four-bit fixtures cover tied
and untied prepared readout. A V4 partition fixture checks full retained captures
with final-position scores and state-only completion dependencies. The direct traversal
fixtures test readout separately; the prepared-session matrix also exercises the
shared scheduler and reservation adapter.

The full architecture numerical suite passed 284 tests after the migration,
including existing TP/PP, media, observation/intervention and speculative
regressions. The runtime backend-independence suite passed 76 tests, including
two-rank readout agreement with an observer on only one rank. Native adapter
conformance confirmed demand survives both array and neutral tensor wrappers
without allocating a native tensor. Native backend tests compile; the linker
reports the existing large unwind-section warning. Released-checkpoint numerical
validation and native memory measurements were still outstanding at that stage;
the later results below cover readout selection and token-only chunking.


Prepared-session chunk conformance now also uses nonzero SafeTensors payloads in
fully resident, host-layerwise and disk-streamed selections. It executes
state-only prefixes, one-position final scores, and three cached decode steps at
four uneven chunk sizes through the production transaction path. A completion
failure test checks rollback, absence of score publication, and no successful
commit. This operation is the scheduler integration seam; it does not itself
choose chunks, prepare media spans, enforce a working-memory reservation, or
expose chunk advancement through the facade.


Verification passes 254 core unit tests, all 608 runtime tests, all 288
architecture numerical tests and 101 portable facade/conformance tests (one
pre-existing checkpoint-dependent test ignored). The full numerical suite and
native test-target type check were repeated after the completion-vote retention
change. The 11 focused numerical tests include composite spans, retained-frontier
rejection and partitioned chunking. The nine-family
reserved-session matrix covers both empty and two-token cached prefixes,
four chunk sizes, three cached decodes, resident/host-layerwise/disk weights,
cancellation and second-span input failure. Mask fixtures verify batch packing,
cached key extents and broadcast queries. A two-rank scheduler test confirms
that one rank's cancellation stops both at the same completed boundary.
The MLX reservation-recovery test confirms failed/blocked native observation
retains capacity until independent settlement. Native test targets type-check
with no default features. Scalar fixture cost contracts do not establish native
workspace bounds or default facade integration.

Official released validation preparation now uses
[Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17),
stored outside the source tree under
`/private/tmp/eredu-bounded-qwen35-2fc06364715b967f1860aea9cf38778875588b17`.
All files match their publisher LFS SHA-256 or Git blob SHA-1 metadata. The
1,746,942,600-byte weight file has SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The hardware is an Apple M3 Ultra with 32 CPU cores and 256 GiB unified memory.
The native backend uses vendored MLX 0.32.0 with the repository's completion
and platform patches. The isolated independent reference installs MLX 0.32.2, MLX-LM 0.31.3 and
Transformers 5.17.0. Its initial five-token prefill plus three cached decode
steps succeeds. The score, readout-memory and native chunk-policy comparisons
below now pass. Enforceable memory-bound validation is still pending.


## Released Qwen readout measurements (2026-09-13)

The [recorded measurements](validation/bounded-qwen-readout-2026-09-13.json)
compare fresh native processes with identical token IDs and three teacher-forced
cached decodes. `last` uses ordinary prefill; `sequence` uses an empty capture
selection that preserves the sequence observer's full-row demand. Both return
one vocabulary row. No hidden captures are retained. Each run processes the
whole prompt in one invocation; these results do not measure native chunking.

| Prompt positions | MLX live peak, last / sequence (GiB) | Process peak footprint, last / sequence (GiB) |
| ---: | ---: | ---: |
| 5 | 3.510 / 3.510 | 5.005 / 5.010 |
| 128 | 4.232 / 4.232 | 5.738 / 5.856 |
| 512 | 6.275 / 6.275 | 7.804 / 8.277 |
| 1,500 | 11.793 / 11.793 | 13.327 / 14.714 |
| 3,000 | 20.893 / 20.893 | 22.558 / 25.331 |

At 3,000 positions, peak process footprint drops by 2,978,283,640 bytes
(2.774 GiB), from 27,199,392,672 to 24,221,109,032 bytes. The observed output dtype
is F32, so a full 3,000 × 248,320 score tensor occupies about 2.775 GiB before
selection. MLX's live peak remains 22,433,750,408 bytes in both modes: other
full-prompt transients establish that earlier peak. This directly demonstrates
why readout selection alone does not finish the working-memory task. MLX cached
allocations, process RSS and process physical footprint are distinct metrics;
the record retains both maximum RSS and peak footprint from `/usr/bin/time -l`.
Neither allocator telemetry nor these measurements establish an enforceable bound.

Across the five lengths, prefill score differences between native readout modes
are at most `2.575e-5`; all three decode score rows match exactly and every argmax
matches. Independent MLX-LM comparison checks every vocabulary score at prompt
lengths 5 and 3,000 plus three cached decodes, with `atol=0.25`, `rtol=0.02`.
Every row and argmax passes. Maximum absolute reference error is `0.210024`;
maximum RMSE is `0.038212`. Native scores are F32, while the independent model
loads the released BF16 weights. Timing includes cold native/JIT effects;
these runs are memory/numerical validation, not a throughput comparison.

Reproduce the reference and one native case with Metal access:

```sh
python3 eredu-backend-mlx/validation/fetch_bounded_qwen_checkpoint.py /private/tmp/qwen-reference
python3 -m venv /private/tmp/qwen-reference-env
/private/tmp/qwen-reference-env/bin/pip install mlx==0.32.2 mlx-lm==0.31.3 transformers==5.17.0
cargo rustc -p eredu-backend-mlx --example readout_memory_probe --features metal,accelerate -- -C link-arg=-Wl,-S
python3 - <<'PYINPUT'
import json
prefix = [760, 6511, 314, 9338, 369]
for positions in [5, 128, 512, 1500, 3000]:
    with open(f'/private/tmp/qwen-input-{positions}.json', 'w') as output:
        json.dump({'prompt_ids': (prefix * ((positions + 4) // 5))[:positions],
                   'decode_ids': [11751, 13, 198]}, output)
PYINPUT
/usr/bin/time -l target/debug/examples/readout_memory_probe /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json last /private/tmp/qwen-last.json 3000
/usr/bin/time -l target/debug/examples/readout_memory_probe /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json sequence /private/tmp/qwen-sequence.json 3000
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /private/tmp/qwen-reference-env/bin/python eredu-backend-mlx/validation/bounded_qwen_reference.py /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json /private/tmp/qwen-reference-scores.json --compare /private/tmp/qwen-last.json
```

Repeat both fresh native processes with each input length for the table. The link
flag strips debug symbols from the probe executable to limit build-disk use; it
does not change numerical compilation settings. A failed initial debug link
exhausted local disk space; deleting generated incremental build caches allowed
the probe to build. The native linker still reports its existing large unwind
section warning. No hardware limitation prevented this single-device matrix.

## Released Qwen chunk measurements (2026-09-13)

The [chunk-policy record](validation/bounded-qwen-chunks-2026-09-13.json) uses
the same checkpoint, hardware and inputs. All 15 runs use final-position readout
and three teacher-forced cached decodes in fresh native processes. `full` sets
the chunk limit to the prompt length; the other policies cap it at 128 or 512.
Both native direct and composite token-only adapters enter the runtime's shared
driver; this released conditional Qwen checkpoint exercises composite ingress.

| Prompt positions | MLX peak, full / 128 / 512 (GiB) | Process peak footprint, full / 128 / 512 (GiB) |
| ---: | ---: | ---: |
| 5 | 3.510 / 3.510 / 3.510 | 5.006 / 5.005 / 5.006 |
| 128 | 4.232 / 4.232 / 4.232 | 5.739 / 5.739 / 5.739 |
| 512 | 6.275 / 4.316 / 6.275 | 7.804 / 5.859 / 7.805 |
| 1,500 | 11.793 / 4.377 / 6.565 | 13.328 / 6.632 / 10.554 |
| 3,000 | 20.893 / 4.483 / 6.775 | 22.559 / 7.721 / 11.332 |

At 3,000 positions, the default 512-position policy reduces measured MLX peak
from 22,433,750,408 to 7,274,563,696 bytes (67.6%) and process peak footprint
from 24,222,780,344 to 12,168,154,064 bytes (49.8%). A 128-position limit further
reduces these peaks to 4,813,825,208 and 8,290,094,032 bytes. These include existing
model residency. Process footprint still grows with prompt length even at a
fixed chunk limit; chunk scheduling alone does not prove total memory bounded.

Every vocabulary score in all four output rows passes native chunk/full
comparison with `atol=1e-4`, `rtol=1e-4`. Maximum absolute difference is
`3.2783e-5`; all argmax results match. Both 3,000-position chunk policies also pass
the independent MLX-LM comparison with the previously declared `atol=0.25`,
`rtol=0.02`. Maximum absolute reference error is `0.210024`, maximum RMSE is
`0.038212`, and all four argmax results match. Native outputs are F32; the
independent implementation loads the released BF16 weights. These tolerances
describe numerical comparison, not a memory-budget guarantee.

After the setup commands above, reproduce each fresh process as follows:

```sh
for n in 5 128 512 1500 3000; do
  for policy in full 128 512; do
    chunk="$policy"
    if [ "$policy" = full ]; then chunk="$n"; fi
    /usr/bin/time -l target/debug/examples/readout_memory_probe \
      /private/tmp/qwen-reference /private/tmp/qwen-input-"$n".json last \
      /private/tmp/qwen-chunk-"$n"-"$policy".json "$chunk" \
      > /private/tmp/qwen-chunk-"$n"-"$policy".stdout \
      2> /private/tmp/qwen-chunk-"$n"-"$policy".time
  done
done
for chunk in 128 512; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /private/tmp/qwen-reference-env/bin/python \
    eredu-backend-mlx/validation/bounded_qwen_reference.py \
    /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json \
    /private/tmp/qwen-chunk-reference-"$chunk".json \
    --compare /private/tmp/qwen-chunk-3000-"$chunk".json
done
```

The native input builder exposes `with_prefill_chunk_positions(NonZeroU64)`;
omitting the probe's final argument selects the default 512-position limit.
The current adapter uses explicit unbudgeted request authority. Media ingress
and observers requiring sequence attribution still retain their whole-request
protocol. The facade cancellation token now reaches ordinary token-only spans. These are
remaining implementation gaps, not architectural exceptions.

## Sampling workspace validation (2026-09-13)

The [sampling validation record](validation/bounded-sampling-workspace-2026-09-13.json)
adds the ordinary sampling contribution to prepared replicated text quotes.
Runtime's configured sampler owns both native and metadata policy execution.
The architecture supplies actual final-position score geometry; the native
adapter retains the same selection and source graph. No artifact payload is
read or native state advanced during inspection.

Quotes cover standard top-k/top-p/min-p and history penalties, static vocabulary
filters, Mirostat cutoff and probability commitment, categorical or greedy
selection, seed creation and retained random-key replacement. Every reserved
sampling step is inspected. Exact boxed U32 history capacities and old/new
replacement overlap are included. Missing seed-initialization host or scratch
facts remain unknown even after later complete spans.

All 330 measured Metal primitive cases and nine nine-step sampling trajectories
fit their tensor-buffer bounds. Cases cover F32/F16/BF16, contiguous and strided
inputs, one and three rows, and vocabulary widths 1, 37, 2,048, 2,049 and 4,097.
The trajectory matrix covers greedy, stochastic standard and Mirostat sampling
with static filters and history growth through capacities four, eight and
sixteen. The independent host penalty equation also matches empty, repeated and
out-of-vocabulary histories across full, zero, one-token and bounded windows.

Reproduce the native checks with a Metal-capable host and serial test execution:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  sampling_ -- --ignored --test-threads=1 --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  mlx_penalty_history_windows_match_independent_counts -- --ignored --test-threads=1
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  prepared_workspace_native_executable_retains_selection_and_projects_live_state \
  -- --ignored --test-threads=1
```

These measurements validate the selected sampling component, not complete
request admission or process memory. Preparation, weight materialization,
changing constraint-controller state, speculative proposals, retained
observations and snapshot lifetimes still need their enclosing bounds. Native
facade requests remain explicitly unbudgeted until the complete quote and shared
reservation are connected before preparation. The released-checkpoint numerical
and chunk-memory results above are unchanged.

## Prompt and direct-materialization workspace validation (2026-09-13)

The [validation record](validation/bounded-preparation-workspace-2026-09-13.json)
adds a runtime quote for complete prompt storage. It charges actual host token
capacity and selected native initialization/shape-copy capacity independently of
prefill chunk size. Composition rejects a different request geometry and keeps
missing backing or enclosing costs unknown. Shared-pool coverage verifies that
shrinking chunks cannot refund complete prompt storage or admit another request
while its reservation remains live.

Native coverage uses the production `prepare_text_prompt` path with 1, 37, 3,000,
4,097 and 16,385 tokens, both exact host capacity and 8,192 spare token slots.
It checks value/identity preservation, no native allocation during quoting, and
full backing retained by a one-position view after dropping the original prompt.
The separate direct-read matrix uses source and concatenated bindings, same-stream
ownership and execution-stream `Copy` aliases, across 37, 4,097 and 16,385
elements. Bounds include allocator capacity rather than logical tensor bytes.
All ten prompt and six direct-materialization measurements fit their respective
bounds. The 40 focused runtime tests also pass, including concurrent prompt
reservations and unknown-bound rejection. An initial alias assertion exposed
that MLX `Copy` shares storage; the corrected test checks identical backing on
both stream paths, and the quote counts it once.

Reproduce on a Metal-capable host, with allocator measurements run serially:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib working_memory
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  direct_materialization_quote_requires_exact_binding_bytes_and_preserves_ordinary_gap
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  metal_text_prompt_preparation_prices_complete_backing_across_chunks \
  -- --ignored --test-threads=1 --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  metal_direct_materialization_batch_peaks_include_retained_sources_and_stream_copies \
  -- --ignored --test-threads=1 --nocapture
```

These are component bounds. Converted recipes, host-residency transfers, full
weight/cache inventory, changing constraints, snapshots, observations, speculative
and media paths still need their enclosing accounting and admission integration.
Native facade requests remain explicitly unbudgeted. Managed payload bounds do
not cover OS caches, allocator caches, driver/JIT memory or total process memory.

## Admitted source execution (2026-09-13)

The [admitted-source validation record](validation/bounded-admitted-source-2026-09-13.json)
connects supplied request authority to the production source gateway. Owned MLX
prompts and borrowed input views carry the same request. Runtime rejects a foreign
execution, different input/chunk geometry, stale cached frontier or incompatible
readout before the source factory. An unavailable source returns a typed rejection;
it cannot fall back to an unquoted whole-input path. One-use authority is consumed
before source construction, so even a failed selection cannot be replayed through
a retention clone.

The replicated neutral fixture exercises supplied reservations in nine family
configurations across resident, host-layerwise and disk-streamed policies. It
compares admitted cached-prefix prefill and three decode steps with ordinary
numerics, checks the retained charge after dropping caller handles, rejects excess
decode and releases after reset. Cancellation preserves exactly one completed
span and its charge. Existing selective/full readout, tied/quantized projection
and composite TP/PP regressions remain in the same focused suite.

The native Qwen hybrid fixture checks the actual input handoff, preserved state
after foreign-target/chunk rejection, retained charge through decode and release
after reset. Its deliberately generous test envelope verifies reservation
transport; it is not a complete production estimate. The ordinary native
cancellation regression verifies compatibility with callers that explicitly
remain unbudgeted.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric bounded_readout
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  prepared_workspace_native_executable_retains_selection_and_projects_live_state \
  -- --ignored --test-threads=1
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --features metal --lib \
  native_cancelled_prefill_settles_one_chunk_without_sampling_and_preserves_session \
  -- --test-threads=1
```

Public preparation still needs complete quoting and shared reservation before
native input allocation. Supplied admission now reaches execution, but this does
not establish complete native budgets for media, speculative/observation state,
snapshots or distributed resources. Sequence-output collection remains on its
existing `SessionPrefill` contract; the ordinary gateway returns final scores.


`InferenceRequest` now distinguishes an actual working-memory reservation from
explicit no-budget execution. Both modes use the same geometry, one-use start,
completion and cancellation driver. Strict or application-budget admission still
requires `WorkingMemoryPool::reserve`; unknown bounds never trigger an automatic
no-budget fallback. The session adapter retains the exact request identity, so a
different charge owner or an unbudgeted request with equal geometry cannot replace
its admitted request. Default facade policy and native workspace pricing remain
to be connected to these contracts.


Exact request identity and explicit no-budget authority share all 14 scheduler
tests. The native recovery test confirms that unresolved request ownership
cannot refund its memory charge. The reproducible fetch script verifies the
downloaded files against the pinned revision endpoint. `git diff --check` is clean.

The cancellation integration passes 254 core tests, 608 runtime tests and 102
portable facade/conformance tests (one existing checkpoint-dependent test
ignored). The 11 focused numerical tests also pass after adding first-span
cancellation through the production source gateway across nine families and all
three weight-residency policies. They verify exact prefix state, no vocabulary
projection and no second span. Ordinary and controlled facade tests agree on
zero committed tokens, cancelled termination and absent first-token time; an
injected concurrent backend error remains a failure in both paths. The full
288-test architecture numerical suite passed before this final cancellation
integration. Native test targets type-check with no default features.
The native cancellation regression also passes on the CPU realization: one
two-position span commits, no scores are published, submission authority becomes
idle, and subsequent decode exactly matches the ordinary two-position prefix.
The nonzero fixture checks that cancellation does not poison a safely settled
session. Shared-runtime two-rank coverage also verifies that peer cancellation
reaches each local commitment token without another submission or terminal vote.
Five existing native control regressions also pass, covering dense state exchange,
convolution and recurrent/MoE continuations, RNG/adaptive sampling, constraints,
captures and interventions. Native input rejection still preserves its original
cause, cached state and capture epoch.

## Shared ordinary and controlled preparation

Core's ordinary machine now owns admission, prompt construction/binding and
sampler construction for iterators, borrowed controlled sessions and detached
continuations. The facade passes host tokens into this path instead of preparing
native input first. Admission observes the vector's full owned capacity, not only
its length, and receives read-only generation configuration and controller state.
The session agrees admission before proceeding to native preparation.

The machine retains the resulting owner until completion cleanup and payload
destruction. Detached forks share that owner without creating another admission
or refunding it when the parent is dropped. MLX carries supplied internal request
authority through its existing detached native recovery; public requests without
such authority remain explicitly unbudgeted.

Neutral conformance checks local/peer startup rejection, original error sources,
host spare capacity, no controller advancement during preparation, completion
retention, detached fork lifetime, and ordinary/controlled output parity. The
native fixture compares host-token and supplied-request startup through two
generated tokens and rejects a foreign execution identity or changed chunk limit
before model work.
These tests verify the shared preparation mechanism, not a complete native quote.

Complete native existing-residency/materialization accounting and shared-domain
pool selection remain unfinished. Dynamic controllers, controlled delivery's
prompt copies, capture, media, speculative lanes and copied state still need
enclosing reservations. Returned token/score handles also need accounting when
callers retain them beyond generation teardown or session reset. An opaque
prepared input has already incurred its input allocation cost before entering
this constructor. No total-process memory
guarantee follows from these managed-resource contracts.

Validation passes 946 tests: 254 core, 555 runtime, 21 evaluation, 106 portable
facade/conformance and 10 native startup, recovery, cancellation and input-failure
tests. One existing checkpoint-dependent facade test remains ignored. The native
test target builds with Metal enabled and the backend checks without default
features. Commands, log hashes, source hashes and limitations are recorded in
[`bounded-shared-preparation-2026-09-13.json`](validation/bounded-shared-preparation-2026-09-13.json).

## Retained checkpoint source payloads

`CheckpointSource::source_storage` now reports source-owned host payload capacity
without reading or converting tensors. Physical owners remain retained while
their bounds are merged, preserving identity and deduplicating shared source
roles. Restricted, prepared and resolved views cannot discount inaccessible
payloads that their underlying source still owns. Memory-backed checkpoints
report vector capacity, including spare space. Load-time quantization reports
both original source storage and packed host output storage.

SafeTensors caches hold weak payload references; live leases are separately
owned operation resources. GGUF file readers now use an explicit 8-KiB buffer,
and sources report the configured reader-cache ceiling, capped by their fixed
reader slots, even before readers open.
Prepared architecture blueprints aggregate primary, companion, complete, target
and prediction source roles once per physical owner. Unknown custom sources
propagate an unknown combined bound.

This supplies the host-source part of existing-residency accounting. It does not
price native weight/state buffers, temporary leases or conversion work, catalog
metadata, operating-system file cache, or other process memory. Complete native
inventory and shared-pool admission remain required.

The neutral reservation fixture composes actual source capacity with the existing
pool baseline: the planner chooses smaller chunks, a second live request is
rejected, and source storage remains charged after request completion. Ordinary
and controlled scheduling produce the same nonzero output. This is conformance
of those contracts, not automatic public-native pool registration.

Validation passes 800 tests: 104 checkpoint, 54 GGUF, 621 architecture, 20 bounded
prefill and one native quantization test. One existing manual scaling test remains
ignored. The backend checks without default features and its native test target
builds with Metal enabled. Commands, hashes and limitations are recorded in
[`bounded-source-storage-2026-09-13.json`](validation/bounded-source-storage-2026-09-13.json).

## Native residency storage inventory

The MLX residency manager now exposes its retained physical payloads, including
allocator padding, immutable host-transfer capacity and exact primary/per-unit
source stores. `RetainedStorage` owns the handles while merging inventories, so
array views, manager aliases and shared source roles count once. Evicting a
logical residency entry does not release storage still retained by an inventory.

The query bypasses recovery reaping and performs no native evaluation, completion
poll, payload read or materialization. In-flight and failed transfers make its
bound unknown; their recovery owners can hold additional resources outside the
manager. Lazy and unrecognized custom-backed arrays remain unknown rather than
being priced by their logical size. Certified MLX host-transfer owners use the
same identity for host buffers and completed array aliases.

This describes current manager-owned payloads, not all model memory or future
allocation. Module replacements, parameter banks, decoder state and escaped
outputs must still be included, and future materialization needs its workspace
quote. Metadata, allocator caches and process memory are outside this inventory.
It does not yet register public native requests with a shared working-memory pool.

The synchronous transfer boundary finalizes array readiness after exact native
completion and before publishing the resident generation. Completed-event markers
therefore do not force a later cold query to perform hidden completion work.

Cross-unit aliases use the neutral controller's validated owner. Native preflight
checks the physical owner's source recipe, and reacquiring an alias retains one
charge for the shared host allocation. This path is now covered by an enabled
native regression, including eviction and reacquisition.

Validation passes all 57 native residency tests with no ignored cases, including
six new inventory tests and the newly enabled cross-unit alias regression. The
Metal test target builds and the backend checks without default features.
Commands, source/log hashes and limitations are recorded in
[`bounded-native-storage-2026-09-13.json`](validation/bounded-native-storage-2026-09-13.json).

## Retained target parameter values

The shared runtime can now traverse retained target-module values using immutable
access at a resolved session boundary. The retained architecture supplies static
modules, and the paired policy supplies resident units or stored reload overrides.
Replicated, routed and partitioned execution use this same traversal. Unloaded
weights are not acquired, checkpoint sources are not reopened, and unresolved or
active transactions reject inspection before any visitor runs.

MLX imports the resulting native values into the physical storage inventory.
Aliases deduplicate by actual allocation, while a retained inventory keeps old
owners alive through later replacements. This is the module-value component of
existing residency, not a whole-model estimate: source/residency buffers, banks,
prediction modules, decoder state and escaped outputs remain separate owners.
Public native shared-pool admission and complete future workspace composition are
still required.

Resident-policy construction now finalizes its weight transfer before returning
loaded modules. Merely ordering a future stream dependency is insufficient to
provide settled physical storage to admission; the completion happens during
construction, while cold inspection remains free of native work.
The manager's synchronous acquisition path finalizes the same readiness before
publishing storage without a caller-owned transfer.

Validation passes 715 tests: 656 runtime tests, the parameter-sensitive TP/PP
matrix, the native target-storage matrix, and 57 native residency regressions.
The neutral matrix covers 18 configurations and 48 rank sessions; the native
matrix covers nine dense/tied/packed and residency combinations. The Metal test
target builds, the backend checks without default features, and the portable
neural crate checks with all features. Commands, source/log hashes and scope
limitations are recorded in
[`bounded-retained-parameters-2026-09-13.json`](validation/bounded-retained-parameters-2026-09-13.json).

## Composed target storage

The MLX target can now combine retained module values, the actual prepared
weight manager and sources, addressable-bank pools, and current decoder state
into one physical payload inventory. Runtime guards immutable inspection of
mechanisms and state at the same resolved session boundary. The native adapter
retains the manager selected during construction instead of deriving physical
capacity from a frozen telemetry report.

Paged state includes its sealed device blocks, immutable host-transfer buffers
and buffered shard payloads as well as layer tails. All actual retained managers
participate, including separately restored namespaces. Buffer aliases are
deduplicated, and a scoped bank handle includes the entire physical pool it
keeps alive. Pending or retiring cache operations and background failures keep
the bound unknown. Inspection performs no payload reads, materialization,
completion polling, recovery, or eviction.

This remains a target-side inventory, not a full request admission: prediction
modules/prototypes, observers, escaped outputs and enclosing request resources
still need composition and shared reservation coverage. The retained inventory
can itself keep old buffers alive after eviction or replacement. Future cache
growth, streaming work and execution temporaries require separate bounds.

Cache-worker accounting covers retained roots after completion publication as
well as records still marked in flight. The ordinary target completion boundary
finalizes native state-array readiness, without changing cold inspection.
The initial combined inventory exposed a gap in host-layerwise and disk-streamed
Metal execution: MLX shares host-transfer backing through a custom array deleter.
Without a certified owner identity, those completed arrays correctly remained
unknown. The native contract described below supplies that identity; logical
tensor bytes never substitute for it.

Validation of that initial inventory passed 766 tests: 656 portable runtime tests, the nine-configuration
target test, 58 native residency/bank tests, and 51 cache tests including all seven
explicit native/Metal transfer tests. That target matrix checked three certified
resident inventories and six explicitly unknown host-layerwise/disk-streamed
inventories after prefill and two cached decodes. The Metal test target builds
and the backend checks without default features. Commands, source/log hashes and
limitations are recorded in
[`bounded-target-storage-2026-09-13.json`](validation/bounded-target-storage-2026-09-13.json).


## Certified shared host backing

The native wrapper now recognizes MLX's own host-transfer owner through a named
retaining deleter and verifies that the array's backing buffer belongs to that
owner. It reads the owner's complete charged capacity and opaque identity without
polling, evaluating or waiting. Arbitrary custom deleters and unfinished arrays
remain unknown. A buffer and any completed array/view that shares its allocation
report the same identity; separate allocator domains cannot collide.

The physical inventory retains both array and host handles while charging that
shared allocation once, regardless of insertion or merge order. Portable state
projection receives the same deduplicated backing capacity. The original host
handle may be dropped while an array or inventory continues retaining the native
owner. This fills the shared-host identity gap in target accounting; it does not
provide request reservations or price future workspace, copied/escaped outputs,
prediction resources, or observers.

Validation passes 118 tests: the three allocation-identity tests under both
CPU-only and Metal builds, all 59 native residency/bank tests, all 51 cache tests
including explicit Metal transfers, the native workspace-projection regression,
and the nine-configuration target test. Every target configuration now has a
certified retained-storage bound after prefill and two cached decodes. The Metal
test target builds and the backend checks without default features. Reproducible
commands, native patch identity, source/log hashes and remaining scope limits are
recorded in
[`bounded-host-identity-2026-09-13.json`](validation/bounded-host-identity-2026-09-13.json).


## Retained prediction owners

The prediction executor now supplies architecture-owned immutable traversal of
its physical modules and retained state prototypes. This covers DeepSeek V3,
DeepSeek V4 sequential and DSpark execution, Inkling, Qwen hybrid and Nemotron-H.
The native inventory includes current module values, unloaded placeholders and
replacement tensors, exact prepared sources, weight-residency buffers and every
prototype's cache storage. The replicated/composite session guard prevents this
inspection while a submission is unresolved. Traversal does not load parameters,
clone a prototype, poll completion or reopen a checkpoint.

Prediction and target inventories retain and deduplicate physical backing when
merged. Lazy replacements remain unknown, and inventory snapshots keep their
original owners alive through later restoration and executable destruction.
This adds retained extension storage to admission evidence. Active prediction
lanes, observers, escaped output/copy owners, future workspace and shared request
reservations are still required for complete public budget enforcement.

The native regression exposed lazy full-shape constructor placeholders as an
additional retained owner. Unloaded slots now contain completed broadcasts of
scalar zeros, preserving parameter shapes and dtypes with bounded scalar
backing. Actual prediction invocations continue to bind their checkpoint leases
and active replacements before execution. Cold inventory neither evaluates the
old constructor graphs nor allocates full placeholder parameter buffers.

Validation passes 34 focused tests: 12 neutral prediction tests and 22 native
CPU/Metal tests, including explicitly enabled Metal cases. The inventory matrix
covers seven prediction variants across all three weight-residency modes (21
configurations), plus nine target configurations with no embedded extension.
Native regressions exercise nonzero state prototypes, scalar-placeholder geometry,
shared replacements, missing checkpoint paths, speculative components,
interventions, quantized prediction parameters, and ordinary prediction execution.
The backend also checks without default features. Reproducible commands,
source/log hashes and scope limits are recorded in
[`bounded-prediction-storage-2026-09-13.json`](validation/bounded-prediction-storage-2026-09-13.json).

## Preparation authority and retained token charges

The shared runtime now arbitrates preparation and direct prefill through one
request-start state. A successful preparation claim precedes native allocations
and binds startup sampling configuration. Its finite output allowance must match
the admitted geometry before startup can be claimed. Move-only prompt/sampler stage permits
retain the request charge until completion or failure. Prompt construction and
binding are separate transitions, so a cloned handle cannot publish readiness
while construction is unresolved. Failure does not refund authority, and the
original retention handle cannot bypass preparation to start prefill.

MLX sampling state and returned token owners retain the same request charge.
This extends its lifetime beyond release of the session's submission lease:
returned tokens remain charged after generation and session destruction. Sampler
restoration merges charge owners, and sampler/prompt copy and reseed recovery
hold independent retention through native work. Pending prompt copies preserve
chunk policy and request identity instead of reverting to unbudgeted input.

This closes startup reuse and sampled-token retention gaps for supplied internal
admissions. Ordinary public calls still need automatic complete native quoting
and shared pool composition. Active prediction/observation resources, escaped raw
logits, independent copy/branch admission, and complete future workspace remain
required. Retaining an existing charge does not establish the cost of a new
allocation, and a copied request handle cannot authorize an additional prefill.

## Shared retained-storage accounting

The working-memory pool now registers changing physical inventories as well as
request reservations. Overlapping inventories count each certified backing
allocation once. Both kinds of admission use one lock and one capacity, so
concurrent requests cannot spend space already registered to loaded resources.
Capacity conflicts, arithmetic overflow, and over-budget registrations reject
without partial charges or an artificial increase in the historical peak.

The native adapter preserves distinct identities for native allocations, source
payloads, and immutable byte buffers. Host and Metal views of the same certified
allocation share one identity. Registered inventories retain their physical roots
through cloning and model destruction. Final handle destruction defers cleanup
outside native locks, retaining the charge until that cleanup actually runs.

This supplies shared existing-storage accounting for complete request admission;
the public session/domain owner still needs to install and retain registrations.
The fixed pool baseline must be disjoint from registered storage. A registration
neither proves a future workspace bound nor admits a new allocation. Strict
unknown-bound rejection remains necessary, and automatic complete admission,
active speculative/observation resources, escaped raw logits, and independent
copy/branch budgets remain unfinished.

Validation passes 879 tests: 664 runtime tests, 105 native session/storage/recovery
tests, four checkpoint storage tests, and 106 portable facade/conformance tests.
The native registration matrix covers seven prediction variants across resident,
host-layerwise, and disk-streamed storage, including overlapping target/prediction
registrations and replacements (21 configurations). Existing target inventory
coverage also passes for nine dense/tied/quantized residency configurations.
CPU/Metal tests cover physical aliases, nonzero retained payloads, ordinary and
controlled sampled-token lifetime, and safe deferred cleanup. The backend checks
without default features. One portable tokenizer/template test remains ignored
because its optional LFM checkpoint environment variable is unset. Commands,
source/log hashes, scope limits, and the distinction between synthetic admission
charges and measured native payload capacities are recorded in
[`bounded-request-residency-2026-09-13.json`](validation/bounded-request-residency-2026-09-13.json).

Resident pooling inspection now includes the local persistence sentinel, partial
source windows, complete pooled history and overlap buffers. Runtime and native
construction share declaration validation. Native projection preserves full
backing identities and unknown lazy allocation capacity without evaluating or
replaying state.

The retained quote driver now constructs gated, ReLU-squared and pooling routed
models and direct/routed composite text. Composite spans use ordinary input
admission with metadata tensor identities; evaluated media metadata is never
invented. Pooled/indexed attention and multi-axis rotary geometry are traceable,
but native bounds not yet supplied remain unknown. Independently cached expert
banks have distinct gather/member-chunk/scatter execution: their quotes explicitly
remain incomplete until that selected strategy is priced. Resident grouped
calculations do not substitute for those costs. These are incremental quote-path
changes, not completion of whole-request public budget admission.

This inspection exposed a cached Muse continuation error: full-history causal
mask widths outlived the retained sliding keys. Layer-local mask construction now
uses the actual visible suffix, while rotary coordinates remain absolute.
Partition ingress preserves that per-layer construction. Empty pooled rotary
history also bypasses invalid native inferred reshapes; its cold allowance still
covers frequency construction.

Validation passes 883 tests: 751 neural/runtime tests, nine architecture tests,
17 focused native tests, and 106 portable facade/conformance tests. Architecture
fixtures include 54 routed and 72 composite text quote trajectories; native
tests cover live cached continuation, pooling restoration and rotary allocation
bounds. The backend also checks without default features. One optional released
LFM tokenizer test remains ignored. Commands, source/log hashes and remaining
scope limits are recorded in
[`bounded-pooling-workspace-2026-09-13.json`](validation/bounded-pooling-workspace-2026-09-13.json).

Native local/pooled attention, indexed attention, pooled-position selection and
pooled-mask gathering now have tensor and host workspace facts. They compose the
selected native implementation, including einsum lowering, matrix workspace,
mask normalization, shared sink softmax, reductions and sorting buffers. Top-k
selection retains the full sorted-index buffer. Broadcast source views do not
incur a fictitious replication of the full pooled bank. Mixed Boolean/additive
mask normalization preserves eligibility and neutral values for absent masks.
Selected addressable expert execution remains incomplete, as does complete
public request budget admission. Positional bounds are described below.

The native validation covers 648 configurations across F32/F16/BF16, strided and
contiguous inputs, empty local or pooled history, per-batch/head broadcast masks,
Boolean/additive mixtures, sinks and 4,097-position pooled banks. Outputs match
independent host equations and every measured active-allocation peak fits its
cold bound. Native top-k backing capacities also fit the retained allowance.
Eight focused tests and 36 workspace regressions pass, and the backend checks
without default features. Reproducible commands, source/log hashes, tolerances
and remaining scope limits are recorded in
[`bounded-native-pooling-2026-09-13.json`](validation/bounded-native-pooling-2026-09-13.json).

Relative-profile attention and multi-axis rotary now have native tensor and host
workspace bounds. Relative attention includes integer masks, profile gathering,
optional logarithmic scaling, head expansion, dense products and softmax. Rotary
pricing follows each selected layout and separates the temporary Rust frequency
vector from its native copies. Round-robin execution avoids constructing unused
independent-axis graphs. Signed position offsets saturate consistently with the
neutral reference, and relative spans at the final I32 endpoint remain valid.

The retained Inkling resident text fixture now reports bounded equation workspace
before initial execution and after cached continuation. DeepSeek and Muse fixture
quotes still identify missing operator costs. Complete public budget admission,
grouped operators, addressable expert scheduling, materialization/transfers,
paging, parallel execution, media, observation, speculation and independent
copies remain unfinished; a finite equation quote alone does not price them.

Validation passes 579 native positional configurations: 432 relative-attention
cases and 147 multi-axis rotary cases. Relative inputs cover F32/F16/BF16,
strides, window and logarithmic policies, and integer origins beyond exact F32
coordinates through the final valid I32 endpoint. Rotary covers all three
layouts, heterogeneous axes, empty rows, strided I32/U32 inputs and signed
saturation. Numerical references, measured active-allocation peaks and retained
rotary backing checks pass. CPU and Metal paged relative-attention regressions
also pass. Nine focused tests and 38 workspace regressions pass (45 unique tests),
and the backend checks without default features. Commands, source/log hashes,
tolerances and scope limits are recorded in
[`bounded-native-positional-2026-09-13.json`](validation/bounded-native-positional-2026-09-13.json).

Native multi-stream residual collapse/expansion and the final learned head now
have tensor and host workspace facts. The bounds include F32 RMS preparation,
dense and batched products, coefficient construction, every selected Sinkhorn
normalization pass, strided reshape copies and retained coefficient outputs.
The final-head observer boundary remains explicit in the quote. These operations
have no disjoint host numerical payload; native scalar arrays remain in the
tensor domain. Pooling contractions share the same dense-einsum derivation.

Empty multi-stream batches preserve their shapes, dtype and coefficient-observer
callback while bypassing empty native reductions that trigger an MLX assertion.
Grouped expert selection/execution and complete public budget admission remain
unfinished, alongside the previously listed execution-path composition gaps.

Validation covers 180 multi-stream configurations with 540 separately measured
collapse, expansion and final-head allocation peaks. Fixtures include one and
multiple streams, uneven batch/token geometry, long RMS rows, zero-token spans,
three Sinkhorn pass counts, F32/F16/BF16 activations, both full- and lower-precision
parameters, and contiguous/strided storage. Independent F64 equations check the
coefficients, mixing matrix and output; actual retained backing fits the quoted
allowance. DeepSeek's cached fixture now has four unpriced operations, down from
twelve before these facts, and continues to report an incomplete quote.

Five focused tests, 40 workspace regressions and the 648-case pooled/indexed
regression pass (44 unique tests). The backend checks without default features.
Commands, source/log hashes, tolerances and remaining limits are recorded in
[`bounded-native-hyper-2026-09-13.json`](validation/bounded-native-hyper-2026-09-13.json).

Workspace operations can now declare that an output shares an earlier output's
backing allocation. The trace validates backward references before mutation,
preserves all possible input owners, and charges one physical allocation across
different logical shapes and surviving sibling views. Joint routing uses this
contract for selected and always-on coefficient slices, while selected IDs keep
the complete partition-index buffer. The native quote includes dense projection,
corrected sigmoid ranking, partitioning, unbiased-logit gathering, shared
normalization and scaling. No disjoint host numerical payload is constructed.

This completes the joint-selector operation fact, not the enclosing routed
model quote or request admission. At this validation point ordinary top-k
selection, grouped execution, addressable expert scheduling and the other listed
path/accounting gaps remained; the next increment below covers top-k selection.

Validation passes 240 native joint-routing configurations across F32/F16/BF16,
full- and lower-precision parameters, contiguous/strided inputs, multiple leading
ranks, tied rankings, all-group selection, empty batches and 4,097 selectable
groups. Independent F64 equations check unbiased coefficients and valid top-k
cutoffs; active-allocation peaks fit the bounds. Native allocation identities
confirm that both coefficient views share their complete backing. The neutral
tests also preserve unknown input capacity through output aliases, so sharing
cannot turn an incomplete inference estimate into a complete one.

All 88 neural-layer tests pass with all features, along with five focused native
tests and 42 workspace regressions (133 unique tests). The backend checks without
default features. Commands, source/log hashes, tolerances and scope limits are
recorded in
[`bounded-native-joint-routing-2026-09-13.json`](validation/bounded-native-joint-routing-2026-09-13.json).

### Ordinary top-k routing and intervention workspace

Top-k selection now has a cold bound for dense, affine, MXFP4 and native GGML
projection, optional input normalization and learned scales, all four scoring
policies, grouped eligibility, normalized coefficients and caller-supplied IDs.
The bound retains the full native partition-index allocation. It includes tie
detection and the possible additional CPU-stream partition without counting
shared native storage as a second host payload. Guaranteed score/coefficient
aliases share one root; supplied index views retain their possible input roots.

Intervention accounting executes the existing neutral intervention driver over
native primitive costs. It therefore includes the same raw-logit, transformed
score and ranking adjustments; exclusions, forced rows and zeroed contributions;
original-decision capture; and successful validation predicates. Existing request
control arrays are borrowed; operation-owned host vectors are priced separately.
Validation avoids allocating duplicate ID sets. Empty routers skip invalid MLX
reduction calls while retaining scale arithmetic and the selected output dtype.

These are selector-operation bounds. Grouped expert execution, addressable
expert residency, complete public admission and the remaining path/accounting
gaps still prevent a general complete routed-model budget claim.

Validation covers 2,592 cold configurations across 36 encodings, plus 552 native
configurations with independent F64 equations and measured active allocations.
Every native peak fits its cold bound. Coverage includes multiple selected
partitions, one- and two-entry partitions, 4,097-way tied selection, empty batches,
all-group selection, strided inputs, three floating precisions, supplied IDs and
every intervention action with and without original-decision capture. Native
affine, MXFP4 and Q8_0 fixtures use independently decoded packed weights.

The eight focused tests pass alongside 88 neutral neural-layer tests and 45
workspace regressions (138 unique tests). The backend also checks without
default features. In the prepared cached fixtures, DeepSeek and Muse now retain
only the grouped-execution operation gaps; the dense Inkling fixture remains
bounded. Commands, source/log hashes, tolerances and scope limits are recorded
in [`bounded-native-topk-routing-2026-09-13.json`](validation/bounded-native-topk-routing-2026-09-13.json).

### Packed expert execution workspace

Native grouped linear, gated-product and ReLU-squared operator facts compose
selection sorting, hidden/weight gathers, actual grouped projection mechanisms,
activation, weighted reduction and TP bias correction. The gated implementation
shares its 64-token threshold and 32-token chunks with the estimator. Its graph
retains all chunks through completion; the estimate includes every chunk and the
final concatenation. Group-size-16 affine projection explicitly prices its real
per-selection packed-weight and companion replicas. Other supported gather
projections do not acquire a fictional selection-sized dense bank.

Unit-boundary estimates preserve possible complete fused gate/up backing and
separate observer-produced tensors from the final projection. The neutral
observer trace now consumes the selected native delivery schedule, so gated
observations above 64 tokens receive the same 32-token callbacks and short tail
as execution. Shapes, offsets, original/intervention/effective order and all
replacement owners survive the boundary. A missing schedule is a typed quote
failure before callbacks. Independently materialized expert providers remain
unpriced until their real member and residency schedule is traced. This is
unfinished accounting, not a model limitation. Complete public admission and
the other listed path-accounting work remain.

Validation passes 2,336 cold configurations across 56 encodings and 635 native
configurations, including strided inputs, F32/F16/BF16, empty batches, uneven
native chunks, sequential reduction, FP8 row-block origins, unit replacement
and TP bias separation. Independent equations check numerical outputs; every
measured active native allocation peak fits the corresponding bound. The
prepared DeepSeek, Muse and Inkling fixtures now have finite initial and cached
workspace bounds. This establishes those equation traces, not complete request
admission or the entire family/path matrix.

Six focused tests and 48 workspace regressions pass (51 unique tests), and the
backend checks without default features. Commands, tolerances, source/log hashes
and scope limits are recorded in
[`bounded-native-grouped-2026-09-13.json`](validation/bounded-native-grouped-2026-09-13.json).

The subsequent observer-schedule validation expands this to 2,780 cold and 911
native configurations, including 315 with unit observers. Native and metadata
callbacks agree on original/intervention/effective order, shapes, absolute token
offsets and total invocation length, including empty input and uneven tails.
Independent numerical and active-allocation checks pass throughout. Neutral
tests verify retention of all replacement owners and early failure without
refunding prior workspace. Missing schedules preserve a typed neutral error.

All 91 neural-layer tests, 12 architecture workspace tests, six focused native
tests and 48 workspace regressions pass (154 unique tests); the backend checks
without default features. The architecture regression fixture also now explicitly
rejects an unexpected pooling-state variant in its compressed-attention checks.
The earlier record preserves the gap as it stood before this schedule change;
the current evidence is
[`bounded-grouped-observer-schedule-2026-09-13.json`](validation/bounded-grouped-observer-schedule-2026-09-13.json).

### Logical host copies of retained tensors

Owned host reads and serialization now follow signed native strides directly.
Capacity-backed prefixes, transposes, broadcasts and reversed views preserve
their logical values without a tensor-sized staging allocation on Metal. Unaligned values
are copied without creating misaligned Rust references. Byte export produces one
host output; value equality produces no payload. Borrowed slices require logical
row contiguity, and empty reads remain empty. Safe strided-view construction
checks source bounds and signed arithmetic before publishing the native graph.
Observation integer widening and routed capture/intervention coordinates use the
same borrowing iterator, keeping existing host-result budgets and native error
sources intact.
Owned vectors and widened observations preallocate their exact logical capacity,
including scalar results, instead of inheriting generic collection growth slack.

This is a prerequisite for correct retained-state copying and serialization.
It does not complete public request, independent snapshot or capture admission.

Validation passes 568 wrapper tests with SafeTensors enabled, ten focused native
host-read tests and 60 cache/observation/capture/intervention tests (629 unique
tests). All fourteen native scalar representations are covered. Sixty-five
observation configurations verify logical values and exact output-vector
capacity, including scalar and empty results. Five completed Metal layouts show
no native allocation during host iteration, byte export or equality. Snapshot
independence checks compare backing-allocation identities directly.

The backend also checks without default features. Commands, source/log hashes,
native source references and scope limits are recorded in
[`bounded-logical-host-reads-2026-09-13.json`](validation/bounded-logical-host-reads-2026-09-13.json).

### Centroid-selected prediction readout

The MLX masked-output primitive previously used one minimum over the complete
batch and sequence. Its independent scalar reference uses a separate selected
minimum for every position. The native reduction now follows that contract, so
partitioning a sequence cannot alter the masked scores of another position.
Empty batches and sequences return an empty F32 score tensor without invoking
native selection or reduction; positive hidden, vocabulary and centroid geometry
is validated by the same portable function used in cold inspection.

The metadata tensor executes this primitive through an explicit descriptor.
The selected Metal bound prices the complete centroid partition, ordering and
index transformations, actual gathered weight replicas for every position,
dense products and their casts/copies, per-position minima and dense F32 fill
and scatter. All lazy children are conservatively retained through completion.
Disjoint managed host payload is explicitly zero. This removes an unpriced
operation from ordered-embedding assistant traces; complete prediction request
admission and retained capture/branch accounting remain unfinished.

Validation passes all 92 neural-layer tests, 13 architecture workspace tests and
six focused native tests (111 unique tests). The native matrix covers 160
configurations across F32/F16/BF16 and mixed weights, I32/U32 token ordering,
strided inputs, empty axes, all-centroid selection, multi-pass sorting and long
selected-score reduction. Every measured active-allocation peak and output
backing fits the cold bound. Sixty uneven-chunk comparisons preserve full output;
the multi-row CPU regression also matches the scalar reference. Architecture
traces exercise three draft steps with both ordinary and ordered assistant heads.
The backend checks without default features. Commands, hashes, tolerances and
scope limits are recorded in
[`bounded-masked-readout-2026-09-13.json`](validation/bounded-masked-readout-2026-09-13.json).

### Concurrent request capacity limits

`WorkingMemoryPool::reserve_with_capacity` binds a complete managed-domain ceiling
to the same reservation that retains execution authority. All later reservations
and storage registrations obey the minimum of the pool capacity and every live
request ceiling. A larger later budget cannot raise a smaller live limit. Equal
limits from independent requests have separate ownership; clones share one owner.
Retirement removes a limit only when its last reservation, completion, recovery
or retained-state reference has retired. Historical peak usage is never reset.

`plan_prefill_with_capacity` reuses the ordinary planner and retries smaller
chunks through the atomic limited-reservation path. Its limit includes the fixed
baseline, registered unique storage and concurrent requests. A limit already
below live usage returns `CapacityBelowUsage`; unknown workspace still rejects
before accounting mutation. Core's existing incremental application-request
budget remains a separate constraint.

This supplies the concurrency contract needed for public request admission.
Public policy, physical-domain installation and complete native request quote
composition remain in progress; this change does not claim their completion.

Validation passes 33 bounded-prefill tests, 558 runtime unit tests and 78 neutral
backend conformance tests (669 unique tests), plus the backend check without
default features. Focused cases cover overlapping storage inventories, independent
and cloned limit owners, atomic rejection, concurrent limits and budget-selected
uneven chunks. Ordinary/controlled numerical parity retains the same domain
ceiling; cancellation and failed completion polling keep it until safe release.
The record is
[`bounded-request-capacity-2026-09-13.json`](validation/bounded-request-capacity-2026-09-13.json).

### Public policy, transaction costs and unquoted domain owners

`TextInferencePolicy` is carried by `TextGenerationConfig` and facade
`PreparedChatGenerationSettings::inference`. Its optional prefill maximum reaches
both raw-token and prepared-input ordinary/controlled MLX startup. A configured
maximum larger than the prompt is harmless; the shared driver chooses the actual
span. No maximum uses the existing runtime policy. Controlled configuration
identity includes both the chunk policy and managed capacity.
An already admitted smaller chunk is preserved whenever it fits the requested
maximum; a tighter maximum requires new admission and rejects a mismatched grant.

A managed capacity requires a finite output allowance before native preparation.
The facade supplies its existing 256-token default when no allowance was resolved.
Direct core requests without a finite allowance fail during coordinated admission,
without advancing a controller or constructing prompts or samplers. Controllers
can supply cold filter geometry and additional retained/growing numerical payload;
absence is an unknown bound. The facade's plain-text controller describes its
validity mask and history growth; dynamic grammar variants remain unpriced.

Public MLX managed-capacity requests currently return `WorkingMemoryError::UnknownBound`.
A supplied private reservation cannot certify the public shared domain. Prepared
speculative facade requests carrying either new policy return
`InferencePolicyUnavailable` during coordinated host preparation, including a
policy on a later batch lane. Direct native speculative entry also validates all
lanes during coordinated admission before sampler or draft execution. These are
implementation gaps, not architectural limitations; the complete goal still
requires these routes to accept and enforce applicable policies.

Prepared resident equation quotes now include transaction checkpoints and a
conservative late-failure rollback. Exact-concatenation ordinary/fixed state and
resident pooling state share their immutable buffers. Compressed state copies
both capacity buffers and logical views; rollback copies the checkpoint again.
All copies remain in the transient graph until completion, while successful
closing state alone supplies the retained-state bound. A missing copy fact keeps
the quote unknown, and cold quoting leaves the supplied state unchanged.

`RuntimeStateEstimate::selected_state_backing` records complete selected decoder
backing separately from logical fixed/context bytes. Explicit refinement takes
the larger logical/backing bound and then adds existing media terms, preserving
semantic categories and overflow checks. It never upgrades missing state/media
coverage. The selected backing retains exact request geometry, checked against
execution workspace in either attachment order and again at admission.
Prepared text composition combines that backing, all transaction and
sampling spans, full host/native prompt storage, and controller payload. Enclosing
materialization, captures, snapshots and prediction remain mandatory inputs;
unknown components are preserved. Native validation now exercises this composition
with real selected mechanisms instead of an arbitrary large request envelope.

`WorkingMemoryPool::acquire_unquoted` provides a shared barrier for owners or work
whose memory has not yet been completely accounted for. It atomically excludes
live request reservations, including zero-byte reservations; reservations reject
while any unquoted owner survives. Cloned leases retain a single blocker. Exact
storage may be registered while the blocker remains live, allowing safe publication
before the last unquoted owner retires. Used/peak counters describe known charges;
`unquoted_owner_count` explicitly identifies incomplete domain accounting.

Native installation still needs participation before model loading, independent
prompt/sampler preparation, session operations, and copies, followed by exact
inventory publication or safe destruction. Idle sessions retain decoder state,
and raw output/native array escapes need allocation-lifetime ownership. A pool
attached only to the requesting session would miss those owners. Independent
snapshot/capture admission, incremental state-charge promotion, streaming and
parallel materialization, media and prediction execution remain outstanding.

Validation passed 225 unique tests: seven core capability tests, 41 bounded-prefill
tests, 49 focused runtime tests, seven prepared-architecture tests, 111 portable
facade/conformance tests, eight native preparation tests and two Metal composition
tests. The architecture matrix includes 297 request trajectories and 12 compressed
copy-gap cases; native chunk policy covers 16 ordinary/controlled and raw/prepared
combinations. The CLI no-default-features plus MLX check passed. One existing
facade test remains ignored, and existing unused-helper/linker warnings remain.
No new released-checkpoint or memory-peak measurements were added. Commands,
source hashes, results and outstanding integration are recorded in
[`bounded-public-policy-2026-09-13.json`](validation/bounded-public-policy-2026-09-13.json).

### Native domain participation and allocation ownership

All production `MlxBackend` instances now share one local managed host/device
domain. Loading acquires an unquoted owner before native communication and model
materialization. The owner survives the prepared model, extracted executable and
session, including deferred teardown. Detached native recovery retains it through
failed or unwinding materialization. A live quoted reservation excludes a new
unquoted load even when its byte charge is zero. Until complete inventory
promotion exists, the loaded model conservatively prevents quoted admission.

The safe native wrapper can attach a `Send` owner to completed certified physical
backing. It preserves existing allocation identity and capacity and survives views,
lazy children retaining that backing, native recovery pins and buffer donation.
Host-transfer owners use shared host storage rather than individual array wrappers.
Failed or unavailable attachment returns the supplied owner. Attachment neither
evaluates an array nor authorizes an independent result allocation. Native final
release enqueues a preallocated retirement node; ordinary host reclamation drops
the Rust owner outside native locks, retaining unprocessed nodes across unwinding.

The shared session output seam attaches the domain owner before publishing model
logits or raw output arrays. Tensor/array conversions and native views retain that
physical attachment. Opaque token submission owners separately keep the domain
owner after their session payload and submission lease become releasable.
Final prefill output indexing completes its new lazy view before leaving the
existing request's native recovery scope. This preserves completed output
publication without evaluating arrays from a cold allocation query.

Public strict managed budgets still reject unknown bounds. Exact inventory
promotion, independently prepared prompts/samplers, snapshots and copies, observer
callbacks retaining lazy intermediates, and distributed coordination of admission
remain incomplete. The loading guard is local; it is not yet a rank-coordinated
capacity decision. Complete paging, parallel, media, capture and speculative
admission remain part of the goal. No new numerical-reference or peak-memory
measurements accompany this ownership increment.

Validation passed 617 unique tests: 558 native-wrapper tests and 59 backend
integration, preparation and recovery tests. The seven focused allocation-owner
tests exercise CPU and Metal and are included in the wrapper count. Two existing
wrapper tests and four backend tests remain ignored. The backend check with
Metal, image and audio features passed; existing unused-helper/linker warnings
remain. Commands, hashes and corrected validation findings are recorded in
[`bounded-native-domain-2026-09-13.json`](validation/bounded-native-domain-2026-09-13.json).

### Independent storage charges and durable allocation identities

`WorkingMemoryPool::register_storage_individually` admits a complete deduplicated
inventory atomically and returns a separate charge for each identity. Grouped and
individual registrations share the same typed registry and capacity limits.
Validation and provider key cloning precede mutation; failed admission changes
neither usage nor peak. Final key destruction runs outside the accounting lock
while its bytes remain charged. This permits reentrant destruction and preserves
coverage for allocation storage retained by identity tokens.

Native physical identities now use non-reused process-local generations rather
than addresses. Views, donation and host-buffer aliases share their backing's
generation. Exhaustion cannot wrap into an earlier identity. Checkpoint identities
and buffered byte identities pin their Arc allocations with weak ownership.
Strong payload destruction can still occur, while inline storage remains reserved
until the last weak key retires. Neither kind of identity is persistent across
processes. This is necessary because native charge destruction is deliberately
deferred until after physical backing has retired and its address may be reused.

Backend publication attaches a payload-free individual charge to every certified
native allocation, including host-only immutable transfer buffers. It retains
source and buffered-byte roots separately, outside attached owners, avoiding
ownership cycles. Complete load-time nonstate inventories use this path. Missing
inventory coverage stays unquoted. Partial attachment failures cannot clear the
unquoted barrier, and already attached charges retain their valid backing lifetime.

This publication records existing nonstate storage; it does not yet promote a
model to fully accounted execution. Decoder-state roots still require exact
overlap handling with request workspace traces, and future operations require
their own admission. Complete quoted preparation/copy admission, observer escapes,
distributed coordination and complete paging/media/speculative admission remain
outstanding. Public strict managed budgets continue to reject unknown
coverage rather than treating these storage charges as a complete request bound.

Validation passed 681 unique Rust tests: 560 native-wrapper tests, six checkpoint
identity tests, five focused runtime unit tests, 44 bounded-prefill tests and 66
backend integration/preparation/recovery tests. The backend publication tests run
CPU and Metal variants. Three additional native test groups exercise concurrent
generation exhaustion, 64 deterministic address-reuse lifetimes and uncertifiable
generation rejection without damaging storage. The Metal/image/audio backend
check passed. Two wrapper tests and four backend tests remain explicitly ignored.
No new released-checkpoint numerics or memory-peak measurements were performed.
Commands, source hashes and corrected validation findings are recorded in
[`bounded-allocation-publication-2026-09-13.json`](validation/bounded-allocation-publication-2026-09-13.json).


### Independent ordinary preparation and snapshot ownership

Public native prompt and sampler preparation now acquire an unquoted owner from
the backend's shared managed domain before allocation. Reserved variants call
the same allocation workers under their one-use request authority. They do not
try to acquire an unquoted lease which would conflict with their own reservation.
Explicitly unbudgeted requests still acquire independent ownership. Reserved prompt
backing retains the existing request charge before the separate bind stage and
through escaped aliases; attached accounting authority contains no native roots.
Prompt token backing retains ownership before constructing the lazy batch-axis
view, protecting raw arrays escaped through the public borrowed-input interface.
Greedy and stochastic samplers retain ownership for host history and native RNG
state; sampling submissions preserve it through completion and recovery.

Ordinary state capture, sampler copies, pending-input copies and reseeding acquire
independent ownership before allocation. Copied prompt/token/RNG backing retains
that owner through native retirement. Native state snapshots remain encapsulated
and carry ownership after their payload; exchange moves owners with state after
all fallible preparation has succeeded. Sampler restore merges retained authorities
without creating duplicate leases or refunding older request charges. Cold snapshot
estimates include added ownership metadata; inspection does not materialize arrays.

A live reservation, even for zero bytes, rejects these independent allocating
operations with `ReservedWorkActive` before mutation. Empty pending input and
nonallocating temperature changes need no fresh owner. This is conservative
participation in the domain, not a successful quoted-copy implementation. Public
strict budgets still reject unknown coverage. Complete model/state promotion,
quoted independent copies, speculative preparation/copies, observer escapes and
the remaining paging, media and distributed admission paths remain outstanding.

Validation passed 72 focused native tests with Metal enabled, including ten new
preparation/admission/lifetime tests and an independent numerical copy/backing test.
Existing controlled continuation, recovery, raw-output ownership and ordinary versus
controlled preparation coverage also passed. The Metal/image/audio feature check
passed. No new reference-model or peak-memory measurement was required for these
ownership changes. Commands, source hashes and validation findings are recorded in
[`bounded-preparation-memory-2026-09-13.json`](validation/bounded-preparation-memory-2026-09-13.json).

### Speculative preparation, RNG and target snapshots

Prepared speculative execution validates every lane before acquiring an unquoted
owner in the existing coordinated admission stage. That owner precedes sampler,
seed and cache setup, remains through detached native recovery, and is bound into
the target/draft execution context. Greedy sampler clones retain it for host history.
The context carries the actual backend pool; standalone native contexts use the
same aggregate domain as ordinary backend instances.

Opaque native seed/random-state wrappers retain ownership through key splitting,
position-addressable draft keys, RNG mutation and controlled snapshot clones.
Reseeding acquires independent ownership before allocation. A key used in another
managed domain must acquire that domain's authority before derivation or mutation;
repeated operations in the same domain reuse their existing authority. Native
failure recovery retains its own handle instead of relying on the caller keeping
the source wrapper alive.

Portable `InferenceRetention` now also retains neutral unquoted leases. Existing
clone, import and restore paths preserve them, including empty or host-only state,
without adding MLX dependencies to portable crates. The architecture's embedded
target-state snapshot hook receives the complete mechanism context. Its native
implementation acquires independent authority before copying, retains a neutral
lease in the returned state, and attaches ownership to completed copied arrays.
External target-cache copies and autoregressive checkpoint/restore follow the same
rule; autoregressive state retains the bound domain for context-free checkpoint
calls. External restore merges installed ownership before replacing its state.

This advances ownership coverage but does not establish a complete native quote.
Standalone low-level logits/capture operators, complete model/state
promotion and quoted-copy admission, exact decoder-state overlap, and the remaining
paging, distributed, media and observation paths remain unfinished. Public strict
managed budgets still return `UnknownBound`; no unquoted owner is cleared by these
changes. No new reference-model numerics or memory-peak claim is made here.

Native identity checks exposed that MLX's ordinary `copy` may retain its source
allocation, and event completion alone does not certify the returned descriptor.
The shared speculative copy helper now constructs an explicit deep copy after
contiguous staging and establishes completion on the returned handle before
attaching ownership. Its estimate already includes staging and destination payloads.
CPU and Metal copy tests require different allocation identities and verify the
remaining native alias keeps ownership after source and wrapper retirement.

Validation passed 124 focused native tests, then all five explicitly selected
Metal split-stream scheduler tests, including CPU-draft/GPU-target execution and
stochastic acceptance. Four neutral retention tests and 44 bounded-prefill tests
also passed. Seven other native tests remain ignored in this increment: five
activation/component tests and two multiprocess ring tests. This increment adds
no released-checkpoint numerical or peak-memory measurements. Commands, source
hashes and corrected copy findings are recorded in
[`bounded-speculative-memory-2026-09-13.json`](validation/bounded-speculative-memory-2026-09-13.json).

Prediction-extension snapshots now use a separate borrowed snapshot context rather
than the tensor-only equation context. The architecture forwards it to each layer
in both target-plus-prediction and prediction-only copies. Native sequential and
pooling caches carry independent ownership in a wrapper, including when no array
exists. Copying admits the destination domain before native work, retains recovery
authority and attaches it to completed copied backing. Cache checkpoint and clone
preserve those handles; restore merges both histories before fallible mutation,
and clear does not refund the authority. Prediction model-state copies reuse the
same mechanism as target-state snapshots. Estimates include ownership metadata;
these changes do not provide successful strict native budget admission. The wrapper
path covers the materializer's resident sequential/pooling caches. Sealed paging
manager arrays require their own complete inventory and lifetime accounting.

Prediction snapshot validation passed 135 selected native tests and one explicitly
enabled Metal prediction-completion test. This includes eight new ownership tests,
with nonzero sequential/pooling copies and strided native aliases on CPU and Metal.
The neutral speculative production case passed with the new context-forwarding
checks; 111 portable facade/backend tests and the Metal/image/audio build also
passed. No new reference-model or peak-memory measurement was made. Ignored fixture
and native cases, commands, source hashes and validation findings are recorded in
[`bounded-prediction-snapshot-memory-2026-09-13.json`](validation/bounded-prediction-snapshot-memory-2026-09-13.json).

### Speculative distribution and callback ownership

Processed speculative distributions now keep their lazy native value and allocation
authorities together. Standalone logits processing, capture callbacks, probability
queries, residual construction, sampling and adaptive commitment use the same
domain admission and native recovery rules as prepared execution. Operations merge
source and destination authorities before invoking callbacks or changing state;
repeat use within an already retained domain does not create another lease.
Verification admits every input before native event/copy handoff, while empty and
other no-work cases return without acquiring authority.

Sampler clones share allocation-authority history. This is necessary because a
custom policy or shared capture state can retain native arrays even from a callback
taking `&self`. Dropping the clone that performed that work must not release its
domain while an earlier clone still holds the shared payload. The authority list
is not borrowed during callbacks. These changes preserve existing sampling and
controlled-session semantics; they do not establish complete byte quotes or account
for host capture records after delivery. Strict native managed budgets still reject
unknown coverage.

Distribution ownership validation passed 142 selected native tests plus all five
explicit Metal split-stream scheduler tests. Seven new tests cover admission
before callbacks/capture-ledger mutation, retained lazy distributions, exact
nonzero probabilities and residuals, cross-domain verification, callback failure
and unwinding, and shared sampler clones. The Metal/image/audio build passed.
No portable contracts or model equations changed in this increment. Commands,
source hashes, ignored cases and remaining accounting gaps are recorded in
[`bounded-speculative-distribution-memory-2026-09-13.json`](validation/bounded-speculative-distribution-memory-2026-09-13.json).

### Idle inventory and ordinary output/observer ownership

Idle executable inventory now includes target parameters, dormant prediction
resources, retained checkpoint sources and explicit enclosing session storage.
Decoder backing stays separate from nonstate storage, and zero decoder bytes do
not imply a zero logical frontier. Parameter-overlay originals and published
replacements contribute their actual native backing identities; cross-map aliases
and aliases with installed model parameters count once. Lazy or unrecognized
arrays remain unknown without evaluation.

Session-payload composition conservatively rejects missing processor or
communication coverage, including a native world retained only by the prepared
target. Review also found constructor-generated rotary frequency arrays outside
the parameter visitor. A separate read-only traversal now includes operator helpers
and explicit architecture static owners. Partial inventories still retain their
known payload when another owner or native backing is unknown. Decoder-state
inspection remains independent; none of these releases an unquoted owner.

Native owners now retain immutable unquoted leases. A checked fresh-session transition
to registered idle storage drops only its own handles after complete inventory;
it cannot revoke an escaped output or snapshot's existing lease. Ordinary operation
entry acquires authority for both the model and the supplied backend context before
work, retains operation-created payloads through session retirement, and moves
authority with displaced state during exchange. Direct prefill/decode outputs
retain request charges even without a sampler. Their completed backing holds
payload-free charge handles, so raw aliases can outlive the session safely.

The native wrapper can now attach these payload-free owners to a lazy descriptor.
Backing publication transfers the bundle before native use, including certified
host storage, shared buffers and donation. Ordinary observer adapters attach before
callbacks for activation, replica, generated, routing and routed-unit tensors, and
to returned intervention values. Skipped generated factories stay skipped, and
attachment does not force evaluation. Independently allocated callback values that
are never returned, installed prediction observers and delivered host records still
require their own complete accounting.

These changes do not enable strict public native budgets. Four positive public
admission regressions remain explicitly ignored until complete admission exists.
In addition to complete storage and workspace coverage, admission needs an exact
operation contract: a reservation retained by state does not authorize unpriced
instrumentation, custom callbacks, sampling changes or independent copies. Multiple
escaped raw outputs cannot all reuse a transient span allowance. Existing
identity/geometry/frontier checks and independent-copy rejection remain in place.

Validation passed 213 distinct backend tests, 15 native allocation-wrapper tests
and four standalone C++ test groups. The final 11 inventory tests include the
operator-coverage correction, positive decoder growth and physical alias lifetime
checks. All 20 focused Metal completion, capture and speculative cases passed.
The Metal/image/audio build passed with 17 unused-helper warnings, including
inventory hooks awaiting admission integration; native test linking retains its
existing unwind-table warning. Seven cases from the broader selection remain
ignored, including four pending public strict-admission success cases.

An overbroad filter also selected distributed matrices. That run was stopped after
20 single-process cases and the complete dense Inkling prediction matrix passed;
the routed matrix had not completed. The focused cases were rerun to terminal
success. This partial distributed run does not establish complete native
distributed validation. No new released-checkpoint or peak-memory measurement was
made. Commands, source hashes, corrected findings and exact remaining cases are in
[`bounded-idle-observer-memory-2026-09-13.json`](validation/bounded-idle-observer-memory-2026-09-13.json).

### Read-only retained numerical values

Neural modules now expose a separate retained-value traversal with explicit
completeness. It includes checkpoint parameters and nonparameter numerical helpers
without changing editable parameter identities. The default visits known parameters
and returns incomplete coverage. Derived modules recursively visit all children;
skipped fields require an explicit metadata, raw-tensor or optional-tensor
classification. An incomplete child does not suppress later known values.

Architecture static ownership uses its own explicit read-only hook. Built-in
models traverse their whole retained static aggregates, while resident, bounded,
routed and partitioned execution use the shared runtime policy traversal. Missing
or loaned units keep completeness false while unaffected static and resident
values remain visible. Physical backing identity, full allocation capacity and
alias deduplication remain backend responsibilities.

The native traversal includes scaled rotary denominators, explicit inverse
frequencies, projection/normalization parameters and quantization companions.
Architecture traversal also includes DeepSeek V4 local/compressor/indexer rotary
frequencies and Muse-Glimmer's optional query scale. These helpers never acquire
editable checkpoint parameter identities. Opaque native representations remain
unknown until their numerical storage can be inspected. Known module ownership
does not certify unmaterialized backing: a lazy helper keeps its inventory unknown
without triggering evaluation. Nonempty-state ownership promotion and strict public
request admission remain separate unfinished steps.

Validation passed 234 Rust tests: 97 neural-module tests, three derive-macro tests,
three neutral architecture tests, 111 portable facade/backend tests and 20 focused
native tests. The scaled-Llama fixtures exercise nonzero prefill and cached decode
in resident, host-layerwise and disk-streamed execution. Resident inspection
retains the lazy helper without evaluating it; after execution it accepts either
certified or unknown backing and preserves each inventory's captured facts.
Explicit numerical validation outside inspection then establishes the helper's
exact allocation identity and capacity. Editable topology, state frontier and
source-read counts remain unchanged by inspection.

Architecture, codec and both minimal and Metal/image/audio backend builds passed.
One portable tokenizer/template test remains ignored because its external LFM2
checkpoint is not configured. No new distributed matrix, released-checkpoint
comparison or peak-memory measurement was run in this increment. Commands, source
hashes, corrected test assumptions and remaining admission gaps are recorded in
[`bounded-retained-values-2026-09-13.json`](validation/bounded-retained-values-2026-09-13.json).

### Initial idle storage publication

Fresh native sessions now publish their complete existing storage after the
initial reset settles. The constructor checks session health, idle submission
authority, exclusive payload ownership, certified nonstate inventory and absence
of decoder storage. The initial operation history must contain only clones of the
same loading authority. Outer discovery and layout caches own declarations and
identities, with no additional numerical payload.

Registration and all native charge attachments finish before local executable,
session and initial-reset loading handles retire. Existing native aliases keep
their physical allocation charges. Escaped loading handles and pending recovery
retain their immutable exclusion leases. Incomplete helpers, sources, processors
or communicators keep the loading barrier, as does nonempty decoder storage.
The transition runs only during construction; a successful session cannot repeat
it to accumulate publication attachments.

Later unbudgeted operations acquire model and context authority before native
work, even when the initial loading handles have retired. A conflicting live
reservation therefore rejects reset and token-input creation without changing
state. This establishes registered idle participation, not a bound or permission
for future operations. Public strict request admission, nonempty state handoff,
quoted independent copies and complete instrumentation/output-lifetime contracts
remain unfinished.

The publication increment passed 55 focused native tests, including six new
regressions for exact initial storage, escaped loading handles, output-alias
lifetime, rejection before native work, lazy-helper fallback and nonempty decoder
rejection. Four positive public strict-budget cases remain explicitly ignored.
Both minimal and Metal/image/audio backend builds passed. No new peak-memory or
released-checkpoint measurement was made. Commands, source hashes and remaining
limits are recorded in
[`bounded-initial-publication-2026-09-13.json`](validation/bounded-initial-publication-2026-09-13.json).

### Text step authority and retained sampled tokens

The ordinary/controlled machine acquires an owned backend permit before its
controller decision. The permit is passed to native submission, retained through
controlled token observation and commitment, and finished without a host token
read for ordinary asynchronous delivery. Exact session admission precedes permit
acquisition. MLX validates current prepared-request bindings, session/domain,
parameter epoch, prediction and phase, then acquires unquoted operation authority.
An existing request reservation never bypasses that acquisition.

Prediction, decision and controlled commitment use the existing bounded rank
agreement so local failures do not leave successful peers advancing into later
collectives. Core-issued run/policy/attempt identities distinguish restored state
and forked machines from execution authority. Unrestricted policy mutation changes
the revision; a snapshot cannot restore a spent attempt or clone a permit.

The sampling quote now retains every emitted token root between invocations,
rather than retaining only the latest RNG state. This fixes undercounting when
callers retain all yielded tokens. Full native backing capacity, real aliases,
random-key replacement, history overlap and unknown bounds remain in the shared
trace. Separately retained completion objects and raw logits require additional
accounting beyond successful serial token emission.

Public native strict budgets still reject unknown coverage. Complete retained-run
quote binding, preparation/instrumentation authority and repricing of mutable
controls remain unfinished; the MLX step permit currently uses unquoted authority.
Native tokens still need exact predecessor receipts connected to that operation
contract, since membership in a retained request only proves charge lifetime.
Nonempty-state handoff, quoted copies and complete media/distributed admission
remain required.

The step/token increment passed 229 focused tests: 28 core, 18 runtime, 111 portable facade
and backend conformance, two evaluation, one architecture sampling integration,
and 69 native tests. Minimal and Metal/image/audio backend checks passed. The
native fixtures retained four greedy or stochastic token outputs through model
retirement and verified that temporary logits and displaced random keys retire
independently. After four stochastic samples, both CPU and Metal retained an
8-byte U32 key; greedy sampling retained no key. This corrects an earlier test
assumption that indexing necessarily retains the full 16-byte split parent.
The conservative capacity bounds remained valid.

Four positive public strict-budget tests remain ignored, along with one portable
tokenizer fixture requiring its external checkpoint. No new peak-memory,
released-checkpoint or native distributed measurement was made. Reproducible
commands, source/log hashes, corrected validation findings and remaining limits
are recorded in
[`bounded-text-step-tokens-2026-09-13.json`](validation/bounded-text-step-tokens-2026-09-13.json).

### Bound run progress and predecessor receipts

The original core context is now bound during admission, before prompt/sampler
creation and mutable policy exposure. The existing runtime preparation authority
owns the run binding, stage readiness and monotone prediction progress under one
mutex. Stage construction cannot race cold binding, and clones cannot bind again.

`InferenceTextStep` requires that original run/policy, the next exact ordinal and
an unused output slot. It allows one active attempt. Successful issuance consumes
the slot before callbacks or peer agreement; dropping an unfinished permit fences
the run. A decode additionally requires the immediately preceding completed
`InferenceTextStepReceipt`. Receipts are provisional until finalization and keep
only identity metadata, while the permit separately retains the request charge.
Restored state, old receipts and copied preparation handles do not reset progress.

At the run-binding checkpoint, MLX retained the original binding while native
entry still used unquoted authority. The subsequent native integration below
connects that logical grant to a complete cold quote and token publication.

Run-binding validation passed 263 tests: 31 core, 52 bounded runtime, 69 native
and 111 portable facade/conformance tests. The new coverage includes three core
ordering/failure regressions and eight runtime cases using genuine core contexts,
ordinary/controlled output, pre-step mutation, stale/provisional/foreign receipts,
quota, fork/restore rejection, callback failure/unwind and eight concurrent claims.
Runtime, minimal backend and Metal/image/audio checks passed. The same four
positive public strict-budget cases and external tokenizer fixture remain ignored.
No new peak-memory, released-checkpoint or native distributed measurements were
made. Commands, source/log hashes and remaining limits are recorded in
[`bounded-run-grant-2026-09-13.json`](validation/bounded-run-grant-2026-09-13.json).

### Native quoted submission integration

Public strict admission now attempts a real cold quote for fresh resident token-ID
execution. It requires complete published static storage, empty decoder state,
one native domain, retained primitive bounds and exact completion. Family
workspace equations, transaction overlap, cumulative sampling, prompt capacity
and the controller declaration are composed by the existing neutral planner.
No synthetic workspace or observed peak substitutes for a missing bound.

The resulting private contract binds the original request/configuration, session,
domains and parameter epoch. Prompt preparation checks the actual Vec length and
allocation capacity and records private provenance that binding and submission
require. A caller cannot relabel another input with a reservation. Each core
prediction claims a runtime step before controller
callbacks, checks the canonical sampler and all native stateful-layer frontiers,
and validates the actual decision's neutral controller contract. Claiming a
submission rechecks the actual token's predecessor receipt. A move-only
native ingress witness then retains the request in the existing submission and
recovery driver. Only successful outputs receive predecessor receipts; failures
and abandoned permits cannot refund their attempts or output slots.

The current quote covers the core's serial completion lifecycle. Escaped tokens
are priced cumulatively; arbitrary separately retained completions/raw logits
still require their own lifetime policy. Capture/intervention installation rejects
before mutation on this strict path, as do direct temperature/RNG overrides.
Detailed neutral reservation failures remain available through the public error
source chain. This initial integration left populated-state continuation to the
increment below. Repricing controls, independent state copies, media, bounded
weight residency and TP/PP quoting remain unfinished. Unknown coverage continues
to reject without native preparation.

Native-quote validation passed 342 tests: 38 core, 54 bounded-runtime integration,
62 working-memory unit, 77 native, and 111 portable facade/conformance tests.
Minimal backend and Metal/image/audio feature checks passed. The four formerly
ignored strict-budget native cases now run and pass, including exact capacity,
one byte short, concurrent preparations and ordinary/controlled parity. Only the
portable tokenizer fixture requiring its external checkpoint remains ignored.

The nonzero tiny Llama fixture ran five prompt tokens in one-position chunks and
emitted three tokens, with receipts 0/1/2 and final decoder frontier 7. Both greedy
and stochastic generation matched ordinary/controlled execution. Its existing
managed residency was 43,392 bytes; the measured controller declaration produced
request reservations of 213,015 bytes for greedy sampling and 227,584 bytes for
stochastic sampling. These are proved reservation bounds, not new native/process
peak measurements. Tests additionally retain state after dropping the run and
retain escaped tokens after model retirement, confirming that charges survive
until their actual owners retire.

No new released-checkpoint comparison or native distributed measurement was made.
Commands, source/log hashes, validation corrections and the unfinished matrix are
recorded in
[`bounded-native-quote-2026-09-13.json`](validation/bounded-native-quote-2026-09-13.json).

### Quoted populated-state continuation

Resident token-ID admission now accepts known populated decoder storage backed by
an exact quoted predecessor. The new geometry includes the cached position in
state and context sizing. Cold inspection records a runtime-owned state revision,
the predecessor request and the native session binding without evaluating arrays.
First submission rechecks that opening under the session lease and atomically
fences the predecessor's future run permission. Failed validation changes neither
run. An already fenced predecessor can be replaced if no attempt is active and
the separate native state proof still holds.

State revision is independent of request identity and logical position. Advancing,
restoring or exchanging a state invalidates old evidence even if its position is
unchanged. Numerical operations without a reservation also invalidate the revision.
Native successful output records the producing revision; decode checks it together
with the immediate predecessor receipt. Revision metadata owns no native payload
or memory reservation. Charges still follow their actual state, completion and
output owners, including failed restoration or typed prediction-state replacement.

At this checkpoint continuation kept both complete reservations. That safely
admitted repeated requests with sufficient capacity, but over-reserved shared state
and expired transient allowances. The funded-storage increment below separates
those lifetimes and transfers physical storage by identity. Subtracting the old
state's byte total would not account for escaped old backing. Quoted reset,
promotion of populated unquoted state and independent snapshot/copy admission
remain unfinished.

This increment passed 909 tests: 718 neutral runtime tests, 80 focused native
tests and 111 portable facade/conformance tests. The native nonzero tiny Llama
fixture reuses a seven-position prefix, appends three input tokens and emits
three further tokens, reaching position 12. Both ordinary and controlled output
match fresh unbudgeted generation over the equivalent full prefix. A competing
stale continuation rejects before callbacks, native input or accounting changes.

The controlled finite-capacity case charges 43,392 bytes of published storage,
213,015 bytes for the retained first request and 221,803 bytes for the successor.
It rejects at 478,209 bytes and executes at exactly 478,210 bytes. Dropping the
successor and session leaves only the first request's charge while its escaped
tokens remain; dropping those tokens releases the charge. These are reservation
bounds, not new allocator or process-peak measurements.

Minimal and Metal/image/audio backend checks passed. The external tokenizer
fixture remains ignored. No new released-checkpoint or native distributed run
was made. Commands, final source/log hashes and remaining limits are recorded in
[`bounded-state-reuse-2026-09-13.json`](validation/bounded-state-reuse-2026-09-13.json).

### Funded physical storage and run retirement

A fresh unique reservation can now separate immutable request identity from a
move-only funding run. Native operations acquire independent scopes before
allocation. Publishing exact physical backing transfers bytes from the reserved
envelope to the existing storage registry without increasing total usage or peak.
Aliases share a charge. Allocations that retire while work remains active return
credit to their originating envelope; after the run closes and all scopes are
certified, surviving storage retires independently of unused workspace.

MLX publishes prompt backing during admitted preparation and decoder, output,
validation and RNG backing after exact completion. Model-source publications stay
with the session so escaped tokens do not retain checkpoint sources. The shared
core machine and its preparation owner retire controller/input/completion payloads
before backend funding, including readiness rejection and unwind. Native sampler
setup transfers the run owner only after its final fallible preparation stage.
Request metadata cannot replace a private preparation quote or live work scope.

Validation exposed a distinction between native event completion and descriptor
settlement. A sampled RNG sibling could be finished while its descriptor still
carried an event, so cold inventory correctly reported unknown storage. The
completion adapter now settles all explicitly submitted descriptors after its
exact event succeeds. These roots are already encoded; this step cannot submit a
new graph or create a host readback payload. Cold inventory remains side-effect
free. A focused sibling test checks backing availability before token observation.

Controller size and lifetime are separate declarations. Strict native admission
now requires `inference_workspace_is_run_owned`, whose default is false. Opaque
numerical storage covered only by run funding must not escape through external
clones, errors or callbacks. A shared-mask test verifies size-only admission
rejects before sampling, callbacks or accounting changes. The built-in fixed
filter owns its storage; the facade constraint controller's shared validity
storage still needs independent host accounting and currently rejects strict
native admission.

Successor admission still reserves a full future bound alongside registered
opening state. Residual quotes must trace borrowed identities across every span,
including displaced old backing and replacement copies. Unknown publication
still quarantines remaining funding; later safe reclamation and overly retained
capacity ceilings need further work. Quoted reset/copy/fork, mutable controls,
media, speculation, bounded weight residency, distributed execution and separately
escaped raw logits/completions remain unfinished parts of the full goal.

Final validation passed 965 tests: 40 core, 729 runtime, 85 focused native and
111 portable facade/conformance tests. Minimal backend and Metal/image/audio
feature checks passed. One external tokenizer fixture remains ignored.

In the final native cached-reuse fixture, published model storage was 43,392
bytes. After the first controlled run retired, only 780 bytes remained beyond
that storage, instead of retaining its full 213,015-byte quote. The successor
reserved 221,803 bytes and executed at a complete domain capacity of 265,975
bytes; one byte lower rejected before preparation. After both runs retired,
792 incremental bytes remained for current state and escaped tokens. Retiring
the session and outputs in either order released their exact physical charges,
eventually reaching zero. Exact physical capacities can vary with the selected
native allocation strategy; the tests compare identity unions rather than fixed
storage totals.

These are managed accounting results, not new allocator or process peaks. No new
released-checkpoint comparison or native distributed run was performed. Commands,
source/log hashes, diagnostic failures and final passing results are recorded in
[`bounded-funded-storage-2026-09-13.json`](validation/bounded-funded-storage-2026-09-13.json).

## Identity-aware cached-resident admission (2026-09-13)

Cached resident successors now reserve a separately proved incremental bound.
One cold native projection maps each actual decoder allocation to the portable
root used by its state views. Runtime pins only already registered identities at
their exact capacities. The equation trace keeps this borrowed-root set immutable
across every prefill/decode span, then unions opening, closing and new allocation
roots, including possible aliases. Only the registered roots are excluded from
the new charge. Displaced replacements, rollback copies, scratch and host workspace
remain included. Full selected-state and workspace reports remain available.

Subtracting an old byte count from a peak is unsound. A neutral regression starts
with allocation A=64 bytes, replaces it with B=16, then replaces B with C=80.
The later B+C overlap needs 96 new bytes. The conservative full report is 144;
subtracting A would incorrectly leave only 80. Identity-aware tracing reserves
96 alongside A's existing 64-byte charge, rejects a 159-byte domain and admits
160 bytes. The ordinary reservation API still rejects a lowered full-report
reservation without the private residual proof.

Core applies the shared context, completeness, application-budget and safety
policies to the incremental requirement. Runtime's ordinary candidate planner
also drives residual chunk selection. The accepted reservation itself retains
the existing-storage pin, so dropping the diagnostic quote cannot release its
credit. Conversion transfers custody to the funding run and independent work
scopes. Certified retirement releases pins outside accounting locks; uncertified
work retains its bound and borrowed charges in quarantine.

Native regression coverage checks actual cache allocation identities for ordinary
KV, compressed, hybrid and pooling state, including aliases, empty windows and
unknown lazy backing without evaluation. Temporary projection witnesses release
their native backing while the numerical metadata remains alive. Ordinary and
controlled cached generation also retain an old raw decoder alias across cache
replacement: its physical charge survives retirement of both runs, their tokens
and the session, and disappears only when the final alias is dropped.

In the final broad native fixture, the cached request reserves 221,419 bytes
instead of its 221,803-byte conservative full report. Its live opening domain is
44,172 bytes, giving an exact admitted capacity of 265,591 bytes; 265,590 rejects
before preparation. The output is `[7, 7, 7]`. After cached runs with an escaped
old decoder alias retire, 1,176 bytes remain beyond the 43,392-byte model storage;
those charges follow the actual remaining state, tokens and alias. These numbers
are managed charges, not new allocator or process peak measurements. Isolated
runs select different physical capacities; assertions use actual identity unions.

The complete goal remains unfinished. Shared facade controller storage, quoted
reset/restore/fork/copy, mutable controls, capture, media, speculation, bounded
weight residency, distributed admission and arbitrary raw-output/completion
retention still require integration. Quarantine cleanup and old capacity-ceiling
lifetimes remain conservative. Quote-construction failures currently stop the
native candidate search; supporting a later complete smaller candidate after an
incomplete initial residual quote was a planner integration gap at this milestone
and is addressed by the later incomplete-candidate increment below. No new
released-checkpoint comparison or native distributed run was performed.

Validation passed 1,317 tests: 285 core, 58 neural workspace, 750 runtime,
93 native, 20 architecture and 111 portable facade/conformance tests. Minimal
backend and Metal/image/audio feature checks passed, as did scoped formatting
and whitespace checks. One external tokenizer fixture remains ignored. Exact
commands, source/log hashes and results are recorded in
[`bounded-residual-storage-2026-09-13.json`](validation/bounded-residual-storage-2026-09-13.json).

## Shared controller and loaded-model host storage (2026-09-13)

The facade's tokenizer-validity masks now use `SharedTokenFilter`. Core owns its
immutable payload, separate identity and per-domain accounting custody. Clones
and snapshots share the same owner even when created before accounting attaches.
The final mask payload retires before its physical charge. Registration keys and
quote metadata retain only identity information, so they cannot keep the mask
alive through an owner/registry cycle.

`TextControllerStorage` distinguishes wholly owned controller state from a
run-owned remainder with explicitly enumerated shared filters. Runtime checks
that their unique capacities fit the additional host quote, then adopts them
from the full funding envelope before prompt or sampler preparation. Shared
storage already registered in the domain receives no second charge. The size
quote remains conservative; this change does not subtract an old mask total
from an unrelated peak. It still reserves the full source-mask allowance when
that source is already registered; independently proved incremental host pricing
remains needed to remove that conservative reservation overlap.

Admission binding and prediction entry check the original inventory. Decisions
carry shared tokenizer-owner provenance, checked again before native submission.
A callback replacing a mask with equal values and capacity cannot substitute its
identity. Controlled choices retain the original and forced masks in their
workspace declaration. Successful shared publication remains charged after a
later preparation rejection while any external alias survives; partial attachment
failure leaves conservative funding in quarantine.

Inspection also found a gap before the first run: native initial publication
finishes before facade tokenizer-mask construction. A never-used loaded model
could therefore retain an unregistered mask. `LoadedModel::from_runtime` now
returns `Result<Self, BackendFailure>` and defers fingerprint/mask construction
through a neutral backend hook. Runtime acquires unquoted domain ownership before
the factory, registers exact storage and attaches custody before releasing it.
MLX binds this to the actual session's pool. A live reservation rejects before
the factory runs. Standard and planned loaders and direct callers propagate the
fallible result. The unquoted construction phase excludes concurrent admitted
work; it does not claim a bounded model-loading allocation peak.

This covers text facade controllers with explicit tokenizer masks, including
ordinary, controlled and pre-staged forced decisions. Grammar parser/vocabulary
storage still lacks complete quotes. Generic `All` controllers need metadata for
optional forced filtering before their later mask can be quoted without allocating
a fake witness. Post-admission control changes still require repricing. Quoted
reset/restore/fork/copy, quarantine reclamation, full media/speculation/bounded
weight/distributed admission and arbitrary raw-output/completion lifetimes remain
unfinished. Shared snapshot mask ownership alone does not prove the full snapshot
budget. No new released-checkpoint comparison, allocator/process peak or native
distributed measurement accompanies this change.

Validation passed 1,286 tests: 291 core, 765 runtime, 26 facade sampler,
112 portable facade/conformance and 92 native tests. Minimal backend and
Metal/image/audio checks passed, and the CLI distributed-control caller compiled
with the fallible constructor. The distributed test was not executed. One external
tokenizer fixture remains ignored. Scoped formatting and whitespace checks passed.

The native fixture retains a 193-byte shared mask with 64 logical entries. Both
the detached ordinary driver and controlled generation produce `[11, 11, 11]`
through a five-position prefill and two cached decodes. After run, outputs and
session retire, the pre-existing external alias keeps exactly 193 bytes charged;
its final drop releases them. Loading-time construction publishes the same exact
capacity before any run. A second factory is not invoked while an admitted run
is live. A pre-staged forced token 17 also completes three native predictions
while preserving the original mask and shared provenance.

Neutral fixtures additionally prove that unused 37- and 53-byte loaded sources
leave exactly 10 bytes available in a 100-byte domain: 10 admits and 11 rejects.
Commands, source/log hashes, corrected test fixtures and final results are in
[`bounded-shared-controller-2026-09-13.json`](validation/bounded-shared-controller-2026-09-13.json).

## Optional forced filtering and retained score aliases (2026-09-13)

Core now separates an exact filter witness from an optional mask declaration.
`TextFilterWorkspace::OptionalMask` holds a logical position bound and maximum
mask allocation capacity; it has no bit payload and conveys no shared storage
custody. Decision validation preserves exact `All`/`Allowed` behavior, and accepts
both mechanisms only under an optional declaration. It checks executable token
membership, logical length, spare allocation capacity and the existing additional
payload bound independently. Unfiltered decisions retain their full executable
domain even when the prospective forced mask has fewer positions.

Runtime choice wrappers derive this metadata without invoking controller callbacks
or allocating a witness. An exact `Allowed` source keeps its prior quote. An `All`
source covers a future canonical-domain mask, and nested wrappers take the larger
possible mask extent while preserving the previous mask's replacement overlap.
Shared-owner provenance and storage inventory validation remain independent.

The existing configured sampler runs one conservative trace across the complete
output allowance. Each optional filter includes both the complete masked branch
and the unfiltered input alias, so mixed schedules retain all possible old output
roots. MLX supplies the same native and host costs as ordinary filtering. Missing
required facts stay unknown; zero-output requests need no optional mask payload
or filter primitive, while retaining independent initialization costs. There is
no second execution sampler.

Review exposed a related score-alias gap: reusing one logical input identity for
every predicted token could undercount distinct earlier score buffers retained by
outputs. Architecture inspection now records the maximum complete backing union
of its actual score rows, including possible aliases, with unknowns preserved.
`WorkspaceSamplingInput` carries that capacity to a fresh score identity on each
step. Current scores remain in the equation quote; earlier scores retained by
outputs join subsequent sampling spans. This deliberately conservative envelope
provides no residual credit. The lower-level bare-layout conversion explicitly
means packed storage; native architecture composition supplies backing facts.

Pre-staged forced choices may execute and then return to unfiltered sampling under
the same admitted policy. Optional-filter pricing does not authorize controller
mutation after admission: the original revision fence still applies. Dynamic
controls, grammar parser storage, incremental shared-host reservations, quoted
reset/restore/fork/copy, quarantine cleanup, complete media/speculative/host/disk/
distributed admission and arbitrary escaped raw-output/completion lifetimes remain
unfinished. No new released-checkpoint comparison, allocator/process peak or
native distributed run accompanies this slice.

A neutral mixed-schedule regression retains three unfiltered score buffers before
its final filtered step. Its 6,677-byte peak exceeds the 6,158-byte maximum of the
two homogeneous runs; the per-step optional union reserves 8,213 bytes. Separate
fixtures cover greedy, stochastic and Mirostat sampling, unknown facts, zero
outputs, nested wrappers, and shared key/input aliases. An architecture regression
uses 2,048-byte backing for a 148-byte score row: after three outputs, its two
retained prior aliases add exactly 3,800 bytes over packed backing.

Validation passed 1,381 tests: 295 core, 773 runtime, 58 neural workspace,
21 architecture, 26 facade sampler, 112 portable facade/conformance and 96 native
tests. Minimal backend, Metal/image/audio and CLI caller checks passed. The CLI
distributed test was compiled, not executed. One external tokenizer fixture
remains ignored. Scoped formatting and whitespace checks passed.

The native fixture forces token 11 while an independent ordinary run chooses 5
first. Both ordinary and controlled admitted runs then match a fresh ordinary
continuation from the prompt plus token 11. Exact-capacity admission succeeds,
one byte less rejects before work, and post-preparation controller mutation
still rejects through the original policy fence. These are managed accounting
and semantic assertions, not allocator or process peak measurements. Commands,
source/log hashes, the corrected isolation fixture and final results are in
[`bounded-optional-filter-2026-09-13.json`](validation/bounded-optional-filter-2026-09-13.json).

## Registered shared-controller reservation credit (2026-09-13)

Registered shared masks now reduce the new reservation through a separate fixed
controller contribution. Their immutable inventory stays live alongside the
controller's other work throughout admission and callbacks. Runtime pins the
complete exact inventory in the same pool, then decomposes the additional host
allowance into that always-live source capacity and the remaining controller
allowance. It performs this decomposition before adding the controller component
to other workspaces. No old total is subtracted from an equation or sampling peak.

Full diagnostics remain unchanged. A privately constructed contribution carries
full and incremental enclosing estimates from the same original input and request
geometry. Full equations and registered decoder equations feed the same terminal
incremental proof, candidate planner and core policy checks. Reservations expose
the original `Admission` after funding conversion, including full state/workspace
reports and the admitted incremental amount; this metadata retains no payload or
accounting pin.

The reservation owns the combined host and decoder pins before returning. Funding
conversion transfers them to the run and outstanding work scopes. Temporary quote
owners retire, while actual source attachment still happens before prompt/sampler
preparation. Existing attachment transfers no new charge. A late failure cannot
refund a source retained by an external alias; uncertified work conservatively
retains both pins in quarantine. Original identity, decision provenance and policy
revision checks remain unchanged.

A failed complete registered-source pin uses the full controller contribution.
This supports previously unregistered custom sources, which receive normal funded
attachment before preparation. Mixed registered/unregistered inventories retain
that conservative full fallback; partial source credit is not implemented in this
slice. Final emitted masks, optional filtering, forced pre-override overlap and
controller history remain fully priced.

The full goal is still incomplete: grammar parser/vocabulary bounds, dynamic
controls and capture repricing, snapshot/reset/restore/fork/copy admission,
quarantine cleanup and old capacity ceilings, complete media/speculation/host/disk/
TP/PP/combined admission, escaped raw-output/completion lifetimes and retrying a
smaller complete candidate after an initially incomplete residual quote were
outstanding at this milestone. The later incomplete-candidate increment below
addresses the last item. No new released-checkpoint comparison, allocator/process peak or
native distributed run accompanies this change.

Validation passed 1,399 tests: 295 core, 787 runtime, 58 neural workspace,
21 architecture, 26 facade sampler, 112 portable facade/conformance and 100 native
tests. Minimal backend, Metal/image/audio and CLI caller checks passed. The CLI
distributed test was compiled, not executed; one external tokenizer fixture
remains ignored. Scoped formatting and whitespace checks passed.

The neutral composition fixture preserves a 228-byte full requirement while
admitting 196 bytes with registered controller credit, or 148 bytes with combined
controller and decoder credit. Its exact 244-byte pool succeeds and one byte less
rejects. Native fixtures reuse a 193-byte source mask across ordinary and
controlled runs, preserve full diagnostic equality, and combine that credit with
cached decoder storage. They also drop every source/controller alias while an
admitted preparation independently retains the charge, then release it despite
historical request metadata. Forced optional filtering still produces tokens
`[11, 11, 11]` against an unforced initial token of `5`, with its final and
overlapping masks fully priced. These are managed accounting and semantic tests,
not allocator or process peak measurements. Reproducible commands, source/log
hashes and native diagnostics are recorded in
[`bounded-controller-host-credit-2026-09-13.json`](validation/bounded-controller-host-credit-2026-09-13.json).

## Host-layerwise managed text admission (2026-09-13)

Selected host-layerwise weights now join resident weights in the native managed
text builder. Architecture inspection constructs and binds real metadata units
inside the same traversal and prompt/decode spans used for execution. The runtime
binding adapter validates every parameter before publication. Constructor helpers
remain in the full span, including one eager scalar seed per unloaded parameter;
the full logical weight fill is lazy and replaced before execution.
The workspace backend also declares its existing masked-output projection
implementation, removing a metadata-construction rejection found by the Gemma4
composite fixture. This declaration grants tracing capability, not memory coverage
without the corresponding native facts.

Runtime owns the shared group-bounded window geometry. MLX supplies the exact
ready host-copy inventory and selected allocator capacities for every dispatched
name, including aliases that can independently allocate destinations. The maximum
window sum bounds fresh materialization; existing host/static/device backing
remains separately charged. Prospective parameter backing covers allocation or
host aliasing and receives no decoder residual credit.

Cold inspection neither settles work nor repairs missing stores. After capacity
reservation, admission atomically revalidates and pins the same ready host stores
so later execution cannot fall back to disk materialization. Policy, layout,
depth, source identity, parameter epoch and opening state remain tied to the
private quote. The existing ordinary/controlled permits, completion and recovery
owners cover execution; final publication charges surviving device-window,
decoder and output allocations before funding certification.

This path requires complete native/host facts, ready immutable MetalShared
transfers, device-resident decoder state and the existing ordinary token-ID
contract. Other host policies, disk, TP/PP/combined execution, media, speculation,
capture and dynamic controls still need complete admission. Quoted branch/copy
operations, quarantine cleanup, old capacity ceilings, arbitrary escaped raw
outputs/completions and complete matrix validation also remain unfinished. This
is an implementation increment, not completion of the full goal.

Validation passed 1,162 tests: 108 neural-library, 797 runtime, 25 architecture,
112 portable facade/conformance and 120 native tests. The native total includes
117 admission/lifetime regressions, two cold metadata tests and one explicitly
run isolated allocator test. That allocator test is intentionally ignored in
the general suite and was run separately with a single test thread. One external
tokenizer fixture remains ignored. Minimal backend, Metal/image/audio and CLI
caller builds pass; the CLI distributed test is compiled, not executed.

The nonzero three-layer Llama fixture compares resident execution with host
windows of depth one and two, through five one-position prefill chunks and three
cached decodes. Greedy and seeded stochastic ordinary/controlled outputs match.
Exact-capacity admission succeeds and one byte less rejects before controller or
native work. Early controlled-run abandonment and escaped token aliases retain
only the appropriate physical charges after completion and teardown.

Additional successful matrices cover routed Qwen3-MoE, mixed linear/full/linear
Qwen3.5 and text-only Gemma4 composite execution. Each compares resident and host
depths one/two in both generation modes, with uneven prefill `2+2+1` and three
cached decodes. They validate state frontiers, selected group windows, unchanged
cold telemetry, post-admission source pins and final physical retirement. An
injected Qwen3-MoE provider failure after an earlier unit preserves its original
native exception and prevents commitment; incomplete publication retains funding
after session teardown and a recovery pass. Direct constructor/transfer faults
and cancellation between prefill chunks are not newly injected in this slice.

The isolated scalar-construction experiment observes 8 Metal bytes for two
Float32, Int32 or Uint32 seeds against a 14-byte bound, and 2 bytes for two Uint8
seeds against a 2-byte bound. Large `[8192, 8192]` and empty `[0, 8192]` logical
weights give the same seed costs; neither full weight is evaluated, and dropping
the placeholders restores baseline active memory. This is not a whole-run peak
measurement. No new released-checkpoint, process-memory or native distributed
measurement accompanies this increment. Commands, source/log hashes and limits
are in [`bounded-host-layerwise-2026-09-13.json`](validation/bounded-host-layerwise-2026-09-13.json).

## Retrying incomplete cold candidates (2026-09-13)

Native managed text admission now tries smaller chunks when the initial numerical
workspace quote is incomplete. Previously, an eager initial sealed quote failed
before reaching the shared candidate planner. Both ordinary and controlled
preparation now send the initial and subsequent candidates through that planner.
The first diagnostic candidate binds output width and controller decisions;
complete candidates must carry the same contract before capacity is reserved.

Runtime's opaque `IncompleteWorkspace` error distinguishes missing numerical
coverage from invalid evidence. Only validated composition can construct it.
Geometry, pool identity, state-span accounting, borrowed-root identity and known
arithmetic must pass first. Missing state-accounting protocol or lost borrowed
association remains a fatal unknown bound. Source readiness, native errors and
changed controller decisions also stop immediately. Complete foreign-pool quotes
now fail identity validation before reaching retryable application-budget policy.

The same planner tries each integer chunk size down to one, without assuming
monotonic workspace demand. It can move between incomplete and over-budget
candidates and retains the final typed rejection if none fits. An incomplete
result carries metadata and its original unknown-bound cause, with no source
registration pin. No candidate with missing coverage can create a reservation;
rejected candidates leave pool usage and peak unchanged. Reservation, host-source
pinning and prompt/sampler preparation keep their existing order.

This addresses the initial-incomplete-candidate gap recorded by earlier
milestones. It adds no missing primitive bounds or new execution paths. Grammar,
dynamic controls/capture, quoted snapshot/reset/restore/fork/copy, disk and other
host policies, complete media/speculation/distributed admission, quarantine and
capacity-ceiling cleanup, and arbitrary raw-output/completion retention remain
unfinished. No new checkpoint comparison, allocator/process peak or native
distributed run accompanies this increment.

Validation passed 969 tests: 806 runtime, 30 native, 21 architecture and 112
portable facade/conformance tests. Nine runtime and five native tests specifically
exercise the new retry boundary using real nonzero metadata operations, including
initial attempts `4, 3, 2` before accepting chunk two. The native suite also reruns
resident and selected host-layerwise exact-capacity, cached reuse, completion,
failure and ordinary/controlled parity cases. Minimal backend and Metal/image/audio
builds, scoped formatting and whitespace checks pass. One external tokenizer
fixture remains ignored. Commands and source/log hashes are recorded in
[`bounded-incomplete-candidates-2026-09-13.json`](validation/bounded-incomplete-candidates-2026-09-13.json).

## Reusable direct-read and cold disk metadata (2026-09-13)

Encoded checkpoint batches and recipes can now be cloned without payload I/O or
payload-cache pins. The copies preserve the original admitted-file contract and
can fill independent destinations after the source object retires. Changed,
truncated or removed files retain their typed errors; partial reads cannot become
valid parameter output.

MLX prepares an owned direct-read plan during loading and reuses the same capacity
facts as ordinary direct batches. Its canonical dispatch allocates each owner
once and shares declared local/external aliases. The manager exposes each unit's
direct allocation bound, actual warm device backing and complete alias-owner
dependency closure. Cold inspection does not reprepare headers, read payloads,
settle work, mutate residency or retain device arrays. Source and manager identity
are weak witnesses; keeping a plan alive cannot retain unrelated source-wrapper
payloads. Native array snapshots also suppress housekeeping and queued-owner
retirement, return a typed busy result under contention and leave lazy backing
unknown.

Neutral host and disk selections now use the same layerwise equation and sampling
provider, including unit construction across uneven prefill and cached decode.
Materialization remains a separate required component. Native managed disk
admission still needs an operation-scoped direct route, completed consumer/transfer
retirement, explicit eviction before refilling the window, and accounting for
retained canonical-owner units. Foreground direct reads are the next integration;
background prefetch and ordinary conversion require their additional bounds and
funding. The full goal remains incomplete.

Validation passed 36 focused tests: 18 checkpoint bulk-read, three native-array
metadata, two architecture quotation, 12 native read-plan/residency and one
explicitly run isolated Metal allocator test. The allocator test is ignored in
the ordinary group and was run separately. It observed 444, 81,920 and 229,376
bytes against bounds of 886, 147,454 and 294,910 for 37, 4,097 and 16,385 positions,
respectively. Each case produced identical physical backing and peak with and
without a cross-stream copy. These are isolated direct-read measurements, not
whole-model or process peaks. No new released-checkpoint or native distributed
validation accompanies this increment.

Minimal backend and Metal/image/audio builds, scoped formatting and whitespace
checks pass. Exact commands, source/log hashes and limitations are recorded in
[`bounded-disk-read-prerequisites-2026-09-13.json`](validation/bounded-disk-read-prerequisites-2026-09-13.json).

## Foreground direct disk admission (2026-09-13)

Selected foreground disk streaming uses the shared managed token-ID builder,
layerwise equation trace, candidate planner and ordinary/controlled step driver.
Loading retains the exact direct encoded read plans. Admission requires zero
host-prefetch budget, device-resident decoder state, complete source/primitive
facts, and the existing finite-output and idle-publication contracts. Ordinary
conversion and background prefetch still require their own materialization and
funding integration; they do not silently substitute for a quoted direct read.

A private receipt retains only read metadata, weak manager/source identity,
policy/layout identity and per-binding capacity envelopes. Current completed
Device hits may change allocation identity between operations while preserving
geometry, alias ownership and the admitted capacity. The receipt neither pins
all device copies nor keeps source-wrapper numerical payload alive. Host copies,
unresolved transfers and changed source/owner evidence reject before allocation.

Each existing SessionOperation activates the receipt after recovery ownership is
established and before input conversion or unloaded-unit construction. The
manager restricts every acquisition, including recursive alias owners and ready
Device hits, to one exact declared group-local window plus its priced persistent
owner closure. Exact retained plans perform missing direct reads. Guard release
requires operation release and settlement of every recovery scope; old token
handles can retain physical charges without extending the active disk route.
Guard destruction performs no lock acquisition, eviction or native cleanup.

The admitted foreground policy settles previous consumers and transfers,
reclaims their references and explicitly evicts completed copies outside the
next window before any initial or subsequent refill. A roomy logical residency
budget cannot accumulate every newly loaded layer. Canonical owner units retained
by alias pins are charged separately from the rotating window, and eviction
failures remain errors. Unquoted disk scheduling retains its existing behavior.

Validation passed 106 focused native tests, including 21 new cases: eight window
policy, seven manager-route, five managed generation and one typed-error test.
The nonzero three-layer Llama SafeTensors fixture uses a roomy 1 GiB logical
device budget with zero host prefetch. Greedy and seeded stochastic outputs match
resident execution in ordinary and controlled runs through uneven prefill
`2+2+1` and three cached decodes. Exact minimum-chunk capacity succeeds; one byte
less rejects before reads, controller callbacks or input/sampler work. Repeated
cold quotes leave source telemetry and pool usage unchanged.

Five unequal units in groups of three and two exercise group cuts, short tails,
a warm second forward and a canonical owner with an additional companion output.
Alias parameters are selected by identity and share their actual canonical
backing. An allocation-owner callback proves the old weight is physically retired
before the next constructor and refill, while an enclosing native submission
scope remains unsealed. Equivalent singleton-group window depths are accepted;
mismatched windows, Host/background acquisitions and changed warm geometry reject.

Cached successors progress while old tokens, raw aliases and submission owners
remain alive. They use one common finite capacity: raising a budget while old
descendants retain the original ceiling still needs handoff/repricing work.
Replacing the admitted source inode after the first output preserves the typed
`AdmittedFileChanged` cause, prevents further commitment and keeps uncertified
funding after teardown. Recipe wrappers now preserve source-less checkpoint and
MLX failures in the public error chain.

Minimal backend and Metal/image/audio builds, scoped formatting and whitespace
checks pass. This increment adds no released-checkpoint comparison, whole-model
allocator/process peak or native distributed measurement. Background prefetch,
ordinary conversion, grammar, dynamic controls/capture, independent quoted
snapshot/reset/restore/fork/copy, complete media/speculation/distributed admission,
quarantine and ceiling handoff, and arbitrary raw-output/completion retention
remain unfinished. The full goal remains incomplete. Commands, source/log hashes
and limitations are in
[`bounded-foreground-disk-2026-09-13.json`](validation/bounded-foreground-disk-2026-09-13.json).

## Completed funding capacity succession (2026-09-13)

A later managed text request on the same session can now explicitly authorize a
larger capacity for completed predecessor funding. A move-only delegation from
the unique original funding owner identifies the exact pool, execution and
account. Runtime performs the change atomically with successful candidate
admission. The old run must be closed, every work scope certified, quarantine
absent and unused workspace released. Legacy reservation APIs retain their
original ceiling behavior.

This changes policy without releasing old memory. Retained tokens, raw aliases,
request metadata and physical registrations remain charged, and each predecessor
keeps the adopted ceiling until its final retirement. A successor retiring first
cannot remove that ceiling. Fixed pool capacity and unrelated accounts still
constrain admission, including accounts with the same numerical old limit.
Accounts whose limits stay unchanged need no new completion claim and remain
ordinary constraints on admission.

MLX keeps the delegation tokens in private session metadata outside the model
payload and escaped output owners. The existing candidate planner, controller
checks and core predecessor/step authority remain shared across ordinary and
controlled generation. Rejected candidates leave limits, used bytes and peak
unchanged. Once admission succeeds, a later preparation failure does not undo
the authorized limit: other work may already use the additional capacity. A
safely settled failure may release its unused workspace; uncertified publication
still retains its existing funding protections.

Validation passed 342 tests: 169 runtime, 61 native and 112 portable facade and
backend conformance tests. Twenty-four new tests cover atomic funding and
candidate policy, actual completed-output succession in both drivers, exact
minimum-chunk admission and one-byte-short rejection, active continuation refusal,
and retained policy after safely settled preparation failure. An existing real
source-failure test additionally rejects a larger subsequent capacity without
changing accounting or starting new work.

The nonzero three-layer direct-disk Llama fixture starts A at its exact finite
capacity, retains all outputs, a raw strided alias and old submission resources,
then runs B at twice that capacity. B matches a fresh resident reference. The
last A alias retains its exact backing, charge and adopted ceiling after B and
the runtime retire. Neutral tests also cover A-to-B-to-C, zero-byte and alias-only
descendants, unrelated equal-valued limits, fixed pool capacity, overflow, poison,
concurrent admission/certification and reentrant physical-key destruction.

Minimal backend and Metal/image/audio builds, scoped formatting and whitespace
checks pass. One external tokenizer fixture remains ignored. No new released
checkpoint, whole-model allocator/process peak or native distributed validation
accompanies this change. Grammar/compiler and semantic-host state, dynamic
controls/capture, independent quoted snapshot/reset/restore/fork/copy, complete
media/speculative/distributed and remaining host/disk admission, certified
quarantine cleanup and arbitrary raw-output/completion retention remain
unfinished. The full goal remains incomplete. Commands and source/log hashes are
recorded in
[`bounded-capacity-handoff-2026-09-13.json`](validation/bounded-capacity-handoff-2026-09-13.json).

## Immutable controller byte sources (2026-09-13)

Shared controller inventories now include closed immutable byte buffers as well
as tokenizer masks. Both use the same identity, per-domain attachment, physical
registry and fixed-source credit. Exact capacity includes spare allocation.
Clones made before attachment retain the resulting charge until the final alias
retires, and payload destruction precedes accounting credit. Source kind remains
part of the contract; opaque bytes cannot stand in for tokenizer validity.

A complete post-callback source witness accompanies managed decisions that declare
byte owners. Existing submission validation rejects missing, unknown or changed
kind/identity/capacity before native work, including same-size replacement during
the decision callback. Forced overrides preserve the witness. Controller bounds
still have to cover allocations performed inside their own callbacks.

The facade replaces nested vocabulary vectors with one packed offsets-and-bytes
owner. It reads immutable trie metadata without copying a GrammarState, preserves
sparse tokens and marker stripping, and allocates only inside the deferred
backend hook. Runtime obtains unquoted domain authority before the factory and
publishes exact custody before releasing it. Ordinary and controlled setup use
the same path and preserve original typed construction errors.

This covers immutable vocabulary/source custody. Compiler/trie internals, active
parser and lexer growth, semantic output state and independent snapshot copies
remain separate work; grammar admission remains unknown until all of them are
covered. The full family/path goal is still incomplete.

Validation passed 434 tests: 15 core ownership, 188 runtime/control, 43 facade
grammar/sampler, 114 portable facade/conformance and 74 native tests. Twenty-nine
are new: nine closed-source custody, 13 mixed-inventory policy, four vocabulary
and facade setup, and three native mixed-controller tests. One external tokenizer
fixture remains ignored. Existing shared-mask, source-credit, capacity-handoff,
resident/host/disk, ordinary/controlled and failure-retention coverage also passes.

The native fixture keeps one pre-attachment byte alias through ordinary and
controlled runs in separate domains. Each domain retains exactly 257 bytes after
its outputs and runtime retire, and both release only with the final shared
owner. The loading hook registers a 513-byte source before inference, refuses a
second factory under active funding, and leaves the original run usable. A
same-size source replacement rejects before native work at preflight and after
the decision callback. Portable real Qwen tool-choice None/Auto setup preserves
typed rejection before vocabulary packing in both entry points, and ordinary
retry succeeds without losing semantic behavior.

Minimal backend, Metal/image/audio backend and MLX/Metal/image/audio facade builds
pass, as do scoped formatting and whitespace checks. This increment adds no
released-checkpoint comparison, whole-model allocator/process peak or native
distributed measurement. The bounded construction hook still uses unquoted
authority, so retained live reservations can exclude another vocabulary factory.
Complete bounded host/parser preparation and independent copy admission remain
separate work. Commands, source/log hashes and the remaining limits are in
[`bounded-immutable-controller-sources-2026-09-13.json`](validation/bounded-immutable-controller-sources-2026-09-13.json).

## Parser table prerequisites (2026-09-13)

The workspace selects archive-verified local derivre 0.3.12 and llguidance 1.8.0
forks. Both inherit the portable unsafe-code ban; llguidance builds only its safe
Rust API. Original licenses, archive hashes and source-file hashes are retained
in `third-party/parser-upstream.json`. The facade's exact version requirement
keeps routine dependency updates from bypassing the local parser changes.

Derivre now supplies cold, allocation-free lookup, checked backing/index geometry,
and exact retained vector/table capacity. A separate closed prepared hash-cons
owner allocates its storage up front, then inserts or rejects without growing.
Duplicate lookup succeeds at full capacity, and rejected insertions leave the
payload, indices and scratch unchanged. Its preparation still requires allocation
authority; this primitive does not certify an enclosing parser's storage.

LlGuidance checks new lexer states before interning or allocating their descriptors
and transition rows. Existing states remain usable at the limit, including after
the limit is lowered. Checked geometry covers the 31-bit state index, u32 backing
offsets, and descriptor/transition layouts. Limit failure keeps the existing
typed parser error and stopped-matcher behavior. Direct transition-row growth
also removes the former temporary row vector.

Validation passed 373 tests: 34 derivre, 38 llguidance, 187 facade and 114 portable
facade/conformance tests. Nineteen are new: nine hash-cons tests, eight lexer-state
tests and two public parser mask/commit/error/clone tests. Three existing external
checkpoint fixtures remain ignored. Minimal parser builds and the facade with
MLX/Metal/image/audio pass. The archived llguidance integration helpers and one
benchmark depend on files absent from its published package; those unavailable
targets remain documented and are not counted as executed tests.

This is a table prerequisite. Upstream still activates the configured state cap
at mask computation, leaving construction and optional warming for the next
change. Derivative candidates, mutable caches, compiler/tokenizer ownership and
copy overlap remain unbounded by this primitive. Finite managed grammar admission
remains unknown. No new released-checkpoint, native distributed or whole-process
peak measurement was made. Commands, exact source changes and remaining gaps are
recorded in
[`bounded-parser-tables-2026-09-13.json`](validation/bounded-parser-tables-2026-09-13.json).

## Lexer construction state limits (2026-09-13)

The following increment closes the state-cap activation gap above. The cap now
applies before building the DEAD/MISSING sentinels and throughout initial-state
selection, first-byte warming, optional large-lexeme precomputation, initial
parser-row construction and initial skip selection. Caps below two fail with
`InvalidLexerStateLimit` before grammar compilation. Any seeding phase that needs
an additional state beyond the cap returns the original typed `ParserError` with
phase context instead of publishing an already-failed parser.

Independent lexer copies retain the same per-instance cap and fail independently.
Matcher errors and their clones preserve closed configuration/parser causes and
their stop classification. Unrelated constructor errors still release their
original payload before matcher publication. A caught recognizer panic can leave
the shared lexer poisoned; diagnostic inspection uses a nonblocking probe and
preserves the original failure without attempting recovery or panicking again.

Validation passed 403 test executions: 54 parser tests with all features, 48
parser library tests without default features, 187 facade tests and 114 portable
facade/conformance tests. Sixteen tests are new: three sentinel/leaf tests, three
lexer tests, four parser seeding-phase tests, three matcher failure tests and
three additional public construction/clone tests. The all-feature and minimal
library runs intentionally exercise overlapping cases. Three existing external
checkpoint fixtures remain ignored. The MLX/Metal/image/audio facade build and
scoped formatting/whitespace checks pass.

Fixtures measure actual construction count N, admit N and reject N-1. Separate
tests reach initial-row and skip growth after successful warming, and optional
precompute with a measured lexeme weight above its 1,000 threshold. Fuel failure
retains its own cause. Nonzero token prefixes and independent suffixes verify
runtime exhaustion, unchanged source state and mask/commit parity after cloning.

This enforces a state count. It does not bound variable expression storage,
derivative/relevance caches, parser rows/history/captures, shared compiler/tokenizer
sources or independent-copy peaks. Managed grammar byte admission remains
unknown. No new released-checkpoint, native distributed or process-peak evidence
was added. Reproducible commands, source/log hashes and limits are recorded in
[`bounded-parser-construction-2026-09-13.json`](validation/bounded-parser-construction-2026-09-13.json).

## Prepared expression encoding (2026-09-13)

Derivre now shares one expression encoder between its ordinary arena, a checked
word counter and a prepared incremental writer. The writer has explicit retained
scratch headroom and writes at the uncommitted frontier. It publishes successful
new encodings in place, detects duplicates before testing remaining committed
capacity, and clears written scratch on ordinary non-commit exits. An ignored
write error cannot turn a partial encoding into a successful entry. Deliberately
forgetting a writer fences later mutations until owner destruction; committed
reads remain valid and the allocation remains owned.

Raw expression encoding preserves tags, flags, byte order, initialized padding
and legacy intern IDs for equivalent insertion sequences. It returns a table-local
`u32`, without pretending to establish an ExprSet's reserved references or child
ownership. Invalid lengths and insufficient scratch/storage reject with typed
causes before publication. Preparing the backing allocation still requires the
caller's allocation authority; requested word counts are not byte admission.

Validation passed 461 test executions: 53 derivre tests with all features, 53
without default features, 54 parser tests, 187 facade tests and 114 portable
facade/conformance tests. The two derivre runs exercise overlapping cases with
different hash implementations. Nineteen tests are new: ten writer lifecycle and
capacity cases, and nine encoding cases. They include exact/one-short capacity,
full-table duplicates, direct publication into the staged allocation, ignored
errors, unwinding, forgotten guards, mixed write boundaries, every expression
variant and byte-concatenation lengths 0/1/3/4/30/31. Three existing external
checkpoint fixtures remain ignored. The native-enabled facade feature build and
scoped formatting/whitespace checks pass.

This supplies bounded encoding, not a complete bounded ExprSet or derivative
engine. Simplifier/traversal scratch, weight and derivative/relevance caches,
grammar/compiler/tokenizer ownership and copy peaks remain separate. Managed
grammar admission stays unknown. No new native distributed, released-checkpoint
or process-peak measurement was made. Commands and source/log hashes are in
[`bounded-expression-encoding-2026-09-13.json`](validation/bounded-expression-encoding-2026-09-13.json).

## Parser copies without a redundant state (2026-09-13)

`TokenParser` now constructs the requested parser copy once, then copies its
remaining fields through one exhaustive initializer. Previously `deep_clone`
created an ordinary parser state and replaced it with a second independent copy
while the first copy was still alive. Ordinary copies still share the lexer;
deep copies retain independent lexer state. Captures, token history, cached
forcing, EOS aliases, configuration, remaining token allowance and terminal
errors preserve their existing copy behavior.

Validation passed 415 test executions: 60 parser tests with all features, 54
without default features, 187 facade tests and 114 portable facade/conformance
tests. Six tests are new: two private state-lifetime witnesses and four public
copy-behavior tests. Nonempty parsed captures establish one completed state copy
and two simultaneously live states, followed by independent suffixes and
source-first destruction. Public tests also exercise cached forcing, alternate
EOS, fresh and terminal copies, and remaining token limits. Three existing
external checkpoint fixtures remain ignored. The native-enabled facade build
and scoped formatting/whitespace checks pass.

The witness counts completed parser states, not allocations or physical bytes.
The change removes one unnecessary temporary; source/destination overlap,
independent lexer storage and all other host copies still need admission.
Managed grammar bounds remain unknown. No new native numerical, distributed or
process-peak measurement was made. Commands, source hashes and evidence are in
[`bounded-parser-copy-2026-09-13.json`](validation/bounded-parser-copy-2026-09-13.json).

## Frozen grammar recipes and host preparation (2026-09-13)

Runtime chat preparation now acquires host authority before compiler work,
validates the tokenizer roundtrip and selected grammar eagerly, then registers
one immutable reconstruction recipe. The recipe contains frozen tokenizer
configuration, the successful grammar, original tool schemas, EOS aliases,
structural IDs/spellings, stop sequences and activation trigger. Prepared-chat
clones share that exact allocation. Temporary matchers, factories, tokenizers,
tries and slicers retire before preparation returns; explicit text generation
leaves the recipe encoded.

The tokenizer envelope also preserves its special-token encoding flag, which
upstream serialization omits. Validation checks complete serialized configuration,
canonical token mappings and normalized trie bytes/metadata. Sparse added-token
configurations that cannot roundtrip unchanged reject before publication.
Semantic reconstruction uses the frozen environment, preserving the selected
syntax fallback and original-schema completion validation.

Core's opaque `HostPreparationAuthority` retains a backend's existing unquoted
lease. MLX validates runtime identity and acquires that lease without allocating
native tensors. The common ordinary/controlled/speculative semantic constructor
acquires before rebuilding parser or validator state. Actual tokenizer, grammar,
controller and shared validator owners retain custody through aliases and
internal forks. A live reservation rejects acquisition, including a zero-byte
reservation. Typed backend causes survive public facade error mapping.

The immutable recipe is charged by its actual allocation capacity. Unquoted
mutable parser work remains excluded from simultaneous finite admission; this
does not establish its byte bound. Public prepared-prompt strings and vectors,
Text pipeline capacity, model tokenizer/decoder caches, escaped application
values and fresh host snapshot/restore/fork destination funding remain separate
work. Source authority inherited by internal forks does not substitute for a
cross-domain copy allowance. Existing logical control budgets are unchanged.

Validation passed 339 test executions: four core authority tests, four native
domain tests, 212 facade tests and 119 portable facade/conformance tests. Thirty-eight
tests are new. They exercise actual compiler-root retirement, registered recipe
alias lifetime, decoder/added-token/EOS roundtrips, original-schema fallback,
reconstruction and fork behavior, validator custody, rejection before work, and
ordinary/controlled text selection with semantic hooks unavailable. Three
existing external checkpoint fixtures remain ignored. Portable and native-enabled
facade builds and scoped formatting checks pass. The zero-byte reservation
fixtures test exclusion; they do not provide a native inference workspace quote.
No new released-checkpoint, distributed or process-peak measurement was made.

Validation commands, counts, source hashes and known gaps are recorded in
[`bounded-frozen-grammar-2026-09-13.json`](validation/bounded-frozen-grammar-2026-09-13.json).

## Host authority before snapshot copies (2026-09-13)

Snapshot capture, restoration and branch construction now acquire destination
host authority before facade metadata, cursor, parser and delivery copies. The
neutral driver checks independently before allocating discovery/estimator
metadata, reserving logical copy allowance or copying controller, capture and
native state. A rejected acquisition preserves the source and consumes no
logical copy allowance. Its typed backend cause remains distinguishable from
an actual native copy failure.

Core and facade continuations keep combined source/destination custody after
their payload fields. Restoration attaches custody before installation; branch
exchange moves it with the logical child. Capture identities retain shared
payload-free custody so preexisting checkpoint and partition aliases survive
parent destruction safely. Forks still receive distinct identities, and schedule
restoration never refunds cumulative capture usage. Speculative activation
checkpoints retain their source capture custody without changing control identity.

The existing finite MLX copy rejection remains; it now happens before previously
unguarded host work. Successful unquoted copies retain authority until their
owners retire. Finite funded copy bounds, direct low-level caller admission and
independently exported generic controller/callback/DTO payloads remain separate
work. The logical snapshot ledger is not a physical allocation certificate.

Validation passed 1,355 test executions: 310 core, 708 runtime, 212 facade and
125 portable facade/conformance tests. Fourteen tests are new. They cover actual
nonzero controller payload retirement, shared checkpoint/partition aliases,
poison and reentry, both admission checks, partial Unicode state, restore,
exchange and post-admission failure cleanup. A direct neutral-driver fixture
counts controller copies and host callbacks and verifies neither runs after
admission rejection. Three external checkpoint fixtures remain ignored. Portable
and native-enabled facade checks and scoped formatting pass. No new native
numerical, distributed, released-checkpoint or process-peak result is claimed.

Commands, validation counts, source hashes and remaining limits are recorded in
[`bounded-host-snapshot-2026-09-13.json`](validation/bounded-host-snapshot-2026-09-13.json).

## Source-bound copy components (2026-09-13)

Configured sampler copies now inspect and consume a plan borrowing the exact
source. It prices the fixed-size history box by its full capacity, preserves
standard and adaptive controls, and performs no sampling or allocation during
inspection. Ordinary MLX snapshot accounting now includes spare and cleared
history slots, matching the actual copy. Speculative sampler clone accounting
uses the same primitive.

Native isolated copies share their contiguous-then-independent-copy operation
sequence with workspace tracing. The existing RNG/pending-input and dense KV
copy workers consume that program. Source allocation aliases remain distinct
from destination copy requests; overlapping copied outputs remain in the trace.
Dense copies preserve capacity, positions, key-only and sliding-window metadata.
Existing-array projection uses nonblocking descriptor snapshots without
housekeeping, evaluation or numerical allocation. Shape metadata does allocate;
this is not an allocation-free whole-copy quotation.

The component traces remain diagnostics. Complete finite snapshot admission,
destination host funding and fresh branch execution authority still require the
remaining controller, metadata, decoder, capture and delivery proofs. The
existing typed rejection for quoted MLX copies is preserved.

Validation passed 1,075 test executions: 714 runtime, 212 facade, 125 portable
facade/conformance and 24 native tests. Sixteen tests are new. They cover fixed
history capacity and adaptive state, matching stochastic continuations, separate
RNG backing, lazy/unsupported sources, absence of housekeeping during projection,
aliased K/V sources with independent destinations, and capacity/window behavior.
Existing hybrid, compressed, paged and pooling snapshot/projection tests also
pass. The backend without default features and the facade with MLX, Metal,
image and audio enabled also compile; scoped formatting and whitespace checks
pass. Three external checkpoint fixtures remain ignored. No new released-model,
distributed or process/allocator-peak result is claimed.

Commands, validation counts, source hashes and remaining limits are recorded in
[`bounded-copy-components-2026-09-13.json`](validation/bounded-copy-components-2026-09-13.json).

## Independently funded sampler components (2026-09-13)

Managed ordinary sampling now constructs the canonical configured sampler inside
its original admitted preparation. A consuming stage binds the actual account,
execution and configuration. A separate closed host scope retains the sampler
through parent-request retirement; it cannot adopt native storage or certify
PRNG work. Its protected balance includes the inline sampler and maximum old/new
history overlap, without adding another charge. Native publication cannot spend
that balance. Sampling quotes cover separate host and tensor maxima so later
retained outputs still fit alongside the fixed history allowance.

Each sampling attempt consumes a move-only permit before native filtering. A
failed or unused attempt cannot refund that allowance. The existing configured
sampler still implements Standard and Mirostat behavior; this wrapper bounds its
history growth and binds custody rather than introducing another algorithm.

The pool can copy that exact owned sampler into a fresh destination account.
Admission counts all live source and destination funding against the tightest
ceiling before allocating the copy. It enforces an optional per-copy application
limit and explicit safety reserve. The fixed history box, including unused slots,
and adaptive controls stay with the destination account until destruction. A
copy can outlive its source and can itself be copied through another admission.
It has no mutable export or authority to run a child generation.

This is a finite sampler-component facility. Whole native and facade snapshots
still reject unknown bounds before copying: RNG, pending input, state layout,
controller/parser, decoder, cursor, capture and delivery need complete aggregate
funding and fresh execution authority. Inline representation allowances are
managed accounting, not additional heap allocations or a total process-memory
guarantee. Local two-process Ring preparation-failure regressions are included
in validation; no new released-checkpoint, distributed numerical, multi-host or
process-peak measurement is supplied by this increment.

Validation passed 1,156 test executions: 729 runtime, 212 facade, 125 portable
facade/conformance and 90 native tests. Twenty tests are new. Coverage includes
Standard/Mirostat copy contents, exact/one-short capacity, source-account and
copy-descendant lifetime, concurrent admission, quarantined source accounts, protected
history funding, 16/32-step host/native peak composition and filter-failure
permit consumption. Existing managed admission, request reuse, host-layerwise,
disk-streamed and ordinary/controlled behavior also passed. Three external
checkpoint fixtures remain ignored.

Two test assumptions were corrected during validation. The local two-process
speculative failure case now counts its accepted Admission and rejected Sampling
agreements explicitly. The filter-failure fixture now permits only an ID beyond
the output width, since shorter valid masks are intentionally padded. Original
failure causes, zero publication, retained-charge and source-state checks remain
in place; neither correction changes the production filtering or agreement path.

The backend without default features and the facade with MLX, Metal, image and
audio enabled compile. Scoped formatting and whitespace checks pass. Commands,
source/log hashes, the initial fixture failures and remaining limitations are in
[`bounded-funded-sampler-2026-09-13.json`](validation/bounded-funded-sampler-2026-09-13.json).

## Independently funded native copies and source policy (2026-09-13)

Native key and pending-scalar copies now use a sealed workspace plan for the
same contiguous-then-independent-copy program executed by MLX. The plan checks
exact imported roots, capacities and destination independence in a private
trace; editable diagnostic reports cannot authorize allocation. Missing native
or host facts stay unknown and arithmetic failures retain typed causes.

The runtime binds that plan to existing registered source storage and admits a
fresh destination through the existing accounting transaction. Source-origin
health, current ceilings, application allowance and safety reserve are checked
before copying. The result has one work scope and closed destination custody.
Source pins retire after successful completion and publication; uncertified work
retains them through quarantine. Native publication includes only final copied
backings, and escaped aliases retain their independent physical charges.

The concrete worker validates the current funded sampler, request, frontier,
parameter epoch, state revision and completed pending receipt. One session lease
covers inspection, admission and execution. Settled read-only copy failures
preserve the source; unresolved native recovery still fences it. Early and late
recovery failures preserve the original worker error. Saved numerical data
contains no source generation quote or ready-to-submit pending token.

Core now provides mutable runtime access for copying while keeping generation
state and pending input immutable. Shared snapshot capture, fork copying and
restoration staging use it without changing the source policy identity. Actual
restored-state installation still changes policy. Portable tests verify these
operations against ordinary outputs and authentic run, policy and attempt
identities, including failed staging and nonrefundable prediction attempts.

Validation passed 1,598 test executions: 311 core, 118 neural contracts, 739
runtime unit tests, 15 text-run integration tests, 212 facade, 127 portable
facade/conformance and 76 native executions. Thirty-one tests are new. The native
coverage includes real stochastic keys and scalar inputs, exact/one-short
admission, source-ceiling preservation, source/model retirement, escaped aliases
and partial-copy recovery. Three external checkpoint fixtures remain ignored.
The backend without default features and the facade with MLX, Metal, image and
audio enabled compile; scoped formatting and whitespace checks pass.

The initial native test build exposed test metadata-access wiring and the missing
read-only copy boundary; both were corrected before the successful runs. No new
released-checkpoint, distributed numerical or process/allocator-peak measurement
is claimed. This remains a native component facility: complete snapshots need
aggregate sampler, native-state and host-payload bounds, and runnable restoration
needs fresh execution authority. Commands, hashes, initial failure evidence and
remaining limits are in
[`bounded-native-copy-account-2026-09-13.json`](validation/bounded-native-copy-account-2026-09-13.json).

## Aggregate saved sampling/input (2026-09-13)

Canonical sampler history and native RNG/pending-scalar copying now share one
admission decision. The runtime joins the actual authenticated history plan with
the sealed native array program, checks both source owners under the same lock,
and commits their combined demand once before any copy. The account initializes
a protected host hold and separate host/native scopes. Native publication cannot
spend the history allowance, and either owner can retire without falsely
certifying the other's work. Their component diagnostics overlap within the one
account and must not be added as separate charges.

The native worker consumes that admitted operation through the existing recovery
and destination-publication path. Saved components preserve Standard/Mirostat
controls, complete history capacity, RNG state and scalar input. They can be
copied again after the original run advances or its model is dropped, using the
saved component's own host account and physical roots. An old live receipt or
frontier is not used as authority for saved-data duplication.

The shared snapshot contract now distinguishes immutable saved sampling/input
from runnable sampling state. Full unquoted capture performs one aggregate copy;
restore and fork prepare one independent live pair before installation. A bounded
saved component exposes no runnable sampler, pending-input grant or mutable
export. Funded resume and conversion to an unquoted copy reject before work.
Fresh resumed-run admission, initial-decode permission and installed-state
revision binding remain necessary for executable restoration.

Validation passed 1,191 test executions: 748 runtime, 23 neutral evaluation, 212
facade, 127 portable facade/conformance and 81 native executions. Sixteen tests
are new. They cover exact/one-short combined admission, rejection before host or
native copying, both source domains and late quarantine, protected host adoption,
both owner-drop orders, aliases, host unwind, partial native-copy failure,
saved copies after source retirement and production trait rejection of unfunded
resume. Shared-driver tests verify one aggregate dispatch and one independent
resume preparation per installation. All checks passed on their first run.

The backend without default features and the facade with MLX, Metal, image and
audio enabled compile. Scoped formatting and whitespace checks pass. Three
external checkpoint fixtures remain ignored. No new released-checkpoint,
distributed numerical or process/allocator-peak result is claimed. Complete
native cache/layout, input identity, controller, decoder, capture and delivery
payloads still need to join admission before whole finite snapshots can pass.
Commands, source/log hashes and remaining limits are recorded in
[`bounded-aggregate-sampling-2026-09-13.json`](validation/bounded-aggregate-sampling-2026-09-13.json).


## Shared live metadata and text identity construction (2026-09-13)

Actual live state layouts and committed prepared-input identities now use closed
immutable owners. Their capacity includes the retained vectors, shapes, names and
fingerprints, including spare capacity. Input descriptor metadata uses sorted
boxed entries while preserving JSON-map and canonical wire behavior. Registry
keys retain no payload. The shared owner releases its payload before accounting
attachments, so aliases created before publication retain the same charge through
their final retirement without a source/charge cycle.

Initial native model publication includes its actual state layout separately
from mutable decoder arrays before releasing loading authority. Key/value,
hybrid and pooling checkpoints, isolated copies and same-layout forks share that
owner. Session transactions, chunked prefill, control slots and cache loading also
retain the actual input identity instead of cloning or rehashing its contents.
Descriptive layouts in selected plans and reports remain distinct metadata; this
change does not certify every planning or diagnostic allocation.

The text prompt quote now includes a closed identity construction plan. One
immutable token borrow supplies both native upload and canonical semantic
hashing. Fixed descriptor/hex allocations and explicit SHA scratch are priced
before construction; no intermediate encoded-word vector is built. Native prompt
preparation publishes the resulting identity with its tensor roots through the
original funding scope before certification. Legacy reservations and unquoted
preparation retain their original authority through escaped identity aliases,
including borrowed reconstruction and fingerprint replacement. Reserved prompts
reject fingerprint replacement before allocating a new identity.

These measures cover declared managed payload and explicit construction scratch,
not allocator/shared-owner bookkeeping or total process memory. Generic media and
per-chunk structural descriptors, mutable native state-copy metadata, complete
facade controller/decoder/capture/delivery payloads, and fresh executable resume
admission remain unfinished. The full finite snapshot guard stays in place. This
increment does not add a released-checkpoint or process/allocator-peak result.


Validation passed 1,613 test executions: 320 core, 768 runtime, 81 neutral session
conformance, six architecture, 99 native, 212 facade and 127 portable
facade/conformance executions. Forty-four tests are new. Existing cache-load,
prompt provenance and state-exchange tests also verify shared-owner identity and
independent final retirement. Three external checkpoint fixtures remain ignored.
The backend without default features and the facade with MLX, Metal, image and
audio enabled compile. Scoped formatting and whitespace checks pass.

Initial checks exposed fixture dtype/import/lifetime issues, remaining shared-API
migration sites, and older tests that excluded newly retained identity storage.
Those were corrected; the failing logs are preserved. Peer review also closed
legacy-reservation and unquoted-replacement alias custody gaps before final
validation. Commands, source/log hashes and limits are recorded in
[`bounded-host-metadata-2026-09-13.json`](validation/bounded-host-metadata-2026-09-13.json).


## Native state-copy preparation (2026-09-13)

Resident key/value, compressed and pooling caches now expose exact borrowed
copy operands without cloning descriptors, making views or doing native work
during preparation. Their existing isolated snapshots use the same copy
program. Key/value snapshots preserve stored padding; compressed snapshots copy
logical latent/rotary views and retain capacity equal to length. Pooling copies
all five optional slots independently, including valid accumulation-before-pool
frontiers and present zero-length arrays. Source aliases do not collapse
destination slots. Each recovery variant retains compaction intermediates and
final copies before the next fallible operation.

Hybrid fixed roles now occupy a sorted boxed table with exact retained slot
extent, checked duplicate declarations, stable role order and independently
mutable optional values. Ordinary checkpoint clones share native handles;
isolated snapshots copy values. This removes the former tree-node-capacity
ambiguity without treating a host-byte diagnostic as allocation permission.

The complete native snapshot still needs mutable live host source accounting,
exact destination host construction, a combined sampler/state account, paged
catalog and transfer plans, and immutable saved-state ownership. Whole-facade
state and fresh resume admission remain unfinished. No new released-checkpoint,
distributed or memory-peak measurement is added by this increment.

Compressed retained-state traversal also includes padded stores when they have
backing independent of the logical views, as can happen after checkpoint copies.
Physical inventory deduplicates shared allocations. The isolated copy still
consumes only its two logical operands, so complete source inventory and
destination-copy operands must remain separate concepts.

Validation passed 286 test executions: 135 native cache, five prediction-memory,
11 saved-copy, eight bounded host-layerwise and 127 portable facade/conformance
executions. Twenty-three tests are new. One external portable checkpoint fixture
remains ignored; native ignored tests in the selected suites ran explicitly.
Minimal-backend and MLX/Metal/image/audio facade checks passed, as did scoped
formatting and whitespace checks.

Review corrected a native-retirement test assumption, narrowed prepared-copy
construction, and found the compressed source-inventory omission. Initial builds
also exposed two old map-access sites, a test import, and a prediction assertion
that conflated retained backing with logical values; all were corrected before
passing validation. Two empty initial test selections are excluded from the final
results. Reproducible commands, source/log hashes and remaining limits are in
[`bounded-native-state-copy-2026-09-13.json`](validation/bounded-native-state-copy-2026-09-13.json).


## Mutable host-state source ownership (2026-09-13)

Runtime's HostSlotTable owns an actual boxed slice without allocating Clone or
an owning payload export. HostSlotMetadata describes only its fixed inline
extent and identity. It has no pointer to table elements. Payload reads and
updates use ordinary borrowed slices; a cold gate serializes attachment against
retirement. Retirement marks the source before dropping elements, outside the
gate, followed by the owner's accounting token. Earlier token aliases can retain
charges after payload retirement but cannot attach a new source charge or
authenticate a future copy.

Portable DeviceState layer storage, native key/value and hybrid layers, and
hybrid fixed roles use this owner. Clone and checkpoint operations preserve
their existing value/alias semantics while giving newly allocated tables their
own identities. Native retained storage includes these sealed extents, dedups
identity aliases and publishes each charge through the existing registry. Model
bootstrap retains loading authority until actual layer/fixed tables and shared
layout are registered. Enclosing model-operation publication sees the actual
installed owners after state changes.

This accounts for the declared inline source tables. Nested native arrays, paged
control/catalog resources, inference-retention payload, constructor/collector
scratch and bookkeeping are not certified by a table's capacity. Complete
destination host construction, paired saved decoder/sampler ownership and fresh
resume admission remain unfinished, as do the wider facade and family/path
requirements. No new released-checkpoint or memory-peak measurement is claimed.

Dormant hybrid prediction prototypes also expose their actual layout, layer
and fixed-role tables through the shared state inventory. Shared numerical
backing still deduplicates independently of these separately owned host tables.

Loaded Llama and Qwen fixtures verify complete initial publication and exact
retirement through escaped table tokens. The DeepSeek V4 fixture deliberately
retains loading authority: its retained rotary frequencies are lazy native Copy
expressions without certified cold allocation facts. Its test publishes only a
separate exact table inventory under that still-live authority and checks table
retirement; it does not claim complete model publication or strict admission for
that path. Closing this native helper-inventory gap remains required.

Validation passed 1,209 test executions covering 1,196 distinct tests: 779 runtime,
81 neutral backend conformance, 209 native and 127 portable facade/conformance
tests. Twenty-eight tests are new; repeated fixed-slot and operation selections
are counted once in the distinct total. One external portable fixture remains
ignored. Minimal-backend and MLX/Metal/image/audio checks passed, with scoped
formatting and whitespace verification. Initial fixture errors and the corrected
DeepSeek cold-inventory assumption are retained in the validation evidence.
Commands, source/log hashes and exclusions are recorded in
[`bounded-host-slot-2026-09-13.json`](validation/bounded-host-slot-2026-09-13.json).

## Paired saved continuation components (2026-09-13)

The shared snapshot driver now stores one immutable decoder/sampler/pending-input
pair. Required backend hooks handle capture, independent duplication, compatibility,
logical estimates and resume preparation. A backend must finish all fallible
preparation before returning the installable tuple; the driver retains its
existing host guard, non-refunding budgets and installation sequence. Saved
sampling diagnostics remain borrowed through the pair.

At this validation point, MLX preserved its existing unquoted copy behavior
through an opaque paired owner and rejected bounded paired requests before
either worker. The subsequent resident-copy section records progress on joint
physical admission; fresh funded resume remains unfinished.

Validation passed 385 tests: seven neutral continuation, 127 portable
facade/conformance, 212 facade unit and 39 native tests. Six are new. The neutral
fixtures exercise full restore/fork continuation parity and late-copy failure;
the native paired fixture checks component preservation and repeatable decoder
logits using a manually assembled sampler, not a complete resumed generation
run. Three existing external/environment fixtures remain ignored. Minimal-backend
and MLX/Metal/image/audio checks and scoped formatting/whitespace checks passed.
Reproducible commands, hashes and limitations are recorded in
[`bounded-paired-components-2026-09-13.json`](validation/bounded-paired-components-2026-09-13.json).

## Decoder destination construction and copy binding (in progress)

A closed slot initializer derives its destination extent from an actual source
table. It creates one `Option<T>` buffer without invoking `T::Clone`, `T::Default`
or caller initialization code. Completion retains that same buffer, and copying
completed slots does not add another optional wrapper. Public preparation only
reports the requested payload and explicit value-temporary allowance; the
allocating worker runs only after the joint runtime account is committed.

For the pinned Rust 1.98.0 implementation, `Vec::with_capacity(n)` records the
requested count, `n` vacant pushes do not grow it, and boxing transfers it when
length equals capacity. The bound covers requested slot payload plus two explicit
slot-value temporaries for a nonempty table. Nested resources, allocator and
owner bookkeeping, descriptors, and compiler frames are separate. The inspected
[Vec implementation](https://raw.githubusercontent.com/rust-lang/rust/1.98.0/library/alloc/src/vec/mod.rs)
and [pinned RawVec implementation](https://raw.githubusercontent.com/rust-lang/rust/88d9e12ae178fab0fb5cc050a94da85685d449ea/library/alloc/src/raw_vec/mod.rs)
are the allocation basis; a toolchain change must review that path again.

Native saved-copy preparation also retains the exact destination session
identity alongside its submission lease. Both entry points reject using that
preparation on another session before admission, while fresh preparation on the
other session still supports same-pool saved-source copying. Cold state
inspection returns an actual borrowed decoder view without performing numerical
work inside inspection.

The runtime join admits sampler host payload Hs, decoder initialization peak P,
one numerical copy demand N and safety reserve in one account. Application and
shared-domain checks use Hs+P+N+safety. Sampler/table-or-saved-host/operand/full-source
origin validation occurs under the same usage lock before either host worker.
The two host holds cannot be spent by native adoption. Each host owner releases
only its own hold after payload retirement; the native scope carries all source
pins into unresolved-work quarantine.

Partial completion errors keep the actual table and its funding together. Saved
slots can become a new copy source through their closed held-account proof,
without reusing the original run or assuming a second physical registration.
A provider key projection rejects another same-sized table identity. Runtime
validates the supplied registered origins; the native closed projection remains
responsible for complete transitive source inventory and actual source access.
The resident KV driver now consumes that joint primitive. It retains the actual
live payload or saved decoder independently of the mutable runtime, preserves
shared layout and committed input-identity owners, and validates the destination
session lease. Complete source inventory is pinned separately from copied
operands. The same submission recovery holds the real source and every partial
array; only final destination arrays are published before native certification.
Saved decoder values carry no live request, predecessor receipt or installable
state. Copy-from-saved authenticates its retained origins rather than replaying
an old live frontier.

The recovery collector holds n source handles plus at most two produced handles
per copied operand, with checked capacity reserved before account commit. Its
handles, quotation descriptors and registration maps use the existing
command/shape/handle bookkeeping exclusion. This is not a total host/process
memory bound: every referenced numerical buffer and actual semantic slot,
history, layout and input-identity payload remains separately accounted for.
The collector does not claim to eliminate allocations made by native descriptor
cloning. Source aliases remain separate destination-copy operands.

This is the resident KV component route. Hybrid, pooling, paged and selected
prediction-state copying remain unfinished and reject before numerical copying.
Fresh funded resume and unpriced enclosing facade state remain unfinished. No
new released-checkpoint or process-memory measurement is claimed here.


The native snapshot composition now routes bounded resident KV capture and
saved duplication through that driver. Its decoder and sampling views share one
sealed pair and account; pair estimates count the aggregate once. A standalone
copy of the borrowed sampling view copies only sampling destinations and creates
its own account. An unquoted source cannot be promoted by choosing a bounded
policy. Pending prefill payloads still need their own complete copy program.
Funded native growth remains unknown and resume returns a typed rejection before
installation. No populated saved state becomes runnable through copy custody.


Validation of resident paired copying passed 1,180 tests: 801 runtime, 81 neutral session,
146 native cache, 17 native saved-copy, two cold decoder-adapter, 6 native snapshot and
127 portable facade/conformance tests. Thirty-four tests are new. Native selections ran
serially with local Metal execution enabled. Minimal-backend and MLX/Metal/image/audio
checks passed. The portable external fixture remains ignored. The initial paged fixture
and array-inspection API corrections, and concurrent cold-inspection rejections, are
retained in the evidence. Commands, source/log hashes, managed-payload scope and remaining
limitations are recorded in
[`bounded-resident-paired-copy-2026-09-13.json`](validation/bounded-resident-paired-copy-2026-09-13.json).


## Sampler resume and Pooling copy prerequisites (in progress)

Cold sampling workspace inspection now borrows populated policy and records only
history length/capacity and adaptive scalar progress. It executes the same filter
order and adaptive update as ordinary sampling without cloning token history
before admission. Primitive descriptors and unknown bounds retain their existing
semantics; metadata tracing is not an execution grant.

For actual history length L, capacity C and new output allowance M, the closed
resume plan starts with C destination slots and follows the real growth rule up
to L+M. Its protected host envelope is the inline sampler plus four times the
largest of C and each old-plus-new capacity overlap. Thus L=5, C=8, M=11 requires
24 words of peak history, with 16 final slots. Zero-output and cleared histories
still retain their actual capacity. Static policy must match; the new stage also
checks the complete configuration and quota. Source health and installation of
the new hold share one accounting lock. History and adaptive state are copied only
after that boundary, and the destination gets a fresh attempt counter and custody.

The source-bound fixed-slot initializer can also price a distinct destination
representation without copying values during preparation. The new resident
Pooling copy binds actual Local/Compressed/Sparse layer tables, preserves their
optional numerical slots and shared layout, and uses the existing joint host and
native account. Retained source descriptors are visited separately from operands.
No global layer index is invented for a representation that does not store one.

These are component prerequisites. Stateless absent-table Pooling still needs a
mandatory complete-source join with no decoder table. The existing model fixture's
cold rotary helpers remain unknown, so physical leaf/account tests do not claim a
complete positive model admission. Hybrid, paged, facade/controller payloads and
fresh runnable resume remain unfinished. No new released-checkpoint or process
memory measurements are claimed.


Validation of these prerequisites passed 1,250 test executions: 824 runtime,
81 neutral session, 218 native sampling/cache/copy/snapshot and
127 portable facade/conformance. Twenty-eight cases are new. Native tests
ran serially with local Metal enabled; the existing portable external fixture
remained ignored. Minimal-backend and MLX/Metal/image/audio checks passed. The
compile-time overflow fixture, trait import, supported overlap layout and partial
root-count corrections are retained in the evidence. The sampling and saved-copy
filters can overlap; the total describes executions, not unique cases. Commands,
source/log hashes and remaining scope are in
[`bounded-resume-foundations-2026-09-13.json`](validation/bounded-resume-foundations-2026-09-13.json).


## Actual stateless state and dense destination construction (in progress)

Stateless Pooling copy now uses the actual absent-table branch, distinct from a
present empty layer table. Its closed admitted source borrow prevents substituting
a different stateless instance. The account joins sampler and numerical demand
with mandatory complete-source pins, with no decoder table or decoder host hold.
Uncopied source roots stay in the native scope after host owners retire and remain
in quarantine if completion is not established. Existing KV and present-table
Pooling continue through their original typed table join.

The dense initializer selects `[D]` directly from the actual source borrow. It
allocates one exact-capacity Vec, pushes sequentially without growth, and boxes
only after the exact count is reached. Incomplete finish retains the same partial
buffer. Its managed bound covers `n * size_of::<D>()` plus two explicit moved
values for nonempty fill; it does not price arbitrary native constructors or
nested resources. Empty/ZST destinations preserve their logical slot count.
Public preparation and completed borrowed access expose no execution grant.

Fresh runnable resume still needs the request-bound dense owner, continuous
host-hold to actual-table charge transfer, checked native installation, controller,
input and new step authority. Dense initialization alone does not fund that work.


Retained native array inventory now observes allocation facts without runtime
housekeeping and retains only necessary descriptors through a dedicated safe
inspection clone. Known views/aliases deduplicate by actual backing; lazy backing
remains unknown. Query or clone failure occurs before inventory mutation and
preserves typed contention/native causes. Neither operation evaluates, polls or
reclaims unrelated owners. The source must remain settled and unchanged across
the separate observation and retention steps. This correction does not eagerly
materialize the current model fixture's unknown rotary helpers or turn its
component tests into complete positive model admission.

Validation passed 1,306 test executions covering 1,294 unique tests: 840 runtime,
81 neutral session, 252 native backend, 6 native wrapper and 127 portable
facade/conformance executions. Twelve native executions repeat tests selected by
overlapping filters. The 25 new tests comprise 8 mandatory complete-source joins,
8 dense initializer cases, 3 stateless Pooling cases and 6 cold inventory/clone
cases. Native tests ran serially with local Metal enabled; one existing portable
external fixture remained ignored. All 13 checks passed, including minimal-backend
and MLX/Metal/image/audio builds. This verifies the component boundaries above;
fresh funded native resume, Hybrid/paged copying, complete model-helper loading
and the wider bounded-inference goal remain unfinished. No new released-checkpoint,
distributed hardware or total-process memory validation is claimed. Commands,
source/log hashes, unique-test identities and retained limitations are recorded in
[`bounded-stateless-dense-2026-09-13.json`](validation/bounded-stateless-dense-2026-09-13.json).


Fresh prompt construction now admits the existing dense initializer against the
actual new request account. Source health and full source registration are checked
under the same accounting lock as the protected inline hold. Exact-budget and
one-byte-short behavior includes the independently protected canonical sampler.
Partial fill/errors retain the original buffer, source borrow and prompt claim.

Completed publication shifts only the actual table payload from reserved to
registered bytes, without increasing total usage or peak. The table owns its
charge before the temporary hold retires, including zero-byte identity lifetimes.
Ordinary registration/pinning/adoption sees the same key and deduplicates. A
provider comparison panic precedes counter changes and defers final key
destruction until owner/account locks unwind; poisoned state stays conservative.
The insertion basis was checked against the pinned Rust implementation.

This is inline host payload accounting, not native constructor workspace or an
end-to-end resume result. The new bootstrap's inherited residual-pin branch has
not received a dedicated behavioral fixture; complete-source and direct source
pins have. Native decoder installation, fresh input/controller authority and the
wider family/execution matrix remain unfinished.


Validation of fresh dense prompt construction and exact table publication passed
1,313 test executions: 853 runtime, 81 neutral session, 252 native backend and
127 portable facade/conformance. Thirteen new runtime tests cover source health,
exact protection, partial/error custody, readiness, alias retirement and a real
provider comparison panic. The 12 checks include minimal-backend and
MLX/Metal/image/audio builds. Native tests ran serially with local Metal; one
existing portable external fixture remained ignored. Overlapping native filters
mean this is an execution count. Commands, hashes, review basis and explicit
remaining scope are in
[`bounded-dense-prompt-2026-09-13.json`](validation/bounded-dense-prompt-2026-09-13.json).


Resident KV dense construction now shares the existing isolated layer-copy
program with immutable snapshots. Its host payload is protected before filling,
its actual native operands supply the separate copy bound, and exact table
publication retains the same allocation. A fresh state preserves cache values,
controls, local layout and global layer offset, with empty inference retention.
Partial copies remain in the caller's ordinary recovery collector; publishing the
host table does not establish native completion.

Prepared-state binding and saved-pair origin checks add exact executable
provenance without retaining model payload or old execution permission. Cold
validation rejects foreign or parameter-invalidated origins before destination
work. Immutable duplication preserves the original identity and remains possible
after model retirement. Fresh request admission, actual resumed input/sampler
composition and the shared driver's installation/readiness sequence remain work.
The dense component tests do not claim a complete future forward workspace; the
origin tests use an actual bounded tiny Llama source run, not resumed inference.


Validation of fresh native KV construction and executable-origin binding passed
1,326 test executions: 853 runtime, 87 neutral session, 259 native backend and
127 portable facade/conformance. Thirteen new cases cover neutral binding,
actual dense native copying, and original-provenance preservation through saved
duplication. All 12 checks passed, including minimal-backend and
MLX/Metal/image/audio builds. Native tests ran serially with local Metal; one
existing portable external fixture remained ignored. One origin fixture was
corrected to load both models before creating reserved snapshot work; its initial
failure is retained in the evidence. Overlapping filters mean these are execution
counts. Commands, hashes and limits are in
[`bounded-native-resume-state-2026-09-13.json`](validation/bounded-native-resume-state-2026-09-13.json).


## Retained helper finalization during loading (in progress)

Native loading explicitly submits and settles the actual numerical roots already
retained by target and prediction modules before first physical publication. The
existing detached loading scope protects allocation authority, and the existing
completion adapter retains the submitted roots. Publication then uses a fresh
cold inventory. The inventory itself never evaluates, polls or upgrades a lazy
root, and finalization does not construct unloaded units or certify missing
traversal coverage.

This addresses lazy retained helpers such as scaled rotary frequencies at the
resident loading boundary. It does not establish a complete model quote by
itself: selected architecture, cache, sampling, host and source bounds must all
remain complete in the actual request. The focused native probe uses a real
resident V4 prompt and multiple decode outputs and reports any remaining typed
unknown rather than supplying an assumed fact. Central validation is recorded below. Layerwise V4 constructor-owned host vectors, fresh funded
resume and the larger bounded-inference goal remain separate unfinished work.


The core fresh-resume entry point shares ordinary Admission/Prompt/Sampling
readiness and the existing generation machine. Its ordered preparation owner
keeps controller, prompt and sampling payloads ahead of staged authority during
failure and unwind. New context and final-controller binding occur before
construction; installation must finish before Sampling Ready. Actual saved-source
quotes, native installation, absolute-versus-local prediction accounting and
complete facade/controller payload admission still require implementation. This
neutral extension alone makes no native bounded-resume claim.


Validation of the shared resume driver and retained-helper loading passed
1,595 test executions across 12 checks, with 13 new tests (nine neutral driver
cases and four native loading cases). Native checks used serial execution with
local Metal access; one existing portable external fixture remained ignored.
Test-only corrections aligned fixture visibility, CPU source-stream selection
and initial-publication expectations; an initial sandbox run could not access Metal. These failures are
retained in the evidence. The actual resident V4 quote reported completeness
`true`; the probe runs finite ordinary/controlled generation only
when actual admission succeeds. Commands, diagnostics, hashes and limitations are
in [`bounded-resume-driver-loading-2026-09-14.json`](validation/bounded-resume-driver-loading-2026-09-14.json).
Full native funded resume and the larger family/path integration remain unfinished.


The native dense KV source projection returns a complete copy-span report,
projected destination state and separate source-storage witnesses. It quotes the
actual isolated-copy program before future equations, preserving large source
backing/aliases and distinct copied output capacities. Unknown source backing
remains a required rejection even when output-allocation bounds are known.
This route traces before returning its source witnesses; it cannot install
borrowed-storage credit afterward. Enclosing admission must use conservative
full composition or a separately proved protocol that binds credit before trace.
Host table construction and full saved-state resume remain separate obligations.

The staged native resume-coordinate prerequisite separates immutable saved cache frontier from absolute prediction observations and fresh request-local receipt ordinals. Actual ordinary finite save/duplicate tests cover source advancement and retirement; internal helper cases cover a nonzero origin, stateless coordinate arithmetic and overflow without claiming resumed inference. Native prompt conversion, complete preparation composition and installation remain separate work; full funded resume and the overall bounded-copy goal are unfinished. Central validation is recorded below.

A staged private opening seal now separates a future provisional quote from a runnable installed opening. Ordinary quotes remain immediately sealed. Semantic empty-retention and single-assignment tests cover zero-byte authority, foreign/stale revisions, busy native boundaries and resealing; native cases use actual ordinary admissions. No pending quote can be built from a bare reservation, and the complete native resume installer is still absent. Central validation is recorded below; funded resume and the full bounded-copy goal remain unfinished.


Validation of the actual saved-source projection, populated-sampler architecture
quote, prediction coordinates, pending-token numerical program and private opening
seal passed 1,704 test executions across ten checks, including 28 new cases.
Native checks ran serially with local Metal access; one existing portable
external fixture remained ignored. Exact commands, source and log hashes,
review findings and limits are recorded in
[`bounded-resume-source-plans-2026-09-14.json`](validation/bounded-resume-source-plans-2026-09-14.json).
These mechanisms do not yet expose full native funded resume: complete pending
host input construction, saved-source admission, installation and failure custody
still need integration. The full family/path bounded-inference goal remains open.


The closed pending-token host plan distinguishes retained payload D (one actual InputPart) from a conservative construction envelope H = 3D + sizeof(T), including explicit moved values. Empty maps and extents allocate no nested payload. H is protected before construction and remains with every shared input alias; a separate boxed part ensures payload deallocation precedes hold retirement. Native tensor backing, copy temporaries and completion stay independently accounted. Reference-count bookkeeping, allocator overhead and compiler frames are outside this payload fact.

Private saved-source quotation now composes actual dense-copy spans, copied key/pending numerical operations, the exact pending host plan, populated sampler growth and future shared architecture equations. Its full copy terms include retained source/destination backing plus transient overlap. Their sum is deliberately conservative and may repeat existing source coverage; it does not subtract registered source credit or claim to be an exact incremental reservation. Unknown required native or host facts remain unknown.

The final typed dense-state bridge moves the same registered host table and native arrays into a control slot after the caller establishes native settlement/publication. Foreign origins, mismatched geometry and incompatible state representations reject without installing the state. Full funded native resume, remaining family/path integration and total-process memory guarantees are not supplied by these mechanisms. Validation of this construction increment is recorded below.


The closed pending-token host constructor, final dense-state binding and actual
saved-source diagnostic passed 1,318 test executions across seven checks,
including 18 new cases. Native tests ran serially with local Metal access; one
existing portable external fixture remained ignored. Exact commands, source
and log hashes, review findings and limits are recorded in
[`bounded-native-resume-construction-2026-09-14.json`](validation/bounded-native-resume-construction-2026-09-14.json).
Source-bound admission, native construction and installation through shared
resume readiness remain in progress; this increment does not enable native
funded resume or complete the family/path bounded-inference goal.


Constructor-owned V4 frequency payloads use the shared fixed F32 initializer. Its validated N-element Vec allocation has capacity N and is filled without growth; the host-domain contribution is N*4 through the native upload. Native source and copy storage are independently included by the selected Metal fact. The actual layerwise quote traverses every selected unit constructor within the completed forward span, including replicas and construction-before-drain overlap. Metadata execution allocates no frequency values, and missing mechanism facts still prevent finite admission. This closes a specific previously invisible caller-owned frequency buffer; it does not establish whole-facade host coverage or enable fresh funded resume. Central validation and native host/disk parity are recorded below.


The fixed F32 constructor and layerwise rotary host accounting passed 705
test executions across nine checks, including 12 new cases. Actual DeepSeek V4
resident, host-window depths one/two and direct disk execution matched across
ordinary and controlled runs with five prompt positions and three cached
decodes. Exact declared capacity admitted; one byte less rejected before input,
controller or source work. The NN tests observed the actual fixed buffer
capacity and preserved the typed realization cause; scalar-frequency tests
matched the previous formula bit for bit. Native tests ran serially with Metal
access; one existing portable external fixture remained ignored. Commands,
source/log hashes, pinned allocation basis and limitations are in
[`bounded-layerwise-rotary-host-2026-09-14.json`](validation/bounded-layerwise-rotary-host-2026-09-14.json).
Fresh native resume and the full applicable family/path goal remain in progress.


The native fresh-resume adapter composes a conservative full quote from actual immutable saved cache, key, pending token and populated sampler values. Both original saved owners and every complete registered source origin are validated under the reservation lock before account/capacity changes; source pins survive in the new funding scopes. Existing source charges remain live. This full source-inclusive envelope deliberately takes no retrospective shared-allocation credit.

Dense host table construction, pending host input and resumed sampler history use separate protected host holds. Native copies retain all intermediates and actual source ownership through completion; only final destination arrays are published before certification. The fresh decoder table/prefix, key and input are prepared before installation, and escaped native allocations retain their physical charges independently of session/readiness lifetime. No old receipt, seed restart or refunded output quota enters the new run. Validation of this native resume integration is recorded below; broad family/path completion remains outstanding.


The routed/composite resume validation exposed a representation gap: ordinary generation could run, but bounded saving rejected the generic Hybrid container even when its actual payload was only KV attention. The new closed adapter validates every layer's real attention and empty fixed-role storage, prices the Hybrid outer slots, and reuses each existing KV copy operation. Empty child tables allocate identity metadata only; their zero-byte registrations are still published so a complete inventory can be pinned before the first resumed prediction. Actual nonempty roles are never discarded or repriced as empty. The integrated routed and composite family tests passed, including full inventory pinning before the first resumed prediction.


The compressed copy projection distinguishes potentially four source allocations from exactly two logical copy operands. The aggregate caller imports the complete source before tracing. New destination capacity is the logical length, so subsequent cache growth starts at that compact capacity. Unknown source backing remains unknown and cannot be erased by finite output-copy facts. Empty logical arrays retain their explicit state and can grow normally. The compressed projection passed fourteen runtime and nine native focused tests, followed by the integrated regression checks.


Grouped decoder host construction sums the exact existing outer and ordered child
initializer envelopes, including absent-valued role slots. Frozen copy validates
the actual sampler, every table origin, numerical operands and complete registered
source together before one H+sum(P)+N+safety account commit. Fresh construction
validates the original request/run and same source set under one lock, then protects
sum(P) within that already admitted account. Each table has independent host custody;
all complete/residual source pins remain with the separate native scope/quarantine.

A completed dense child transfers actual D through the existing attached table
registration before its residual host temporaries retire. Other table holds remain
protected, and the original account still controls unassigned residual budget.
Partial outer/child values and publication errors retain their respective custody.
Checked descriptor counts use the established bookkeeping exclusion; actual table,
role and numerical payload is included. This group adds no arbitrary scalar byte
authority or whole-copy/native-completion certificate.

Gemma4 exposed an actual nonempty optional prefix-role table, requiring grouped construction rather than the earlier zero-payload fast path. Full-source collectors now separately count all retained descriptors and requested copy operands, so compressed padded stores stay in recovery while only logical arrays are copied. Native mechanism tests cover nonzero floating and integer roles, aliases, absent roles, complete compressed sources, exact domain capacity and one-short rejection, recopy after source retirement, and a late fixed-copy failure. Actual shared-driver tests cover Gemma4 plus recurrent Qwen and compressed DeepSeek, including ordinary/controlled numeric-state and token continuation. Grouped integration validation passed, including actual recurrent and compressed family continuation through the shared driver.


Fresh resident native resume and grouped state construction passed 1,392 test executions across
eleven checks, including 47 new cases. Routed Qwen3-MoE, sliding composite Gemma4 and affine packed Qwen3
passed continuation parity. Recurrent Qwen3.5 and compressed DeepSeekV3 also preserved numeric state through several cached predictions. Standard and Mirostat continuation matched
uninterrupted future tokens through ordinary and controlled runs. Repeated
resumes retained fresh local receipts, original prefix, saved key/history and
decoder frontier. Exact capacity admitted and one byte less rejected before
work. Pre-install cancellation/failure preserved the current branch; failures
after exchange and Sampling readiness rejection fenced the exact target while
escaped arrays retained charges. Commands, hashes and limitations are in
[`bounded-native-fresh-resume-2026-09-14.json`](validation/bounded-native-fresh-resume-2026-09-14.json).
Broader family/state, capture, media, speculation and distributed integration
remains outstanding.


Table-backed Pooling fresh resume prices its actual fixed host destination and
isolated numerical copies, using the same source-bound prompt construction and
future workspace projection as the existing typed native route. Shared primary and
index source aliases produce independent requested destination streams.

Saved resume now avoids reserving already registered decoder/key allocation bytes
for a second time. The sealed copy program supplies this credit; no old account's
bytes are subtracted wholesale and full diagnostics remain source-inclusive.
Complete source charges and pins survive through completion or quarantine.
Destination arrays, protected host buffers, pending-token conversion and future
state/sampling remain priced. Pending/controller source credit and tighter
composition of preparation and future peaks remain separate work.

Pooling, incremental admission and the capture host destination below passed
central integrated validation. These additions do not complete bounded inference
across the broader execution matrix.


The closed F32 capture destination derives a host plan from an actual immutable
admitted selection and its phase/invocation geometry. Unknown axes, overflow and
rank above the current typed 32-axis constructor limit reject before buffers.
For output rank r and n elements, retained payload D includes the inline
TensorObservation plus r*sizeof(usize) and n*sizeof(f32) actual Vec capacities.
The protected construction envelope adds two inline DTO moves and, for nonempty
shape/data respectively, two usize/f32 scalar moves. Checked geometry and the
pinned Vec requested-capacity path establish this envelope; telemetry does not.
The full envelope remains charged until the final shared alias retires.

Admission uses the existing domain counters/ceilings and loading exclusion under
one lock, minting only a private host-only scope. Incomplete and failed construction
retain surviving payload with that scope; final host cleanup cannot certify native
work. This is one fresh destination with no source credit. Native source storage,
transforms/transfers, capture quota, serialization output and complete observation
records remain separately owned and unproved here. Existing bookkeeping exclusions
are unchanged; the actual shape/data payload is included. No whole-capture bound
or production managed-capture support is implied.


This integration passed 1,388 test executions across eight checks, including 24
new behavioral tests. Actual Pooling continuation crossed a primary/index pooling
boundary and matched ordinary and controlled future tokens under Standard and
Mirostat sampling. Exact reduced reservation admitted; one byte less rejected
before work. Neutral tests cover distinct copied destinations sharing old roots,
foreign/quarantined source identities, unchanged physical charges and retirement.
The capture host leaf passed payload, alias, serialization, capacity, incomplete
construction and cleanup tests; production managed capture remains unfinished.
Commands, source hashes and limitations are recorded in
[`bounded-resume-capture-accounting-2026-09-14.json`](validation/bounded-resume-capture-accounting-2026-09-14.json).


The no-decoder Prompt constructor has no decoder host payload because its actual
source and destination have no table. It adds one native scope to the original
account after atomic source validation, with no additional reservation or invented
metadata. Sampler/history, key/pending copies, closed pending host construction,
future execution and facade payloads retain independent accounting obligations.
Together with parent-account capture storage, this prerequisite passed central
validation; it does not expand the current public managed-admission matrix.


A parent-funded capture tensor destination protects its exact host construction envelope within the original account's remaining reserve. It creates no second reservation, and native publication cannot consume its protected hold. All concurrent independent destinations must already fit the enclosing quote; aliasing one completed owner adds no payload allocation. Closed, metadata-retired or quarantined parents reject new allocation/fill/finish. Existing payloads retire safely and completed read-only aliases retain their charge. This prerequisite does not price native source creation, transforms, extra host transfer vectors or capture record envelopes, and managed capture ingress remains gated.


The inherited first-pin-only quarantine limitation recorded in bounded-resume-capture-accounting-2026-09-14.json is addressed by linked custody for every abandoned scope. Six new regressions reproduce the old loss and cover distinct/overlapping/zero-byte sources, all drop orders, successful siblings, concurrent drops, poison and actual residual Prompt composition. The fix passed combined validation; no native completion is inferred from cleanup.


The parent capture/no-decoder integration passed 1,406 test executions across seven
checks, including twenty-two new behavioral cases. Parent host destinations and native
storage share one exact reservation; wrong identities and closed/quarantined
parents reject without releasing surviving payload custody. Actual absent-state
construction preserves original Prompt authority and source pins through explicit
completion, cancellation, native error and unwind. Public managed capture and
selected absent-state resume remain unfinished. Commands and evidence are in
[`bounded-parent-capture-no-decoder-2026-09-14.json`](validation/bounded-parent-capture-no-decoder-2026-09-14.json).


Fresh saved-source quotations include the existing exact host transfer or foreground direct-disk persistent-closure/moving-window materialization bound when those mechanisms are selected. The same temporary source workspace feeds the populated architecture equations and the subsequently admitted pins/receipt. Existing registered decoder/key source credit remains confined to its sealed copy program; old physical charges and full-source health remain independently enforced. Successful shared reservation precedes host pin mutation or direct-route activation, and ordinary per-operation receipt validation prevents falling back to unquoted reads or transfers. This closes the resident-weight-only resume gate for proved host/direct-disk mechanisms; it does not prove whole-facade host memory, arbitrary asynchronous schedules, or every remaining family/state integration. The native scenarios passed centralized validation, recorded below.


The host/disk integration regressions exposed and corrected two shared-path gaps: retained host inventory entered ordinary housekeeping during a cold quote, and the text driver forwarded cancellation only to prefill. Inventory now uses cold metadata snapshots; the existing Prediction readiness exchange handles between-step cancellation while preserving completion failures and peer permit retirement. The original failing checks and focused regression coverage are retained with this validation increment.


Selected host/direct-disk saved resume passed 369 test executions across seven
checks, including ten new behavioral cases (three native resume, two native cold-inventory and five neutral cancellation matrices). Actual three-layer fixtures exercised
host depths one/two and foreground disk, Standard/Mirostat sampling and matching
ordinary/controlled continuation. Exact reduced capacity admitted; one byte less
rejected without controller/native/residency work. Each completed operation released
its disk restriction while retained outputs kept their physical charges. Cancellation
and final escaped-token retirement also passed. These fixtures supplement the
resident family evidence; they are not a new released-checkpoint or distributed
validation. See
[`bounded-layerwise-saved-resume-2026-09-14.json`](validation/bounded-layerwise-saved-resume-2026-09-14.json).


A private F32 Metal capture leaf now traces and executes one static slice and, for Preview, an explicit reshape plus prefix slice. Certified narrow views preserve their whole backing allocation in the trace and registry pins. Slice aliases and possible reshape copies use the existing exact selected facts; unknown facts stay unknown. Evaluated host iteration reads shared Metal storage directly into the existing exact host destination, with no second data vector. Parent H, the actual source and exact borrowed native scope are joined before work; native allocations remain an obligation of the original enclosing quote. Pre-buffer rejection/unwind rollback and post-native failure have separate pin lifetimes. The leaf does not enable managed capture or intervention or price observation record composition. The F16/BF16 extension is described below; F64 and other transfer mechanisms remain unfinished.


Shared capture delivery is a lifetime prerequisite, not complete managed capture support. CapturePayload::SharedTensor preserves an existing protected tensor owner, and SharedCapturedStep retains a complete frame with final custody through all aliases. A source-bound frame construction plan, original-request host allowance and public driver delivery integration are still required; the hidden retain bridge grants none of these.


Nonempty F16/BF16 capture selection has one additional explicitly priced native AsType(F32) result. Under the reviewed selected Metal view invariants, Vector copies a contiguous nonbroadcast extent no larger than logical N; General allocates N*4. The fact charges the allocator's full capacity envelope and retains source/selection/cast roots without relying on donation. Shape/command/handle metadata retain their existing bookkeeping classification. Direct settled F32 iteration adds no numerical host Vec; the original exact host destination H remains protected separately in the same parent account. A shared host alias retains custody until final drop, while the caller's exact native scope retains source pins through settlement or quarantine. This is a leaf bound: invocation attribution, transform admission, the full native N contribution and the ordinary capture ingress still require their enclosing integration.


The native capture prerequisites passed 1,610 test executions across eleven
checks, including thirty-two new behavioral tests. Core shared delivery preserves
the tensor wire format and custody through aliases. Runtime transfer construction
binds source pins and host storage to the exact original native scope. Native
F32/F16/BF16 Full/Slice/Preview cases cover nonfinite values, narrow and reversed
views, empty results, unknown facts and post-cast failure recovery. Four native
descriptor regressions and existing wrapper/session tests passed against the
rebuilt pinned patch set. These are component proofs: source-plan admission,
cumulative frame/tensor storage, observer-aware original workspace quotation and
public ordinary/controlled delivery remain unfinished. No managed capture gate
was opened. Commands, source identities, corrections and limitations are in
[`bounded-native-capture-prerequisites-2026-09-14.json`](validation/bounded-native-capture-prerequisites-2026-09-14.json).


The protected raw-tensor frame primitive accepts only unpartitioned Full/Slice/Preview geometry and keeps every selection record, including missing, skipped and failed outcomes. It allocates final record and sidecar buffers before work, preserves exact capacities through shared finish and retains P through aliases. Nested protected tensor data is separate. This prerequisite does not enable managed instrumentation: the original request still must own/charge its capture plan, include all simultaneously retained scheduled frame/tensor envelopes and numerical work, and bind the actual source/collector before admission.


Origin-aware ordinary capture geometry uses cached opening positions + new prompt_tokens + request-local prediction for Context, with prefill prediction zero and first decode one. The immutable plan carries origin through known-shape/slice checks, estimates and readmission; maximum-one requests have no decode capture. Zero-origin and independent-invocation identity formats remain unchanged. This geometry prerequisite does not enable managed instrumentation or supply the original capture quote/source ownership.


The shared admitted capture source reports its inline DTO and actual nested Vec/String capacities, including transform vectors, selected discovery payload and semantic digest. Plan aliases retain per-domain source custody; a registry identity alone retains neither payload nor charge. This is a retained-source measurement, not an admission or construction-peak certificate. Original source attachment and cumulative frame/tensor reservation are still required.


Capture planning prerequisites passed 1,436 test executions across six checks,
including twenty-eight new behavioral tests. Fixed frame construction protects
its actual record, string, shape and sidecar buffers in the original reservation.
Cached opening positions now contribute to capture Context geometry and semantic
identity without changing request-local prediction coordinates. A shared immutable
plan retains its actual nested capacities and any attached source custody through
aliases. Future native selection tracing uses the same Full/Slice/Preview program
as execution, preserves the enclosing workspace context and conservatively prices
a possible F32 conversion. This increment does not yet connect cumulative capture
claims, source publication, observed-equation quotation or public delivery to
ordinary/controlled admission. Existing managed capture gates remain unchanged.
Commands, exact sources, review notes and the test-fixture correction are recorded
in [`bounded-capture-planning-2026-09-14.json`](validation/bounded-capture-planning-2026-09-14.json).


Borrowed capture revalidation removes a temporary whole-plan/digest reconstruction from session preflight and checkpoint validation. It retains the exact admitted selections for backend estimation and rejects changed selected declarations or capabilities before estimates run. This is not a bound for subsequent preflight shape vectors, checkpoint copies or full observed generation, and does not open managed capture gates.

Shared capture options can now enter the original core text preparation path without late reconfiguration or a second reservation. The default still rejects Some(SharedCapturePlan), including empty plans with physical source storage. A complete managed route still needs source registration, original-account cumulative host/native quote, closed per-step claims, source/step-bound native collection, and owned delivery. The options/installation hooks themselves provide no finite-memory proof.

Exact native shared capture-plan inventory includes the owner's real nested capacities, including spare vector/string capacity. Complete inventory registration is atomic; one-byte-short credit and incomplete coverage reject before attachment. A later attachment failure retains earlier source charges and requires the existing funded scope to remain uncertified until its normal failure policy resolves it. This retained-source component does not establish plan-construction, instrumentation, frame or whole-inference bounds.

A closed cumulative capture host primitive prices every frame P (including prefill metadata and schedule-gap records), every eligible floating tensor P with actual context growth, and fixed claim/control payload including all potentially coexisting coordinate receipts. One original-account host hold protects the total before the claim buffer exists; native adoption cannot spend future host capacity. Per-frame/tensor constructors consume one-use claims and share that hold instead of acquiring P again. Completed aliases conservatively retain the entire H until the last owner. This is destination custody only: actual SharedCapturePlan registration/source health, native transforms/completion, logical capture quota and outer delivery storage still require their original admission integration. No managed capture support claim follows from this primitive alone.


Capture admission prerequisites passed 1,494 test executions across seven checks, including forty new behavioral tests. The runtime protects the entire finite schedule before allocating its fixed claim table; child frame and tensor constructors share that original hold and cannot reissue a spent coordinate. Core options carry the exact shared source through ordinary, controlled and detached preparation with one Instrumentation agreement. Native inventory publishes that source by actual retained capacity, and borrowed revalidation avoids rebuilding a plan or digest. Public managed capture still needs observer-aware quotation, source/step-bound native collection and protected delivery integration. Commands, reviewed source identities, test cleanup corrections and limitations are recorded in [`bounded-capture-admission-prerequisites-2026-09-14.json`](validation/bounded-capture-admission-prerequisites-2026-09-14.json).


Original-scope capture ingress retains a lazy activation before evaluation and publishes its backing without certifying unrelated pending work. A terminal tensor failure conservatively retains whole-run host storage after partial buffers retire; claim issuance remains nonrefunding. Component fixtures combine the inspected native activation/copy span with cumulative scheduled host storage. The complete model observer quote, quota ledger, protected delivery and facade preparation envelope remain unfinished.

Scheduled settled-source native capture uses the already protected cumulative host H and the original account native allowance N, without a child reservation or duplicate host hold. Full actual source allocation pins remain attached to the exact mutable native scope, including failure/quarantine paths. Leaf tests derive H from the actual schedule and N from the shared Metal trace, check original H+N admission at exact capacity and one byte short, then adopt actual completed roots while H remains protected. This is one-coordinate mechanism coverage; full architecture-hook quote integration, lazy-source ingress and application capture policy remain separate.

An already registered SharedCapturePlan can now pass into CaptureSession::from_shared_plan and additive shared partition receipt constructors without another semantic plan allocation. Its exact per-domain source charge follows every actual source alias, including aliases created before attachment. Receipt/session/record construction and native transforms still require their own original account and preparation contracts. CaptureCheckpoint deliberately continues copying its own raw destination plan under preparation and retains the original shared source separately; its logical destination estimate includes only the source handle, not a fabricated new copy of existing source storage. No finite checkpoint-copy or whole managed-capture claim follows from this shared-source lifetime integration.


Capture native ingress and shared observation-source prerequisites passed 1,436 test executions across eight checks, including thirty-five new behavioral tests. An additional diagnostic run passed the four native ingress tests. The closed worker retains a lazy source before evaluation, publishes actual backing inside the original scope and consumes a one-use scheduled host claim. Terminal errors keep original custody; prepared traversal paths and capture sessions retain their actual shared sources. Two initial native fixture expectations were corrected: host aliases conservatively retain the original account balance, and asynchronous allocation-owner retirement uses the existing bounded wait. Public managed capture still needs the complete observer-aware quote, ledger/transaction delivery and facade envelope. Sources, corrections, exact commands and limitations are recorded in [`bounded-capture-native-ingress-2026-09-14.json`](validation/bounded-capture-native-ingress-2026-09-14.json).


The closed frame bound includes precommit, encoding-counter, pending-abort and owner-preserving error controls. Wire-size validation uses the existing serializer without a second JSON buffer; an underestimated successful record becomes a bounded Failed record before its tensor retires. Original whole-run custody covers these constructors once. Observer-aware resident/layerwise equation diagnostics include live capture roots, keep closing retained state decoder-only, and add the concrete borrowed-hook host overlap without changing source identity or tensor-only diagnostics. Full logits precede the actual final-row sampling operation; future sampler allowance and unknown operator/host facts are preserved. Actual native observation paths are registered source residency, while cold construction/rebinding and internal hook allocations retain their separate obligations. Native generated-input provenance, funded transaction/drain composition, the facade envelope and full managed capture admission remain unfinished.


Observer-aware equation, actual path-source and protected capture-delivery components passed 2,453 test executions across ten checks, including twenty-seven new behavioral tests. Original whole-run custody covers precommit and detached abort delivery; actual native paths are shared with cold metadata equations, and full logits precede final-row sampling. These are component validations: funded transaction/drain composition, native generated-source mechanisms and the complete facade capture envelope still require integration. Exact sources, commands, fixture corrections and limitations are recorded in [`bounded-observed-equations-delivery-2026-09-14.json`](validation/bounded-observed-equations-delivery-2026-09-14.json).


The original cumulative capture host plan now includes the closed funded session, lexical observer and error/delivery controls, identity owner and exact Option<CapturedStepDelivery> machine slot before the existing checked move multiplier. Frame and tensor payloads retain their original hold across delivery, legacy parking and escaped aliases; no second payload charge or fresh claim is created. The wrapper accepts only an unused original bank and preserves spent coordinates, logical quotas and failed-drain owners. Prepared traversal avoids rebuilding outer hook paths during forward, while internal/generated operations and native/facade controls keep separate obligations. Selected generated capture, routed/partition prepared hooks, native quote/installation and the complete facade envelope remain unfinished integrations.


Prepared observer dispatch, the cumulative funded capture session and retained core delivery passed 2,478 test executions across ten checks, including twenty-five new behavioral tests. These components preserve original custody through precommit, abort, drain failure and legacy parking. Native installation/quotation, selected generated capture and the facade envelope remain unfinished. Exact reviewed sources, commands and limitations are recorded in [`bounded-funded-session-delivery-2026-09-14.json`](validation/bounded-funded-session-delivery-2026-09-14.json).


Selected Metal FP8 generated-input quotation now includes its real fixed reconstruction operations between preparation and projection finish. Skipped factories emit no reconstruction work; prepare roots and decoded weight-scale scratch remain held through the observer and finish. Missing selected provenance rejects explicitly and missing operation/host facts remain Unknown. The prior creation_bytes formula remains a logical capture quota only, independently of selected physical allocation facts. This prerequisite does not by itself open managed Tensor capture or distributed capture support.


The shared generated-projection reconstruction and selected physical equation facts passed 1,695 test executions across eight checks, including nine new behavioral tests. Real Metal F32/F16/BF16 callbacks preserve projection values and generated-input reference values; compact roots, reconstruction and projection finish share the original equation trace. These components do not yet enable funded generated capture or certify partial-factory recovery. Exact sources, commands and limitations are recorded in [`bounded-generated-projection-equations-2026-09-14.json`](validation/bounded-generated-projection-equations-2026-09-14.json).


The scheduled native adapter consumes one already protected tensor claim and
reuses its original FundedWork scope. Before lending the model it checks the
bank's original account and that active scope; exact semantic plan/quote/source
binding remains an original installation obligation. Each transfer retains full
actual source backing and every native intermediate until whole-work publication,
including failure/unwind. Host receipt completion does not certify native work.

Legacy and scheduled Full/Slice/Preview logical quotas now share the same checked
count arithmetic, retaining the full selected count before Preview truncation.
These quotas remain separate from actual H/native bounds. The adapter owns two
borrowed pointer fields; its metadata estimator uses two fixed 32-element u64
stack arrays plus scalar controls. These fixed controls are recorded for original
quote composition; they are not another tensor payload grant. No complete
originating observed quote or public managed capture path is added by this unit.

The retained generated-factory protocol and actual BlockFp8/workspace producers are available as prerequisites. Existing collectors keep their bare-factory compatibility behavior and legacy logical creation quota. Selected generated managed capture still requires its closed funded consumer, original observed admission and matching span retention; no managed gate is opened by the protocol.

Cold capture diagnostics include behavioral coverage for shared logical policy and original-context equations, including actual resident, host-layerwise and disk fixtures. Host H is returned separately from numerical spans. The managed path still requires accepted semantic/source binding, H protection and collector installation, plus the remaining prefill/generated/distributed mechanisms.

Additional registered inputs retain their existing physical charges with no extra scalar credit. Overlapping registrations and heterogeneous key domains remain governed by the pool identity registry, including zero-byte identities. Healthy cold association is revalidated at reservation; a newly quarantined origin rejects without changing usage, peak or an eligible predecessor capacity ceiling.


Original-scope scheduled capture, retained generated factories, shared cold observation policy and mandatory registered-source joins passed 2,478 test executions across thirteen checks, including 34 new behavioral tests. The actual resident, host-layerwise and disk diagnostic routes preserve the live decoder and source identities. This validates components; original observed admission, cumulative H installation and public managed capture remain unfinished. Exact commands, source hashes, integration corrections and limitations are recorded in [`bounded-original-capture-components-2026-09-14.json`](validation/bounded-original-capture-components-2026-09-14.json).


The next private capture admission composes native observed equations with cumulative H and the actual newly registered source capacity C. A previously registered plan contributes no duplicate C; its origin is validated both before original reservation and before pending-bank consumption. If another request publishes the source during preparation, the original conservative C allowance remains reserved and the actual resulting registration is validated. Quarantine between publication and installation is rejected without refunding that origin or consuming the pending bank.

Generated capture now shares legacy per-selection creation quota while invoking the retained producer only once per hook. Empty selected slices stay lazy, while Preview(0) on a nonempty slice preserves legacy construction. Actual compact inputs and all seven reconstruction outputs remain in native recovery through later failures, and cold workspace roots remain until the enclosing span ends. Missing mechanism facts remain unknown. These are component and private-admission changes: managed public capture, selected-prefill attribution, intervention, snapshot and whole-facade delivery coverage remain unfinished.


The retained generated consumer, borrowed capture/source binding, registered-origin witness and private original captured admission passed 2,491 test executions across nine checks, including 27 new behavioral tests. Exact commands, source hashes, integration corrections and scope limits are recorded in [`bounded-original-capture-admission-2026-09-14.json`](validation/bounded-original-capture-admission-2026-09-14.json). Public managed capture installation, execution and delivery remain unfinished.


Original core TextPreparationOptions capture now connects the pre-Prompt source/C plus cumulative host/H admission to installed collection and shared delivery in MLX. The exact same admitted request, source payload and native funding scope follow ordinary and controlled execution. Request installation checks readiness, identity, origin health and current prepared paths without acquiring a new reservation. Each operation rechecks that binding; completed frames retain physical custody through aliases and legacy parking. These aliases can conservatively retain the complete remaining original account balance, not merely H. Managed captured copies reject before projection or copy because copied claim/ledger state is not implemented. The public facade trace envelope, selected-prefill capture and the broader media/speculative/parallel observation matrix remain separate unfinished work.

For decode-selected plans, bounded prefill spends one p0 schedule-metadata row across all native chunks. The final frame becomes committed only after final indexing and scope settlement; cancellation, error and unwind retain an aborted frame and its original charge. Per-chunk state already committed remains committed. Selected split-prefill tensor capture still returns a typed attribution rejection pending its portable implementation.


Original core-options captured admission, authenticated installation, native execution and completion-first shared delivery passed 1,958 test executions across nine checks, including 38 new behavioral tests. Actual resident, host-layerwise and disk fixtures preserve four-token ordinary/controlled output and nonzero decode Preview values; exact/-1 admission, empty-plan drains, cancellation and settled controller failure retain their original identities and accounting. One p0 frame spans independently committed prompt chunks and is published only after outer cancellation, score indexing and reservation settlement succeed; late failures abort it without refund or whole-prompt rollback. Captured copies and the complete facade/media/speculative/parallel observation matrix remain unfinished. Commands, exact sources and corrections are recorded in [`bounded-original-capture-integration-2026-09-14.json`](validation/bounded-original-capture-integration-2026-09-14.json).


Facade post-token termination now uses a closed core query, preserving the existing mutable controller query/error semantics without revising the bound policy. Ordinary, detached and managed continuations share it; explicit mutable policy access still invalidates an existing finite request binding. The query issues no step, drains no capture and does not itself stop core advancement. Complete facade host/capture admission remains unfinished.

The closed termination query passed 796 test executions across six checks, including five new behavioral tests. Genuine finite request contexts preserve run/policy identity and three cached predictions across ordinary, detached and managed consumers; explicit mutable policy access still rejects the next admitted step. Original typed query errors and pending completion/input remain intact. Commands and exact sources are in [`bounded-controller-query-2026-09-14.json`](validation/bounded-controller-query-2026-09-14.json).


Native registry retirement now scans the complete entry-time registry at ordinary host retirement boundaries. It reclaims only positively settled current-owner records outside the registry lock. Polling remains bounded and destructor-free. This fixes a reproduced 128-record starvation case in which 16 successful empty-completion waits reclaimed none of the 64 eligible current-owner payloads. The separate foreign thread retains its 64 payloads until its own retirement. This changes real owner retirement, not byte estimates or a completion certificate.

Native retirement fairness passed 868 test executions across the clean CPU patch suite, rebuilt Metal wrapper/session suites and the same isolated fairness fixture linked against the actual Metal library. The CPU patch suite passed 49 existing cases plus the new two-owner regression; each configured regression reclaimed exactly 64 current-owner payloads and no foreign-owner payloads before owner-local cleanup. Full commands, native patch/source identities and pre-fix evidence are in [`bounded-native-retirement-fairness-2026-09-14.json`](validation/bounded-native-retirement-fairness-2026-09-14.json). This repairs native retirement; selected-prefill capture, whole-facade admission and the wider bounded-inference matrix remain unfinished.


### Shared delivery does not price a facade envelope

Moving `CapturedStepDelivery::Shared` through a facade callback retains the existing `SharedCapturedStep` custody. No new hold, reserve, allocation claim or native completion certificate is introduced. Alias retirement remains governed by the original frame owner; explicit raw DTO cloning is separate caller-owned allocation.

`TraceBudget` still counts compact JSON transport bytes without an encoded buffer. Its exact/one-byte-short tests are transport-quota tests, not physical host admission. Facade provenance, configuration, prompt/decoder/output state, journals, event construction and callback-owned envelopes still need an actual closed source/program bound in the original account. Shared records do not inherit that missing protection from their captured frame. Admission gates are unchanged in this increment.

Shared facade delivery passed 1,451 test executions across six checks, including seven new behavioral tests. Actual neutral ordinary/controlled drivers deliver the same tokens and shared frames after exact completion. No-token cancellation frames, empty-plan error drains, original typed failures and retained alias lifetime are covered; raw/shared event serialization remains compatible. Complete facade host-envelope admission remains unfinished. Commands, exact sources and the test-only public control-handle correction are in [`bounded-facade-shared-delivery-2026-09-14.json`](validation/bounded-facade-shared-delivery-2026-09-14.json).


Selected prefill capture has a geometry-only prerequisite: CapturePrefillRowAssembly borrows the exact admitted ordinary selection and fixed chunk schedule, deriving checked Sequence-row fragments, global strided slices and global Preview placement, including batch/head scatter and nonzero cached origins. Context/TokenRows/media/unknown or multiple Sequence axes and independent invocation authority are explicitly rejected by this mapper. Declared row geometry alone does not prove that a hook's chunk values equal its full-forward values. Existing PrefillAttribution, PartialPrefill and output-demand gates remain in force until the exact semantic activation, persistent original-H partial targets, one-use coverage/quota, per-chunk capture-root/source-pin retirement and matching native/workspace readout are integrated. This unit grants no bytes, completion, new capture support or full-prompt fallback.

Closed prefill row geometry passed 397 core test executions, including eight new tests, plus portable architecture/facade and native feature checks. Independent nonzero full-source references verify uneven chunks, cached coordinates, global strided slices/Preview, batch/head scatter, zero intersections and checked extreme metadata. This proves placement only; original host partial targets, actual native fragment realization/retirement and semantic activation remain unfinished. Commands and sources are in [`bounded-capture-prefill-geometry-2026-09-14.json`](validation/bounded-capture-prefill-geometry-2026-09-14.json).


A temporary capture source channel can now retain and revalidate actual registrations separately from permanent request sources without another hold or work scope. Failure or handle Drop keeps all pins in original recovery/quarantine. The removal primitive is crate-private and has no production caller: selected prefill capture and its per-chunk credit remain unfinished. Tests verify that only final registration/physical-owner retirement returns credit; surviving aliases retain their charge and the full cumulative host H is not refunded. A native integration must also allow settled submission-graph owners to retire before relying on actual available credit for the next chunk.

The exact capture source-segment mechanism passed 1,588 test executions across eight checks, including seven new neutral custody and accounting regressions. Existing native session and observation checks also pass. These tests cover exact scope/schedule identity, all prior source origins, constructor rollback, quarantine union, alias-dependent credit and retirement outside the usage lock. Retirement remains crate-private without a production caller; native chunk/carrier integration and selected-prefill activation are unfinished. Exact commands, source hashes and limits are recorded in [`bounded-capture-source-segment-2026-09-14.json`](validation/bounded-capture-source-segment-2026-09-14.json).

CaptureRunHostPlan includes the fixed p0 target slot box and all measured target/writer/mapping/error owner controls with explicit construction moves. Each final logical tensor shape/data allocation is the existing single P term; fragment writes acquire no additional hold. Multiple partial targets and aborted sidecars retain their original H. Alias retirement releases custody only after payload retirement. Source storage, native transformation scratch, logical capture quota and facade envelopes are independently required and are not supplied by this host placement program.

Persistent host prefill targets passed 1,598 test executions across eight checks, including ten new neutral tests. One original destination per target preserves values and pointer identity across interleaved fragments; incomplete and aborted targets retain their original custody. A reviewed correction checks the complete inference frontier even when all p0 captures are inactive. Native fragment/source binding, settled chunk retirement and selected-prefill activation remain unfinished. Exact commands, sources and limits are recorded in [`bounded-prefill-host-target-2026-09-14.json`](validation/bounded-prefill-host-target-2026-09-14.json).

A closed prefill-fragment selection program is now available as a prerequisite: checked derived local axes feed the same MLX operation sequence used by whole-value capture, and Preview uses each fragment's contribution to the global selected prefix. Cold traces preserve their existing context, source aliases and unknown bounds; signed native geometry failures remain typed. This does not yet connect partial host targets or retire native capture sources between chunks. Existing selected-prefill attribution/readout gates remain unchanged, and neither a finite fragment report nor Sequence row extent proves full-forward semantic equivalence or original admission.

The derived fragment-selection program passed 561 test executions across six checks, including seven new core, metadata and Metal tests. Nonzero F32/F16/BF16 arrays match independent whole-source filtering and global Preview; cold tracing preserves larger backing aliases, lazy sources and unknown bounds. This validates the shared selection leaf, not native source-bound fragment transfer, chunk retirement or full-model capture admission. Exact commands, sources and limits are in [`bounded-prefill-fragment-selection-2026-09-14.json`](validation/bounded-prefill-fragment-selection-2026-09-14.json).

Original capture H includes the measured short-fragment transfer and source-rollback owner controls with explicit construction moves. The target's existing shape/data P is counted once across all fragments. All previous and newly supplied source origins validate with the exact original active scope under the existing Usage lock. Constructor failure restores its source channel; entered-work failure preserves that channel through recovery/quarantine while the frame retains partial data. No per-fragment hold, reservation, source credit, scope or completion grant is introduced.

The private source-bound fragment writer passed 1,606 test executions across eight checks, including eight new neutral tests. It reuses one target buffer, joins exact scheduled source/scope custody, preserves rollback before construction and retains all origins after entered failure. Public construction remains closed pending canonical chunk-stamp validation. Existing native and portable paths pass compatibility checks; this does not establish native fragment transfer or per-chunk memory reuse. Exact commands, sources and limits are in [`bounded-prefill-fragment-transfer-2026-09-14.json`](validation/bounded-prefill-fragment-transfer-2026-09-14.json).

An observable completed-record retirement pass distinguishes a scan that ran from runtime/registry contention. `CompleteSnapshot` does not imply that all physical aliases or charges retired, and reports no byte credit. Unsettled and foreign-thread records remain retained; queued allocation-owned Rust resources are reclaimed separately outside the native runtime lock. A later per-chunk capture path must preserve roots/pins on Busy/error and independently justify actual remaining coverage before reusing a workspace peak. This wrapper increment alone adds no capture integration or managed support claim.

The observable native retirement pass passed 874 test executions across seven checks, including four new Rust tests and the new isolated native executable in full CPU and Metal builds. The executable covers no hidden progress, a complete 129-record snapshot and real registry contention; existing owner-thread fairness remains passing. `CompleteSnapshot` reports traversal only: completion, caller-root disposal and actual storage credit remain separate obligations. Commands, exact native source/library identities and limitations are in [`bounded-observable-native-retirement-2026-09-14.json`](validation/bounded-observable-native-retirement-2026-09-14.json). Selected-prefill integration and the broader bounded-inference goal remain unfinished.

The original-account capture bank now supports claim-free prefill retention bootstrap and an exact request/chunk/epoch success association. Its actual controls and stamped identity payload are included in cumulative host H; escaping registration/ticket aliases retain that same H. Bootstrap spends no frame/tensor claim and opens no scope or reservation. Default observers retain the existing whole-span workspace. The addition does not implement native carrier retirement, per-chunk memory reuse, fragment execution, or selected chunked-prefill admission. Completion-agreement guard creation failure remains an explicit distributed-retention contract gap.

The canonical post-chunk cancellation vote now observes cancellation requested during the retained source retirement callback. The same original reservation guard covers callback, readiness and cancellation; host claims and source pins remain charged. This timing correction adds no native retirement or per-chunk budget credit.

Canonical prefill retention association passed 1,619 test executions across eight checks, including thirteen new neutral tests. The original bank can register an exact chunk without spending a frame claim, and the shared runtime issues its ticket only after the exact model commit and both guarded completion checks. Real neutral session tests cover final delivery, initial and final-callback cancellation, original preparation errors and retirement/settlement failure precedence. The corrected fixture reproduces pre-fix Complete after callback cancellation, then verifies Cancelled and Aborted with the same model/guard path. No source channel is released by this increment; native carrier integration and the inherited distributed guard-creation failure contract remain unfinished. Commands, source identities and limits are in [`bounded-prefill-retention-protocol-2026-09-14.json`](validation/bounded-prefill-retention-protocol-2026-09-14.json).

The stamped short fragment constructor reuses the original full-target H and same native scope; it adds no second hold or scalar capacity certificate. Complete backing pins join the existing capture segment through the private atomic source mechanism. Failed/partial targets stay in the frame and source pins stay with scope recovery. The leaf does not retire records, clear segment pins or assume next-chunk credit. Real stamped native transfer and collector/control composition still require original-work integration.

The actual-source native fragment leaf passed 1,474 test executions across seven checks, including two new Metal cases. Nonzero F32/F16/BF16 strided physical chunks preserve full backing identity and agree with independent global Preview selection; cold preparation checks known backing and actual cast specialization. Whole-value, scheduled and fragment destinations share the existing native selection/settlement/read worker. These cases do not execute the stamped source-bound transfer: that composition still needs the original-step fragment callback and carrier integration. Exact commands and limits are in [`bounded-prefill-native-fragment-leaf-2026-09-14.json`](validation/bounded-prefill-native-fragment-leaf-2026-09-14.json).

Registry publication preserves the existing immortal sizeof(Registry) control payload. Concurrent first-use contenders may allocate one temporary candidate each, freed after a failed single compare-exchange. Scope and owner-tag storage are unchanged. No numerical buffer or memory reservation is introduced; native allocator metadata and ordinary allocation/destructor latency remain explicit control-runtime costs.

Recovery try-begin moves the existing supplied resource/request owner into either the ordinary Node or an inline RecoveryBeginError<T>. Busy allocates neither native scope nor Node and copies no backing storage. Error/owner controls have their actual Rust type sizes; native failure keeps the existing Exception String capacity. A future enclosing host plan must cover those fixed controls and move overlap, and the existing native Scope/owner-tag/registry bookkeeping; this primitive grants no reservation or arbitrary byte credit. A successful try-only user must arrange explicit/ordinary reaping separately, because try does not register housekeeping. Unresolved Drop retains custody under the existing quarantine rule.

Nonblocking native scope entry and owner-preserving recovery passed 889 test executions across ten checks, including six new Rust tests and a new isolated concurrent first-use registry test in CPU and Metal builds. Busy returns before native entry and preserves the supplied resources; successful entry uses existing completion/quarantine. Existing callers remain unchanged. Fixture-only corrections pre-reserve worker containers, retain counters through registry-owned records and use actual runtime entry when asserting housekeeping behavior. Exact source/library identities and commands are in [`bounded-native-submission-try-entry-2026-09-14.json`](validation/bounded-native-submission-try-entry-2026-09-14.json). Canonical communicator fencing and failure-path caller integration remain required; this does not yet repair distributed timeout cleanup.

New native capture controls enter the original request quote: the always-present optional carrier field for at most M inference plus Prompt/Sampling owners; one active box/capsule/attempt; actual native prepared-fragment and metadata snapshot controls; bounded snapshot shapes; and fixed collector capacities derived from the actual plan. The neutral detached parcel enters the original bank H through size_of. Copy operations checked-add their introduced common control to the existing retained safety reserve, preserving user reserve and one atomic account. No second host hold or native allowance is acquired. Channel retirement moves source pins, never subtracts bytes; backing and other aliases retain their actual charge. Existing registry/allocator bookkeeping exclusions remain explicit.

The shared p0 logical-row progression mechanism keeps one full-value logical charge while validating an expected hook in every admitted physical chunk, including zero-output fragments. A private host preparation route stores its monotone state in the actual fixed target slots; their shared-plan handles and construction controls enter the existing measured H. Chunk completion preflights all semantic rows and existing destination coverage before advancing any cursor. This mechanism releases no native source and creates no new hold. Its constructor remains crate-private until an architecture-bound causal/readout companion authenticates activation; current capture gates are unchanged.

The native capture carrier and shared logical prefill progression passed 1,665 test executions across nine checks, including sixteen new cases. Real one-chunk SessionPrefill tests cover canonical source retirement, initial cancellation, original failure and Busy quarantine, and independent native/frame ownership. Neutral tests cover nonzero scatter, full logical quota across uneven chunks, zero-output hooks, Skip/Fail, no-refund failure and exact host envelopes. These results do not establish chunked capture delivery: original fragment callbacks, architecture binding and pre-allocation span enforcement remain required. See [`bounded-prefill-carrier-progression-2026-09-14.json`](validation/bounded-prefill-carrier-progression-2026-09-14.json).

The canonical communicator fence adds an inline atomic bool and root shared_ptr, including ABI padding, to each native GroupImpl. No separate fence state or mark/query allocation exists; split descendants deliberately retain the root communicator. This remains native communicator/control infrastructure within the existing metadata exclusion, not numerical payload or a new inference workspace fact. Native tests report actual built sizes rather than assuming a byte bound. Terminal marking performs no reserve, refund, certification, source unpinning, or native owner retirement. A future complete communicator-construction host budget must account for the actual selected group allocation at its original construction boundary.

Canonical native communicator terminal fencing passed 953 test executions across twelve checks, including two new isolated Rust tests and a native executable with three scenarios run in CPU and Metal builds. The built ABI reports GroupImpl=32 bytes, atomic bool=1, and root shared pointer=16; these are measured representation facts, not a full communicator-construction budget. Tests cover split/cache aliases, a split/mark race, rejection before runtime/status setup, and unchanged unfinished recovery ownership. Portable guard-failure activation and real multi-rank failure remain unvalidated. See [`bounded-native-group-terminal-2026-09-14.json`](validation/bounded-native-group-terminal-2026-09-14.json).

Prefill declaration storage is part of SharedLayeredObservationPaths: boxed declaration entries and each String's actual capacity retire before its existing per-domain custody. Exact aliases preserve the original physical owner. PreparedCaptureSelection owns only source/path aliases and its readout join; BoundCaptureSelection borrows it with exact candidate geometry. control_peak_bytes measures fixed construction/move overlap, without supplying allocation or execution authority. Original admission must price those controls and register the loaded source before enabling the later callback path. No new numerical buffer, source discount, hold, native completion or retirement grant is introduced.

The causal selection companion's fixed control cost includes one sequential CapturePrefillRowAssembly construction/move overlap, including its logical geometry and fixed shape/stride arrays, alongside the retained companion and bound candidate view. This does not cover transient declaration Vec/String buffers rebuilt by cold collection/rebind: those remain original loading/preparation allocations, with the final shared boxed/string payload separately registered. Hot prepared-path token validation performs no declaration reconstruction. No new host hold, execution authority or source discount is implied.

The retained architecture causal/readout companion passed 1,884 test executions across eight checks, including eight new cases. Nonzero common decoder equations compare full Sequence rows with uneven chunks after a cached prefix; exact source/candidate/catalog binding, readout joins, missing/duplicate declarations and source lifetime/capacity are exercised. The fixed control bound includes its actual temporary row mapper. This validates declarations and cold association, not native split-prefill capture or pre-allocation enforcement. See [`bounded-prefill-causal-selection-2026-09-14.json`](validation/bounded-prefill-causal-selection-2026-09-14.json).

The opening-state source closes the runtime access gap needed for exact per-chunk inventory, but does not itself enforce a memory limit. Its fixed control estimate covers the borrowed views, runtime source and both representation adapters in the native observer chain with explicit construction overlap. Concrete state iterators can allocate; their buffers, native inventory tables, host state metadata, model parameters, residency-manager buffers and independent retained history require original admission/custody accounting. The old allocation-free layer-iterator wording is corrected. No production observer opts in in this increment, and complete opening publication, original span-plan custody and the native remaining-capacity gate are still required before activation. Behavioral coverage uses actual `SessionPrefill`, borrowed nonzero state, F4/P5 chunks 2/2/1, cancellation and failure ordering, local/global index separation, true stateless absence and a typed partial traversal failure through the lexical representation adapter.

The canonical opening-state source passed 1,892 test executions across eight checks, including eight new cases. Actual SessionPrefill tests cover current borrowed nonzero state before preparation, cached-prefix uneven chunks, both error boundaries, initial cancellation, legacy fallback, local/global slot separation and true stateless absence. The private source test preserves a typed partial-traversal failure through a lexical adapter. Native/portable builds and existing session/numerical suites also pass. This is source-access validation, not a complete native inventory or an enabled memory gate. See [`bounded-prefill-opening-state-2026-09-14.json`](validation/bounded-prefill-opening-state-2026-09-14.json).

Original span retention has an explicit host contribution: actual record Vec capacity, fixed plan/Arc storage and measured candidate/seal/association/control moves. It is added once to both full and incremental demand; registered controller, decoder or copy-source credit never discounts it. Original source-preparation diagnostics conservatively include the full activation and state-update contributions before equation composition (ordinary native full prompt source/initialization/identity, or full source-inclusive copy preparation), plus original outside attention and materialization. These values may exceed future newly allocated preparation storage; they are not a remaining-balance grant. Unknown required terms remain unknown. Sampler/controller/cumulative capture H stay separately owned. Native opening-state iteration is not uniformly allocation-free: Hybrid/compressed/pooling iterators and inventory collection require their own exact preflight host contribution or allocation removal before activation. No claim is made that equation tracing prices those new preflight controls.

### Retained span plan P

Opt-in span retention prices actual record capacity, fixed attachment/control storage and compact/error-owner moves before finite admission. Successful consuming promotion protects this entire P once inside the original account and attaches it to the actual shared plan. `remaining - host_held` consequently excludes live schedule storage. Independent equal seals, wrong accounts and duplicate attachments reject; failure preserves original quote ownership. Escaped earlier aliases and unwind paths retain the same charge until the records retire. Existing accounts may conservatively retain additional unassigned balance while that host scope remains live.

The owned receipt continues to borrow exact original span geometry and full unreduced source-preparation/materialization terms. Additional source validators remain health evidence only, not scalar credit or an all-source proof. Unknown original terms still cannot be sealed. Current native opening-state traversal and original receipt activation are subsequent integration work.

### Bound prefill fragment observer composition

An additive observer entry consumes the actual `BoundCaptureSelection` and the original scheduled host bank. It keeps one p0 frame and ledger across canonical chunks, charges each accepted full logical selection once, and writes fragments into the existing one-buffer targets. Full/Slice/Preview mappings, including global Preview and batch/head scatter, remain geometry-derived. A nonempty generated Preview(0) still invokes its physical factory; a truly empty slice or quota skip does not. The retained FP8 factory runs once per hook while creation quota is charged once per accepted full selection.

Cold quotation consumes the same bound physical geometry and loaded path owner. Its existing Selection runner retains every transform output through the current equation span; compact inputs and generated intermediates from earlier hooks remain live until that span ends. Missing operation/host facts remain Unknown. Fixed observer controls include the full logical reconstruction shape/plan, and native carrier descriptor capacity reserves the two compact operands plus seven outputs for each eligible selection (shared factories use that envelope only once). These handle controls are separate from traced numerical bytes.

This is a private/native mechanism and pure-quote extension, not selected-prefill admission support. Before activation, the original request must authenticate the accepted per-span numerical envelope and publish/check the actual settled opening state and dynamic source inventory **before source preparation or eager model work**. Existing claim H, a healthy native scope, and a semantic row declaration do not prove remaining numerical capacity. Installation, physical output delivery, native carrier settlement, and managed facade policy remain enclosing obligations; no gate is opened by this increment.

The explicit-source bundle is the immutable bundle on the quote consumed by promotion, including later registered-source health constraints added by original preparation. Existing capture preparation may publish its newly constructed shared source under the original scope and append that exact pin before conversion. Such joins strengthen health/custody validation without changing numerical requirements, the seal, or source credit; they are not an all-source proof.

### Opening-state traversal without a temporary tensor list

The directly retained native layer tensors can be visited without allocating a per-layer vector: the whole-state visitor dispatches to the native layer callback, while portable implementations keep the existing iterator default unless they override it. Compressed resident storage includes both backing stores and both logical views, which can own independent allocations after a checkpoint. Pooling includes local KV plus every pending, pooled and overlap slot in each stream. Aliases are reported as actual fields; physical storage deduplication remains the receiving inventory's responsibility.

This removes the native iterator-container obligation from that direct source visit only. It supplies no original numerical-span receipt, registration, remaining-capacity proof, completion or new capture support. Manager-owned sealed/host blocks, active residency workers and host metadata still require the separate complete opening inventory; existing manager storage reports explicitly preserve unknown completeness for pending work. Callback-created inventory buffers and earlier/later per-unit iterator allocations are not made free by this change.

The combined span-accounting, fragment-observer and direct-state-visit increment passed 1,976 test executions across ten checks, including 44 new cases. Coverage includes original-account schedule lifetime, exact/one-short funding, competing spends and source quarantine, full logical quota across physical fragments, generated/empty/skip cases, and direct native borrowed fields. Validation exposed and corrected an unconditional encoding check on skipped rows; the existing shared delivery seal retains the successful-tensor check before commit. The native fragment observer remains unjoined to a complete live opening inventory and original installed receipt, so this record does not enable selected-prefill execution. See [`bounded-prefill-span-capture-2026-09-14.json`](validation/bounded-prefill-span-capture-2026-09-14.json).

PrefillOpeningExecution is a lexical source view, not registered storage or headroom evidence. Its fixed control_peak_bytes reports the actual view plus the selected runtime-reference/function-pointer wrapper and three explicit construction/move overlaps. No owner table is allocated by that adapter. The enclosing original preparation must separately account for an opted-in mechanism's inventory containers, retained errors/handoff controls, concrete traversal and native metadata. This contribution does not reserve memory, certify a bound, subtract a retained-state scalar, create a source registration or replace the complete settled opening inventory. State and execution remain owned by the same exclusively borrowed session; no new reservation or hold is made by this hook. The mechanism must preserve partial resources on failed collection, while incomplete execution coverage remains explicit even after known values were visited.

Native live opening-source collector foundation (activation remains unwired):
`MlxReplicatedTextMechanisms` can privately bind a request-scoped collector for
its canonical pre-prepare callback. It borrows the actual state and current
execution visitor supplied by the shared runtime. Four separate inventories
retain state/managers, host slot/layout tokens, weight-manager/checkpoint sources,
and execution arrays; duplicates retain their full physical backing and unknown
or partial coverage cannot become complete. No family/residency branch, model
reacquisition, publication, origin-health grant or new funding scope is introduced.

The private guard closes collection on drop without clearing pending inventory;
exact request/chunk/epoch checks protect the one-use successful handoff. Error and
unwind preserve completed domains and visited execution prefixes. Existing
Result-returning builders can discard their own partial temporaries on error;
those underlying owners remain with the original session/recovery, and this
foundation does not claim a complete replacement recovery inventory. Host-slot
tokens retain accounting custody, so actual state/table ownership remains required.

Fixed field/slot/guard/inventory construction controls have measured helpers.
They are not complete collector bounds: BTreeMap and Vec capacities, SourceStorage
reconstruction, actual state/manager traversal, temporary inventories, Rc allocator
overhead and original enclosing controls remain separate. These obligations,
original-account publication/health validation and retention alongside the actual
operation must be resolved before wiring the private binder. Prediction banks,
outer model/blueprint sources, controllers, copy/sampling joins and current work
roots are not authenticated by this four-domain collector. Existing manager locks
may block; this is not a new bounded nonblocking manager-inspection guarantee.
Managed gates, span debit exclusion, fragment activation, admission and installation
are unchanged. Seven source/ownership tests exercise the private collector; they
do not constitute canonical runtime-hook, managed budget or native transfer proof.

Selected prefill cold inspection can now enter through the actual MLX executable for resident, retained-host and direct-disk parameter routes, including registered opening decoder roots. It reuses the bound observer and existing configured sampling equations: a full Sequence readout remains observable before the final sampling row is selected. Host-plan H is returned separately, and original source/control ownership, accepted-span admission, current opening publication and native activation remain enclosing obligations. Ordinary and raw-capture quotation retain their previous behavior and attribution gates.

The original private capture quote now pays retained span-plan P before finite admission and promotes that exact accepted plan under the original funding run after capture-source publication. An installed alias keeps the same plan charge after quote/collector retirement. New selection/pending/install fixed controls are measured and included in the original enclosing reservation, separately from capture H, source C and P. That reservation inclusion is not yet a live-control host hold: these controls and existing native carrier/common controls must be protected from remaining-span reuse before activating the marker. Native collector dynamic buffers and complete outer-source health/publication are also still required. The current change enables no remaining-balance or selected-prefill execution gate.

The current-source hook and original capture quote/install increment passed 1,949 test executions across nine checks, including 18 new cases. Neutral tests traverse actual current state/execution under SessionPrefill guards; native collector tests cover returned domains, source aliases and partial/error custody; executable and original-admission tests cover resident/host/disk cold quotation, physical readout, exact/one-short admission and retained plan lifetime. The native opening binder remains unwired. Retained native/control holds, bounded collector construction, complete outer source health/publication and actual fragment activation remain required. See [`bounded-prefill-live-install-2026-09-14.json`](validation/bounded-prefill-live-install-2026-09-14.json).

The neutral original text-span promotion protects both span-retention P and named native control Q=A+C+W in one atomic original-account host hold. Merely reserving Q in an outside retained estimate does not protect it from remaining native headroom; sealing adds P+Q once, and promotion excludes that aggregate from remaining minus host_held. Unknown or overflowing facts cannot be promoted. Earlier shared plan aliases and compact control guards retain the same hold, including failed post-attachment validation and unwind. Tests cover original H coexistence, exact/minus-one capacity, zero-byte source health, alias lifetime, and the actual internal text-span activation dispatch. Native control measurement and owner wiring, opening inventory publication, dynamic publication collector bounds, and public route activation are separate unfinished obligations.

### Native leaf visitation within opening inventories

Audited MLX leaf retained-value visits use field references and the existing
borrowed tensor representation. The visit itself allocates no tree/map/name or
array handle, evaluates nothing, and invokes no native housekeeping. Callback
work belongs to the caller and is outside that claim. Native leaves retain no
new fields or controls, so this change introduces no retained host capacity.

This removes one temporary allocation source from the opening inventory path.
It does not bound `RetainedStorage` maps, checkpoint source inventories,
manager/state collection, metadata snapshots, publication or pin controls.
Their complete original host bound and recovery ownership remain prerequisites
for activation. No source-health, remaining-span, native execution, or release
authority follows from this visitor.

Retained native storage scanning uses allocation-only Array and immutable-host inspection rather than allocating shape snapshots. Results preserve full backing capacity, alias identity and unknown lazy storage. Inventory maps, ownership slots, checkpoint and manager traversal still require their own bounded construction and original-account custody before native activation.

The nonblocking cache-worker retained-work query is a prerequisite for a bounded manager inventory pass. It avoids waiting for the worker registry and copying its contents, preserving busy/unknown rather than reporting an empty worker. Manager traversal, owner-slot capacity and source-health publication remain enclosing requirements.

The original control-custody and inspection increment passed 2,078 test executions across thirteen checks, including 29 new tests. Original P+Q is sealed and held once, follows actual pending/installed/historical/work owners, and survives aliases and certification. Native leaf visits borrow fields directly; allocation-only metadata and nonblocking worker inspection remove additional temporary allocation or wait paths. Native opening-source slot capacity, complete bounded publication and actual chunk activation remain unfinished. See [`bounded-prefill-control-inspection-2026-09-14.json`](validation/bounded-prefill-control-inspection-2026-09-14.json).

### Checkpoint source visits for bounded inventory construction

Borrowed checkpoint storage visits build no temporary source map or payload
copy. Their successful per-owner facts and same-Arc retention have fixed
representations. Visits may repeat aliases; a collector must preprice its
slots, check consistent capacities, deduplicate identities and checked-sum
the unique bounds. Completeness describes visited owner coverage, not a
successful aggregate sum, source-health check, or admission.

Callers can retain each owner directly into their prepared storage before
continuing. Later source errors or unwinding do not discard that caller-owned
prefix. Final payload destruction remains ordinary Drop and must occur
outside manager/accounting locks. The visitor itself does not price callback
work, exception construction, catalog/recipe metadata, external leases,
publication/pin controls, or operating-system caches. Original-account host
custody and full opening-inventory composition remain separate prerequisites.

### Borrowed manager inventory prerequisite

The private MLX manager visitors enumerate current physical fields without
constructing `RetainedStorage`, source maps, shape vectors, names or an owned
snapshot. Weight completeness checks every selected ledger unit for pending
host/device copies, including units absent from the storage map, and retains the
failed-transfer flag. Sealed-cache completeness retains worker, pending-ticket,
background-error, write-reservation and retiring-read/demotion evidence. A busy
or poisoned worker registry is incomplete; it is never interpreted as empty.
Independent worker flags/registries are checked before and after callbacks.
This is point-in-time evidence, with no barrier against later work.

This addition creates no retained manager field or heap object. Its local enum,
iterator, mutex guard, error carrier and caller-error slot are inline controls;
a future original collector must price those actual representations alongside
its concrete callback/control peak. Array inspection clones, owner-slot storage,
pins, publication records and source-provider error controls remain separate
obligations. There is no pre-admission maximum count, ledger-entry lease, growth
seal, complete outer-model inventory, budget credit or activated gate. Existing
managed quote/accounting limits are unchanged.

The borrowed checkpoint/native-manager increment passed 397 test executions across eight checks, including 21 new tests. Checkpoint source references retain the same physical Arc without temporary source maps. Weight and cache managers borrow actual source/storage fields under nonblocking state loans, preserve unknown and pending-work evidence, and return callback failures after unlocking. Closed owner-slot bounds, original control-custodied construction, complete bounded publication and chunk activation remain unfinished. See [`bounded-prefill-borrowed-manager-2026-09-14.json`](validation/bounded-prefill-borrowed-manager-2026-09-14.json).

### Opening inventory slot-count prerequisite

Cold diagnostics can now bound fixed native state fields, actual host slot/layout
roles, immutable physical module fields, and weight-manager declared bindings.
Absent concrete tensor fields reserve their future slots. Weight maps allow one
host and one device owner per declared binding, including unloaded units and
aliases; source bounds count every physical source branch. Unknown/overflow and
busy policy state do not become complete bounds. Paged state counts include its
fixed tails and manager roles, not a guessed cache-catalog growth allowance.

This prerequisite allocates no collector and adds no host hold. Original quote
sealing, measured owner/publication/pin controls, retained exact source bindings,
request-bounded cache growth and later entry/lease validation remain required.
Idle bounded modules absent from opening are future materialization work; the
original span must actually protect that work before execution. Existing dynamic
publication maps and manager/worker/catalog metadata are not priced by these
handle counts. No original or controlled capture gate is opened.

### Terminal prefill failures retain unresolved work

The shared source/prefill runner attempts reservation retention before source construction, cancellation agreement, chunk preparation, nested model work, completion agreement, and final indexing. Failed entry fences the session and returns before the next callback or vote. A peer already waiting in its selected bounded agreement may time out; that error marks the retained communication incarnation and abandons both nested and outer guards without calling successful completion waiting. No settled chunk ticket, native certification, capture-source release, host-budget refund, rollback graph, or replacement agreement follows a terminal failure. The first local cause wins over a later communication error, and an active epoch is not rewritten as an agreed abort.

A successfully agreed ordinary source rejection keeps its existing `BeforeStateMutation` classification and cause. Cancellation remains an ordinary settled status result. Local failures after a completed negative agreement retain existing rollback policy, while their guard cleanup still uses nonblocking abandonment. The enclosing observation guard aborts provisional delivery once. On unwind, the shared wrapper removes its field-held inference guard, fences without native work, and resumes the exact panic payload; outer guards then abandon normally.

Neutral tests exercise actual shared sessions with two rank-local states and bounded host rendezvous, all seven guard-entry cuts, ordinary source retry, original error/panic identity, healthy phase counts, and unresolved account retention. Native tests exercise the actual MLX entry under runtime contention, the portable terminal-mark bridge to canonical native aliases, and recovery owners whose probe remains pending with neither failure nor blocked status. Native multi-rank injected-entry/timeout validation remains required; singleton and neutral tests do not prove remote notification, collective cancellation, or native peer completion. Pending native work may remain quarantined indefinitely without independent completion evidence. Original communicator host construction coverage remains a separate accounting obligation; no capture or remaining-span gate is opened here.

The combined count integration also forwards all sixteen remaining built-in architecture static aggregates. These replace the initial generic-decoder-only coverage described above; nested unknown module bounds, native cache growth, original fixed-capacity construction, storage publication and final activation remain separate requirements.

The capacity-diagnostic and terminal-cleanup integration passed 2,458 test executions across 17 checks, including 29 new tests. Validation covers module/source/state/weight slot counts, all built-in static count forwarders, exact nonblocking native entry, canonical terminal aliases, original-cause preservation, and shared-session failure/unwind cleanup. These results do not activate bounded native capture or complete owner publication. See [`bounded-prefill-capacity-terminal-2026-09-14.json`](validation/bounded-prefill-capacity-terminal-2026-09-14.json).

### Finite existing-only grouped-pin controls

`PreparedPrefillStoragePinPlan<K>` derives rows from the actual accepted equation
schedule and binds the key type plus per-row slot ceilings before sealing. Its S
term sums every row's potentially surviving grouped pin or terminal failure, the
fixed input/unique-key/ordinal buffers, independent layout, Arc payload/counters
and measured constructor/move controls. Original text promotion protects P+Q+S
with the existing single host hold. Bank extraction and pinning acquire no second
hold or reservation and transfer no numerical capacity. Existing allocations,
including zero-byte entries, retain their original registry origins.

An issued row remains spent on Busy, validation failure, panic or Drop. Terminal
errors keep their input owners and custody; there is no replacement/retry API.
Busy before issuance leaves the bank unchanged. Group commit validates the exact
canonical stamp/slot/account and all group origins under one Usage lock; all
provider comparisons finish before owner increments. Scratch and key destruction
occurs after unlocking. Escaped groups remain healthy source pins after normal
run closure, while quarantine is still rejected. Native/source owner allocations,
key-owned heap payload, collector capacity proof and new-entry publication remain
independent obligations; this unit does not replace the existing registry engine.

The subsequent topology-count correction closes two earlier diagnostic gaps: generic absent module children now report zero for their actual selected topology, and native packed embeddings expose their extra physical array. Tied/untied dense native construction and nonzero packed alias fixtures cover these cases. Explicit same-topology lazy fields reserve future slots; opaque custom owners, supplementary prediction/codec domains, paged catalog growth, original fixed collector construction and storage publication remain separate requirements.

The strict Qwen causal-row fixtures exposed missing effective input/output callbacks in the shared outer unit traversal. Both prepared and legacy traversal now emit those callbacks from the actual consumed value; tests keep every declared hook required instead of synthesizing effective rows in a test observer. Prepared source accounting includes the additional retained string capacities. Behavioral coverage also compares callback order and replacement values and verifies effective-input failures stop state work and release the active unit.

Outer unit discovery includes the actual effective input/output emitted by shared and custom traversal. Catalog completeness checks remain strict: neither unknown support nor synthetic captured values stand in for missing points. These read-only declarations preserve the original geometry and do not grant chunk-assembly semantics, native capture capacity, publication authority or a managed capture gate.

The finite existing-pin, topology-count and Qwen causal-row integration passed 2,280 test executions across eleven checks, including 24 new tests and six updated existing tests. These validate original one-time pin custody, atomic existing-only registration, terminal failure ownership, absent optional topology, packed embedding backing visibility, Qwen dense/routed causal observation rows, and actual effective unit-boundary delivery and discovery in shared, custom and partitioned traversal, including strict recovery of scalar resident loans. Bounded native collection/publication and managed activation remain unfinished. See [`bounded-prefill-fixed-pin-topology-causal-2026-09-14.json`](validation/bounded-prefill-fixed-pin-topology-causal-2026-09-14.json).

### Exact original C publication and its lifetime

The original text seal now combines measured P, native Q, optional finite pin
controls and fixed single-source publication controls under one host hold. Exact
new SharedCapturePlan capacity C is an additional original reserved term outside
that hold. The backend enclosing estimate contributes H once and no longer adds
C separately. An exact retained healthy one-key source pin derives C=0; an
independent source cannot supply that fact. A same-domain race preserves the
original nonzero C reservation rather than granting retrospective credit.

Pending construction follows the original plan attachment. Publication stages
one key and registration, validates every original source and the exact scope
under Usage, then mutates the ordinary registry without further provider key
comparisons or clones. Typed source reuse retains the established private owner;
an opaque or wrong-type attachment cannot stand in for that owner. The final
fixed S+C witness is joined without rebuilding a variable source buffer. Busy,
provider rejection, late quarantine, panic and Drop produce no replacement
attempt or readiness grant. Source and plan aliases keep actual original custody.

After the original run closes and all native scopes settle, a healthy account
retains only its protected host balance and independently registered live storage.
An earlier C source alias therefore keeps original P+Q+S and registered C, without
keeping unused equation/controller headroom. Native scopes, quarantine and an
active span still retain the full remaining envelope. The original capacity
ceiling remains until ordinary account retirement or an existing explicit handoff;
a later larger requested ceiling cannot override it. A later independently quoted
request may use released terminal headroom within that ceiling, with every old
live charge still included. No alias receives new execution credit, reservation
or quota refund. Final owner tests require complete release after all genuine
aliases retire.

Tests cover original exact/one-byte-short admission, existing and raced C, typed
reuse and physical-source mismatch, opaque/wrong-type custody, all-origin health,
late original-scope spending, Busy/poison and provider unwind outside locks. These
source-publication tests do not establish native opening-inventory completeness
or authorize additional managed capture geometry.

### Fixed opening collection is a separate unactivated mechanism

Private native opening collection separates array, immutable-host, byte-buffer,
checkpoint-source, shared-layout and host-slot owner capacities. Aliases and
zero-byte owners consume slots. An actual weight manager contributes its entire
immutable binding ceiling and every physical source role; settled current
visitation must still succeed under its nonblocking state loan. Unknown topology,
excess visits, pending/incomplete inventories and unknown physical backing reject
without evaluation, polling, or an allocating collector fallback.

The exact plan's fixed Vec payloads, per-entry fact slots, borrowed bindings, per-retained C++ array handles and
constructor/error controls join the original P+Q seal before allocation. No
second reservation, later arbitrary-byte hold, refill or rebind is introduced.
The capsule cannot reset or clone; facts are readable only after complete
collection. Its error path retains the source prefix and original host custody; an escaped
collection error keeps a compact original control guard after its cause.
Host-slot tokens cover inline tables while the lexical state borrow retains the
actual tables; source/catalog and numerical payload obligations remain separate.

This mechanism does not close paged catalog growth, exact later lease acquisition,
persistent session rebinding, bounded publication/attachments, existing shared ArrayDesc/graph
and manager/catalog enclosing ownership, or native span activation. Tests of the
original control seal do not certify those separate domains or enable capture.

Inspection retention includes the single native C++ handle allocated for every
possible array slot, including duplicate aliases and currently absent fields.
The linked native implementation supplies its handle size without opening a
device or acquiring the runtime lock. Successful cloning shares the existing
`ArrayDesc` and backing; it creates neither a copied descriptor nor a second
C++ handle during a Rust move. This fact excludes allocator bookkeeping and
error-reporting allocations and is not a total-process memory guarantee. Fixed
collector installation and later-acquisition proofs remain separate requirements.

### Terminal host-only funding retention

Existing funding accounts now distinguish native work scopes from closed host
custody when minting them under the same Usage lock. The total scope count still
retains identity and domain policy; a separate checked native count controls
transient workspace retention. Original sampler, decoder/pending input, capture
and accepted span host holds cannot publish native storage or become native span
authority. Their source pins and exact held contributions remain intact.

Terminal settlement releases only `remaining - host_held` after run closure and
last native certification, while quarantine and an active span prohibit trimming.
Physical storage retirement then releases its own charge without refilling an
inactive transient allowance. Host-to-storage handoff still atomically moves the
same amount from held/reserved to registered, including after terminal trimming.
Historical quote/peak values, logical capture consumption, capacity-token policy
and all managed applicability gates are unchanged.

Recovered poisoned accounting locks quarantine existing funding accounts before
host cleanup, metadata retirement, storage retirement or run closure can settle
them. Physical owners still retire outside the lock, but their retired funding
returns to conservative custody; poison never becomes reusable headroom. Normal
closed host tails retain exactly their live held bytes and registered roots.

Original capture-source publication, healthy terminal host retention and fixed native opening prerequisites passed 2,454 test executions across eleven checks, including 39 new tests. Coverage includes the exact original one-source seal, typed custody and reuse, failures retaining original ownership, fixed native owner slots, a cold native handle-size fact, and exact surviving host charges after run/native closure. These checks do not activate the remaining native span path or certify its full enclosing inventory. See [`bounded-original-publication-terminal-opening-2026-09-14.json`](validation/bounded-original-publication-terminal-opening-2026-09-14.json).

### Lexical native opening pairing

The private fixed-opening plan can be prepared from one actual quiescent typed
MLX session. State, execution, source store, weight manager and the retained path
token all come from that session. Existing fixed-capacity derivation, source
checks, one-time seal and original P+Q custody remain in force. The paired
preparation error's concrete representation is included in the existing measured
control peak; borrowing the tuple creates no allocation, reservation or grant.

This closes the lexical pairing seam only. Persistent erasure/rebinding, later
cache acquisitions, provider/bank and full enclosing-owner inventory, bounded
publication, canonical capsule retirement and remaining-span activation still
require their own joined contracts. Partition-local execution/path pairing is
unimplemented here, not a model-family limitation. No capture gate changes.

Paired session inspection, nonblocking selected-policy visits and additional causal target-row declarations passed 2,694 test executions across eleven checks, including 34 new tests. Same-session lexical pairing retains exact owners and prepared-token identity; actual selected-policy contention rejects incomplete inspection. Nonzero GPT-OSS, K2, Qwen hybrid, LFM2 and Nemotron-H cases compare real callbacks and complete mutable state across a cached prefix, uneven chunks and repeated decode. This does not activate full native opening collection or establish persistent pairing and complete enclosing inventory. See [`bounded-expanded-family-causal-inspection-2026-09-14.json`](validation/bounded-expanded-family-causal-inspection-2026-09-14.json).


### Existing-only opening snapshot pin slice

The private native snapshot adapter derives its finite descriptor bound by
checked addition of actual array, host-buffer, byte-buffer, checkpoint-source,
layout and slot-table categories. Aliases retain separate owner slots but share
the canonical physical charge. Empty native allocation sentinels contribute no
backing key; real zero-byte source and metadata identities still validate.

The original proposal measures the detached snapshot, preparation failure and
escaped error controls in Q, alongside the existing fixed-owner buffers and each
inspection clone's native handle. Its typed finite pin layout contributes S
before reservation. The same final P+Q+S custody protects handoff, key staging,
registered pins and escaped failures. No diagnostic count or foreign control
receipt can construct a second native snapshot under that charge.

Prepared resident, host-layerwise and disk-streamed test sessions exercise the
real canonical callback and intentionally stop before input preparation. A
separate resident forward changes KV storage; a new snapshot then rejects the
missing keys until actual funded publication. These tests do not establish
current-inventory freshness or complete bounded host/disk execution. Future
paged acquisitions, changed cache/weight windows, full enclosing owner
inventory, comparison-free new-key publication and bounded native attachment
are still required where those resources occur. Healthy terminal trimming may
release only the original non-host tail; snapshot/error custody retains its
original aggregate until the final owner retires.

Existing-only native opening snapshots and neutral stamped opening-group custody passed 2,066 test executions across nine checks, including 14 new tests and one strengthened activation test. Native snapshots retain actual inspected owners under the original finite bank; neutral installation validates exact original stamp, account and all physical origins before moving grouped custody into canonical retirement. These are separate prerequisites: persistent current-session binding, complete opening inventory, new-key publication and bounded native attachment still need to be joined before managed activation. See [`bounded-opening-snapshot-stamped-group-2026-09-14.json`](validation/bounded-opening-snapshot-stamped-group-2026-09-14.json).

Prepared native allocation-owner nodes close the fixed handoff mechanism for
completed certified Array and immutable host backing. Cold preparation may wait
for the existing one-time error-handler initialization; final attachment cannot
enter that initializer. Physical backing and the charged native callback node
retire before enqueueing the Rust accounting owner; Rust-node storage is also
freed before its payload can release the original charge. Fixed normal representation
facts include both heap nodes, the named constructor state and result/error
wrappers; a future closed opening publisher must derive finite counts from its
actual original proposal and include those costs before the same original seal.
Busy, absent/uncertified backing and allocation failure return the complete owner.
Logical emptiness alone does not imply absent backing: the CPU allocator can
retain a zero-capacity physical allocation, while Metal returns null for an empty
allocation. A known allocation-info result can describe that allocation-free
empty value using the zero identity sentinel; it is not proof of physical
backing. Prepared attachment requires a nonzero certified physical identity and
keeps its owner through the final backing alias. An exceptional native error retains
its original diagnostic source, whose dynamic
allocation remains an enclosing failure-budget obligation. This mechanism neither
funds nodes nor publishes registry keys. Original P/Q/S association, canonical
new-key commit, partial-prefix recovery, metadata attachment, persistent current
pairing, future paged acquisitions and complete outer inventory remain explicit
publication obligations. Existing deferred owner attachment remains dynamic.

Canonical finite capture publication, Muse causal rows and prepared native allocation-owner attachment passed 2,161 test/case executions across twelve checks, plus four existing standalone native regression executables. The increment adds 25 Rust tests and six C++ cases. Original finite publication validates every origin before atomic canonical insertion; prepared native handoff preserves aliases and frees its native/Rust control nodes before releasing accounting custody. Muse uses the shared causal-row contract through actual prepared composite paths. These mechanisms still require the original native session, opening/completed inventory and outer-resource composition before managed activation. See [`bounded-publication-causal-attachment-2026-09-14.json`](validation/bounded-publication-causal-attachment-2026-09-14.json).

DeepSeek V3/V4 ordinary causal declarations passed 1,129 test executions and portable/native compilation across four checks, including ten new tests. Nonzero fixtures compare actual outer/readout hooks and complete persistent state after uneven chunks and three cached decodes; V4 includes a mature ratio-128 compression boundary and same-next-decode verification of per-call attention scratch. Declarations retain original source/readout identity and existing proposal/observation availability. These family equations do not themselves activate managed native capture or cover every runtime wrapper. See [`bounded-deepseek-causal-prefill-2026-09-14.json`](validation/bounded-deepseek-causal-prefill-2026-09-14.json).

Kimi Linear ordinary causal declarations and the width-one KDA correction passed 1,136 test executions plus portable/native compilation across four checks, including seven new tests. Fourteen nonzero configurations exercise real callbacks, every fixed/compressed state field at uneven continuation frontiers, and three cached decodes. Width-one KDA keeps its recurrent matrix and uses no convolution histories; exact source/readout and wider-kernel rejection behavior is preserved. Selected-wrapper opt-in and native managed capture remain separate work. See [`bounded-kimi-linear-causal-prefill-2026-09-14.json`](validation/bounded-kimi-linear-causal-prefill-2026-09-14.json).

Selected ordinary causal forwarding and the LFM2 stateless convolution/local-mask correction passed 1,143 test executions across four checks, including seven new tests. Actual prepared owners preserve original callback paths and physical readout across resident, host-layerwise and disk execution. Width-one NoState frontiers remain zero, while attention reads its actual local cache offset. Prepared TP/PP regressions exercise leading and trailing stateless cuts. Kimi selected opt-in and original native managed activation remain separate work. See [`bounded-selected-shell-lfm2-2026-09-14.json`](validation/bounded-selected-shell-lfm2-2026-09-14.json).

### Finite native opening/end rows under the original text seal

The private joined proposal derives both fixed capsules, descriptor mappings, all possible native attachment nodes, output slots and wrapper/error controls from the actual prepared native topology and original prefill schedule. It seals its own named A/C/W facts with original C publication, finite pin rows and finite publication rows before reserve. No arbitrary count-plus-guard constructor, second reservation, later quote or refill is introduced. Accepted P+Q+finite S is held once; source payload C and scheduled capture H remain separately owned. The completed batch validates every existing/new origin before its non-fallible registry commit, then maps each unique result through the batch's first-input ordinal. Nodes and their Rust/native fixed representation are prepared before inspection/registry commit. Existing aliases deduplicate in the canonical registry, while per-retain handle and attachment storage remains priced independently.

Q includes two owner capsules per chunk (including linked native array clone-handle bytes), all potential attachment allocations and list/control representations, fixed index/output buffers, retained identity/fingerprint control blocks, row/bank/move/error wrappers and the joined mechanisms weak slot. Allocator bookkeeping, exception-message allocation and earlier construction of that always-present loaded mechanisms slot are not proved by these sizeof facts. The latter remains an enclosing loaded-owner admission obligation. An attached node can outlive the row bank, historical quote and run; its opaque raw custody keeps the original aggregate until the actual backing dies. Terminal host-only trimming therefore preserves that charge without retaining the unused full original ceiling.

The real native fixture quotes actual captured equations, drives nonzero 2,2,1 chunks through SessionPrefill, and compares final KV, logits and captured values to a full-span run for resident/host/disk weight routes. It also tests exact/minus-one original admission, source-generation rejection, completed-source ordering, cancellation, publication/attachment prefixes and escaped alias/error custody. These are source-staged tests until the central validation record reports execution. They do not prove the remaining accepted-span activation check, full enclosing graph/recovery/parameter/processor/communication inventory, paged growth, future non-native acquisitions or native partition composition.

The joined native fixture calls the actual row proposal/seal, original core reservation/C promotion, installer, SessionPrefill and carrier, but does not invoke the new private `admit_with_capture_opening_rows`/`CaptureAdmission` entry. Its inactive-source cases prove proposal rejection. Direct private TextQuotation coverage of returned bank installation and inactive-source skipping remains an explicit composition test obligation.

The original native opening/completed-owner composition passed 2,124 test executions across nine checks, including eighteen new tests. Actual 2,2,1 native capture matches a full-span reference across resident/host/disk weights, with canonical original publication and prepared allocation attachment. Failure, cancellation, stale binding and escaped-alias/error tests preserve the original accounting owner. Direct private CaptureAdmission entry/skip-branch coverage, complete enclosing inventory and gateway activation remain explicit follow-up obligations. See [`bounded-native-opening-join-2026-09-14.json`](validation/bounded-native-opening-join-2026-09-14.json).

Gemma4 causal text, corrected sliding-history execution and declared decoder partition ownership passed 1,433 test executions across six checks, including fourteen new tests. Actual group2 callbacks and complete publisher/receiver state preserve uneven continuation and three decodes; eighteen prepared TP/PP/residency worlds cover shared-cache boundaries. Shared mapping now retains complete-unit observations on their actual logical execution replicas. Native model-session regressions also pass. This does not activate managed capture or establish complete outer-resource coverage. See [`bounded-gemma4-causal-prefill-2026-09-14.json`](validation/bounded-gemma4-causal-prefill-2026-09-14.json).

Selected Kimi causal forwarding passed 1,160 test executions across four checks. Four new tests replace the old hook-only rejection test and construct168 actual prepared sessions over eight KDA/MLA configurations and three residency modes. Every real declared row and complete prefix/final/decode state compare across full and uneven cached continuation. Original source identity, physical readout and routed/MTP constructor rejections remain intact. See [`bounded-selected-kimi-causal-2026-09-14.json`](validation/bounded-selected-kimi-causal-2026-09-14.json).


### Original private opening admission composition fixture

The native opening join has additive source coverage for its previously untested private original-admission entry. A full three-token captured reference is compared with one-token chunks on resident, host and disk routes, including every nonzero readout-embedding value and token across prefill and three decodes. Capture limits remain one selection per logical step and four for the run. Exact capacity and one-byte-short cases use the actual original candidate quote; inactive empty/decode-only/skipped-prefill sources produce no row bank. Original frame H and source P+Q+finite S+C are checked through their distinct last aliases. The fixture uses the real core context and installer through a scoped test-only entry selector, without a public gateway opt-in or replacement native facts. Test-only diagnostic containers remain fixture ownership; no complete enclosing-host bound is claimed. Compilation and native execution remain pending central validation.

Private original capture admission and installed selection forwarding passed 265 native test executions plus a backend test build without default features. Three new tests drive thirteen real continuations through original admission, core context binding, bank installation and four predictions, including twelve exact/minus-one admission pairs. The actual installed companion now reaches the existing bounded prefill observer through shared native checks, preserving LastPosition projection and complete captured pre-readout rows. Original Q includes the introduced borrowed controls. Full versus chunked capture agrees across resident/host/disk, inactive capture creates no row bank, and original accounting survives frame/source aliases. Public eligibility and full enclosing inventory remain unfinished. See [`bounded-private-capture-opening-admission-2026-09-14.json`](validation/bounded-private-capture-opening-admission-2026-09-14.json).

Inkling causal prefill passed 1,170 test executions across six checks, including eight numerical/semantic tests and two typed error-adapter tests. Actual group-two rows, four convolution histories, width-one omission, scaled readout, bound expert values and full physical logits match direct/prepared and TP/PP/residency execution. The four auxiliary state adapters retain typed missing-role errors. Native capture activation and complete enclosing error accounting remain separate work. See [`bounded-inkling-causal-prefill-2026-09-14.json`](validation/bounded-inkling-causal-prefill-2026-09-14.json).

The combined Qwen integration passed 1,331 test executions across five checks, covering all twenty-four new tests from the hybrid width-one, zero-section mRoPE and conditional causal-row changes. Full numerical architecture regressions, neural contracts, native rotary behavior and portable facade checks validate the integrated sources together. Nonzero loaded parameters, complete recurrent/KV/history state, every physical observation row and final public output are preserved across the tested prepared/parallel/residency matrix. These results do not activate native managed capture or certify complete enclosing memory ownership. See [`bounded-qwen-causal-integration-2026-09-14.json`](validation/bounded-qwen-causal-integration-2026-09-14.json).


The final-only prefill consumer retains only the final span output. Previously an
intermediate physical Sequence output survived until a later result replaced it;
that owner is now released immediately after its own settled callback. Consumers
which need every score still use the original per-chunk API. This removes that
specific overlap without changing the quoted geometry or claiming that all
enclosing/input/native graph owners are accounted for. The source package requires
central execution before these new tests constitute validation.

The combined prefill and ownership stack passed 3,427 test executions across eleven checks, covering all eighty-six new cases from its thirteen reviewed constituents. Core/runtime and full architecture numerics, native patch CPU tests, Rust CPU/Metal completion tests, native model sessions, portable facade and feature builds validate the final sources together. Original token-result/capture/pin/plan lifetimes and typed error retirement share their original admission; settled intermediate outputs are released through the common prefill driver. Selected Moshi resident/host/disk frame conformance is included. Full native span activation, facade result/copy integration and remaining outer host ownership remain incomplete. See [`bounded-prefill-ownership-integration-2026-09-14.json`](validation/bounded-prefill-ownership-integration-2026-09-14.json).


### Original native retained sequence preparation (staged source)

Native sequence preparation now binds R to the genuine core admission before
Prompt or Sampling. Ordinary, controlled and detached core entries share the same
planner and quote. The accepted quote stores one consuming bank; its historical
aliases retain the same aggregate after extraction. Static native preflight and
replay errors hold no account. A consumed foreign/fenced bank and a failed mutable
provider keep their original custody through core's concrete error retirement.
Freezing the provider keeps raw aggregate custody on the same result storage;
reservation metadata alone does not retain a credited decoder root after its
funding run ends. Only declared registered-source witnesses make that stronger
source-retention claim.

Ten staged native tests cover actual resident/host-layerwise/disk startup through
all three core drivers, capture composition, exact admission and one byte short
with a one-position candidate, original token/EOS layout deltas, busy/replay and
genuine foreign claims, fenced construction/provider errors, dormant/zero cancel,
unwind, alias/iterator lifetime, and saved-copy rejection before work. A real
three-token 2+1 prefill followed by three decodes commits sampled token ids into
the original fixed provider and compares all routes. These tests manually prepare
the extracted storage; they do not establish facade first-finish-step wiring.
They are source-reviewed fixtures awaiting the root's serial native validation;
no execution result is claimed by this stage.

Original admission failures carrying promoted R use the direct core error owner.
Later native row/callback error erasure and complete opening/outer inventories
remain unactivated boundaries. The fixed cold-capacity builder accepts the same
claim before sealing, but its separate path has no new execution test here.
Copying an R-owning continuation remains rejected pending destination admission;
this is an integration obligation, not a model-family limitation.


The native retained-sequence fixture correction supersedes its initial lifecycle assumptions. Both the consumed-bank fence and the dormant-provider fence occur in the genuine extraction callback before Prompt; the latter first constructs the actual provider and then closes the still-local original funding run. Each prediction drains `take_completed_delivery`, including the no-capture `None` result, before advancing again. Capture source/span/R assertions use scalar facts observed after successful original installation, since core's retained-copy boundary intentionally rejects even after bank extraction. The successful driver separately asserts that typed `CopyNotAdmitted` rejection. Native copy-entry coverage uses the actual completed Sampling state moved by a scoped test-only hook before its return into core, followed by intentional setup rejection and release of the core borrow. It checks unchanged account/path/PRNG/bank state on that exact intercepted value; it does not obtain a copy borrow from a live retained continuation. Ten existing test cases remain ten; the corrected cases await central execution.

The integrated original sequence gateway and ownership changes passed 2,606 test executions across eight checks, including thirty-nine new cases. Original capture/token-result ownership, native quote/work/submission/carrier retirement, genuine native sequence hooks and the private retained facade cursor are validated together. Facade original-envelope/result/copy integration and complete managed native activation remain unfinished. See [`bounded-original-native-sequence-integration-2026-09-14.json`](validation/bounded-original-native-sequence-integration-2026-09-14.json).


### Fixed original sequence consumer layouts

A consumer contribution derives named cursor/step/error/result representation
costs from its actual concrete types and joins original R before acceptance.
It is immutable, has no arbitrary-byte constructor, and exposes no guard or
allocation authority. Runtime retains the descriptor with the original
request/context and accepted bank, checks it before extraction, and returns
that consumed association through the provider. Native capture and sequence
producers retain their original P+Q+S+C/H composition; expanded binding/provider
controls are measured from actual types.

Only exact consumer association permits the private facade's fixed consuming
failure envelope. Constructor mismatch returns the unchanged sequence. Failed
advancement retires the non-owning value first: empty cursor before owning
preparation error, or ordinary cause before owning cursor. One closed core
error erasure and its retirement overlap are part of the fixed contribution.
Successful results retain the existing raw account through immutable aliases.

None of these fixed facts completes the bounds for input/tokenizer/parser,
source/decoder/error payloads, events/journal or copied/public result owners.
The public loaded and controlled facade routes remain unchanged; unknown
required domains still need rejection before their eventual managed work.


Single-segment native text extraction validates the complete typed input, then
returns its existing tensor handle directly. It no longer allocates an
intermediate vector of cloned array handles. Multi-segment extraction borrows
the original handles for the existing concatenation. This removes known
ingress containers; it does not certify the remaining source/control/native
allocation inventory or change the original full-prompt payload charge.


### Recovery node ownership and finite population boundary

Recovery keeps the same preallocated Box through typed ownership, erasure, deferred quarantine and guarded retirement. Consuming concrete unboxing now precedes probe destruction and final original custody destruction. Callback panic retains the entire affected node even when a prior native observation was settled; later inspection cannot turn local callback unobservability into safe release. Detached orphan snapshots use closed links so an earlier callback panic cannot destroy an unvisited native payload. Actual original-account fixtures retain the existing P+Q/C/path tail through this boundary; these tests do not add a node grant or establish allocator event timing.

Recovery::node_control_bytes is a checked diagnostic for one requested Node allocation and its actual named constructor, owner, callback and extraction controls. It is not currently composed into original admission, and it is not a compiler stack-size or total population bound. The existing Work M+2 schedule does not count all recovery nodes. Repeated terminal observations may retire before later calls, but unresolved stream-consumer nodes can overlap; a complete original route still needs its exact simultaneous bound or one-shot/reusable slots that cannot replace an unresolved owner. No guessed global multiplier or later hold is introduced. Native Scope parents and records can outlive the Rust probe, so accepted-record readiness and probe Drop do not prove that separate allocation's final release.


### Ordinary-host node versus its enclosing owners

OrdinaryRetirement retains one existing Node Box and now deallocates it through a concrete consuming helper before dropping T. Its erased retirement does not reconstruct or copy the payload, and its typed into_inner path transfers the same T after unboxing. This is a destruction-order boundary; it does not make an enclosing Rc allocation, payload child storage or queue population priced by an original quote. Node and queue representations are unchanged; the concrete returned Node and its destruction/transfer value are explicit controls for a future complete original composer. No guessed count, later hold or activation is supplied. The existing five behavioral tests cover transfer identity, native-lock deferral, recursive snapshot behavior, TLS retention and panic retention; this source increment does not claim their execution or allocator event timing.

Original consumer admission, guarded recovery/ordinary retirement and native ingress integration passed 2,743 test executions across fifteen checks, including twenty-six new cases. Existing ordinary retirement cases also pass. Full public payload, copy destination, native recovery population and managed activation obligations remain unfinished. See [`bounded-consumer-recovery-integration-2026-09-14.json`](validation/bounded-consumer-recovery-integration-2026-09-14.json).


### Ordinary retained result within original R

The consumer's closed ordinary-output mode adds the actual GenerationOutput<(), GenerationTokenIds>, Result<Output, Cursor>, terminal-method/constructor timing values and constructor token/reason arguments to its checked named controls. The existing tuple extraction and original R token/EOS/provider storage stay independently measured once. Enlarged descriptors, sealed bindings, providers and bank/error controls continue to use their actual sizeof terms; no fixed architecture constant or arbitrary multiplier replaces them.

The same original request/context, one-use bank and returned-provider comparison authenticate this mode before any cursor work. Terminal construction only moves the existing raw token owner. Tests retain exact charged-tail and missing residual-only source checks, and separately show explicit full source pins retire on freeze while the raw result/iterator preserves its original host hold. No new quote, late contribution, hold, refund or output allocation is introduced.

These named representations are not a compiler stack-frame bound or a bound on arbitrary application wrappers. Public result aliases, dynamic tokenizer/parser/source/event/error payloads and copies remain outside this fixed component. Tokenizers 0.23.2's current stream decoder replaces its ids vector during drain/collect and creates decoded/error strings; reserving one ids vector is not a complete physical decoder plan. Public managed activation remains closed.


### Tokenizer sharing does not supply a payload bound

A neutral tokenizer wrapper now allocates one shared HF tokenizer Arc at construction. Fresh facade decoders retain an immutable snapshot of that exact object rather than allocate a deep tokenizer copy and a new Arc each time. Wrapper configuration mutation uses Arc::make_mut and may allocate a new configuration/header when snapshots remain alive; old snapshots keep the old payload. Template environment/cache/variables are separate.

These owners are not charged by an invented zero-cost shared-source fact. The actual new header, transitive tokenizer state/caches and possible COW construction overlap remain part of the unfinished original tokenizer/source inventory. Decoder ids/prefix, tokenizers' drain/collect and decoded/error strings, parser/event payloads and earlier facade controls remain separately unclosed. No new hold, quote, arbitrary-byte contribution, source registration or public managed route is introduced.

### Native original row-error allowance

One complete error per actually claimed quoted text operation is now enclosed
under its original accepted controls before it can escape the native driver.
The allowance precedes Work construction, and the final core-owned source follows
all existing runtime/observer/recovery wrapping. Pending capture consumption has
one analogous allowance through collector conversion and row installation.
Failed preflight/replay checks cannot clone another allowance; row diagnostic
helpers no longer retain Q independently on each rejection.

Original A prices the installation envelope once and original W composes the
operation envelope under the existing M+2 ceiling once. The fixed facts enumerate
the real native/core/row/runtime/funded-observer and retained NN Arc layouts; they
do not certify dynamic String capacities, allocator overhead, independent NN
clones, raw Rows Rc/Weak allocation retirement, native Scope lifetime/counts, or
full outer/paged/partition inventories. Existing native payload safety guards,
failed prefixes and canonical marker retirement stay unchanged.

The pending source tests cover actual claimed Work rejection using two real
admissions, consumed installation with retained replays, and source/classification
plus outer-core error lifetime. The internally injected scope mismatch is not a
public-reachable route. Existing exact/-1 admission and real 2,2,1 row/cancellation
checks are preserved. The low-level fixture's own original finite allowance does
not substitute for a genuine core claim or refresh a stale prepared token. No
public native activation or new quote/hold is introduced.

The consumed-installation native error fixture retains the original aggregate P+Q+R+S together with completed C and the actual registered shared-observation-path source. It records only detached path identity/capacity, validates that key while the error survives and verifies missing-key retirement after error drop with replay errors still retained. This distinguishes explicit original registered-source custody from reservation metadata and from the lower-level row fixture's smaller source set.

Original retained result controls, native operation/install error retirement and immutable tokenizer sharing passed 2,665 test executions across ten checks, including fourteen new behavioral cases and two compile-fail checks. Full native/facade payload bounds and public managed activation remain unfinished. See [`bounded-native-errors-result-integration-2026-09-14.json`](validation/bounded-native-errors-result-integration-2026-09-14.json).


### Native Scope owner primitive remains unintegrated

PreparedSubmissionScopeOwner<T: Send> and SubmissionScope::try_begin_retaining close one constructor-owned native Scope and one Rust retirement-node allocation. Preparation and native failure preserve the exact original owner, with no replacement hold. Actual Scope, Rust node, prepared/error and named extraction-control sizes are available as cold diagnostics; they neither count owners nor establish an original reservation. The existing ordinary-host queue remains responsible for safe payload destruction, including reclamation on another host thread. Quiescent or settled native work does not release the Scope owner while a child or Record reference remains.

A managed producer still needs a sealed original Scope contribution, exact attempt association and a finite simultaneous population before reserve. No M+2 multiplier, arbitrary bytes-plus-guard attachment, native completion, full graph/Record closure or gateway activation is supplied. Ordinary native Scope layout grows as well; this change makes no claim that an earlier incomplete ordinary inventory already covers it. The six Rust and six native cases exercise failure ownership, real nonzero CPU work, nested/Record lifetimes and post-release host reclamation; execution is recorded only by the later central validation.

Constructor-owned native Scope custody passed 571 test executions across eight checks, including six new Rust cases and six new native cases. Cold diagnostics and safe deferred retirement are validated; original producer funding, finite population, Record/transitive inventory and public managed activation remain unfinished. See [`bounded-native-scope-owner-integration-2026-09-14.json`](validation/bounded-native-scope-owner-integration-2026-09-14.json).


### SessionPayload and FrozenDecoder retirement boundary

The closed native owners preserve the existing Node Box plus Rc allocation requests, and queue the same Node only after final Rc extraction returns. Their constructor/alias/retirement operations create no production replacement allocation, late host hold or grant. Existing actual-type control facts reflect changed inline fields; no unused public pricing API is added.

This does not establish original funding of loaded or copied owner populations. A future accounting join still requires actual Rc/Node and named extraction layouts, the original load/destination-copy admission point, finite simultaneous active/queued/copied owners and dynamic children. Physical registration alone is not that outer inventory, and the existing M+2 work count is not a session or saved-decoder count. Test-only active-retirement status is a separate payload-free observation allocation, absent in production.

Closed session and frozen-decoder active ownership passed 375 test executions across three checks, including one new real-session retirement case. Identity, checked exclusive access, completed-output custody and saved-copy behavior remain covered. Other shared-owner populations, original load/copy funding and public managed activation remain unfinished. See [`bounded-session-payload-integration-2026-09-14.json`](validation/bounded-session-payload-integration-2026-09-14.json).


### Plain-join and ByteLevel fixed decoder destinations

For a compiled actual source with maximum piece B and N successful future token steps, eredu-text derives N u32 lookbehind slots. ByteLevel uses raw N*B and two distinct UTF-8 destinations of 3*N*B bytes; plain join uses no raw scratch and two destinations of N*B+max(N-1,0). All arithmetic and individual slice-layout extents are checked. Construction validates exact extents before writing caller storage. Compaction does not refill the total call ceiling.

The compiled source owns exact record/byte boxes and exposes their concrete payload plus source struct; separate diagnostics name actual layout/stream/result/error controls without inventing a summed funding grant. Cold HF vocabulary/string copies, packing capacity and box-conversion overlap remain outside that live payload fact. Source compilation/registration, original provider/bank binding, enclosing HF ownership, facade parser/events and consuming error/copy envelopes are still required. A borrowed chunk avoids a hot owned String only within this kernel; public managed decoder activation remains closed.

The fixed-destination decoder prerequisite passed 68 test executions across three portable checks, including ten new behavioral cases and one compile-fail ownership check. HF output/frontier parity and fixed-buffer limits are covered; original source admission, other decoder lowerings and facade parser/event/copy integration remain unfinished. See [`bounded-fixed-decoder-integration-2026-09-14.json`](validation/bounded-fixed-decoder-integration-2026-09-14.json).


### Finite observation prerequisite, before Scope producer funding

Repeated successful model finalization and reads through clones of the same
native token no longer allocate a fresh observation Scope. The fixed text
completion query reuses submitted-work owners, preserving token-before-model
ordering and current health checks. First validation/scalar failures retain
the same concrete source for later calls; a failed or interrupted observation
does not receive a replacement attempt. Pending polls still retain their
existing recovery owners. No settled/readiness result is used as a proof of
final native allocation destruction.

This change adds closed shared token-result and error-source representations.
Existing enclosing sizeof facts observe changed inline layouts, but do not
price those new allocations, arbitrary aliases or a Scope role schedule. Those
remain explicit original pre-reserve integration obligations. No M+2 Scope
count, new hold, later guard attachment or complete bounded-native claim is
introduced. Focused source fixtures cover nonzero native results, exact repeated
errors, current-health rejection, reentry, state-loan cleanup and final shared
source retirement; their execution remains part of central validation.

One-shot native output observation passed 492 test executions across five checks, including eight new cases. Scalar/model/text replay, exact failure custody, current health, contention and retirement are covered. Original pricing for the new shared allocations, finite native Scope roles and public managed activation remain unfinished. See [`bounded-output-observation-once-integration-2026-09-14.json`](validation/bounded-output-observation-once-integration-2026-09-14.json).


### Contiguous text ingress avoids packing containers

PreparedTextPrefill retains a single native segment directly, releasing the supplied segment Vec before returning the source and avoiding a selected-view Vec for its later spans. A one-row host span borrows its exact token subrange; a complete multi-row span borrows the existing row-major host storage. Narrow multi-row spans still pack the current chunk, and genuinely segmented input keeps its existing intersection/concatenation behavior. Validation, mask coordinates, prompt identity and the reservation-before-native-construction boundary are unchanged.

This removes specific temporary or retained containers; it adds no funding authority and makes no complete source-memory claim. The existing original prompt payload must not be counted again. Multi-segment views, narrow batched packing, mask index containers, composite prepared-input metadata and enclosing source/owner costs remain separate inventory obligations. The existing neutral nonzero source fixture now also compares singleton segments, uneven single-row host spans and a complete batched host span.

Contiguous prefill ingress passed 11 neutral numerical test executions in the bounded-readout suite. The existing source fixture was strengthened for singleton segments, uneven single-row host spans and full batched spans; no new test case is counted. Original source ownership and remaining multi-segment/mask/composite container inventory remain unfinished. See [`bounded-prefill-contiguous-ingress-integration-2026-09-14.json`](validation/bounded-prefill-contiguous-ingress-integration-2026-09-14.json).


### Native opening-row Rc/Weak retirement

The original finite row proposal prices its actual closed Rc allocation and named construction, upgrade, installed-weak and extraction controls, replacing superseded raw-owner/header terms. A and W see the final concrete optional strong fields under their existing original producers; no later hold, refill or model-lifetime count is introduced. The weak keeps full original accounting/source custody until control-block retirement while semantic rows still die with the last strong owner.

One genuine no-prediction core startup fixture installs the actual original row bank, checks live/active-authority/slot-busy/fence preservation, exact expired-weak host-plus-C tail with the actual registered paths already in the loaded baseline, and repeated exact admission after explicit synchronization on the same runtime. The existing exact/-1 test also covers row-enabled admission; successful nonzero 2/2/1 and actual failure/parcel/source tests retain independent survivor assertions. These staged tests do not imply universal refund, full native inventory or activation. Compilation and execution belong to the central validation record.

Closed native opening-row retirement passed 383 session test executions and two feature checks. One new genuine startup fixture proves live, active-authority, slot-loan and poison exclusion plus exact expired-weak custody and repeated budget reuse; existing exact and one-short admission now includes row-enabled capture. Complete native inventory and public managed activation remain unfinished. See [`bounded-native-opening-rows-retirement-integration-2026-09-14.json`](validation/bounded-native-opening-rows-retirement-integration-2026-09-14.json).


### Exact ordered decoder scratch remains separate from original admission

For the fixed ByteFallback pipelines, N is the total successful-call ceiling and fixed lookbehind slot count, B is the largest actual packed ordinary piece, and F=N*max(B,3). Three is the UTF-8 width of one replacement scalar for each byte of a wholly invalid fallback run. Replacing before fallback or after fusion uses N raw bytes plus F bytes for each candidate/prefix. An appended ByteLevel stage uses three explicit raw regions (N pending-run bytes, F fused UTF-8 bytes, F converted bytes) and 3F bytes for each candidate/prefix. Every product, extent and aggregate is checked, and all four caller slices must match before mutation. Singleton ByteLevel reuses the existing direct form's layout.

The complete fused token is retained before ByteLevel chooses mapped bytes or entire-token UTF-8 fallback. Ordinary empty records still flush fallback runs, and borrowed output prevents another mutation while its view is live. Actual source record/mode/control sizes remain derived from the final concrete types. These are storage diagnostics, with no late hold, grant, replacement quote or managed activation. Original source publication/construction, shared HF payload, parser/grammar, event/journal/output ownership and public copying remain separate unfinished accounting. Canonical R token history is already owned independently; decoder lookbehind is its own necessary algorithm state.

A separate external differential harness can validate the pinned complete local tokenizer artifacts against HF 0.23.2 without placing large tokenizer files in tracked source. It verifies file hashes and actual decoder configurations before comparing emitted text and complete stream state. Its ordinary validation allocations and any future execution results do not constitute managed admission evidence.

Exact decoder pipeline lowering passed 76 Rust test executions across text, documentation and portable facade checks, including eight new cases. A separate external differential replay passed for ten pinned full tokenizer artifacts; its case and step counts are recorded independently from Rust tests. Original source/admission, cold compilation, parser/event ownership and public managed activation remain unfinished. See [`bounded-decoder-pipeline-extension-integration-2026-09-14.json`](validation/bounded-decoder-pipeline-extension-integration-2026-09-14.json).


### The original two-role preparation Scope bound

Prompt and Sampling each have one presealed original allocation role,
independent of the output count. Their native Scope, owner queue node,
Recovery Box and named constructor/extraction controls enter existing Q
before admission. Final P/A/W representation facts use actual enlarged
types. Mixed capture/sequence consumes one bank from its same promoted span;
no accepted replacement quote, second hold or M+2 Scope count is used.

A rejected native begin preserves the same pending owner for retry. Success
or unwind cannot refill a role. Real original-account fixtures distinguish
quiescence, final parent/child native allocation deletion and later unlocked
queue retirement, including Send custody reclaimed on another host thread.
Never-started Work is not given an invented settled observation: abandoning
it can keep a conservative nonzero envelope after model teardown. Reusable
cancellation cleanup requires a separate untouched-Work proof. The fixed
pair does not close later prediction/copy/observation roles, Record or registry
populations, dynamic Work inventories or the full managed-inference graph.


Preparation-pair conformance keeps capture-only sampler-copy rejection distinct
from sequence-bank rejection: the fixture uses actual core-bound preparation and
asserts zero sequence claims/R bytes before examining its original sampler.
Named-control tests compare the complete pre-reserve pair enrichment and retain
exact/minus-one admission checks. The capsule lifetime test waits for deferred
model retirement to reach its exact original hold, then preserves each parent,
child, queued-owner and final-zero assertion. No expected balance is relaxed.

The original two-role preparation Scope integration passed 984 Rust executions and two feature checks, including ten new neutral and native tests. Same-owner retry and distinct native/Rust retirement custody are verified; unused Work quarantine, other Scope/Record producers and complete managed activation remain unfinished. See [`bounded-original-preparation-scopes-integration-2026-09-14.json`](validation/bounded-original-preparation-scopes-integration-2026-09-14.json).


### Scope custody survives complete native record deletion

Native record cleanup now releases its retained Scope reference after the
record allocation is deallocated, rather than from the base destructor body.
A focused native regression checks both parent and child custody immediately
after the derived allocation deallocator returns, for successful and failed
terminal records. It also checks that subsequent retirement cannot release
custody again. This is a lifetime ordering fix: Record populations, payload
capacities, preparation allocations and complete managed coverage still need
their own original bounds.

Native Record deallocation ordering passed 117 C++ and Rust test executions, including one new focused native regression. The same retained Scope reference now outlives the complete Record allocation. Finite Record populations, payload bounds and complete managed activation remain unfinished. See [`bounded-record-postdelete-scope-integration-2026-09-14.json`](validation/bounded-record-postdelete-scope-integration-2026-09-14.json).


### Original source-bound decoder prerequisite

One concrete text-compiled source can be consumed before the original adaptive admission loop and installed in the same one-use generation-sequence bank. Candidates borrow immutable actual source/N/skip facts; no candidate consumes, recompiles or refills it. Original R prices the actual program, four fixed destinations and named owner/error controls; native Q names staging moves/guards. The existing canonical result token storage is charged once. Local destination preparation is the existing first readiness attempt and failure retains all partial buffers. A pointer-free progress engine shares the exact borrowed fixed decoder equations; the facade private cursor lends each suffix into its existing parser/commit body without an intermediate output Vec.

The accepted staging source retires before its actual reservation and funding run on error/unwind, using owner-preserving start_funding and immediate lexical guards. Immutable token freeze retires decoder source/buffers before full source custody while preserving the same result allocation/raw hold. These mechanisms do not activate public managed decoding or fund the initial HF/compiler peak, caller input/platform mutex lifetime, parser/events/errors/copies, or unclosed native populations. Runtime/native admission, actual HF frontier and private facade cursor tests are separate evidence; central execution remains a root-owned integration requirement.

The original decoder source bridge passed 1,218 Rust executions and two feature checks, including nineteen new behavior tests and the new suffix-borrow compile-fail contract. Test-only initializer, empty-history assertion and capture-shape corrections preserve strict production identity and behavior. These checks separately cover original-account admission, native selected-residency predictions, HF frontiers and private cursor ownership; public managed decoding and compiler/parser/native ownership work remain incomplete. See [`bounded-original-decoder-source-bridge-integration-2026-09-14.json`](validation/bounded-original-decoder-source-bridge-integration-2026-09-14.json).


The untouched-Work successor closes the preparation pair's previously documented
conservative Busy-abandonment tail. It cancels only a funding scope whose native
interface has never escaped its closed prepared owner; an empty roots collection
is not the proof. Repeated genuine admissions can reuse their exact original
capacity after all pending/control owners retire. The claimed role never refills.
Active Work still uses the original publication/certification/quarantine rules.

Conformance separately covers actual nonzero native execution followed by an
eager native validation error, preserving its original concrete source and normal
safe cleanup, and an exposed scope dropped with an actual unresolved native child,
which remains quarantined even after child lifetime retirement. It does not call
the first case a failed asynchronous Record. Existing native late-failure suites
provide that mechanism evidence. Neither scope cancellation nor record quiescence
establishes complete native allocation/graph accounting or public managed support.


Complete UTF-8 decoder suffixes now pass by borrow through the shared UTF-8 input adapter when it has no partial bytes. This removes its pending-buffer and intermediate String copies on the original fixed-decoder path. Fragmented legacy input retains the same joining and failure semantics. Parser, stop-matching and semantic event storage remain separate admission obligations.

The untouched original Work and borrowed UTF-8 increment passed 1,239 Rust executions and two feature checks. Nine behavioral tests were introduced, replacing one old conservative-tail fixture (net eight). Unused funding now returns its own scope count after owners retire; active unresolved work remains fenced. Complete decoder suffixes avoid UTF-8 staging copies in the shared parser path. Full compiler/parser/native accounting and public managed activation remain incomplete. See [`bounded-unused-work-and-borrowed-utf8-integration-2026-09-14.json`](validation/bounded-unused-work-and-borrowed-utf8-integration-2026-09-14.json).

### Five finite outer prediction Scope roles

Original Q may now contain one bank for the actual accepted M prediction attempts, with distinct ModelExecution, Sampling, SamplingEvent, ModelValidation and original TokenScalar controls. Cold native facts and neutral bank/set controls enter the original seal; unknown facts and overflow reject before acceptance. The bank consumes the genuine active step in order, including Started cached decode. No receipt, copied token or arbitrary count grants an issue, and failure/drop never refunds it.

Each role protects its native Scope/Rust queue owner and closed Recovery allocation independently through final retirement. The same ordinary/controlled drivers, cancellation boundaries and observation-once cache are used. Initial cancellation/zero output use no roles; an issued outputless prefill can use model plus validation only. The scoped new fixture matrix compares token IDs and retained state for legacy/sequence/capture across all three residencies, and the existing admitted opening-row route in resident execution. No new host/disk opening inventory is inferred. Tests and exact/short admission are staged for central execution; this source package makes no execution claim.

The contribution prices concrete role layouts and named moves, not native Record/graph/root buffers, arbitrary errors/aliases/copies or complete source/media/communication payloads. The existing M+2 Work count is not a Scope count. Original capacity/retirement policy and public managed activation remain unchanged.

The five original prediction Scope roles passed 1,150 Rust executions, including twelve new behavioral tests and a compile-fail authority example, plus both feature checks. Actual shared ordinary/controlled and resident/host/disk fixtures retain the same original reservation through the five named consumers. Copied scalar observations cannot refill original roles. Native Record/graph/root/error populations and full host preparation/source ownership remain incomplete; this increment does not activate public managed inference. See [`bounded-original-prediction-scopes-integration-2026-09-14.json`](validation/bounded-original-prediction-scopes-integration-2026-09-14.json).

### Checked fixed-decoder construction prerequisite

The fixed decoder now has a non-allocating borrowed source plan and a consuming
compiler. It checks sparse/added ID and packed-byte extents, reserves the three
compiler buffers once, and retains every partial prefix in its closed failure.
Successful publication moves the same record/byte Vec capacities to the existing
kernel without shrink allocation. Final source size and simultaneous compiler
peak are distinct facts; named control diagnostics use the actual types.

Ordinary `PreparedDecodeSource::prepare` delegates to this compiler, retiring its
partial owner before returning the historical error. Admitted construction must
instead retain the owning failure under its original construction allowance.
Neither entry admits the HF graph, its compilation/input/native-Mutex owners,
or facade preparation/parser/events/errors/copies. No source budget is attached
retroactively and no public managed-decoding gate is enabled by these facts.

The checked borrowed decoder compiler passed 2,062 Rust executions, including eleven new behavioral tests and a consuming-plan compile-fail example. The unchanged full-tokenizer oracle also matched ten pinned artifacts across 246 cases and 18,634 decoder steps. Portable/native feature checks passed. Construction-source admission, prior HF residency and the public managed gateway remain separate obligations. See [`bounded-borrowed-decode-compiler-integration-2026-09-14.json`](validation/bounded-borrowed-decode-compiler-integration-2026-09-14.json).

### Original literal-stop source and fixed visible-text working storage

An explicitly requested plain projection joins the actual immutable stop source
to the existing original decoder input before acceptance. Its one source take,
checked identity and consumed bank bind decoder/M/skip/stops and the concrete
consumer mode. A suffix-only provider defaults to rejection. For nonempty stops,
one fixed destination uses `L + max(K-1,0)`, with L from the actual decoder's
per-call text layout and K from the actual longest stop; no-stop input borrows
the decoder suffix without stop scratch. This bounds working storage, not a
transcript. Source tables, final concrete controls and preparation/error overlaps
join the original contribution; no late quote, generic parser bytes or new guard.

The private borrowed batch and token result remain under the same consuming
provider/cursor lifetime. Partial destination failures cannot retry allocations;
zero output/initial cancellation skip them. Public owning events, Delivery
history/records, HF/tokenizer/compiler input, stop-packing peak, tool normalization
and snapshot/copy destinations remain explicitly outside this completed slice.

The original plain stop-text provider and borrowed event path passed 2,527 Rust executions, including fourteen new behavioral tests and two borrowed-output compile-fail examples, plus portable/native feature checks. Legacy and fixed stop handling share one kernel and the same commitment/readiness driver. Cold source packing, HF ownership, public owned events/records, tool parsers and snapshot/copy destinations remain required work; no public managed route is activated. See [`bounded-original-plain-stop-events-integration-2026-09-14.json`](validation/bounded-original-plain-stop-events-integration-2026-09-14.json).

Original cold decoder-source admission and shared request transport passed 2,085 Rust executions, including thirteen new lifetime, boundary and genuine-request tests. The unchanged full-tokenizer oracle matched ten pinned artifacts across 246 cases and 18,634 decoder steps. Portable/native feature checks passed. Shared stop sources, original HF/input ownership and full public managed inference remain required work. See [`bounded-loaded-decode-source-integration-2026-09-14.json`](validation/bounded-loaded-decode-source-integration-2026-09-14.json).

The original submission-tracking component passed 2,607 C++ and Rust executions, including fourteen new regressions and thirty-two ordinary/controlled native matrix sessions. One fixed arena is sealed into the original quote and retained through actual Record/container deallocation. The explicit ceiling does not predict graph fit; saved/copy integration, graph/task/error allocation closure, default selection and public managed inference remain required work. See [`bounded-original-record-quota-integration-2026-09-14.json`](validation/bounded-original-record-quota-integration-2026-09-14.json).

### Descriptor retirement is a prerequisite, not a graph grant

The native descriptor drain uses a fixed intrusive link/phase in each existing
descriptor and one automatic registry frame per simultaneously draining thread.
It performs no heap worklist allocation or first-access C++ TLS setup. Its
Expand/Finalize ordering preserves parent deferred authority through known
input/sibling final releases; the actual Rust reclamation queue remains free to
run concurrently on another ordinary host.

No graph capacity or original reservation is added by this change. The separate
descriptor and shared-control allocations, dynamic shapes/strides/edges,
primitive/event payloads, surviving external roots and arbitrary callbacks
remain explicit accounting obligations. A descriptor member alone cannot pay
for its own allocation through operator delete, and a shared_ptr deleter alone
cannot pay for a raw descriptor queued beyond that control block's lifetime.
The Record arena continues to cover only its declared objects/containers.

### Original stop packing and cold ownership prerequisite

The concrete borrowed stop compiler derives checked first-occurrence Entries/Bytes requirements before reserve, preserves real partial buffers on failure, and stores the same Vecs without shrink/boxing. A closed runtime source retains its whole original amount after compiler activity ends and through final Arc/payload retirement. Idle sources reduce later admission capacity while permitting existing host preparation; poison does not settle. This source-only increment does not duplicate its charge into generation R, fund prior caller/HF owners or activate managed decoding. Shared plain request/header/destination composition is the dependent next unit. Ten new behavioral tests and one consumed-plan compile-fail are authored; execution belongs to central validation.

### Shared plain decoder and original stop sources

One original request header now binds the exact decoder and optional original stop owner, N, skip policy and plain/suffix mode. Both source accounts validate before its one-use CAS; source compilation and payload charge stay outside generation R, which counts only destinations and actual controls. The existing provider/cursor prepares and retires the same decoder/stop kernels, with no-copy token freeze. Thirteen authored tests cover paired claims, original source tails, per-request override composition and actual native cached continuation; central execution remains separate. Prior InferenceRetention metadata can keep unquoted reset unavailable, so original-funded reset remains a required lifecycle successor. HF/caller input/parser/owned public output/copy and native inventory closure are not implied.

The descriptor retirement and shared literal-stop integration passed 2,251 C++/Rust executions, including 35 new behavioral regressions and one consumed-plan compile-fail example. Intrusive two-phase retirement preserves parent authority through known descendants; original stop compilation and paired request transport reuse the common stop kernel with source charges retained once. Native graph allocation closure, original-funded reset, public input/HF/output ownership, remaining family/path work and public managed activation remain incomplete. See [`bounded-descriptor-shared-stop-integration-2026-09-14.json`](validation/bounded-descriptor-shared-stop-integration-2026-09-14.json).

### Moshi/PersonaPlex decision demand evidence

The neutral nonzero fixture exercises actual selected resident, host-layerwise,
and disk-streamed construction with ten canonical frames, bounded advancement,
forced/mixed sampling, diagnostics, required final-row observations/interventions,
and failure/cancellation after a committed prefix. Projection traces record the
actual vocabulary operator input shapes. Separate 3/2/1/1-position low-level
calls compare complete temporal/depth cache state and full temporal hidden values
across Sequence, LastPosition, and StateOnly demands, including cached calls.
These are source-staged tests until central validation records execution. They
use a Moshi fixture; distinct PersonaPlex nonzero numerical and released-checkpoint
validation remain open, with its strict released-profile gate unchanged.

Optional readout changes neither the original admission boundary nor source,
state, parameter, graph, record, or completion-accounting obligations. Native
completion retains the actual optional scores, full temporal hidden state, and
existing cache/token-validation owners. No graph bound is inferred from fewer
vocabulary projections.

Moshi decision-demand integration passed 2,005 Rust executions, including six new neutral/runtime and nonzero numerical tests. Resident, host-layerwise and disk-streamed calls select hidden rows before vocabulary projection and preserve complete temporal/cache state; required observations and diagnostics retain their scores. Native tiny-model/operator and scheduler checks also passed. Distinct PersonaPlex numerical coverage, native TP evidence, PP, snapshot/fork/persistence, PCM lifecycle and original managed-memory closure remain incomplete. See [`bounded-moshi-decision-demand-integration-2026-09-14.json`](validation/bounded-moshi-decision-demand-integration-2026-09-14.json).


### Realtime executable replicas versus shared checkpoint sources

An immutable checkpoint recipe shared by independent realtime units does not
imply shared executable backing. The normalized per-unit binding plan contains
one recipe-backed local representative per canonical source family, plus any
local aliases. Existing binding-byte and host-capacity queries count these real
replicas before residency admission. Same-unit aliases count one backing; actual
source reads and cached-shard counts are separate observations. One-unit windows
do not hide an additional pinned logical-owner unit.

The focused conformance cases exercise complete alias-chain/source validation,
real independent materialization, exact and one-less native host/device capacity,
local and cross-unit alias identities, eviction/reacquisition and promotion.
These checks concern immutable parameter residency. They add no native graph
quota, original request hold, public managed activation, or released PersonaPlex
checkpoint claim. The private PersonaPlex numerical fixture follows separately.

Realtime unit-alias integration passed 2,051 Rust executions, including seven new neutral/native cases. Independently evictable units carry counted replicas; local aliases share actual backing. Original source validation survives dense-to-packed conversion, and final outputs require authoritative publication and exact geometry. Native host/device budget boundaries, eviction, promotion and existing Moshi quantization paths passed. Public managed-memory closure and distinct PersonaPlex numerical/released validation remain incomplete. See [`bounded-realtime-unit-alias-integration-2026-09-14.json`](validation/bounded-realtime-unit-alias-integration-2026-09-14.json).

### Original TokenIds input destination (source-only increment)

A positive `TokenIdsInputPlan` is borrowed throughout original adaptive admission.
Candidates derive geometry and actual layouts without allocating or copying input.
After acceptance and run binding, one bank constructs exactly one U32 destination;
I joins the existing protected P/Q/R sum. A typed original prompt report removes
only the superseded legacy caller-capacity term and preserves native upload,
reshape, staging and identity costs. Caller spare capacity is never adopted as
funded storage. Original native A names the enlarged quote, pending Prompt owner,
slot/return/retirement controls and concrete installation-error source; no late
grant, replacement quote or copy of the canonical R charge is introduced.

Borrowed claim preflight rejects foreign slices/claims/preparations before I/R
take; repeated fixed failures allocate no source and retain no custody. Once a
real bank is consumed, allocation/validation failures own that bank or actual
partial Vec in core's closed source envelope. Input payload retires before its
original guard. Busy before native Prompt start retains the same pending IDs,
Work and Recovery node. Synchronous C copying allows the input to retire after
upload and identity hashing, independently of later Scope/Record retirement.

Authored tests cover all three core routes, zero output, original exact/one-short
admission, caller mutation, same-session equal-valued foreign claims, replay,
capacity-overflow and post-fill unwind, original registered-source tails, adaptive
candidates, and native resident/host/disk cached state. A private facade test uses
real runtime banks and the existing plain cursor/Delivery readiness. These tests
have not been executed in the author package. Native comparisons use actual
logical K/V rows, excluding unused allocation capacity; the neutral facade backend
has no numerical/native operations. Public managed activation remains closed.

Original TokenIds integration passed 2,497 Rust executions, including 15 new behavioral cases and one compile-fail. The shared startup borrows input through admission, constructs one originally funded immutable destination and preserves it across native Busy. Native resident/host/disk ordinary/controlled/detached runs compare four predictions and full logical K/V through two cached requests; capture/opening-row exact/one-short admission and private neutral plain-facade composition also pass. Graph integration and remaining public managed-memory closure are still incomplete. See [`bounded-original-token-input-integration-2026-09-14.json`](validation/bounded-original-token-input-integration-2026-09-14.json).

### Distinct original graph metadata arena

An explicit `graph_metadata_capacity_bytes` selects a fixed original-Q-backed arena for synchronous native graph metadata. Quotation includes the actual arena header/capacity, constructor-owned Rust queue node and named construction/retirement controls before reserve. Covered constructors use actual allocation requests, including alignment, fragmentation and block headers; there is no outer-Scope multiplier, caller-supplied node count, fit guess, fallback or late refill. The graph and submitted-work tracking arenas are independent optional components and may be used together.

Covered blocks are ArrayDesc object/control, Shape/Stride spills, migrated array edge/root vector storage, and concrete built-in primitive shell/control blocks. Owning allocators retain the original contribution even when their containers are empty. Every allocated block separately retains it through return, including a queued descriptor whose shared control has already died and a primitive control retained only by Weak. Current allocator copy/move/swap rules select the active domain for new construction and preserve the domain of existing destinations. Foreign-buffer import allocates in the destination before publication and leaves the source valid on refusal.

This enforced capacity can reject execution; it does not predict success. Primitive maps/strings/function captures and unmigrated outer containers, fresh worker construction, outer handles, Event/Fence/task/future/encoder/backing/Data/registry/thread allocations and general error storage remain distinct obligations. No complete managed-bound claim, default capacity or eligibility activation is introduced.

Original graph integration passed 4,268 native/Rust executions, including24 new graph cases and one simultaneous original-input/graph case. Original Q funds one finite graph arena and checked constructors preserve typed exhaustion and owner retirement through real execution; TokenIds remains a distinct original component using the same upload worker; the combined case compares full logical K/V and four predictions through two cached requests across resident/host/disk and ordinary/controlled/detached routes. The arena is one enforced component, and full managed-memory activation, saved/copy graph and remaining native/facade inventories remain incomplete. See [`bounded-original-graph-metadata-integration-2026-09-14.json`](validation/bounded-original-graph-metadata-integration-2026-09-14.json).


### Private PersonaPlex numerical follow-on

Six architecture-owned numerical cases add distinct reduced PersonaPlex semantics to the earlier Moshi demand proof, using the applied generic alias normalization. Actual source/selected binding, resident/host/disk continuation, all mutable state, delayed mixed targets, pre-vocabulary demand, initialization rejection, physical shared norms and packed final-slice sensitivity are exercised by the test source. This synthetic single-rank fixture preserves strict public admission and adds no managed request/accounting activation. Central execution results must be recorded separately. Released-checkpoint/reference validation, native parallel numerical proof, PP/persistence/PCM and complete original-budget closure remain open.

Private PersonaPlex integration passed 648 Rust executions, including six new numerical cases. Real admitted SafeTensors and the shared selected binding plan support nonzero resident/host/disk continuation, full mutable-state parity, delayed demand, initialization rejection and physical parameter sensitivity. These six cases supplement the separately validated alias prerequisite; released PersonaPlex/reference, parallel, persistence/PCM and full managed-memory work remain incomplete. See [`bounded-personaplex-numeric-integration-2026-09-14.json`](validation/bounded-personaplex-numeric-integration-2026-09-14.json).


### Loaded tokenizer raw-access boundary

The loaded model's vocabulary accessor is now metadata-only. It cannot export
HF configuration, a serializer, or an owning snapshot; borrowed strings and its
direct-map ID iterator cannot outlive the loaded borrow. Queries allocate no
internal container, and fingerprint inspection reads the existing value instead
of recomputing the allocating vocabulary hash. Caller-owned collections built
from the iterator remain caller allocations.

This closes one public raw-access exit, not HF residency or construction. The
checked decoder compiler and loaded compiled-source owner remain unchanged.
TextDecoder and controlled snapshots still retain their existing HF aliases;
encoding, decoding, templates, parser state and BPE thread-local caches need their
own complete original contracts. A lifetime-long unquoted model lease is not
silently introduced: it would prevent the existing original compiler and request
reservations. No finite HF size or complete managed-load claim follows here.

Loaded tokenizer metadata-view integration passed 459 Rust executions, including one new behavioral test and three compile-fail boundaries. Queries preserve sparse/normalized/overlapping vocabulary semantics and borrow existing storage; raw HF/snapshot exports are unavailable through the loaded accessor. Existing decoder and generation fixtures remain valid. HF construction/cache ownership and public managed-memory closure remain incomplete. See [`bounded-loaded-tokenizer-view-integration-2026-09-14.json`](validation/bounded-loaded-tokenizer-view-integration-2026-09-14.json).


### Original resident reset destination and control ownership

The fixed resident KV prerequisite has a separate original reset allowance. Its
source-borrowed dense worker reserves one exact buffer, shares the actual layout
and constructs empty policy values without tensor allocation. Checked original
facts include final metadata/identity/revision/account allocations, partial-fill
and returned-error representations, the core source-error retirement envelope
and the original account's entry/deallocation overlap. Safety is included before
acceptance. Existing source bytes and tighter historical ceilings remain live.

Ending construction activity does not refund its allowance. Slot payload dies
before metadata; every metadata/identity/revision alias keeps custody through
its final Arc allocation retirement. Registry keys cannot retain that account.
One fixed domain avoids new generic attachment Vec/Box publication; foreign
attachment rejects. Poison keeps the original numeric obligation quarantined.
A scalar minimum continues enforcing retiring ceilings until their paid entry
allocations have been destroyed outside the accounting lock.

Tests include exact/one-short, genuine foreign claim/source rejection, partial
capacity-overflow failures, unwind, concurrent admissions and retiring ceilings,
and escaped source/identity/revision tails. The prepared-session fixture performs
ordinary nonzero neutral forward before creating a provisional empty destination;
it does not publish a native reset or prove native completion. These cases are
source-staged pending central validation. Full native reset must still supply
original operation/Scope/Recovery/error/publication facts and recognize the new
source witness in later inventories without a second charge. No managed public
gateway or new permission to spend historical generation accounts follows.

Original fixed resident KV reset construction passed 3,723 Rust executions, including thirteen new behavior cases and one compile-fail case. Genuine synchronized core claims bind the actual selected source and retain original construction custody across errors, escaped metadata, revisions and concurrent retirement. Native reset publication, source-witness adoption and other state families remain incomplete. See [`bounded-original-resident-reset-integration-2026-09-14.json`](validation/bounded-original-resident-reset-integration-2026-09-14.json).

Tokenizer model-cache construction integration passed 678 Rust test executions, including 12 new behavioral cases. Explicit constructor policy prevents BPE/Unigram cache storage and survives copying/private recipe restoration while preserving tokenization results. The unchanged pinned 10-tokenizer artifact harness also passed. Finite tokenizer construction/residence and public managed inference remain incomplete. See [`bounded-tokenizer-model-cache-policy-integration-2026-09-14.json`](validation/bounded-tokenizer-model-cache-policy-integration-2026-09-14.json).

Static ByteLevel alphabet integration passed 685 Rust test executions, including seven new behavioral cases. Constant forward mapping and scalar inverse preserve normalization/decoding/offset semantics; disabled regex work does not initialize the regex. The pinned 10-tokenizer artifact harness also passed. Full tokenizer construction/residence and public managed inference remain incomplete. See [`bounded-tokenizer-static-alphabet-integration-2026-09-14.json`](validation/bounded-tokenizer-static-alphabet-integration-2026-09-14.json).

### Fixed reset source-custody prerequisite

The next fixed resident reset source slice adds a closed original-table alternative and one independently retained immutable layout entry. Exact original comparison still occurs once, before destination allocation. Checked demand uses the actual new plan, layout-pin, temporary source, return/error, Arc/Box, registry retirement and iterator representations. The constructor preserves current session/selection/execution/control/revision binding and the previous state on rejection. A successful destination releases its temporary old-table custody; a partial/error destination retains it. Existing concurrent retirement ceilings and poisoned accounting remain unchanged.

Ordinary source vectors are library-owned preparation, not caller-external/free storage and not charged to the old reset. Their opaque population retains the genuine existing unquoted lease, including the first failed construction step. Only an existing unquoted native publication owner supplies this path. Pool-only and funded consumers lacking that owner return a typed unknown-bound rejection before creating an original witness Vec. Fixed original opening uses its prepriced retained table slots instead. There is no new quote, no late finite hold, no copied-token refill and no public reset activation.

Source-only regressions cover genuine exact/one-byte-short repeated reset with four independent predecessor retirements, partial repeat failure, foreign current source, callback-free repeated pin/final key retirement, ordinary transport with a dropped caller lease, existing ordinary native inventory pins, and native owning error retirement with the caller lease dropped. Prior nonzero state, source identity, current revision, partial-error, concurrent-floor and poison cases remain. Successful escaped-owner tails now retain layout + new reset allowance; partial-error tails still retain the complete old source + new allowance. These tests are staged and unexecuted until central validation.

Required immediately following integration coverage: an actual positive native original-table source through the checked native session projection/publication, including old/new native completion and final-owner retirement. That projection is intentionally absent in this slice; ordinary native forwarding and genuine neutral claims do not establish native reset support. Distributed/paged/hybrid reset and the public loaded/controlled reset routes likewise remain unimplemented/default-rejected.

Reset source-custody integration passed 3,277 Rust test executions, including seven new behavioral cases. Successful repeated reset retains independent shared-layout ownership, while partial errors preserve their actual predecessor. Original-source inventories avoid duplicate physical charging and ordinary source populations retain their actual unquoted participant. Native reset publication/readiness and full managed inference remain incomplete. See [`bounded-reset-source-custody-integration-2026-09-14.json`](validation/bounded-reset-source-custody-integration-2026-09-14.json).

Packed BPE construction integration passed 712 Rust test executions, including eleven behavioral cases and one compile-fail case. The existing BPE algorithm consumes fixed source-derived storage, while partial failures retain actual construction buffers. Pinned full-tokenizer decoder regressions and the separate per-model construction/token parity report were validated. Runtime original residence, complete tokenizer construction and public managed inference remain incomplete. See [`bounded-packed-bpe-construction-integration-2026-09-14.json`](validation/bounded-packed-bpe-construction-integration-2026-09-14.json).

### Reset readiness is separate from original reset admission

`ModelRuntime::reset_admitted` validates the retained capability report and creates
the genuine reset claim without first synchronizing. A provider must reject
unknown readiness, or return typed Busy while preserving old state and unresolved
owners. It may not allocate, reap, commit or wait before original acceptance.
The optional consuming preparation owner enables an explicit ordinary preparation
phase followed by that same source-bound admission. This is useful core plumbing;
ordinary preparation does not become funded by the later reset comparison.
Existing active accounts and unquoted-owner exclusions still apply.

The core owner adds references, an address and the actual backend readiness value,
without a core heap allocation. That fact is not a zero-cost native claim. Before
a native implementation opts in, its original plan must include the concrete
prepared-owner, readiness, claim, result/error and publication/retirement controls
and their live overlaps. Waiting still requires either legitimate ordinary
custody before preparation or a separately specified originally funded settlement
mechanism. No native readiness implementation, wait inventory, allowance refill,
new hold or public managed reset route is introduced here. The neutral six-case
fixture's fixed 64-byte comparison exercises core ordering only; runtime reset
account/source/retirement tests retain their concrete demand calculations.


### Original reset host publication

The resident-KV reset constructor can now be composed with an exact native host-publication profile. First-reset source keys are borrowed/projected from the actual table and layout; canonical pin increments and the original charge commit together without a grouped preparation allocation. Repeated reset adopts the actual original table and keeps only an independent layout pin after success. One prepared ordinary node retains displaced state, prompt and moved state authority under the same reset allowance, including failed destination construction. Existing source bytes are not charged twice.

This is a private prerequisite, not complete native reset admission. The fixed node/type controls exclude ordinary queue/TLS infrastructure, external synchronization and generic native teardown. Unquoted owners still reject. Native readiness is not activated, and funded generation's original-table publication consumer remains unimplemented and explicitly rejecting. Post-reset generation parity is therefore not claimed by the publication tests.

Core reset readiness and native resident-KV publication integration passed 3,751 Rust test executions, including fifteen behavioral cases and two borrow/consumption compile-fail cases. Direct admitted entry no longer waits implicitly; private native publication uses actual source pins and one originally prepared retirement owner. Resident/host/disk ordinary and controlled histories, repeated original reset and failure/alias custody were validated. Public native readiness, funded waiting and subsequent funded generation publication remain incomplete. See [`bounded-reset-readiness-publication-integration-2026-09-14.json`](validation/bounded-reset-readiness-publication-integration-2026-09-14.json).

Original packed BPE residence integration passed 2,019 Rust test executions, including ten behavioral cases and six compile-fail cases. A single original source allowance precedes all compiler reserves and final closed-owner construction; partial failures, shared aliases, poison and retirement preserve that custody. Full tokenizer construction, input/operation accounting and public managed encoding remain incomplete. See [`bounded-original-bpe-residence-integration-2026-09-14.json`](validation/bounded-original-bpe-residence-integration-2026-09-14.json).


Generation after a privately prepared original resident-KV reset has a fixed
source-table publication slot, included in the first native Q/Work type facts.
The original table is authenticated, never charged again, and cannot be replaced
or refilled by a later publication. Both actual inventories are validated before
adoption; unchanged repeated publication is idempotent even after certification.
Missing, changed, foreign, extra or retired original tables reject. This does
not bound the existing general publication Vec/BTreeMap/registration population
or activate native reset readiness or a complete managed gateway.

Original resident-KV table consumer integration passed 1,578 Rust test executions, including five new native behavioral tests. Ordinary and controlled generation retain one authenticated table without duplicate charging across resident, host-layerwise and disk-streamed weights; full KV parity, exact/short admission, refused and repeated publication, retirement and rollback are covered. Public native reset readiness, complete tokenizer/facade custody and general native/publication allocation bounds remain unfinished. See [`bounded-native-original-table-consumer-integration-2026-09-14.json`](validation/bounded-native-original-table-consumer-integration-2026-09-14.json).


### Selected parallel prefill boundaries (source-staged 2026-09-14)

The conditional Qwen hybrid numerical fixture now exercises the existing
`SessionPrefill` / `PrefillDriver` pair through both `run` and explicit `step`.
Two focused cases cover TP2/PP1, TP1/PP2 and TP2/PP2, each with selected resident,
host-windowed and disk-streamed mechanisms. A five-token prompt produces the
actual width-two schedule 2/2/1. The successful case compares ordinary final
scores and three cached decode steps against a separate unsplit selected
session. Both modes compare every local KV, attention-history, recurrent and
convolution tensor, layout, offset, position and reset counter using the existing
full-state helper and unchanged absolute `2e-4` numerical tolerance.

The second case requests cancellation on rank zero only, after its first
completed span has been delivered. Every rank must stop at the two-token prefix,
match an independently executed prefix, return no scores, prepare no later
source span and enter no later model transaction. The exact vocabulary operator
is observed in the positive reference; scheduled intermediate spans must never
call it, and successful final readout must project a `[1, 1, 8]` hidden input.
Repeated terminal advances leave source, transaction, observation, projection and
mechanism records unchanged.

These are two new neutral behavior cases (36 worlds, 96 rank trials), with
compilation and execution pending central validation at source freeze. They use
ordinary unbudgeted requests and existing prepared checkpoint/residency drivers;
they add no managed-admission or native-distributed validation claim. Released
checkpoint results, media ingress, speculation and persisted snapshots remain
separate coverage. Run the focused cases with:

```sh
RUST_MIN_STACK=33554432 cargo test -p eredu-architectures --test reference_numeric \
  selected_qwen_hybrid_parallel_prefill_ -- --test-threads=1
```

The command must report two executed tests. The custom `reference_conformance`
runner registers eight aggregate cases; it does not register these two unit tests.
Its `composite_production_and_family_dispatch` case separately checks the shared
composite construction and execution paths.

Selected parallel prefill conformance passed 6 test executions across four checks, including two unique new behavioral cases. TP, PP and combined TP/PP cover resident, host and disk selection with uneven spans, complete nonzero state and three cached decodes; one-rank cancellation stops subsequent source and model work. This adds neutral coverage, not a complete public memory-budget or native distributed validation claim. See [`bounded-parallel-prefill-boundary-integration-2026-09-14.json`](validation/bounded-parallel-prefill-boundary-integration-2026-09-14.json).

Packed added-vocabulary construction passed 722 Rust test executions, including nine new behavioral tests and three compile-fail examples, plus a ten-artifact extraction/encoding differential and timing harness. Four checked destination buffers preserve canonical IDs, special-token filtering and Unicode offsets through the existing HF pipeline; partial errors retain actual allocated prefixes. Complete tokenizer admission and normalized/word/strip profile construction remain unfinished. See [`bounded-packed-added-vocabulary-integration-2026-09-14.json`](validation/bounded-packed-added-vocabulary-integration-2026-09-14.json).

Packed added-token prefix lookup passed 328 Rust test executions, including one new exhaustive Unicode/normalization/special-filter differential test, plus ten released-artifact extraction/encoding comparisons and timings. It reuses existing sorted spelling indices without allocating search storage. Complete tokenizer construction and public managed encoding remain unfinished. See [`bounded-added-prefix-search-integration-2026-09-14.json`](validation/bounded-added-prefix-search-integration-2026-09-14.json).

Independent-draft selective prefill passed 331 test executions across eleven central checks, including seven new behavior cases. Target prefill projects only its required final row; draft prefill updates state without scores through the shared scheduler. Local/peer cancellation prevents sampling and publication, restores the joint checkpoint, and retains completed target-work telemetry. Nonzero state and native ordinary/controlled parity cover uneven chunks and all three residency selections. Embedded/assistant/media migration and full managed speculative funding remain incomplete. See [`bounded-independent-draft-selective-prefill-integration-2026-09-14.json`](validation/bounded-independent-draft-selective-prefill-integration-2026-09-14.json).


### Additional selected dense sliding-prefill coverage

The named `prepared_mistral_and_qwen2_sliding_chunks_preserve_state_and_cancellation`
case reuses the existing `CompletedChunkVisitor` with four actual configurations:
tied and untied Mistral with two sliding-attention layers, and tied and untied
Qwen2 with one full-attention and one sliding-attention layer. Both use a two-row
window, a five-token prompt, and resident, host-layerwise and disk-streamed
selected mechanisms. The shared assertions cover widths 1/2/3/5, run and step,
fresh and cached input origins, three subsequent decodes, complete retained
state, omitted intermediate vocabulary projection, cancellation, source failure
and reservation retention. There is no new chunk loop or model implementation.

This closes those explicit single-rank configurations in the selected scheduler
matrix. It adds neither native distributed evidence nor a complete public managed
memory bound; parallel owner-only projection/cancellation and other routed/pooled
intersections remain separate. Run with:

```sh
RUST_MIN_STACK=33554432 cargo test -p eredu-architectures --test reference_numeric \
  prepared_mistral_and_qwen2_sliding_chunks_preserve_state_and_cancellation -- --test-threads=1
```

Additional selected dense-family prefill coverage passed 2 test executions across two checks, including one new parameterized case. Tied/untied Mistral sliding and Qwen2 mixed full/sliding configurations reuse the existing complete-state scheduler conformance across resident, host-layerwise and disk-streamed selection. The pre-existing family matrix also passed after helper extraction. Native/distributed intersections and the broader managed-memory goal remain incomplete. See [`bounded-additional-dense-prefill-integration-2026-09-14.json`](validation/bounded-additional-dense-prefill-integration-2026-09-14.json).


### Fresh inline tokenizer source under one original allowance

The private aggregate producer plans borrowed root JSON without serde/vocabulary/regex allocation, compares the complete source-derived requirement once, and constructs fresh HF+BPE+added+decode storage under that allowance. It accepts only fully constructed inline profiles and rejects selected regex before admission. The ten pinned complete released tokenizers still require unfinished regex construction; component parity is not full-tokenizer completion.

The decode compiler envelope is Vm+Va ID visits and Bm+2Ba spelling bytes. Successful unique model/reverse-added IDs permit at most two added-spelling visits; ByteLevel's whole-token fallback preserves raw UTF-8. Largest sparse IDs do not size dense storage. Construction failures retain real partial buffers and completed components; idle source aliases retain all original bytes. One atomic request header binds source/N/skip/optional stops before adaptive candidates and uses the existing provider/cursor. Caller JSON, owned input, encoding scratch, parser/chat/public delivery/copy and remaining native inventories are separate unfinished obligations.

Fresh inline tokenizer aggregate construction passed 2,152 test executions across eleven checks, including nineteen new behavioral cases and seven compile-fail examples. One original allowance covers actual fresh HF, packed BPE, added vocabulary, supported pipeline and decoder construction; closed owners retain it through idle sharing and final error/source retirement. Shared request admission preserves ordinary/controlled decoder behavior. All ten pinned complete released tokenizer artifacts still reject their unfinished regex profile before construction. Owned input, encoding workspace, other selected components and public/native managed-memory completion remain outstanding. See [`bounded-original-aggregate-tokenizer-integration-2026-09-14.json`](validation/bounded-original-aggregate-tokenizer-integration-2026-09-14.json).

Native publication scope ownership passed 413 test executions across three checks, including three new behavior cases. Real native owner reclamation is fenced from recursively publishing or certifying; the same move-only scope survives attachment, late failure and unwind without an open cell borrow. Existing CPU/Metal session, quote, capture and reset tests passed. Complete native inventory/backing accounting and public managed activation remain unfinished. See [`bounded-native-publication-owned-scope-integration-2026-09-14.json`](validation/bounded-native-publication-owned-scope-integration-2026-09-14.json).

## Original retained-file tokenizer input

`PreparedArtifactFileRead` consumes an actual regular-file handle and derives its
exact extent. Runtime compares original input I before one Vec reserve/read.
Fixed offsets, a one-byte growth probe and shared before/after metadata checks
reject incomplete or observably changed input without publishing it. Source
preparation and path opening remain separate; the new admitted read is currently
implemented for audited Unix positional I/O.

A successful read ends its active count while retaining all of I. The existing
aggregate root plan then derives C from those immutable bytes; C's comparison in
the same pool sees I's live capacity. Success destroys input/control storage
before returning the unchanged C owner. Failure retains actual file/read buffers
and any by-value C failure inside one concrete core error envelope. No premature
refund or second quote of an existing tokenizer supplies the overlap.

The exact/one-short, real OS error, actual reserve-frontier, source-change,
concurrency and retirement fixtures use source-derived I/C facts. A file-source
case reuses the genuine ordinary/controlled/detached decoder-provider helper.
All ten released files still stop at their selected RegexProfile after admitted
input reading; that does not establish complete released-tokenizer support.
Path opening, other platform bounds, encoding scratch/results, regex constructor
and execution work, parser/events and copy obligations remain unfinished.

Original retained-file tokenizer ingress passed 1,931 test executions across nine checks, with thirteen new behavior cases and two compile-fail examples. Ten unchanged released tokenizer files exercised actual admitted input reading and retained RegexProfile rejection. I stays charged through fresh C construction; success retires I and errors retain actual input/partial compiler owners. Unix file reads are implemented; path opening, other platforms, encoding, regex and complete public managed inference remain unfinished. See [`bounded-original-tokenizer-file-integration-2026-09-14.json`](validation/bounded-original-tokenizer-file-integration-2026-09-14.json).


### Internal captured prefill span component (2026-09-14)

The shared SessionPrefill driver now has an explicitly ordinary custom operation seam used by materialized embedded prediction: selected target readout and declared capture, then non-nested auxiliary StateOnly seed, then aggregate completion and the existing chunk finalizer. For a five-token prompt with width two, target spans are 2/2/1; sequential shifted seeds are 1/2/1 and aligned DSpark seeds are 2/2/1. Only one hidden carry row persists between spans. Actual target and auxiliary frontiers determine capture identity and placement. Verification and committed-prefix attribution remain unchanged.

This component removes avoidable prompt-wide capture/readout on its unobserved route and adds exact window translation for slice collectors and evidence-free activation edits. It does not supply an original speculative quote, price native graph/task populations, admit media/assistant seeds, select a default memory ceiling, or activate end-to-end public managed inference. It also does not make unsupported global collector reductions disappear; their aggregation is a required follow-on. State storage, checkpoints, captures, native allocations and general ordinary preparation retain their existing accounting boundaries. Central validation records describe executed coverage.

Shared captured speculative prefill passed 2,856 test executions across eleven checks, including seven new behavioral cases. Materialized V3/V4, DSpark, Inkling, Qwen hybrid and Nemotron explicit chunks share the ordinary prefill driver, target capture, auxiliary completion and controlled registration. Native nonzero fixtures compare three residency routes, thirteen generated tokens, at least three rounds and snapshot replay; row-window tests retain global strides and edits without refunding cumulative limits. Default embedded whole-prefill, assistants, captured media, global observation aggregation and original managed speculative accounting remain unfinished. See [`bounded-captured-speculative-prefill-integration-2026-09-14.json`](validation/bounded-captured-speculative-prefill-integration-2026-09-14.json).


### Shared Qwen media spans: ordinary source unit

The selected neutral Qwen VL and conditional-Qwen media path can retain encoder outputs across the existing `SessionPrefill` / `PrefillDriver` decoder spans. It executes the encoder in the first actual decoder transaction, imports the completed cut for later spans, and completes all future-only projected/DeepStack roots before publishing span one. It preserves full temporal decoder state, architecture-owned masks/positions and exact vocabulary demand. LastPosition selects hidden rows before projection; Sequence retains every requested row.

Six new `reference_numeric` cases exercise nonzero selected resident/host/disk, TP/PP/combined, run/step, widths 1/2/3/full, real 2/2/1 ordered image/video/projected input, prefix cancellation, cached decode and complete rollback. Failure/source checks include cancellation before encoder work, required-observer and stale-revision rejection, and retained future roots after first-span completion failure. Numerical root visitation is not native completion evidence.

This entry is explicitly unfunded. Its prepared input, source metadata, global axes, compact encoder outputs, span/transient and native owner populations are not made free by reuse. `InferenceRetention::admit` does not install an admission for an unfunded request, so ordinary cached decode from a cancelled committed prefix remains compatible; funded prompt-end validation is unchanged. A terminal driver has no resume-offset contract, and restored revisions cannot revive a source. Original managed media admission, captured-media attribution, unfinished-source fork/resume and native routing remain unfinished work.

Conditional Qwen's retained-media decoder boundary normalizes compact DeepStack
features to the current decoder-row interval before transport. Vision edges
remain compact. The complete boundary arrays are real sender transients, whose
construction uses the same mask/scatter equation as local decoder execution;
this correction introduces no original media allocation claim. Existing selected
full/prefix/cancellation tests remain, with an additional nonzero single-rank
comparison for PP cuts after layer zero, both DeepStack additions, widths 2/3/4,
all three residencies and three cached decode steps. Central execution is
required; staged source/replay alone is not a numerical validation result.

Selected retained Qwen media prefill passed 2,164 test executions across nine checks, including seven new numerical cases and one borrowed-state lifetime regression. Qwen VL and conditional Qwen share encoder-once retained ingress and decoder spans across resident/host/disk and TP/PP/combined neutral fixtures, preserving original positions, all state, requested rows, prefix cancellation, cached decode and rollback. Source/observer rejection and failed future-root retention also passed. This is an explicitly unbudgeted neutral component; native/public media activation, complete original media bounds, other-family hooks and captured-media lifecycle remain unfinished. See [`bounded-composite-media-prefill-ingress-integration-2026-09-14.json`](validation/bounded-composite-media-prefill-ingress-integration-2026-09-14.json).

### Original identity-profile encoding destinations

The checked no-transform, packed-literal, no-dropout/affix/fallback BPE profile now has one original E operation before symbol, merge-heap and ID reserves. It uses the existing HF span/symbol/merge workers and preserves real partial reserve errors. Success and error retain the same C source lease and all E bytes through actual destination/control retirement; ending the active count does not refund bytes. E is never adopted as request I: the existing genuine TokenIds plan borrows it and admits a separate destination. Source reconstruction, raw HF access and a second driver are absent. Normalization, other added-token flags, ByteLevel/template encoding, regex, pairs/offsets/batch work and owned public delivery remain unfinished obligations. No released full-tokenizer or public managed-encoding completion is claimed.

Original identity-profile token-ID encoding passed 2,754 test executions across eleven checks, including fifteen new behavioral cases and four compile-fail examples. The shared HF matcher and merge worker populate originally admitted E destinations while retaining the exact C source. Actual partial reserves, concurrent operations and poison preserve accounting; existing genuine I request input consumes borrowed IDs in ordinary, controlled and detached paths, with native prediction and full logical KV parity across three residencies. Released regex/pipeline encoding and complete public managed activation remain unfinished. See [`bounded-original-tokenizer-encode-integration-2026-09-14.json`](validation/bounded-original-tokenizer-encode-integration-2026-09-14.json).

## Muse retained-media source: ordinary subset

Muse-Glimmer now supplies its own compact normalized raster-media source to the shared selected media lifecycle. It preserves image/video placeholder IDs, text-embedding normalization and per-layer cache-derived masks, completing all future projected rows under the first span. Received encoder boundaries rebuild placement metadata without repeating patch/learned projection. The nonzero local and selected TP/PP/combined cases cover run/step, resident/host/disk, full/last local rows, mixed image/video 2/2/1 and ordinary cancellation-prefix cached decode. Central validation records describe executed coverage.

This is an ordinary/unfunded mechanism, with no native/public retained-media activation or complete peak claim. Gemma4/Inkling still require a neutral observed-inactive dependency state and their own per-layer/shared-attention or zero-unit audio semantics. Native original source/completion, funded continuation and remaining-family integration remain required follow-ons.


## Shared local external-assistant prefill component

Gemma4 and Muse-Glimmer DFlash now have a selected text-span consumer on the existing shared prefill lifecycle. It preserves target commit identity, original selected capture/input witness, cancellation work counts, derived-root completion, snapshot state and whole-sequence verification. Gemma keeps one hidden row plus its required complete K/V prefixes. DFlash keeps the ordered rolling raw suffix and performs initial encoding only at the first proposal. This removes avoidable full-prompt capture/score retention; it is not a total-memory quotation.

The adapter is explicitly unbudgeted. Host metadata/envelopes, checkpoints, native operator graphs/completions and temporary seed overlap still need original speculative admission. No late grant, quota refill or managed activation is added. Selected TP/PP capture-bundle publication, captured media and global observer aggregation remain unfinished mechanisms, not architectural limitations. Native nonzero tests cover Gemma ordinary/control/snapshot on resident/host/disk; DFlash has released-geometry protocol and exact nonzero suffix-equation coverage with the full native numerical gap retained.

## Gemma4 and Inkling optional media: ordinary scope

The selected retained-media cut now distinguishes Unseen, observed Inactive, and Produced dependency outcomes. Existing layered traversal and partition scheduling establish inactivity; later spans import it without fabricating a tensor. Missing active dependencies still reject. Inkling's active zero-unit dMel group executes its existing completion equation. Default hooks preserve other executors.

Gemma4 and Inkling provide architecture-owned retained-source/span hooks through the existing selected media visitor. Gemma4 keeps independent vision/audio projection owners, original placeholders, per-layer inputs, cache positions and shared-KV transport; each input projection runs only at its existing root begin. Received encoder continuations use original metadata, without dummy shape tensors. Inkling preserves exact hMLP folds, dMel offsets/normalization/valid frames and sconv state. Full future media roots settle under the first span; original semantic vectors move into the source while encoder preprocessing temporaries remain under first-span completion.

Five nonzero numerical cases cover: compact optional combinations across resident/host/disk and TP/PP/combined, ordinary versus shared full/last rows, run/step, cancellation at two actual boundaries, complete local state and three cached decodes. The exact large Inkling hMLP receives focused resident/host/disk and bounded PP coverage with 512 MiB fixture budgets. Its remaining raw-image topology/residency cross-products remain explicit unexecuted validation obligations, not unsupported behavior.

Central validation records report executed coverage. This is ordinary/unfunded selected-driver support. Public/native retained-media activation, full original source/control/graph/completion admission, captured-media attribution and funded continuation remain Unit C work. New source and context allocations are not caller-owned or free; no native peak or whole managed execution claim follows from these changes.


### Captured-prefill host Summary/Histogram reduction component

A source implementation adds fixed-per-selection Summary/Histogram accumulation over the existing captured embedded prefill spans, with separate logical target/seed attribution and preallocated terminal delivery. All actual partial transforms retain their existing capture charges. The added persistent host/control/wire reservation uses actual type/rank/string/bin facts; abort, Skip, restore and Drop never refund it. Final status follows the whole prefill, including final indexing and state exchange.

This component requires central validation; it does not establish end-to-end public bounded inference. Original managed destination custody, ordinary/funded collector adoption, external prefix observations, parallel time×space coverage, other global collectors and native backing/encoder/task/future populations retain their separate obligations. Existing native transform estimates remain conservative component envelopes.


### Explicit ordinary native media spans (C1)

The existing chunk option now reaches selected native retained-media adapters. Unconfigured ordinary ingress retains its existing behavior. The exact token-only predicate is shared with the existing text source constructor so media-capable model selection does not intercept token-only original/captured paths. The media route preserves the supplied request and source identity and cannot fall back after a selected-source error. A real original reservation, converted or unconverted, is rejected before the media plan factory; the tests use genuine scalar text admission only to prove that rejection, never to fund media work.

Native source completion includes all future roots in the same output/state/validation event. New source tests are staged for dense/routed Qwen VL and conditional Qwen, resident/host/disk, unequal decoder spans, ordinary cancellation prefixes, nonzero full mutable state and three cached decodes. Pure-text capture, exact request replay/foreign rejection, first validation failure and core iterator/manual advancement are covered separately. These are execution obligations until the integration record reports results; native TP/PP/combined media execution and native Muse/Gemma4/Inkling numerical matrices remain unexecuted here, alongside the existing neutral selected matrices.

Managed media still needs original bounds and custody for actual prepared host metadata and native inputs, semantic plans/cuts, retained encoder/DeepStack/per-layer roots, each decoder and rollback transient, native vector/event/record/graph populations, source/result/error controls and observation/copy ownership. Existing I/R, text and graph contributions do not authorize these populations. This unit adds neither a quote nor a late grant.


### Inkling hMLP pipeline boundary correction

The shared selected media path now carries the exact hMLP source cut and derives actual row geometry from the admitted patch count. The fixed released tower emits `[p,2,8,8,128]`, `[p,2,4,4,w]`, and `[p,2,1,1,4800]` after units 1, 2 and 3 respectively (`w` follows the released configuration). Their wires are `[1,128p,128]`, `[1,32p,w]`, and `[1,2p,4800]`; the receiver restores the original rank-5 shape. Cold communication capacity checks use the same cut and the selected maximum decoder extent, which bounds the patch population because each patch contributes one decoder position. All extent products are checked.

This remains ordinary shared-driver behavior. The route gains one concrete optional endpoint field and native reshape/transport retains the existing owner/completion policy. There is no new original media grant, source/control or graph bound, or claim that reshape is allocation-free on every backend. Managed media activation, public captured-media startup and the remaining raw-image topology/residency cross-products remain separate obligations.


### Inkling received dependency regression

The actual hMLP PP test exposed a second missing handoff after wire packing was corrected: a received completed vision tensor was retained by the shared pass, but the default family acceptance hook left the forward context empty. Inkling now installs the final dependency before retained-source construction and accepts decoder continuations on ranks that already hold encoder context. The existing PP2/PP4, cancelled-prefix, image-first and complete-state numerical case is the central regression; no new case count or executed result is claimed by this source correction. Original managed media and the wider validation matrix remain unfinished.

The combined media and observation prefill integration passed 3,653 test executions across seventeen checks, including 33 new behavioral cases. Muse and optional Gemma/Inkling retain semantic media roots across selected residency/parallel spans; native dense/routed Qwen media settles future roots and preserves state, cancellation and cached decode. External assistants share the span lifecycle, and logical Summary/Histogram reductions preserve whole-prompt results and failure accounting. Native control/token ownership and portable regression suites pass. Complete original media/native bounds, public prepared-media control/capture, remaining observation paths and default managed activation remain unfinished. See [`bounded-media-observation-prefill-integration-2026-09-14.json`](validation/bounded-media-observation-prefill-integration-2026-09-14.json).


### Ordinary prepared controlled input remains outside managed activation

The V2 prepared-input controlled entry is ordinary preparation. A provider acquires actual unquoted custody before attribution vectors, descriptors, token-value reads and shared-owner construction. The facade retains that custody on the closed source, records, escaped snapshot metadata and new owning failures, including failure of its later host acquisition. All strong source/record/metadata aliases retire through `Arc::into_inner`; their payload and shared shell retire before their last custody. This reuses the existing payload-free `HostPreparationAuthority` contract and does not claim that its erased token or arbitrary transitive input payload is free.

New concrete storage includes `MlxControlInput`, `PreparedControlBinding`, the private pending attribution option, `AttributionOwner`, `PreparedRecordOwner`, `PreparedSnapshotOwner`, public/private snapshot facts, source descriptor/maps and copied strings/ID buffers. Ordinary logical snapshot accounting includes the actual source descriptor heap, segment/ID buffers, concrete shared attribution and metadata shells, actual generic snapshot/branch controls and both copies of private/public snapshot strings. Existing original token-prompt control formulas use the actual `Result<MlxModelInput, BackendFailure>` layout, so the always-present optional field is reflected while original inputs retain None. Logical retained sizes and JSON transport bytes are not construction peaks, allocation quotas or original admission. Registry/tokenizer/input/graph and erased custody costs remain under their existing boundaries.

Facade managed/submission/graph policies reject before the new attribution factory. Native opaque inputs carrying an existing quote/reservation also reject rather than stripping authority or acquiring a later grant. Installed empty capture is not unobserved. This unit contributes no new R/G/I/provider allowance and does not activate a managed media completion path. Mandatory next units are complete ordinary pending-media copy/continuation extent integration and actual captured-media attribution/collector applicability, followed by complete original source/event/control/transient facts before managed activation.

### Prepared-media pending copy (ordinary Unit 2)

Pending V2 media can use the existing ordinary snapshot/continuation mechanism through a checked source-borrowed inventory and independent contiguous per-slot copies. Logical estimates count every payload/metadata replica and the newly retained source/partial/control containers; they are not original physical peak bounds. Actual source and partial-root ownership survives failure/unwind under native recovery, with completed escaped backing retaining ordinary exclusion. Authenticated decoder extent drives checked continuation growth; no new request, refunded quote or metadata-shape-derived attribution is introduced. Managed activation and full original source/graph/transient bounds remain open, as does separately reviewed Unit 3 captured-media composition. The staged tests add all-value/view/metadata/final-owner checks and public resident/host/disk snapshot/restore/fork parity; they are not execution results until centrally validated.

The combined prefill follow-on integration records 4,034 passing test results across twenty checks, including 450 revalidated archived regex results, with 66 newly introduced behavioral tests and three new source-ownership compile-fail examples. Terminal TopCandidates, closed tensor retirement, V2 prepared-input control and independent pending-media snapshot copies, and global speculative Preview use shared drivers and actual retained source ownership. Complete original native/media bounds, captured prepared-media integration, other tokenizer profiles and default managed activation remain unfinished. See [`bounded-regex-top-candidates-integration-2026-09-14.json`](validation/bounded-regex-top-candidates-integration-2026-09-14.json).


### Fresh regex constructor scope

Exact source-selected private recipes now describe fresh fancy instruction/delegate construction with one attempt for every real buffer and one shared bounded finalization scratch. Anonymous capture metadata avoids name-map initialization; final NFA properties are recomputed by the ordinary walk. Owning failures preserve partial or completed source storage. Source/control layouts use the pinned concrete Rust implementation; static recipe image residency, process-OOM and future aggregate original C/E ownership remain separate. No construction execution evidence or full released-tokenizer support is inferred from this source change.

The first original regex tokenizer semantic profile uses exact decoded pattern/component selection, never a family label. Fresh aggregate C adds actual constructor and closed source-allocation facts; synchronous E adds actual workspace requirements and a checked 2B mapped buffer for B borrowed input bytes. Real workspace failure prefixes retire before the owning result/error returns, while full E and the exact C source remain held. The private file path retains original I through C, and request I/R retain their existing independent destinations. Released-profile execution, allocator observations and source-test outcomes must be recorded separately from source acceptance. NFC/Digits, normalized added matching, templates, owned input/events, other regex profiles, native inventory and public gateway work remain incomplete obligations.

The original implicit ByteLevel profile derives its exact default pattern from the same constant used by ordinary ByteLevel, without initializing that global. C includes the actual fresh source and final closed wrapper controls; E adds numeric iterator controls while retaining the existing B/3(B−1)/B/2B destinations and one actual workspace. Numeric boundaries use char::is_numeric, with no inserted prefix or per-piece split allocation. FirstLiteral and nested reserve errors retain original custody. Full released-file parity and execution evidence must be recorded independently; NFC, normalized added matching, templates, owned public input/events and broader native/gateway work remain incomplete.

The original regex/tokenizer integration passed 3,053 test executions across twenty-one checks, with 48 newly introduced behavior cases and five explicit source-ownership compile-fail examples. Fresh construction and one source-bound encoding workspace preserve original C/E/I/R ownership for the actual complete Inkling and SmolLM2 tokenizer configurations, using shared HF workers. Other tokenizer configurations, native/media bounds and public managed activation remain incomplete. See [`bounded-original-regex-tokenizer-integration-2026-09-14.json`](validation/bounded-original-regex-tokenizer-integration-2026-09-14.json).

Ordinary prepared-media capture integration records 3,305 passing test executions across eighteen checks, including thirty-six new behavioral tests. FullTensor/Slice, Summary, Histogram, Preview, terminal TopCandidates and explicit empty Capture share media-prefill scheduling and one logical frame; source/checkpoint identity, cumulative budgets, copies, and final-alias frame retirement are preserved. Original managed media, standalone error/callback custody, encoder/intervention and distributed capture remain unfinished. See [`bounded-controlled-captured-media-integration-2026-09-14.json`](validation/bounded-controlled-captured-media-integration-2026-09-14.json).

### Original prepared host source: bounded residence with an ordinary consumer

`PreparedHostInputPlan` now derives complete source capacities by borrowing ordered U32/I32/F32 payloads, shapes, known metadata and extents. `compile_prepared_host_input` compares once before seven reserve sites and the final shared owner, retaining the whole source/construction allowance through partial errors, poisoned settlement and final alias retirement. Metadata is canonicalized; equal content digests do not confer identity. Bool/scalar input is a typed current source-encoding/shape scope, not model-family inapplicability. The old ordinary processor still lowers Bool.

An actual selected-native consumer uploads the source under a genuine ordinary owner and uses shared architecture lowering and the existing media prefill driver. Source-authored tests compare Qwen VL and conditional-Qwen nonzero prefill/full state plus three cached decodes in resident/host/disk; another case checks compiled-source core Prepared iterator/manual advancement and actual five-span completion. They remain pending central execution at staging time. Uploaded prompts cannot be relabelled as managed, and a real prior ordinary owner still rejects new original source compilation in that pool.

This source amount covers requested buffer payload and named target control layouts, not preceding file/processor storage, allocator/RSS, native upload/backing, encoder/private workspace, selected plan/cut/context/completion controls or facade records. Original managed media still requires A bounded source-derived selected compilation; B original upload/numerical work; C complete source/driver/completion/rollback controls; D genuine one-time request/source/run admission; E original V2/controller/parser/records/transport/copy overlap. Capture retains its separate source/frame obligations. No managed, captured, released-checkpoint or native distributed success follows merely from this source owner and ordinary consumer.


### Prepared-media Stage A: original semantic residence

An original Qwen VL/conditional-Qwen semantic body can be compiled before ordinary model loading and consumed by the actual selected native prepared-input route. Its source-derived fixed record count is the actual number of input parts. VL uses exactly four i32 destinations per decoder position (three global axes and every committed-prefix delta) in one vector; conditional Qwen has zero coordinate elements. Grids remain borrowed from the original host source. One source-compiler admission precedes both reserve sites, filling and final shared-owner creation. Errors retain real prefixes; all strong semantic owners finalize through `Arc::into_inner`, with buffers and shells retiring before allowance.

The legacy native runner now passes its existing paired admission forward. Compiled unobserved execution bypasses that allocating inspector/admission/fingerprint path. Captured startup still has its distinct ordinary validation/attribution costs; these are neither removed nor originally covered. Existing semantic workspace scalar equations are preserved estimates, not new native workspace facts.

The source package adds unexecuted acceptance fixtures for exact/one-byte-short capacity, both actual reserve failures, poison/unwind/final aliases, source/selection/current-state rejection, both Qwen consumers over resident/host/disk with prefill and three cached decodes, actual core iterator/manual advancement, cancellation before/inside media, and independent-copy fallback. Central integration records determine execution status; source review is not a passing run. Existing neutral TP/PP/combined media cases remain required regression, and native distributed/released-checkpoint gaps remain open.

The mandatory remaining sequence is B (native upload/encoder/private storage), C (shared source/driver/context/output/completion and native graph/control capacity), D (one genuine original generation comparison and source/run bank), and E (original V2 attribution/controller/parser/delivery/copy plus applicable capture accounting). Cold model inspection/configuration, prior processor storage and ordinary native owners keep their existing boundary. This source residence never adopts prior allocations or activates managed prepared media.

The packed-template original profile derives piece/special/ID/spelling/byte capacities from the complete immutable postprocessor object, including pair metadata. Each of five real reserves is attempted once and its full partial owner survives failure under C. Single-input E validates exactly one A and reserves B+S IDs from the actual special flag, reusing the existing raw matcher, regex, ByteLevel and BPE workers. It inserts no second destination or source charge. Muse complete-file parity is a required independent execution check; normalized-added LFM E, NFC and empty-affix E, original pair input, full owned Encoding/events and public/native closure remain unfinished.

The null-normalizer two-phase profile matches normalized literals only within spans emitted as unmatched by the filtered raw phase. Accepted raw tokens remain boundaries; skipped raw specials retain their existing consumed-match/coalesced-gap semantics. The original E layout keeps B+S IDs, B symbols, checked merge capacity, 2B mapped bytes and one regex workspace, adding the exact two live iterator/callback/offset controls where a normalized phase exists. Four unchanged LFM files are the required next full-artifact oracle; C buffers and source charge are reused without adoption or requotation. These source changes require execution evidence before a verified support claim.


### Ordinary prepared-capture standalone error custody

The ordinary shared-frame lifetime component now has a separate standalone-error path: runtime typed diagnostics, native observer/prediction errors, asynchronous completion/scalar failures and escaped retained neural aliases carry existing host custody independently of the frame or session. Exact private Arc/source layouts are named; retained Clone shares diagnostic storage and final consuming retirement removes control/source allocations before custody. Legacy NN error construction is unchanged.

This internal lifetime component does not provide a finite diagnostic byte ceiling or complete end-to-end managed admission. Native exception/String extents, application formatting, unrelated error APIs and incomplete native/private workspace bounds remain explicit. Mandatory frame control cannot fall back to Skip or fund an error after refusal; completed capture/copy work is never refunded. Central compilation/behavior validation of the source package is required before recording it as validated.

Original host, media-semantic and tokenizer integration records 4,303 passing test executions across twenty-nine checks, including seventy-two new behavioral tests and eight ownership compile-fail examples. Source-derived original reservations precede fixed host/semantic/template buffers; selected graph/session binding and completed ordinary native upload feed shared controlled and uninterrupted drivers. Shared postprocessing and ordered raw/normalized literal matching write directly to original ID output. Qwen VL and conditional-Qwen prefill plus cached decode parity cover resident/host/disk; independent complete Muse, Inkling, SmolLM2 and four LFM files validate token IDs and decoded suffixes. Standalone ordinary capture failures retain complete diagnostic/control custody independently of frames. Native allocation/encoder/control bounds, remaining tokenizer profiles and families, and public managed activation remain unfinished. See [`bounded-original-host-template-integration-2026-09-14.json`](validation/bounded-original-host-template-integration-2026-09-14.json).

The CPU TopCandidates prerequisite records 14 fresh Rust test executions across focused CPU/Metal extraction, copy and shared Metal terminal-program checks, plus native/feature verification. The seven named native standalone passes are separately retained historical evidence and are not included in this fresh execution total. Rank-one F32 CPU Argsort uses its existing U32 output without auxiliary numerical heap or recursion. The strict managed CPU workspace and capture gates remain closed; whole-inference, task/stream/metadata, other kernel and allocator bounds remain unfinished. See [`bounded-cpu-top-candidates-integration-2026-09-14.json`](validation/bounded-cpu-top-candidates-integration-2026-09-14.json).

The original NFC E plan derives canonical scalar D and decomposed UTF-8 N from the exact input using pinned Unicode 9.0.0 tables. Three once-reserved destinations retain D records, D recomposition pairs and N text bytes; the unchanged encoding worker prices N symbols, its checked merge heap, N+S IDs and 2N mapped bytes. Empty BPE affixes preserve Some("") identity while avoiding unnecessary formatting. Raw match priority/special filtering and source-range boundaries remain shared with ordinary HF. Actual partial NFC reserves and every later result keep storage under the same original E/source owner. Ten complete pinned files and pre-refactor Unicode parity are required validation; pair-input E, transforming added-token profiles, path/caller input, owned delivery and broader native inventory remain unfinished and public managed activation stays closed.

Original NFC and empty-affix encoding records 2,543 fresh Rust test executions across fourteen checks, including seventeen new behaviors and three new compile-fail examples. All ten complete pinned tokenizer files are separately required by the original C/E/file-I oracle; a distinct pre-refactor Unicode oracle covers all scalars and long nonstarter runs. These oracle comparisons are reported separately from Rust execution counts. Three once-reserved NFC destinations stay under the same original E/source account through later regex/BPE errors and retirement; existing input and decoder drivers remain shared. Public managed activation, original pair E, transforming added-token profiles, other normalizers/templates, caller/path/owned event and broader native bounds remain unfinished. See [`bounded-original-nfc-empty-affix-integration-2026-09-14.json`](validation/bounded-original-nfc-empty-affix-integration-2026-09-14.json).

### Prepared-media B1: original native source leaves

B1 adds a usable original native source producer and an ordinary selected consumer for the existing Qwen VL and conditional-Qwen compiled path. The quote derives one backing per actual canonical payload/metadata slot, the actual native leaf/descriptor/Data/control blocks, rank-dependent Shape/Strides spills, one arena with its real Rust retirement node, the exact typed leaf destination, and closed source/account/partial-error controls. Native construction reserves the actual libc++ requests first and has no heap fallback. CPU pages include their header; Metal pages include an intrusive prepared residency node instead of allocating an ordinary hash entry. Opaque driver internals, allocator/RSS overhead and runtime initialization are not assigned invented bounds.

One genuine source-compiler comparison precedes these allocations. An unarmed Arc is allocated before the existing allowance transfers, leaving exactly one refund-capable account. Source and native callback references use consuming finalization; Box/Arc shells and native blocks retire before the final allowance. Every real prefix survives failure, poisoned settlement is fail-closed, and equal content never supplies source authority. Ordinary array aliases retain B even after the logical source drops.

Staged tests cover exact/one-short and foreign/unquoted refusal before constructors, actual typed destination reserve failure, native block/backing refusal with retained prefixes, independent object/control/alias and concurrent reclaimer retirement, both selected families in resident/host/disk with full state and three cached decodes, core Prepared iterator/manual parity, and independent pending-copy fallback. These tests are unexecuted at package freeze; central records determine passing status. Positive native tests require the pinned libc++ realization; Metal residency coverage additionally requires actual residency-set hardware. Native distributed/released-checkpoint validation remains open.

This is the B1 leaf increment within the mandatory native-input work. B2 original input/handle/control construction, B3 encoder/private workspace, C shared driver/context/output/rollback/future roots/completion/Graph/Record/Scope, D complete original request binding, and E original V2/copy/capture remain required. Ordinary compiled consumption is useful behavior, not managed media admission or a whole-operation memory bound.

The B1 native source leaf increment passed 1642 fresh Rust test executions after reviewed test-only compatibility corrections. Native production source and all historical failed-attempt evidence are separately pinned; complete original managed prepared-media admission remains unfinished. See [`bounded-original-native-prepared-input-integration-2026-09-14.json`](validation/bounded-original-native-prepared-input-integration-2026-09-14.json).

### Full original prepared-input adaptation B2 (source package)

`MlxPreparedInputMaterializer::model_input_plan` chooses complete B before comparison, from A's exact original host source. It covers B1 leaves plus both handle/part populations, fixed metadata/extent/shape storage, both identity descriptions, the two 64-byte fingerprints, shared owner controls, the inline completed packet, and one terminal bind-error population. Completed B1-only objects cannot be upgraded. Exact/-1 tests refer to the original B comparison; native block queries retain their explicit alignment costs and are not a promise that every physical arena has zero padding.

Completed prepared views, parts, cache aliases and owning diagnostic identities retain the same account. Their controls use all-strong consuming retirement, with allocation/control and actual payload destruction before final custody release. Source or cache aliases do not recreate bind authority. Original cache attachment returns only authenticated existing residence, never fresh credit.

The new source tests cover checked host/native prefix refusal, exact/short/foreign admission, concurrent escaped aliases, target identity, one-shot binding rejection and actual shared ordinary/manual consumers for Qwen VL and conditional Qwen across resident, host-layerwise and disk selections. Source/generator/native replay is separate from central execution. B3/C/D and controlled snapshot/capture/distributed admission remain required follow-ons; no whole-path finite bound or public activation is claimed.

The B2 original model-input increment, including root export/API documentation and the shared-identity test adaptation, passed 2258 fresh Rust executions in its final continuation. Seven previously verified Rust gates (536 executions) and six earlier native/safemlx gates remain authenticated historical evidence with no fresh credit. Complete managed prepared-media admission remains unfinished. See [`bounded-original-model-input-b2-integration-2026-09-14.json`](validation/bounded-original-model-input-b2-integration-2026-09-14.json).

Original plain-string composition selects a canonical C-domain destination before fresh compilation: checked max(max_model_id+1, raw_model_entries+raw_added_entries), marking only canonical reverse-to-forward IDs. JSON declared added IDs do not replace actual fresh assignment. C remains charged once; E is copied once through the genuine original I claim. The terminal R byte destination uses the conservative M×L envelope where L is the actual whole-history DecodeStreamLayout(M,skip).text_capacity(), so this envelope may be quadratic. Actual reserve failures retain successful prefixes and full original custody; text/ID aliases retire through the same payload. Chat rendering, owned caller acquisition/path opening, structural/owned live events, managed copies/resume and missing native whole-inference inventory remain obligations, with public managed gates closed.

The original private plain-string increment passed 1 fresh Rust executions in the ten-file90-request oracle and passed the explicit maximal portable and MLX/Metal facade feature checks.3739 earlier core/text/runtime/facade/portable/conformance/native/HF executions and2compile-fail examples are historical. CUDA/NCCL platform validation, the CPU numerical matrix and public managed activation remain unfinished. See [`bounded-original-text-generation-integration-2026-09-14.json`](validation/bounded-original-text-generation-integration-2026-09-14.json).

### Original media encoder tables: first B3 increment

The complete B input recipe now measures and constructs two checked source-table buffers before its one original comparison. Actual selected Qwen VL and conditional Qwen consume these tables during ordinary resident, host-layerwise and disk-streamed prefill. Existing fixed source/cache/native-handle ownership and one-shot binding remain intact. Original table aliases share actual payload custody; no new source credit is inferred from a digest or descriptor.

Ordinary diagnostic execution follows the same selected retained-ingress traversal, recording the encoder cut and complete first decoder interval without resetting the trace. Native numerical totals and closing/opening unions stay separate from ordinary report/configuration/driver metadata and unpriced host controls. Missing native facts remain unknown. No diagnostic report is an original allocation grant or native completion witness.

Authored validation covers shared legacy kernels and no-write failures, genuine full-B exact/short and actual reserve/fill errors, final typed/erased aliases, nonzero two-family three-residency state/three-decode parity, original-table cancellation before/future/inside media, and native finite/unknown/error diagnostics. Execution results belong in a later central record; this source change claims none. Complete encoder/private workspace enforcement, C graph/completion controls, D request admission and E original controlled/copy/capture remain mandatory. Original no-encoder and other-family/distributed matrix work remains unfinished.

The first B3 encoder-source increment completed 3,170 authenticated historical test executions across the original NN/runtime suites and root diagnostics, including all 21 introduced cases and one compile-fail example. Numeric coverage combines 443 successful cases from a failed full run with the one corrected isolated rerun; Metal coverage combines 29 successful cases with two corrected workspace reruns. Three actual compile/artifact-binding gates add zero behavioral executions. Managed media and complete B3/C/D/E remain unfinished. See [`bounded-original-media-encoder-b3-integration-2026-09-14.json`](validation/bounded-original-media-encoder-b3-integration-2026-09-14.json).


### C1 fixed shared-prefill role and root component

The first genuine original text step now owns the finite local prefill scope bank derived for its exact chunk candidate, plus two fixed native root buffers and an inline completion destination per inner transaction. Smaller chunks may require more simultaneously retained controls. Busy preserves the same prepared slot; cancellation/failure cannot refill it. Shared source, encoder, first decoder, displaced prefix and future-root ownership remains through the existing finalizer and final indexing.

This is a source component awaiting central execution evidence, not complete original prepared-media admission. Collector Graph occupancy is reserved in the existing arena; the residual physical ceiling does not prove the other producers fit. Generic validation metadata/errors, native completion/backend tasks, checkpoints/contexts/publication, manager extras and parallel composition remain explicit C/D/E obligations. Ordinary B3 media consumes the same collector under ordinary ownership; no original request authority is manufactured.

### C2 scoped completion and retained native source — source milestone

Each original prefill InputTransaction collector contributes one concrete native
FailureCarrier and one OwnedNode<OriginalPrefillRootCustody>, with actual wrapper,
return, error, UTF-16 scratch and retirement controls, before the existing original
comparison. The same finite role geometry and same Graph/Record arenas remain in
use. Clone/error delivery retains the same allocation; no replacement grant or
post-admission adoption occurs. Smaller chunk candidates still recompute all
collector/control occupancy.

Fixed scoped observation is separate from terminal lifetime proof. A spent
collector or failed query never releases roots: Recovery must establish its own
terminal evidence or retain them. New source tests cover real exact/-1 core
admission, nonzero CPU/Metal roots, failed submission source retention, scheduler
reset/teardown, concurrent aliases and native text after autorelease drainage.
Execution is pending central validation.

The new shell is not a universal control budget. Event/shared_ptr/task/queue,
command/receipt/callback and exception producer/ABI contributions remain in scope
and unfinished. Platform-private driver/error contents are distinct from measured
Eredu-managed library allocations and their bounded object counts. Hard
Graph/Record caps do not prove combined producer fit; original-media activation
still requires the remaining B/C/D/E and applicable parallel components.


## Original prepared whole-request boundary (D1)

OriginalPrepared uses the existing core sequence claim, scalar policy, runtime candidate loop and atomic Q grant. Its tag cannot adopt an ordinary prompt. Runtime can construct the seven diagnostic Strings and window vector after Q under a closed source/error owner. Common account, request, receipt and continuation identities no longer force the identified pre-Q map/identity allocations or early control-shell refunds. Actual numeric node floors and exact ceilings remain active through detached destruction, including concurrent and poisoned paths.

The complete neutral producer uses actual original parameter/input sources, distinct text/media extents, nonzero encoder/recurrent/window state, full core/controller/sequence/output controls and requested Rust 1.98 buffer layouts. Exact-Q, Q−1, eight target-reserve failures, filled poison/unwind, source substitution, all-owner retirement and run/manual 2/2/1/full-state tests are source-authored; execution evidence must come from the root validation record. The completion replacement bound includes both old and new capacity-two buffers for its synchronous three-prediction profile.

Production native managed media remains closed at the exact first missing contribution after authenticated I/A/B and current selected-state checks. Required follow-ons include original no-encoder semantic admission, numerical/private workspace enforcement, complete graph/record/completion/event/task/error producer fit, source/context/checkpoint/publication and parallel controls, then actual native whole-request success and V2/capture/copy/continuation composition. Missing contributions never become zero, an ordinary diagnostic lease never becomes strict authority, and existing original source charges are never retrospective credits.


### D2 first inspection component

Original prepared-media cold inspection can now obtain fixed scalar state/position facts and canonical windows from the exact retained source without pre-Q diagnostic Vec/String allocation. Legacy ordinary report and V2 attribution ownership remain separate. D1 keeps seven description reserves and one exact window reserve under the already accepted Q, with complete partial/error/source retirement. Window iterator/result controls and all changed C2 boundary/result shapes enter their existing checked recipes; no second grant or late source adoption is introduced.

The native positive fact consumer remains a precise early rejection, not managed admission success: `MissingInspectionStorage` precedes WorkspaceContext, native state projection, layerwise workspace or B3 diagnostic tracing. Output width is not guessed from parameter names or vocabulary IDs; the current ordinary path derives it through tracing. Full fixed inspection storage, genuine native backing/encoder/first-decoder bounds, complete C control fit, no-encoder original admission, other applicable families and distributed paths remain required before native Complete. B3's ordinary diagnostic lease never becomes strict authority.

### Retained text output geometry

Cold original-media inspection reads the protocol-visible text vocabulary directly from the retained architecture plan. Tests cover tied/untied nested Qwen heads, DeepSeek prediction-target projection, Moshi padding, Inkling trimmed output, and admitted GGUF geometry. Native Qwen fixtures preserve the same missing-inspection rejection while checking the actual output width through ordinary and controlled claims in every weight residency. This scalar is not a complete transient-workspace estimate; padded projection temporaries, native state backing, and the remaining request populations still require their own bounds.


## Original fixed report destinations

The D3 report component adds four actual scratch buffers (canonical nodes, ordered alias indices, DFS/order frames and marks), three retained flat-source buffers, and their exact concrete control/error/owner shells to the existing private D1 recipe. Layouts are checked before Q; buffers are constructed after publication of that same account. Reserve failures retain actual prefixes and original sources through the existing owning error. No new grant, registry entry or ordinary-lease conversion occurs. Reduction performs no growth or callback and preserves legacy unknown/overflow order. Public flat metadata remains non-authoritative.

The original neutral consumer exercises actual nonzero shared run/manual execution. Native B3 reports remain ordinary diagnostics: their enclosing source/lease does not make their metadata or downstream native work fully bounded. Native request completeness remains gated on the actual remaining inspection/context/shape/parameter/trace/backing and completion producers. Required validation is source-listed; this source component alone grants no execution credit. The inherited D1 request/preparation Mutex first-use allocations require a separate common synchronization correction before Complete/exact-Q validation; this report component does not fund them.

## Event/task control hard caps and remaining producer fit

Original Event allocate_shared controls and typed TaskNode<F> allocations now consume the same actual Graph arena as descriptor/collector producers; Record-owned storage keeps its existing cap. All finite prefill roles preprice their actual carrier and updated concrete wrapper/result controls before Q. The collector occupancy, residual arena ceiling and other producer fit remain separate; smaller chunks can increase role/collector storage. No numeric safety credit or late account is issued.

Actual native cause/string/carrier retirement occurs outside worker and registry locks. Queued captures die before their counted completion notification, and stack Scheduler drain completes while notification state is still alive. Scoped Roots failure still requires independent Scope/Recovery terminal evidence. Persistent runtime baselines, task inner owners, Metal command/receipt storage and initial arbitrary exception ABI remain unfinished managed contributions, distinct from opaque platform-private bytes.

## Scoped synchronous original controls (source-only)

The prefill role recipe now includes concrete native/safe synchronous evaluation transport, Completion, fixed-error/result and runtime-loan controls before the same Q. Each actual role uses its existing carrier and original Graph/Record. The one-root Graph vector moves into the existing evaluator; its physical occupancy coexists with retained collectors and native producers. Graph/Record remain hard ceilings, and complete producer fit remains unknown. No extra account, post-allocation owner adoption, numeric credit or guessed safety bytes is introduced.

NeedsFundedProgress/Unobservable/Busy cannot fall back to global commits/cleanup or spin; partial work requires independent retained Scope/Recovery terminal evidence. CPU inner payloads, Metal commands/receipts/workspaces, arbitrary exception ABI, persistent baseline internals and D inspection destinations remain explicit required contributions. The new fixed error does not make the native path Complete. All tests are source-only until the root executes the consolidated gate inventory.

External Gemma4/Muse default prefill selects vocabulary rows before projection
through the shared prefill driver. Whole-invocation observers and the ordinary
default whole-media bridge retain their complete actual source span; this span
must be priced or rejected by any strict admission adapter, never silently
shrunk or exempted from a budget. Managed speculative requests retain their
existing rejection, and this change introduces no complete original-native
producer. Split external media still needs the selected retained-media source
and root-lifetime integration already used by ordinary media prefill.

Embedded default prefill selects vocabulary demand before projection through the
shared driver, independently from full target capture and shifted draft seeding.
The whole callback/media compatibility span is actual request geometry and must
be priced or rejected by strict admission. Its ordinary admitted source and
semantic token extraction provide no complete original-request or native budget
claim; selected split-media ingress and existing managed speculative admission
remain separate unfinished mechanisms.

SamplingEvent's concrete Rust population is one existing prediction node, one prepared observed cleanup node, one same-node cleanup slot following actual inner resources, one OperationEvent and two fixed optional array slots (token plus optional next RNG key). Its role recipe prices actual node/owner/result/error representations; the original guard survives complete teardown and escaping errors. The explicit selected-stream producer owns real frontier work even for evaluated roots. Cold descriptor settlement is a retained-observer validation operation, not another evaluation. Source-only nonzero, Busy/drop, missing-arena and foreign-stream and delayed-inner-retirement tests remain pending execution. Whole native producer/arena fit and remaining prediction roles remain mandatory unfinished work.

### Original TokenScalar continuation

The original TokenScalar path uses its configured role plus an independently retained SamplingEvent source; the newer role does not substitute for producer ownership. Named cold controls include the changed token Rc, source observer/guards, actual scalar ScopeRetention and closed Arc<Error> requested layout. No Array/Vec/event producer is introduced by scalar reading. Exact native same-role producer fit and finite escaping-error populations remain mandatory integration work: repeated SamplingEvent polling can currently allocate multiple retained source errors, so no Complete quotation follows from the single role guard or these named layouts. Ordinary conversion remains available, and the selected original uint32 profile is explicit.
### Background prefetch lifecycle prerequisite

Selected ordinary prefetch now reserves its exact unit-domain slots and bounded FIFO once, uses ordinal work tokens, and leaves at most one outstanding work notification. Checked per-state attempt sequences reject stale retries within a cancellation generation; typed exhaustion refuses new issuance while existing work can finish. Retaining transitions and partial reserve errors keep real error and buffer ownership explicit.

These finite lifecycle populations do not close the worker's physical channel/OS/thread storage or its source-to-host producer. The existing background workspace rejection remains mandatory pending the original source-bound epoch, direct host-fill/read destinations, worker claim, exact native/control facts and physical owner overlap. No ordinary worker or diagnostic lease becomes original authority.
### Native descriptor count/fill companion (source unit)

A fixed native kernel now reports the retained descriptor's rank, dtype, logical bytes and the existing classifier's certified full backing. Exact fill validates the same source/current-state witness even for rank zero, then copies only the caller's shape destination. There is no rank cap, tensor evaluation, completion polling, backing adoption, generic native error formatting, or new allocation owner in this kernel. A separate borrowed safe loan uses an existing RuntimeCallGuard; ordinary snapshots/owning try-locks retain their ordinary resource behavior.

Concrete kernel/loan/error/control representations are exposed separately from source/shape destination storage. They are not a whole-request bound. Outer guard acquisition/final-unlock bookkeeping, source/parameter projection frames and destination/workspace construction remain unclosed selected-owner obligations. Source-only tests and ABI probes require CPU/Metal execution before conformance is claimed; this addition does not activate managed native media or alter C+D completion/error semantics.

### Native descriptor owner and private runtime acquisition

ExistingArrayProjection uses the fixed descriptor loan under its actual source/runtime owner rather than allocating a separate shape snapshot. The private reentrant mutex uses bounded short spinning/yielding followed by capped sleeping; it has no parking wait route. This removes parking-table allocation from this lock's final release without adopting a warmed global allocation. Concrete mutex/guard/wait representation facts are exposed, while actual per-thread identity TLS remains explicitly unknown until the compiler/target receipt.

The shared descriptor control recipe includes the new fixed layout result. Static runtime and per-thread baselines remain separate. Ordinary reclaim/hooks, inspection destination maps/layouts, handle cloning and generic errors are still required follow-ons before a complete original request claim. No capability or admission comparison is widened.

Constructor format sources now have borrowed row workers shared with ordinary map construction. They retain exact selected names, source/native versus selected/executable encoding, role filtering and iteration order. The ordinary selected-matrix adapter still allocates its output map and a temporary borrowed-name membership set; it no longer clones source names into a second map. An inspection scan can consume the same source rows without those maps, with quadratic membership work if it uses repeated scans. This is not a full constructor population or storage bound: configuration clones, parameter specifications, modules, context/trace storage, ordinary errors and native resources still require their own admitted construction.


### Host/disk parameter metadata count/fill component

A retained host/disk `LayerwiseWorkspace` can lend its actual selected rows to a strict neutral count/fill worker. Counts cover complete canonical-owner closure units, requested units, every binding name/shape, canonical root prototypes and actual permitted-window members. Exact element sizes and named concrete wrapper/result size/alignment facts are exposed; there is no guessed rank or name cap. Element payload is separate from allocation headers, destination-owner controls, source construction and full request admission.

The caller supplies all seven destinations. Short capacity or fixed semantic/geometry failure precedes any write; the successful table retains the same source borrow and destination lifetime. Host prospective roots remain per dispatch name, while disk aliases retain the existing invocation/persistent trace rules and include complete persistent-owner unit capacities. Warm observed backing remains a snapshot envelope, not current residency or an existing-storage credit. Count/fill does not validate or activate the live receipt.

The component has portable destination/lifetime/layout tests, actual Metal host/disk source and ordinary-map parity cases, and a standalone allocation diagnostic for the actual neutral worker. These source tests are not execution evidence. Root-owned validation must compile/run the named inventory and standalone probe; Metal-only provider cases cannot be credited from an empty CPU filter. The standalone diagnostic places its allocator instrumentation outside all production crates; the neutral and backend unsafe-code prohibitions remain unchanged. Resident parameters, allocating snapshot construction, ordinary maps/context roots, selected module construction, native control fit and final public admission remain unfinished.

### Original chat J/H source and rendering

The next private chat slice constructs real owned source/render storage: three fresh J buffers from the executed 39-instruction compiler image and six fresh H buffers for fixed operands, loop frame/local, concatenation spans and both exact output variants. Input-dependent measurement runs the same selected dispatcher without heap allocation, checks finite loop/fuel/offset arithmetic, and produces source-bound requested capacities. Actual target-reserve failures retain preceding real buffers and genuine J/C/error controls under the original allowance. Partial/completed errors and all strong aliases preserve retirement ordering. Static recipe image residency remains explicit program data.

A retained config File enters exact original byte I before read; I remains charged through fresh J construction in the same pool. This does not adopt a prebuilt template/tokenizer. H transfers no credit when its selected prompt enters the existing fresh S/E→I/R sequence. The immutable canonical C domain, authenticated controller witness and terminal R text/ID ownership remain shared with plain input. By-value operation/startup errors preserve neutral causes without a new early allocating erasure. File opening and the full caller-owned chat request are outside this first private allocation boundary. Existing platform qualifications for retained file reads remain unchanged.

Whole-tokenizer support does not imply arbitrary chat-template/config support or whole-inference admission. The actual selected decoded template, name/settings, integer numeric profile and string-only request fields are checked. General numeric config and tool/history/media/extra-variable/reasoning/template profiles, owned request ingestion, and native global/graph/control/copy/capture/default activation remain required unfinished work. Source-derived layouts supply bounds; standalone allocator observations only check requested allocations and do not prove process heap/RSS bounds. Central validation must record the new 31 behavior cases, three compile-fail examples, complete-file oracle and default/all-feature allocator controls; no execution is claimed by this source package.

### Resident borrowed parameter source population (source-only)

The first resident companion counts named and auxiliary slots, exact retained metadata text bytes, observed ranks and genuinely unknown backing observations. An available zero-sized null-backed sentinel remains known; a zero-capacity physical allocation also remains known. `allocation().is_none()` alone increments unknown backing. The count retains the exact policy borrow and loaded transfer, but not an ongoing readiness certificate; future publication must revalidate changed native state. Aliases remain separate source slots and supply no deduplicated byte credit.

Native named traversal validates present fields, every retained topology key and duplicates before external callbacks. Source coverage always remains fallible. If both an observer failure and a traversal failure occur, the first observer descriptor/count cause wins; otherwise traversal's fixed cause is returned. This is deterministic precedence between independent channels, not a claim of global chronological error ordering. No polling, evaluation, new runtime owner or late budget grant occurs in the count body. Existing runtime-owner initialization/controls, ordinary setup, full module/frame populations and strict request admission remain separate obligations.

### Static, prediction and displaced owner-count component

A fixed borrowed companion now observes actual StaticModules, prediction inner/placeholder/replacement values and NativeParameterState originals/published values. It shares resident's descriptor counter and the ordinary retained-map worker, preserves observer-first fixed causes, and rejects unsupported source coverage rather than publishing partial counts. Actual prediction id and manager installation and pooling/model prototype occurrences remain distinct facts. The same source borrow survives in a counted view; copied scalars convey no authority.

No managed admission is activated by this component. Counts do not deduplicate storage, bound manager/source/prototype payloads, prove readiness or settle native work. Original source/request/control/whole-workspace and distributed ownership obligations remain required. Source-only tests and a positive-controlled standalone allocator probe are provided; compiled/behavioral results must be recorded separately after execution.

The borrowed operation companion shares ordinary workspace validation and equations, preserving unknown results, host/tensor separation, aliases and error ordering while permitting exact caller destinations without a rank cap. It supplies no grant and does not adopt an ordinary diagnostic lease. Source ownership, destination/control funding, complete graph and parameter construction, state/native backing and all actual execution producers must still compose under the existing single original comparison. Background-prefetch and missing native producer paths remain rejected. Repeated borrowed grouped-identity scans replace temporary collections with quadratic metadata validation; this is an explicit time/storage tradeoff, not a model-family limitation.


The shared decoder static-module construction seam now permits a concrete
borrowed declaration/count/fill policy to follow the actual ordinary construction
order. Fixed format declarations carry trainability, aliases, groups, semantic
roles, companion attachments and row layout as well as name bytes. Their closed
neutral validator shares encoding/cardinality/identity/role checks with the owning
format. These are constructor-input mechanisms, not a complete storage quote.
Caller row and UTF-8 destinations, their partial errors and fixed wrappers require
source-bound custody; an arbitrary construction callback does not become bounded
by implementing the public sink trait. Ordinary source Strings, generated names,
format lookups, native parameters and modules, graph/control/workspace populations
and whole-request producers remain unclosed until their real owners compose
under the existing admission authority. No background strategy, family or public
managed request gains an unsupported-as-complete or original activation claim.

The foreground source/controller integration compiles with the native backend and
portable runtime. Three selected telemetry cases pass: prepared/ordinary parity,
forward lifecycle validation, and maximum-peak accumulation. Its source backing
now uses one request-owned live window allowance, released only after actual
native backing retirement. Finite metadata for all permitted attempts remains
charged separately. The later selected public foreground disk fixture passes ordinary, managed and
controlled generation, cancellation, one-byte refusal and retained-output checks,
as recorded in the current integration status and
`validation/bounded-public-resident-chat-2026-09-16.json`. Background prefetch and
broader family/state combinations remain unfinished.

### Native graph construction lookup

A short active-generation profile of the measured 3,000-token managed Qwen run
found 504 of 973 sampled main-thread stacks in GraphConstruction::take.
The closed construction bank now indexes its ten existing recipe size classes.
Each lookup preserves smallest-fitting extent and first-slot tie order while
visiting at most ten classes; returned blocks remain consumed after physical
retirement. The index lives in the original Graph header and is included in
the existing native layout query before admission. No extra allocator, budget
credit, fallback or execution authority is introduced. The focused best-fit,
spent-slot, fragmented rollback and CPU completion cases pass. An older Metal
fixture fails stream admission before dispatch and remains unresolved; actual
prepared-stream public capture and released Qwen execution pass. The same managed
Qwen run fell from 65.796 to 4.705 seconds with identical output token IDs. Exact
commands, artifact identity and memory counters are recorded in
`validation/bounded-public-qwen-2026-09-16.json` under
`graph_construction_class_index`.
