# Ordinary generation front-door consolidation

The selected-executor optimized public native matrix passes all seven cases,
including ordinary tools and child sampling, combined image/audio tools,
single-device media/speculation,
and TP2, PP2 and combined TP2/PP2 image/tool execution.
The actual executor's group-submission mechanism closes the earlier TP2 quote
regression while retaining the resident registry. All six distributed CLI
Resident/Host/Disk cases also pass. Both native readiness fixtures pass, including
pending-child custody and exact final source-account retirement.
The producer checkpoints below preserve earlier failures and their subsequent
corrections. Later passing results supersede those failures only within their
recorded scope. Current native and application verdicts are maintained in
[native validation](bounded-followup-native.md) and
[the overview](bounded-followup.md).

The initial ordinary public `generate_prepared_chat` and
`generate_prepared_text` methods in `api/loaded.rs` entered an independent
startup worker: ordinary tokenizer encoding, decoder construction, semantic
controller/parser setup, `ControlledTextGeneration::from_input`, and committed
pipeline driving. `generate_observed_chat` / `generate_observed_text` fed that
same worker but constructed a separate owning observed-record family.

The canonical worker is `start_prepared_chat(PreparedChatRequest, cancellation)`
and its manual `advance` / uninterrupted `run`. Explicit token prefixes use
`PreparedChatPrompt::TokenIds`; authenticated media uses `Media`; output policy
selects semantic or eligible literal text. Sources, captures, interventions,
termination, sampling and retained errors stay attached to this worker.
Recorded operation uses the canonical controlled-session adapter and its shared
`ControlledGenerationRecord`, whose progress event contains the existing
`ObservedGenerationEvent` without another string-copying record projection.

The migrated direct ordinary calls were in backend conformance `timing.rs` /
`text.rs` and native `native_tool_checkpoints.rs`. Observed consumers include
capture/control conformance, native execution-control tests, distributed CLI
control tests, and component/observation examples. Their exact prefixes,
observations, numerical comparisons, cancellation and failure assertions must
survive migration. Shared mock source/capture mechanisms belong to the
conformance fixture owner. The controlled wrapper now uses the same canonical
session and committed delivery worker.

The ordinary methods, independent worker and observed request/record family
are now deleted. Recorded callers join the canonical adapter. The separate
speculative migration also removed the old allocating engine; canonical
`PreparedChatSpeculativeRequest` consumes the same source-owned chat and
`PreparedChatPrompt` input contracts.

Initial migration checkpoint: portable facade production compiled after deletion. The six migrated
observation/control/component examples pass native-feature Rust metadata checks
(`DOCS_RS=1`, no native execution). Direct ordinary timing/text/tool-checkpoint
consumers, native execution-control ordinary callers, and distributed CLI
control callers are migrated. Native execution-control metadata passes with
`mlx,metal,image,audio`; distributed CLI control metadata also passes. Logs:
`/private/tmp/tokenizer-native-controls-check5.log` and
`/private/tmp/tokenizer-cli-control-check.log`. Both use `DOCS_RS=1`; numerical
and multi-rank execution are separate evidence.
Neutral control fixtures now execute actual funded host-summary source claims.
The broader snapshot/branch conformance matrix is validated independently.

The focused core admission tests pass 2/2 after fixed-workspace sorting, covering
all eight transforms, exact admission identity, source-order errors and refusal
at every reached producer. Runtime source tests pass 2/2: cross-pool rejection,
shared source aliases, failed-admission custody, and final account retirement.
Logs: `/private/tmp/tokenizer-capture-admission-tests2.log` and
`/private/tmp/tokenizer-original-capture-source-tests.log`. These are portable
checks, not native numerical or distributed execution evidence.

The removed rendered observed path also exposed a cold producer gap. Capture geometry
was derived by ordinary encoding in `prepare_observed_chat`; `CapturePlan`
admission then allocated duplicate-detection trees, point copies, temporary
resolved shapes and diagnostics before the later paid source copy. The retained
source copy alone does not bound those producers. The replacement joins borrowed `capture` and `intervention` declarations to
`PreparedChatRequest`, after the same paid prompt encoder or retained media
layout supplies exact geometry. It shares the actual capture admission worker,
including its reached scratch/error producers. Existing
capture identity serialization and source-limit revision remain canonical.

The shared capture validator now prospectively reserves the source copy, selected
point copies, duplicate-key index, resolved axes/slices, identity destination and
diagnostic strings. Duplicate detection uses one sorted borrowed-key table with
original indices, preserving the first source-order error; iterative heap sort
uses fixed control storage. The score-ID limit remains 64. Ordinary admission
uses the same worker with explicit unenforced allocation policy. The core copy
worker is shared with immutable source copy, without an alternate declaration
serializer or validation algorithm.

`WorkingMemoryPool::compile_capture_declaration` pays the intermediate admission
under the actual retained metadata owner, then calls the existing fresh original
capture-source compiler. Failures retain that admission payer; shared source
aliases retain their independent original source account. Native bindings borrow
selected catalog/support and use the actual retained cached origin. Partition
bindings now construct partition discovery through the same prospectively funded
source worker, including fresh Unix artifact identity. They no longer require
ordinary lazy-discovery warmup; see [partition-source evidence](bounded-followup-partition-source.md).

Core machines now expose the actual immutable capture/intervention sources;
record context reads these after startup. Saved/resumed machines retain aliases
of the actual installed source, including revised child interventions. They do
not reconstruct identities from the caller's raw plan.

Constraint runtime-state consolidation and its focused evidence are recorded in
[Canonical constraint controller](bounded-followup-constraint-controller.md).

## Scope of the final source review

The September 18 source review checked the remaining facade generation,
observation, raw-token and media entry points. The removed allocating
prepared-chat/text and observed engines have no production replacement outside
the canonical session. Remaining managed plain-text and prepared-input APIs
require explicit sources and capacity, authenticate input, and use the shared
paid cursor and core generation machine. Ordinary `generate_tokens` remains a
lower-level numerical and token-iterator API over that same core machine.

Raw `prepare_multimodal_input` supports processor inspection and input-count
conformance. It returns a backend prompt, not an authenticated original input.
Its chat convenience method also only constructs a raw prompt. Canonical
`PreparedChatPrompt::Media` requires `OriginalModelInput` and validates its
source; native finite preparation rejects an ordinary raw prepared prompt.
These lower-level construction surfaces are not fallbacks selected after a
source-funded preparation failure. This review was bounded to those ownership
and call paths; it is not a claim that every public low-level operation has an
enforced allocation bound.

The CLI review confirmed that argument validation restricts `--stop` to
`--tools` and rejects tools with `--raw`; raw-mode stop strings are therefore
not an accepted option silently lost during migration. The CLI documentation
now states that restriction explicitly.

## Distributed completed-media readiness

The final caller review found a concrete consolidation gap: distributed native
startup accepted original token input for the paid readiness coordinator but
rejected `OriginalPrepared` before the existing completed-media admission could
run. The old public prepared-input path could reach distributed media execution;
the new rejection was not an architectural limitation.

Completed media now constructs the same paid readiness owner as original token
input. The existing original prepared-source admission still authenticates the
packet, parts, cache, pool, current media binding/frontier and sequence request.
It runs after readiness construction so any local authentication failure enters
the shared admission agreement before prompt or sampling construction. The enum
route itself grants no media or model-work authority and introduces no additional
allocation producer or ordinary fallback.

The neutral sequence-preparation suite passes 22 tests, including a new explicit
readiness refusal case that preserves the typed source, submits Admission/Failed,
and constructs no prompt, sampler or sequence storage. Native-feature metadata
for the public test passes with `mlx,metal,image` and `DOCS_RS=1`; logs are
`/private/tmp/tokenizer-distributed-media-sequence-tests.log` and
`/private/tmp/tokenizer-distributed-media-metadata2.log`.

The public `prepared_chat_native` test
`media::distributed::two_rank_image_tools_preserve_ordinary_manual_and_recorded_output`
reuses the existing image pixels, tool schema, deterministic model and uneven
prefill chunks. The same worker also supplies separate
`two_rank_pipeline_image_tools_preserve_ordinary_manual_and_recorded_output`
and `four_rank_tensor_pipeline_image_tools_preserve_ordinary_manual_and_recorded_output`
filters. Metal ranks use the Ring transport for TP2, PP2 and TP2/PP2, with the same
8 GiB capacity on each rank. Required and automatic tool policies compare exact
committed IDs, finish reason and semantic events across ordinary, manual and
recorded runs and between ranks. The existing single-device fixtures retain their
source geometry and capacity. The first actual two-rank run passed readiness
and reached equation quotation, then both ranks refused `Retained(Layout)`:
the projected TP-local decoder state was being compared with the serial global
selection. The log is
`/private/tmp/eredu-native-media-readiness-distributed-media.log`.

The correction retains the completed partition constructor and its local state,
executor and communication source throughout the existing media interval loop.
The Qwen-VL constructor now publishes its exact shared configuration, local
geometry and completed parameter/graph declarations for paid cold construction.
The initial and saved media adapters select the same parallel workspace and
native recipe recorder used by token input; they do not substitute a resident
recorder for partitioned equations. Encoder-cut retention and prepared observation
paths pass through the existing partition worker. Source constructor tests cover
exact aliases, foreign owners, reached refusals and payer retirement.

The first corrected actual TP2, PP2 and TP2/PP2 runs passed local-state validation
and reached first-prefill quotation, then refused a missing paid Qwen-VL collective
wave producer (`WorkspaceMetadata(Unqualified)`). All three single-device cases
continued to pass. The follow-up uses the same Qwen wave and boundary algorithms
with prospective destinations for stage vectors, shapes, roles, schemas and
receive-side containers; it also retains the exact workspace metadata account
in the prepared ingress plan. Focused tests compare eager/retained and
ordinary/funded waves, segmented lookup intervals and all three PP boundary
forms, exact boundary tensor aliases, and every reached wave/schema/role allocation.
Ordinary and funded vision/decoder schemas consume one immutable family
specification. The declaration's exact shape arrays and fields are funded before
construction, and retained-wave diagnostics use the same paid destination.
The final fixed-control census rerun passed both focused tests in
`/private/tmp/tokenizer-media-paid-controls-tests4.log` (41.40 seconds to build,
0.01 seconds to run). Shared partition dispatch now skips only
activation observation when the observer explicitly reports none, preserving
semantic media-cut and retention callbacks.

DeepSeek V3/V4 routed blocks likewise treat disabled component instrumentation
as an ordinary invocation. Their serial, resident-TP and provider-TP entries use
the same block sequence; an absent activation observer is not an exceptional
state. The ordinary TP routed entry joins the shared routed worker, retaining
its exact coordinate loan and fused reduction while omitting activation paths.
The independent workspace test mechanism explicitly publishes its F32 outputs
and leaves alias strides unrestricted, allowing uneven vocabulary padding to
consume real scalar evidence without changing native qualification.

The affected workspace-estimation suite passes **13/13** (37.03-second build,
0.50-second run), including both prior failures and strict unknown-primitive
refusals: `/private/tmp/tokenizer-deepseek-workspace-estimation3.log`. Two new
neutral numerical tests compare ordinary, disabled and active observation through
all three entries over nonzero prefill and cached decode; outputs and cache
positions match and disabled observation emits no activation callbacks
(`/private/tmp/tokenizer-deepseek-observation-numeric.log`, 48.96-second build,
0.05-second run). Existing V3 TP2, V4 TP2 and V4 intervention tests also pass,
recorded respectively in `tokenizer-deepseek-v3-tp2-parity.log`,
`tokenizer-deepseek-v4-tp2-parity.log` and
`tokenizer-deepseek-observation-intervention.log` under `/private/tmp`.

These checks use the portable architecture crate, without a native backend:

```sh
env RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
  CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu-architectures --test workspace_estimation
env RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
  CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu-architectures --test reference_numeric deepseek_observation
```

The next native diagnostic build passed those paid declarations. TP2 and TP2/PP2
then reached the strict collective source validator and refused missing physical
scalar evidence (`WorkspaceRepresentation`); PP2 independently exposed an encoder
continuation that rebuilt rotary tables instead of using its retained projection.
The TP trace starts at vision LayerNorm: its scalar representation was absent
from the native fact emitter and therefore remained unknown through projection.
The correction follows the pinned MLX 0.32.0 `fast::layer_norm` dtype selection,
including promotion across both affine parameters. The neutral workspace event
now retains the exact optional weight and bias roles already known by the tensor
worker. The native producer can therefore distinguish mixed-precision weight-only
promotion from the bias-only cast to the input type. The old role-losing string
declaration is removed. Unknown required sources stay unknown, and the collective
validator is unchanged. The actual neutral tensor trace preserves all four
optional-affine forms, and the owning/borrowed operation-view test passes all
60 existing and added declarations. Backend library and test metadata pass with
the typed event and its complete consumers in
`/private/tmp/tokenizer-layernorm-representation-check2.log`. The subsequent
native TP2 run still refused missing scalar evidence. Following the actual
vision block identified a second loss: generated/prepared multi-axis rotary
tables lacked the F32 fact established by the selected native worker's explicit
position cast, F32 frequencies, cosine and sine. The fact now reuses the same
geometry validator as the existing allocation bound. A cold regression invokes
the real vision block with generated and prepared tables, three activations,
segmented attention and all four projections; an unknown table still prevents
qualification. The existing native rotary parity test also checks actual dtype.

The same source-chain review found that feature insertion's masked row-scatter
lost its physical dtype. Its selected worker casts source rows to the destination
dtype before broadcasting, so the fact producer now retains the known destination
type after the existing geometry check. Mixed source precision cannot promote
it, and an unknown destination remains unqualified. Cold refusal coverage and a
nine-combination native CPU numerical test exercise that distinction. These
scalar facts do not change numerical workers, storage bounds, capacity or source
authority. Collective refusal now retains fixed operation, group, rank and input
geometry diagnostics under the existing prospective error/source funding. Actual
parallel reruns were still required at this metadata checkpoint. The later
TP2, PP2 and combined TP2/PP2 public cases pass, as recorded in
[native validation](bounded-followup-native.md#current-validation-scope);
metadata acceptance alone did not establish that result.
The combined backend library and all test metadata check passed in 36.85 seconds
with Metal and image enabled, including these scalar producers, the shared
text/media parallel-context loan and CPU InputProducts support:
`/private/tmp/tokenizer-rotary-representation-check2.log`.

The final neutral boundary checks also pass independently of the native adapter:
`cargo check --offline -p eredu-nn --all-features` (9.92 seconds) and
`cargo check --offline -p eredu-architectures` (22.05 seconds). Both use
`RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc`,
`CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=/private/tmp/eredu-prefilter-reference/target` from the repository
root. Logs are `/private/tmp/tokenizer-final-neutral-nn-all-features.log` and
`/private/tmp/tokenizer-final-neutral-architectures.log`; no native feature or
`DOCS_RS` override is selected for these checks.

The next native checkpoint passed PP2 image/tool generation (both tool policies
and all three advancement modes), plus the three existing single-device cases.
TP2 and TP2/PP2 passed the scalar/collective checks but exposed a distinct decoder
startup omission: their prospective trace emitted generated `MultiAxisRotary`
without the source-paid frequency workspace. The failing decoder axes were
`[2, 2, 4]`; the exact log is
`/private/tmp/eredu-native-media-transaction-distributed-media.log`.

Qwen-VL and conditional Qwen hybrid now route ordinary and tensor-parallel startup
through the same family worker, carrying the admitted input's metadata destination.
The parallel choice selects the existing vocabulary lookup and exact local state
layout. Position vectors, decoder rotary frequencies, part vectors and retained
forward context use the same paid producers. Gemma's prepared TP adapter likewise
passes the retained destination into its existing common worker and error-custody
scope. The old TP-only Qwen startup bodies are removed; their local-geometry checks
remain in the shared workers. Arbitrary generated rotary operations are still
unqualified in the bounded native mechanism. Completed media continues to use its
separate, authenticated original/projected encoder-table source through the shared
media ingress worker. Architecture metadata compilation passes in
`/private/tmp/tokenizer-shared-media-startup-check.log`. Eleven focused Qwen-VL
architecture tests pass in 0.04 seconds, including new serial/TP0/TP1 startup
checks with an independently funded request, exact geometry, prepared rotary,
TP collectives, forward-account custody and refusal before tensor work. Source
identity/constructor and paid boundary/wave refusal tests pass in the same run:
`/private/tmp/tokenizer-shared-media-startup-tests2.log`. Native TP execution was
still pending at this checkpoint; the later public matrix now passes.

The native CPU YaRN recipe test separately exposed an exact completion-record
underestimate for shape `[2, 3, 3, 8]`: its six flattened outputs feed one
concatenation, which publishes one output Data owner and two weak-copy Data owners
per input. The selected native record clears these captures after each primitive,
so the required peak is `1 + 2 * 6 = 13`, rather than the record's minimum of eight.
The shared CPU concatenation producer now reports this checked peak; ordinary and
addressable recipe composition preserve its maximum across sequential primitives.
The existing numerical worker, buffer populations and capacity remain unchanged.
Cold regressions cover two-, six- and eight-input joins, including repeated
composition without summing non-overlapping peaks. Combined backend library and
test metadata passes in 34.67 seconds:
`/private/tmp/tokenizer-cpu-rotary-captures-check2.log`. The actual native YaRN
test now passes all three shapes, including exact ordinary/scalar parity and
final-owner retirement, in 0.23 seconds:
`/private/tmp/eredu-native-cold-emitters-cpu-yarn-native.log`.

The same binary identifies the separate InputProducts `Domain` refusal at the
input leaf check, before any retained-frequency check:
`/private/tmp/eredu-native-cold-emitters-cpu-input-products-native.log`. The fixture
had excluded transpose from its quoted span while treating the resulting view
as an independently detached input. The corrected fixture quotes and constructs
that exact alias inside the original graph recipe and strictly validates the
actual dense source before construction. It retains both frequency checks and
checks completed backing identity, exact permuted strides, noncontiguity,
ordinary/scalar results, positions above 2^24 and final-owner retirement. No leaf
guard or capacity changes. The fixture also publishes settlement through its
retained observer before dropping it: CPU output readiness can precede the
signal task's accepted-frontier update, and record retirement does not poll
unfinished work. Settlement and final retirement share the original ten-second
deadline. The combined fixture metadata checkpoint passes in 41.80 seconds
(`/private/tmp/eredu-native-observer-settlement-metadata2.log`). At that checkpoint,
the corrected native fixture and new cold alias-population regression still
required execution. Ordinary readiness alone is not detached-view provenance.

The next actual InputProducts run completed execution and numerical comparison,
then found that its separately inspected singleton transpose had been optimized
out of the result dependency and remained unevaluated
(`/private/tmp/eredu-native-source-settlement-cpu-input-products-native.log`).
The fixture now quotes and completes both result and view as two explicit roots
through the existing `inspect_cpu_outputs` recipe and one completion. Its root
capacity and traversal counts include that second witness; it keeps the strict
known-backing and exact-stride assertions. Subsequent native execution passes
exact ordinary/paid parity and custody with the source-derived SIMD numerical
oracle described in [native validation](bounded-followup-native.md).

## Final portable public checkpoint

After ordered pre-tokenizer source compilation and feature-selected regex
convergence, the September 18 portable checkpoint passes **126
backend-conformance tests**, **38 portable-facade tests** and **372 facade unit
tests**. Two external-checkpoint cases and six unit cases are ignored. The only
filtered unit is the unchanged exhaustive schema refusal sweep, which passed in
the preceding full run. The final run includes retained-configuration ordered
pre-tokenizer coverage and the corrected negative profile fixture: it refuses an
unlisted `Split` regex while preserving the zero-backend-work and payer-lifetime
assertions, rather than rejecting newly supported ByteLevel behavior.

Rust 1.98.0 command:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check \
  cargo test --offline -p eredu --no-default-features \
  --lib --test backend_conformance --test portable_facade -- \
  --skip runtime::chat::tool_schema::registered::tests::facade_schema_row_reservations_refuse_before_growth_and_preserve_the_cause
```

The final log is `/private/tmp/tokenizer-final-portable-onig-coherent.log`:
61 seconds to build, then 14.48 seconds for units, 3.14 seconds for backend
conformance and 15.09 seconds for portable facade. The earlier full schema sweep
is in `/private/tmp/tokenizer-final-portable-public-and-lib.log`. These results
cover the portable no-default-feature profile; default Onig and native execution
have separate evidence. A subsequent private-recipe visibility correction leaves
the nine admitted expressions unchanged; its focused producer verification is
tracked separately and is not evidence of another full public rebuild.

## Canonical neural error ownership

The eager-format `eredu_nn::Error::backend_source` constructor and its `Legacy`
storage are removed. All typed causes now use the existing retained-source worker:
construction and cloning do not format or copy the original diagnostic, aliases
share one source, and final retirement removes the shared control before the
concrete source and its payer. Caller-owned diagnostic strings remain distinct;
the message constructor consumes their buffer. Cold observation binding,
layerwise failure propagation and sampling quotation now use their existing
context-funded source/error workers where the audit found direct erasure.

`retained_source_construction_bytes` is the single NN constructor quote. It prices
the actual source Box, shared shell and fixed constructor transports; source
payload storage and caller-specific controls remain separate. The allocation-only
query is removed, and prepaid producers consume the same constructor quote. The
native observation adapter no longer needs `RetainedActivationFailure` or its
manually described eager-format Arc layout; it retains the actual native cause.
Its existing generated-factory regression verifies the direct typed source and
shared alias without invoking the sentinel's deliberately panicking formatter.

The final neutral NN run passes **213 unit tests and 5 doctests**, including exact
one-byte-short refusal before source-owner construction, no further funding call,
shared original-source identity, concurrent/unwind retirement and payer custody
through the last escaping alias. Runtime sampling and sampling-copy filters pass
**61 tests**. Commands use Rust 1.98.0 and the isolated host target:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu-nn --all-features
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu-runtime --lib working_memory::sampling
```

Logs are `/private/tmp/tokenizer-canonical-nn-errors-final-tests2.log` and
`/private/tmp/tokenizer-canonical-neural-sampling-tests.log`. Native adapter
metadata and public facade validation are separate checkpoints.

Against the same canonical error/query changes, both public no-default-feature
suites pass: **126 backend-conformance tests** in 3.26 seconds and **38
portable-facade tests** in 14.87 seconds, with two external-checkpoint cases
ignored. Reproduce with the same target and
`cargo test --offline -p eredu --no-default-features --test backend_conformance
--test portable_facade`; the log is
`/private/tmp/tokenizer-canonical-neural-public-tests.log`. This is host public
behavior evidence; it does not substitute for the separately scheduled native
observer and distributed execution reruns.

## Canonical layered metadata and observation binding

Layered unit counts, unit/group paths, ingress validation, and row declarations
now take the actual optional metadata destination through their existing
contracts. The allocating `_with_metadata` compatibility defaults and paired
public forwarding methods are removed. Ordinary loading supplies `None`; paid
construction supplies its retained workspace context to the same concrete
producer. `LayeredMetadata` preserves the architecture's typed error conversion
when a generic runtime carries that destination.

Cold observation rebinding uses that destination for every regenerated row,
unit/group path and ordered routing declaration. It compares effective-path
suffixes by borrowing the existing strings. A reached construction refusal
returns through the closed observation/session error wrappers without another
funding callback. The retained declaration source stays physically identical;
rebinding supplies a new runtime identity, never model execution or source
admission authority.

`RoutedObservationPoints` uses the existing ordered map worker. New paths,
ordered nodes, lookup/insertion controls and the immutable shared shell are
funded before production. Cloning shares the exact owner and payer. Adding a
bank to a shared owner copies its actual paths/nodes through the same producer,
leaving earlier aliases unchanged. Both strong runtime bindings and escaping
weak fingerprints retain their original payer until their physical identity
control retires.

The current focused runtime evidence passes **10 observation integration tests**
and **45 observation source/capture unit tests**, including exact source
rebinding, semantic mutation rejection, every reached metadata refusal, no
subsequent callback or model work, and payer custody through the last weak
fingerprint. The full runtime integration binary passes **206 tests** in 6.04
seconds. These runs precede the final movement of the routing-map control charge
before its duplicate-bank lookup; that correction changes no declaration or
runtime binding semantics. Logs:

- `/private/tmp/tokenizer-canonical-observation-binding-tests.log`
- `/private/tmp/tokenizer-canonical-observation-source-tests.log`
- `/private/tmp/tokenizer-canonical-observation-runtime-all.log`

The final source, including lookup controls before duplicate-bank inspection,
passes **768 architecture unit tests** with one ignored. The three focused path,
media-error and row/routing tests all pass; the row test compares ordinary and
paid declarations, checks immutable alias/copy-on-add independence, rejects a
foreign payer, and sweeps every reached refusal. The two closed-error tests now
also cover a partition session's original architecture failure without another
callback. The full runtime integration rerun passes **206 tests**.

Final-source public suites pass **126 backend-conformance tests** in 3.37 seconds
and **38 portable-facade tests** in 15.40 seconds, with two external-checkpoint
cases ignored. Their build took 55.61 seconds. Reproducible Rust 1.98.0 commands:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-grammar-check \
  cargo test --offline -p eredu-architectures --lib
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu-runtime --test backend_independence
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/private/tmp/eredu-core-driver-check \
  cargo test --offline -p eredu --no-default-features \
  --test backend_conformance --test portable_facade
```

The architecture test binary was built by the focused `module_metadata::path_tests`
command (59.72 seconds), then run without a filter (4.74 seconds). Logs are
`/private/tmp/tokenizer-canonical-observation-architecture-tests.log`,
`/private/tmp/tokenizer-canonical-observation-architecture-all.log`,
`/private/tmp/tokenizer-canonical-observation-runtime-final.log`, and
`/private/tmp/tokenizer-canonical-observation-public-tests.log`.

Native equation and distributed execution evidence remains in the maintained
public validation records; neutral metadata tests do not substitute for those
runs.
