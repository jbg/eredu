# Prospective intervention plans

`LoadedModel::intervention_discovery()`
returns versioned mutable targets for the actual loaded session. These are separate
from architecture nodes and read-only observations. Each target joins an existing
node identity, declares an execution stage, semantic axes, exact supported dtypes,
operations, and prefill/decode support. Unverified and unsupported phases cannot
be admitted. Conditional phases may be selected while retaining their stated
input or execution condition. Every scheduled intervention must execute before
prediction commitment; an absent conditional target retains a `Missing` outcome
and fails the attempt through the ordinary state-recovery driver.

Supply borrowed `capture` and `intervention` declarations on the canonical
`PreparedChatRequest`, or retain admitted declarations in its `options`.
`start_prepared_chat` and `start_controlled_chat` compile raw declarations after
exact prompt geometry is known, using the same preparation and execution driver.
An empty intervention plan retains its source identity and lineage while reporting
that no intervention operations are active. Preparation validates retained source
and session identities before native model work.

The plan digest binds operation order, target declarations, request geometry,
prepared-source content, backend session identity and facade session identity.
Plans contain host data only. They cannot contain devices, streams, native tensors,
application callbacks or executable code. An admitted plan cannot be deserialized.

## Schedules and exact activation semantics

Prediction zero is prefill. Later predictions are decode. `CaptureSchedule`
selects prediction indices independently from the operation's semantic axis
slices. Selecting prefill prediction zero does not select one prompt row: use an
explicit `sequence` slice for that. Slices retain their axes and use nonempty
half-open ranges and positive strides. Unmentioned axes remain whole.

Activation operations run after the ordinary observation at that exact hook and
before its value is consumed downstream. Operations on one value compose in list
order. Before/after evidence for a later operation therefore includes earlier
operations' effects. No second unmodified forward pass is used as a baseline.

| Action | Exact effect |
| --- | --- |
| `Zero` | Set every selected element to zero |
| `Scale` | Multiply every selected element by a finite scalar in native dtype |
| `Mask` | Keep true elements; replace false elements with zero |
| `Replace` | Replace every selected element with the complete host tensor |
| `Add` | Add the complete host tensor elementwise; includes vocabulary logit bias |
| `MaskLogits` | Set the specified vocabulary IDs to negative infinity over the selected token rows |

Replacement, addition and explicit mask shapes must exactly equal all selected
extents, including singleton axes. There is no payload broadcasting. A preview or
summary cannot stand in for a complete replacement. Float32 payloads use finite
numbers; float16 and bfloat16 payloads use finite exact 16-bit representations.
The plan dtype must equal the actual value's dtype; mismatches fail before a
replacement is constructed. A backend can support several dtypes while the actual
point dtype remains request-dependent. Use `CaptureRecord.source_dtype` from the
relevant parameter version to distinguish actual precision from host export
precision; BF16 source values can be exported as F32 numbers. A packed-to-F32
parameter overlay can promote component precision, so refresh that evidence before
admitting a new exact-dtype action. Unknown precision is not an F32 assumption.
Scalar scaling uses native dtype rounding;
its input must be finite and representable. Arithmetic otherwise follows the
native dtype's ordinary floating-point rules.

Logit masks use unique canonical vocabulary IDs, retain the full vocabulary axis,
and cannot mask the entire vocabulary. They precede the existing sampler's token
constraints, penalties, temperature, top-k/p and other processing. Replacement
logits are returned to that same sampler. Interventions introduce no sampler or
random draws. Independent sampler constraints can still reject a distribution
with no eligible token.

## Sparse expert-unit edits

Loaded targets with `routed_units: Some(...)` use a virtual `[token, component]`
shape. The component index is `expert * units_per_expert + unit`, checked by
`RoutedComponentGroup::component_index`. This is a stable global expert/unit
identity, independent of the current route slots. `MaskComponents` accepts compact
arbitrary deletion or keep-only sets across this axis. Token slices select explicit
flattened positions; separate schedules select prefill and decode predictions.
Use the shorter set or its complement for large masks. An empty deletion set is
an all-keep no-op and avoids enumerating every expert unit.

Zero, Scale, Mask, Replace and Add use the same native-dtype semantics as ordinary
activation edits. Region payloads have the exact selected virtual shape, and only
entries for participating experts are gathered. An absent expert does not acquire
an activation. Duplicate expert slots each receive the edit. Keep-only survivors
are recomputed in the current forward, so earlier layer edits can change their
values and routing. All operations run before expert down projection and route
weighting, with optional projection input quantization occurring afterward.

The runtime reserves the whole invocation before reading route metadata or
gathering values. It charges participating storage and copies, not a fictional
expert-dense tensor. Operations compose in plan order within each native chunk.
`InterventionRecord.routed_units` reports full and completed token counts and the
number of actual values addressed; compact masks count removed values.
`Unmatched` means the invocation completed without an addressed value, including
all-keep masks. `Missing` and `Failed` never establish successful chunk completion.

Use `InterventionEvidence::None` together with bounded original and effective
`CaptureTransform::RoutedUnits` selections for attributed evidence. Ordinary dense
Preview/Summary operation evidence cannot describe sparse expert participation and
is rejected at these points. Captures use `[token, route, per_expert_component]`,
which deliberately differs from the edit plan's virtual global component axis.
The original capture is before all edits at the boundary; the effective capture
is after all edits. Per-operation sparse evidence is not currently exposed.

Ordinary and controlled execution share this driver, including re-admission,
replay, snapshots, forks and cumulative accounting. Loaded partitioned execution
uses the same admitted operation plan when discovery verifies its sparse provider
hooks, retained ownership and exact native invocation-group transport.

Low-level distributed owners can use `RoutedComponentCoordinateMap` and
`lower_partition_routed_intervention` to lower current participating rows across
unit shards and source peers. The recipe preserves selected global token/component
payload coordinates, including exact half-precision bits, and returns local flat
indices only. A successful recipe does not prove complete route coverage or grant
budget, peer, transport or commit authority. `SessionPartitionIntervention` supplies
that authority through the existing partition session. Preparation reserves native
work, coordinate copies, source/output completion, two invocation votes and global
outcome delivery before model work. It validates complete expert/unit ownership;
replicas execute every edit while logical affected counts are reported once.
Every received source peer is edited before reverse expert exchange, including
peers whose rows are not exported in the public capture.

The shared observer applies operations in plan order to the current native chunk.
Source geometry is agreed before work, and all members must finish their chunks
before the final vote permits reverse exchange. Idle owners explicitly complete
the invocation. Global `Applied` or `Unmatched` outcomes publish only when the
ordinary distributed forward commits. Malformed chunks, missing invocations and
local failures cannot produce successful receipts. Operations also work without
sparse captures; separate original/effective captures remain the evidence API.
Selected pipeline-stage groups include every tensor/expert replica. Their connected
Ring status reduction uses bounded native completion, including intermediate sends.
Disconnected native groups still need coordinated relay support.

## Routing stages and ownership

Routing control is applied before the expert provider receives its request.
The architecture's ordinary selector supplies its score transform, group selection,
normalization epsilon, coefficient scale and any learned coefficient multipliers.
The backend realizes these mechanisms without selecting a model-family policy.

| Action | Exact effect |
| --- | --- |
| `ExcludeExperts` | Mask IDs out of ranking eligibility, then run the architecture's ordinary selection algorithm; gathered scores retain ordinary coefficient semantics |
| `ZeroExpertContribution` | Preserve selected IDs and set their specified coefficients to zero; other coefficients retain their magnitudes; no renormalization or avoided computation is promised |
| `BiasRoutingScores` | Add per-expert biases at the explicitly selected score stage, over the selected token rows |
| `ForceExperts` | Replace IDs with a complete `[selected_token_rows, top_k]` payload and gather their scores and coefficients using the ordinary normalization/scaling policy |

Raw-logit bias precedes the score transform. Transformed-score bias precedes
selection corrections and affects gathered scores. Ranking-score bias follows
learned ranking corrections and changes only selection; gathered scores remain
unbiased. These stages are distinct capabilities, not interchangeable controls.
Bias parameters are finite F32 host values, explicitly rounded to the selected
score stage's native dtype. Unrepresentable biases, non-finite effective scores,
and invalid selected coefficients fail before expert dispatch. The bias is
replicated only across the explicitly selected token rows.

Routed IDs use the global `0..expert_count` namespace. Shared experts have separate
ownership and remain unchanged. Forced IDs must be distinct within each token row,
match top-k exactly, and satisfy the declared group count. Exclusion conservatively
requires enough eligible experts in every combination of selectable groups so an
unchanged group selector cannot dispatch a masked ID. Effective coefficients must
be finite and nonnegative; zero-contribution mode permits a zero total. There is no
caller coefficient override or implicit renormalization option.

Only token-row slicing is supported for routing. Multiple routing operations at
the same target are rejected when their phase/range envelopes overlap, even if
frequencies might avoid coincidence. Use disjoint schedules for separate routing
experiments. This avoids ambiguous interactions between forcing and exclusion.

The initial routing declarations cover ordinary Qwen3-MoE and text-only Qwen hybrid
(Qwen3-Next/Qwen3.5) execution. Other families do not acquire routing support just
because they emit routing observations. Partitioned, speculative, media and
realtime intervention combinations are outside this protocol. Activation coverage
follows genuine declared activation hooks and their loaded-session support.
Controlled capture validates the architecture-admitted decoder batch and prompt
length before model execution. For a prepared composite prompt, this is the
assembled sequence after media projection and merging, including its placeholder
positions. The ordinary architecture and processor admission still validates
multipart and embedded-tensor inputs. A mismatched capture request fails through
the same prepared-input result path without advancing model state.

## Evidence, budgets and state

`InterventionEvidence::None` requests outcome metadata only. `Preview` requests
bounded values per side (and per route field); `Summary` requests activation
statistics. Route previews include original and effective expert IDs and
coefficients. They are taken from one router evaluation, not a second model pass.
The ordinary capture catalog continues to describe activations before intervention
and effective route observations after dispatch.
Raw candidate payloads now include `source: "original"` or `"effective"` in
addition to `stage: "raw_logits_before_sampling"`. Ordinary candidate captures
are original at the logits hook; interventions at earlier hooks may already have
affected them. Candidate scores are never processed sampling probabilities.

Every `CapturedStep` may include intervention records carrying plan and operation
identity, target/node, phase, prediction index, application outcome and optional
evidence. `Inactive`, `Applied`, `Missing` and `Failed` are different outcomes.
A scheduled missing target fails before prediction commitment. A failed operation
does not erase earlier successful records. Replacement payloads are not repeated
in records.

Capture and intervention share one per-step/cumulative ledger. Diagnostic envelopes
are reserved before forward work; evidence transformations are reserved before
copying. Original route-decision temporaries are reserved before the controlled
selector runs, using the same backend estimator used during cold admission.
The estimate covers additional original-decision storage and work, including
retention of original scores when raw-logit bias replaces them, and native
selection/weight temporaries and any host materialization. Ordinary projection
and inference allocations remain separate. Evidence transforms are charged once
through the existing capture estimator, and diagnostic encoding through the runtime.
MLX includes the full selected-group dimension of its group masks and the host
partition-ID lookup. These are conservative logical allowances, without a physical
allocator or private-workspace guarantee. Missing or unsupported estimates and
arithmetic overflow fail explicitly. An operation with no original evidence does
not request this estimate or construct an original decision; raw-logit bias also
skips the unused ordinary score transform in that case.

`TraceLimits` additionally bounds the complete serialized facade
stream. Even a plan with no ordinary captures needs capture budgets for intervention
metadata and requested evidence. The hard plan limits are 1024 operations, 32 MiB of
payload storage, and 64 MiB of compact JSON. This admits complete component-index
sets for 128 layers with 32768 FFN units and 8192 attention channels per layer;
execution and transport reservations remain independently required. Preview evidence has at most 4096 elements
per side/field. These are logical accounting limits, not a physical allocator cap.

Plans apply prospectively. They never retroactively recompute cached KV or recurrent
states. For an independent experiment: complete or cancel the baseline, call
`reset`, prepare a new request, then generate. Hot plan replacement and resumable
snapshots are not supported. A failed forward can already have changed state even
when the particular failing intervention did not apply. The existing rollback,
reset, poisoning and retained-native-completion rules govern reuse. Cancellation,
consumer `Break`, early drop and unwinding keep the ordinary completion owner.

Routed provider failure agreement runs independently of capture and intervention
selection. Disabling masks or unit observations therefore preserves the same
execution votes, including participation by idle experts and inactive pipeline
stages. An agreed provider rejection prevents later model collectives; it does not
establish native completion or successful intervention evidence. Applied outcomes
still require the shared forward commit.

LM Inspector should display controls from intervention discovery, join targets to
node identities, and expose only supported operations/stages/phases. Keep schedule
controls separate from within-tensor row/axis controls. Show dtype and execution
conditions. Preserve missing/inactive/failure outcomes and before/after positions;
never turn them into measured zero. Use the single facade generation workflow and
reset between independent experiments.

## Implementing another backend

The operation recipes run in portable code. A backend implements native mechanisms
and integrates the shared run with its existing submission lifecycle:

1. Implement `eredu-core::intervention::InterventionBackend` alongside
   `CaptureBackend`: exact dtype/geometry queries, selected-region reads and
   updates, zeros, scalar multiplication, explicit mask/column fills, typed host
   payload realization, and elementwise addition. The runtime dispatches the
   activation action and validates native result shapes/dtypes. It never copies
   source activations to the host to execute a patch.
2. Implement `eredu-nn::routing_intervention::RoutingMechanism` for the ordinary
   configured selector. Supply projection, score transform, ranking correction,
   selection, coefficient calculation, indexing/arithmetic and native validity
   predicates. Call `execute_routing_intervention` from the selector's intervened
   path. The driver owns all four actions and all score-stage ordering. It consumes
   the architecture's policy unchanged, including groups, normalization, epsilon,
   scaling and learned multipliers. Shared experts remain outside routed control.
3. Supply a side-effect-free `InterventionEstimator`. `validate_geometry` reports
   native indexing limits, `capture_usage` matches the ordinary evidence estimator,
   and `original_route_usage` estimates only additional original-decision work.
   Its capture count and encoded bytes must be zero: the runtime accounts for those
   separately. Known scheduled shapes are checked during preflight, including
   operations with no evidence; runtime-dependent shapes are checked at the hook.
   Retain the same estimator facts for exact runtime reservation before selection.
   An unavailable estimate returns an error, never a fabricated zero cost.
4. Check native execution-mode support, then use `capture::validate_session` and
   `intervention::validate_session` against retained loaded-session discovery.
   These revalidate admission identity, resolve scheduled geometry and check budgets.
   Use `intervention::install_session` to install one immutable run and
   `CaptureObserver` to forward activation, routing, evidence, failure and finish
   hooks. Supply only native primitives and typed error conversion to the observer.
   Keep opaque prompt inspection, native completion retention, cancellation and
   recovery in the backend's existing owner.

Ordinary MLX capture and intervention observers carry structured backend errors
through native and neural adapters. Portable admission errors remain typed causes
of the facade's neutral `BackendFailure`; native exceptions retain their original
message and creation location. Bounded failure records still contain bounded text,
and consumed reservations are not refunded by restore. The error source does not
replace the existing completion and state-recovery checks. See
[the observation error evidence](component-validation.md#observation-error-causes).

A new model family declares its genuine hooks and ordinary selector policy in
`eredu-architectures`; these drivers do not require a new action implementation for
that family. Discovery still needs family-specific validation and truthful support
declarations. This refactor does not expand the established model/session coverage
listed above, and it does not add a production backend.

The lower-level Rust interfaces changed: the coarse
`InterventionBackend::apply_intervention` method was replaced by native primitives;
intervention preflight and `CaptureSession::enable_interventions` now take backend
estimates. Backend implementers must update those call sites. `RoutingMechanism`
and the shared session/observer helpers provide the replacement execution path.
Serialized plans/records and the application facade workflow are unchanged.

The intervention schema is version 1. Capture/trace records gain optional additive
intervention fields, defaulting to absent/empty when older JSON is read and omitted
on ordinary runs. Existing observation paths, capture positions and generation
entry points retain their meanings.
The candidate `source` field is additive and defaults to `original` when reading
older records. Rust applications constructing candidate structs must provide it.

Behavioral verification includes portable admission, ordered mutation, budgets,
session binding, consumer cancellation/unwinding and failure records; loaded MLX
dense and Qwen routing fixtures; and an expert-provider probe with distinguishable
contributions. Native routing checks cover prefill row selection, decode schedules,
all three bias stages, exclusion/reselection, zero contribution, forced IDs,
group feasibility, normalization epsilon, routing scale and learned multipliers.
The native suite runs on CPU. CUDA, multi-rank, speculative, realtime and media
interventions are not established by these tests. Other model families retain
only the activation points advertised by their loaded-session discovery; their
routing observations alone do not authorize routing control.

`eredu-evaluation::intervention` supplies reusable activation, routing and grouped
routing conformance checks. Both MLX and a host fixture execute those checks through
the same portable drivers. Runtime tests vary indexing constraints and original
resource estimates, check admission/runtime agreement, and reject missing estimates,
overflow and insufficient budgets before returning an executable routing control.
The facade mock also uses shared admission, installation and observer forwarding;
loaded native fixtures continue to check actual expert-provider inputs, shared
contributions, downstream activation consumption, RNG progression and recovery.

The complete facade example discovers support, emits a baseline, resets the
session, zeros the final prefill logits row, and emits bounded before/after evidence:

```sh
cargo run -p eredu --no-default-features --features mlx --example intervened_generate -- /path/to/artifact "Explain gravity" f32
```

Pass the actual activation dtype (`f32`, `f16`, or `bf16`). Discovery lists native
mechanism dtypes; runtime-dependent exact dtype constraints remain checked at the
hook. The example intentionally changes only prediction zero and its final prompt
row. It uses the same ordinary sampling settings and seed for both experiments.


Component boundaries additionally support `MaskComponents`: a compact list of
unique indices and a keep-only/deletion flag over the complete final `component`
axis. Explicit token-row slices are independent of prediction schedules. Surviving
values are recomputed in the current forward. Activation-edit storage and host
uploads are now reserved via `InterventionEstimator::activation_usage` before
native work, in addition to evidence and diagnostic charges. See
[component analysis](component-analysis.md) for the API and current coverage.

For partition drivers, `intervention::localize_component_mask` translates this
action through an architecture-derived `ComponentCoordinateMap`. It validates
global identities before filtering local indices and preserves the explicit token
region. Empty local keep sets zero every local component; empty delete sets leave
them unchanged. The helper performs no native work and does not replace admitted
resource reservations or distributed agreement. The shared partition observer now
composes these operations with live session authority and global agreement.

Nemotron-H target execution uses this path for dense/shared ReLU² units and attention
channels, alongside the sparse expert-unit driver. Native deletion, keep-only,
ordered edits and controlled replay are verified across SafeTensors/GGUF and all
seven TP/EP/PP combinations in each residency mode. See
[the family validation](component-validation.md#nemotron-h-partition-component-execution)
for exact scope and tolerances. Shared-unit edits occur before down projection;
shared-input edits affect only that branch, and effective shared writes feed the
sparse sum. The [shared-expert extension](component-validation.md#shared-expert-scalar-extension)
documents its parameter joins and position-specific keep-only reconstruction.

The preparation layer also exposes `PartitionActivationProjection` and
`ReservedPartitionActivation`. It preserves global admission, projects arbitrary
positive-stride selections and payloads through architecture coordinates, charges
host copies before projection, and reserves native updates before execution. Zero,
Scale, Mask, Replace, Add and compact component/logit masks use the same activation
mechanism as ordinary runs. F16/BF16 payload projection preserves bit patterns.
An empty local overlap acknowledges the invocation without creating a zero tensor;
nonexporting replicas still apply edits. Dropping or failing prepared work does not
refund its charges.

`AdmittedInterventionPlan::intent_identity()` lets peers compare exact operations,
semantic declarations, request geometry and source while retaining distinct local
session-bound `identity()` values. It is descriptive and grants no execution or
branch authority. `SessionPartitionIntervention` binds prepared work to the live
capture owner, exact admission and forward epoch. Pre-forward agreement includes
intent, ordered coordinate maps, source geometry, invocation membership and all
prepaid costs. Replicas apply the operation; inactive pipeline ranks do not enter
its hook collectives. Empty local selections acknowledge without synthetic values.

Every invocation member validates geometry before dependency execution, then
settles native edits and per-operation evidence before agreeing local success.
A bounded world receipt after shared forward completion rejects missing members.
Global `Applied` outcomes and evidence become visible only after final commit.
Before/after evidence surrounds each ordered operation and uses the ordinary
partition capture engine under separate operation/evidence keys. The parent's
capture plan and pre-intervention observation meanings remain unchanged.

The MLX public text adapter selects this path from its retained partition layout
and native invocation-group facts. Unsupported or unverified observation ownership
continues to produce an accurate intervention capability result. The portable
runtime owns agreement, accounting and publication; backend code supplies tensor
operations, transport and safe completion.

Shared routed Qwen and GPT-OSS attention hooks consume modified channels in their
output projection. Their normalized FFN input and completed expert contribution
are separate activation targets. The complete contribution hook follows expert
reduction and is distinct from provider routing events. Ordinary native control
uses the same decoder equations in ordinary and partitioned execution.


A declared intermediate read stage names the effective value consumed downstream.
Intervene on its corresponding original observation to change that computation.
For V3 KV latents, the effective current-position value is inserted into the cache;
changing a future plan leaves earlier cached rows intact. Pipeline input
interventions similarly carry the effective embedding in the typed boundary.
Use a snapshot taken before the tested computation, or exact token-ID replay, to
compare trials that change those earlier values. Stage slices use their declared
`hidden` axis; compact FFN/attention masks use the separate `component` axis.

All-dense V3 TP and combined TP/PP use these same latent, channel and gated-unit
interventions. Native tests compare compact masks and coordinated parameter
overlays with a local reference, including rejection rollback and restoration.
The matrix covers F32 SafeTensors/GGUF and SafeTensors load-time affine
4-bit/group-32; both executions retain the same selected parameter encoding.
Query rotary positions follow tokens independently of local head count. MLA
equation revision 2 invalidates earlier V3 cache fingerprints; rebuild those
prefixes by exact token-ID replay before running new trials.


The neutral partition projector supports architecture-declared additive activation
terms through `as_sum_term(offset_owner)`. Zero, Scale and masks affect each term;
Add is submitted only by the designated owner; Replace clears the selected region
on other terms. The owner and sum mode are part of exact geometry agreement.
Nonowners still validate dtype/shape and participate in completion. Reservations
charge the projected action, so an Add acknowledgment creates no native update or
payload copy. Per-term rounding has numerical, not bitwise, equivalence to an edit
after reduction. Vocabulary sentinel masking is rejected for this contract.
Target-only mixed V3 TP now invokes these hooks through both resident and provider
traversals. Native F32 SafeTensors/GGUF and load-time affine acceptance passes all 63 TP/PP/EP and
residency cases, including Zero, Scale, Add, Replace and strided masks at shared
write/output seams. The affine cases retain 4-bit/group-32 packed weights before
overlays; effective F32 edits restore to the original packed sources on removal.

The same mixed V3 matrix passes all 63 independently cached expert cases, including
atomic edits to both packed expert read halves and write banks, all five additive
actions and restoration. Exact selected tasks retain generated scale/bias ownership,
and ordinary/bank affine materialization uses the same source precision. These are
CPU Ring results with default bank budgets; they do not establish forced-eviction
coverage. Separate cache-eviction and additional encoding matrices are identified
explicitly in the component-analysis guide.

A V3 prediction scope's fused residual and pre-head residual are distinct
intervention targets. Changing the latter changes both returned hidden state and
its draft scores; the legacy after-head hidden output affects subsequent consumers.
Scope membership does not enable intervention admission: runtime first requires
verified prediction hooks in the selected call path. Typed extension hooks preserve
these timings through public phase-specific speculative admission.


The shared embedded strategy now forwards explicit internal observers through
actual target and V3 prediction operations. Component replacements reach downstream
equations under their invocation phase; no-op execution retains ordinary values.
Public phase-specific plans additionally require the loaded activation authority;
distributed capability comes from the retained producer and transport bindings.


`InterventionPlan::admit_invocations` binds the same invocation bounds as its paired
capture plan. Runtime validates actual shape and active token slices before the
forward, then uses the ordinary native edit and evidence mechanisms. Original and
effective evidence carries that physical geometry. Replacement and mask payloads
still require exact selected shapes; invocation admission introduces no broadcasting.
Inactive scopes cannot become missing interventions. Restore and child re-admission
retain accounting and the admission mode. Public speculative plan composition uses
the same bounds and partition invocation agreement.

The speculative collector uses the ordinary admitted activation operations and
original/effective evidence. Architecture-projected applicability disables edits
outside the invoked target/prediction scope before work. Repeated physical phases
share one cumulative ledger even when their generated prediction coordinate is
unchanged. Exact masks retain their declared shape; differing invocation widths
require compatible slices or compact component masks. Typed admission failures and
original native error sources survive speculative propagation. Public phase-aware
plans install through the combined activation authority and use selected partition
producers where loaded discovery reports support.


`SpeculativeActivationPlan` combines captures and optional edits into one admitted
authority bound to the loaded model, effective execution and native session. Public
installation uses `ControlledSpeculativeOptions.activations`. Ordinary
`SpeculativeInterventionPlan` retains its existing one-row sampler semantics.
Internal edits are installed once before forward execution and are removed with
the borrowed executor scope. Prospective changes use
`readmit_activation_interventions` at a drained canonical boundary with a plan
prepared before borrowing the controller. They preserve spent budgets and cannot
rewrite existing KV or recurrent history; replay from an earlier snapshot or exact
prefix is required to test a changed earlier forward.

For fused predictors, the admitted speculative scope determines when an edit can
execute. Context-input edits affect the cache update that consumes them; proposal
edits affect the temporary proposal computation. DSpark's proposal cache is kept
through native completion and discarded, while accepted-context state remains
separate. A dynamic vocabulary addition has its own original/effective input and
output seams, so an edit to its anchor-dependent projection input remains distinct
from changing the primary vocabulary head or the final combined score. Schema-2
speculative activation authority retains these phase bindings across controlled
execution and re-admission.

Qwen prepared prediction applies component masks to the actual fusion, attention,
FFN and readout values consumed downstream. Shared parameter identity does not
merge logical depth identities: a depth-specific activation edit affects that
invocation, while a shared fusion parameter overlay affects every consumer.
Neutral dense/MoE fixtures verify exact unit selection, causal score changes and
failed-observer replay; native TP/PP trials verify global masks and survivor
recomputation through the public controlled path.

## Original windowed execution scope

The selected Qwen embedded dense-window intervention path passes ordinary,
managed and controlled execution with the same ordered edits, cross-window
Preview/Summary evidence, immutable source identity and cumulative lifetime
accounting. The public case
`native_original_qwen_window_intervention_evidence_matches_ordinary_and_controlled`
records ordinary 0.453s, managed 0.620s and controlled 0.650s in the
[bounded native evidence](validation/bounded-media-and-submission-2026-09-16.json).
The original speculative outer checkpoint retains the collector and monotone
invocation/ledger source; it is distinct from the plain capture checkpoint.

Sparse internal/distributed interventions still need the original coordinate and
lowering destinations, native source and shared callback join. Their current
original admission rejects before those ordinary allocation/evaluation workers;
the shared sparse intervention semantics remain applicable implementation work.
The general plain managed request also still needs an intervention-source handoff
alongside its existing capture option. These remaining joins do not reopen the
validated dense-window scope.
