# Prospective intervention plans

`LoadedModel::intervention_discovery()` and `LocalModel::intervention_discovery()`
return versioned mutable targets for the actual loaded session. These are separate
from architecture nodes and read-only observations. Each target joins an existing
node identity, declares an execution stage, semantic axes, exact supported dtypes,
operations, and prefill/decode support. Unverified, conditional, and unsupported
phases cannot be admitted as supported phases.

Call `prepare_intervened_chat(chat, settings, capture, intervention, trace_limits)`
and pass its result to the existing `generate_observed_chat`. The returned
`PreparedObservedGeneration` owns immutable capture and intervention admissions.
Preparation uses retained source and session identities and performs no native
model work. Existing `prepare_observed_chat` callers remain valid. An empty
intervention plan takes the ordinary path and has no intervention trace identity.

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
point dtype remains request-dependent. Scalar scaling uses native dtype rounding;
its input must be finite and representable. Arithmetic otherwise follows the
native dtype's ordinary floating-point rules.

Logit masks use unique canonical vocabulary IDs, retain the full vocabulary axis,
and cannot mask the entire vocabulary. They precede the existing sampler's token
constraints, penalties, temperature, top-k/p and other processing. Replacement
logits are returned to that same sampler. Interventions introduce no sampler or
random draws. Independent sampler constraints can still reject a distribution
with no eligible token.

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
The text entry point requires one token-ID prompt with the admitted batch and
prompt length; the facade constructs it. Embedded-tensor or multipart prompts
are rejected before a native submission.

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
metadata and requested evidence. The hard plan limits are 64 operations, 512 KiB of
payload storage, and 1 MiB of compact JSON. Preview evidence has at most 4096 elements
per side/field. These are logical accounting limits, not a physical allocator cap.

Plans apply prospectively. They never retroactively recompute cached KV or recurrent
states. For an independent experiment: complete or cancel the baseline, call
`reset`, prepare a new request, then generate. Hot plan replacement and resumable
snapshots are not supported. A failed forward can already have changed state even
when the particular failing intervention did not apply. The existing rollback,
reset, poisoning and retained-native-completion rules govern reuse. Cancellation,
consumer `Break`, early drop and unwinding keep the ordinary completion owner.

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
