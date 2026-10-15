# Speculative decoding and multi-token prediction

`eredu` supports lossless speculative decoding with an external assistant,
embedded multi-token-prediction (MTP) heads, or a masked draft over a shared backbone. The target model always verifies proposals; accepted output has
the same target distribution as the corresponding non-speculative path.

The shared speculative transaction carries explicit proposal evidence.
`SpeculativeProposal::causal` retains a processed causal assistant distribution
for likelihood-ratio verification. `SpeculativeProposal::target_matching` retains
only a proposed token, with no invented causal draft probability. Target matching
samples each processed target row against the evolving accepted prefix, accepts
matching tokens and commits the first mismatch or full-acceptance bonus token.
It uses the target RNG independently of proposal randomness, updates adaptive
sampling and semantic constraints for each commitment, and uses the ordinary
transactional accepted-prefix cache retention and rejected-suffix rollback.
Loaded shared-backbone executions expose this policy through
`DecodingMode::SelfSpeculation`. External assistants and embedded MTP retain
likelihood-ratio verification with processed causal draft distributions.

Prepared executor requests carry `GenerationRequestConfig.decoding_mode`.
`SpeculativeDraft::Target` selects the loaded target without an external assistant. Its
prepared session handoff uses the same request table for causal sampling or
architecture-declared ordinary refinement. A target with a retained shared-backbone
draft also realizes self-speculation and all advertised mode handoffs. Refinement-only
targets do not need causal proposal-score admission; targets without that retained
draft reject self-speculation with `NoBackendDraftCapacity`.
Their shared request table also runs AR with zero proposals. Controlled sessions
and the request's optional `GenerationControlHandle` queue AR/speculation changes
at settled boundaries. A handoff appends its pending committed token once,
retains the causal next-token row and acknowledges the effective output position
before sampling resumes. Snapshots preserve this row and both mode states;
serial branches use independent mailboxes, including while inactive.

Likelihood-ratio verification requires causal conditional target probabilities.
Iterative block architectures such as DiffusionGemma report
`SpeculativeCapability::IncompatibleScores` and reject this verification with
`RequiresCausalConditionalScores` before proposal execution. Their tentative
canvases use the independent refinement schedule and draft-event contract.

Prepared chat, native tools, sampling penalties, stochastic acceptance,
cancellation, and semantic events use the same generation pipeline with or
without speculative decoding.

The shared public execution API uses `Speculative*` terminology. `Mtp*` is
reserved for architecture and checkpoint concepts that specifically represent
embedded prediction heads.

## Assistants

An external assistant is loaded separately and validated against the target's
tokenizer, vocabulary, and model-specific interface. Gemma 4 assistant and
Muse-Glimmer DFlash artifacts are supported through this path. Prepared Gemma 4
and Muse-Glimmer targets report a `Declared` separate-checkpoint capability so
automatic execution-plan realization can prove compatibility and construct the
matching assistant. `Ready` is reserved for a speculative realization that is
executable without further pairing.

For embedded prediction, readiness is reported from the installed, typed
executor retained by the session. The ordinary target projection deliberately
has prediction depth zero, so its target-only configuration is not used as a
proxy for installed execution capability.

External assistants currently pair only with replicated target topology. An
external plan requesting a partitioned target is rejected before assistant
weights or draft queues are materialized.

Embedded MTP is available for registered checkpoints with executable prediction
weights, including supported DeepSeek, Inkling, Nemotron-H, Qwen3-Next, and
Qwen3.5 variants. DeepSeek-V4 SafeTensors checkpoints support both ordinary
embedded MTP heads and fused DSpark draft blocks, including architecture-owned
draft-cache transactions in each live lane. A base `deepseek4` GGUF contains target weights only;
its metadata may describe omitted companion prediction weights but does not
advertise an embedded-draft capability. The usable proposal depth is capped by
the checkpoint's validated capability. For sequential MTP that capacity is the
number of prediction layers. For a fused DSpark head it is
`dspark_block_size`, which is independent of the number of DSpark layers used
to compute the block.

Applications should query `speculative_capability` or run model inspection
instead of assuming support from a family name. A declaration admits planning;
it does not claim that assistant compatibility, placement, and construction
have already completed.

Target-capture contracts fix semantic paths, order, ownership, observation
identity, fixed tensor dimensions, and bounded request-sized dimensions before
loading. At execution, the prepared-input identity, exact tensor extents, and
target-cache generation close the per-lane identity before assistant proposal
work. Mismatched or stale captures fail transactionally.

Embedded and external assistants use one neutral scheduling path. Architecture
composition constructs each `SpeculativeExecutor`; the selected backend binds
its generic caches, sampling state, queues, transfers, and completions, then
lends the paired executor through `SpeculativeGenerationVisitor`. The
facade-selected
`eredu-runtime::SpeculativeScheduler` owns token or semantic publication,
request lifecycles, fair action selection, verification resolution, and final
statistics. Embedded heads do not maintain a second acceptance loop or a
parallel set of generation wrappers.

`PlannedModel` owns the loaded target, drafting resources and selected policy.
`session.generate(GenerationRequest { .. })` handles disabled, embedded and
external drafting with one result. Requests carry sampling, input, cancellation,
caller stops, an output mode and a committed-event callback; they never borrow a
drafter. `output.stats().drafting` contains scheduler diagnostics when that
executor runs, including zero-work cancellation. Target-only execution leaves it
absent. Backend failures retain their original sources in `BackendFailure`.

```rust,ignore
let mut session = LoadedModel::load_execution_plan(&factory, path, &plan)?;
let chat = session.model_mut().prepare_chat(chat_request)?;
let request = GenerationRequest {
    input: PreparedChatInput::rendered_prompt(&chat),
    output_mode: OutputMode::Semantic,
    settings: PreparedChatGenerationSettings::default(),
    caller_stop_sequences: &[],
    cancellation: Default::default(),
    control_handle: None,
    on_event: |event| publish(event),
};
let forecast = session.forecast_generation(&request, &Default::default());
let output = session.generate(request)?;
let drafting = output.stats().drafting.as_ref();
```

A backend implements `SpeculativeGenerationBackend` and lends typed execution
resources to the shared scheduler. `LoadedModel::into_session` supplies a
portable target-only owner without requiring that trait. Factory loads can use
`into_target_only()` to release drafting and retain that same portable interface.
Explicitly assembled
sources use `LoadedModel::drafting_session`; factories retain their complete
realization in the returned owner.

`OutputMode::Semantic` requires recognized executable semantic support.
`OutputMode::Text` requires `PreparedChat::text_generation_support()` and emits
literal `TextDelta` and `Finished` events. It honors checkpoint EOS and caller
stops and skips special tokens. Native tool declarations and required calls are
rejected; explicit thinking requires `allow_unparsed_reasoning`. Applications
can emulate tools using ordinary prompt text and their own output parser. The
low-level `generate_speculative_batch` API takes one explicit output mode for
its independently scheduled lanes.

`output.timing().time_to_first_token()` measures the first committed token,
even when Unicode buffering, a stop, or EOS prevents visible text. Cancellation
without a commitment returns `None`; batch timing includes queueing from the
batch call. The CLI chooses text output explicitly for unrecognized templates.

## Execution placement

Prepared-chat requests do not accept backend queues. The portable execution
plan selects target/draft placement before native queue construction. The
backend binds streams and devices to that selected topology and validates the
binding; it does not classify semantic placement by comparing streams after
loading. Tokenizer compatibility is proven by the facade before assistant
payload materialization. The selected modes are:

| Placement | Behavior |
| --- | --- |
| one target stream | target and assistant are ordered on one stream; no same-request lookahead overlap |
| two streams on one device | dependencies use completion events; arrays keep the same physical storage |
| streams on different devices | dependency boundaries synchronize before required arrays are copied |

Two streams on one GPU are an experiment, not an automatic optimization. The
target and assistant can contend for the same compute and memory bandwidth.
Compare the placement with lookahead disabled before adopting it.

The owner inherits proposal and lookahead controls from its execution plan.
Explicit request modes can use stricter bounds; initial requests and later mode
commands cannot expand those ceilings. `drafting_options()` exposes the retained
defaults for inspection. Forecasting and generation resolve the same defaults.
The ceilings remain attached to control state through snapshots and forks.

## Verification and cache state

The target cache contains tokens already used as target inputs. After a fully
accepted proposal block, the target may sample one bonus token from the next
distribution. That bonus is emitted but remains the leading uncached input for
the next target verification.

A draft block owns its ordered proposal tokens, the processed distribution used
for each token, and the exact assistant frontier. Rejection rolls back the
target's speculative suffix and discards the corresponding assistant branch,
sampler branch, semantic events, and statistics changes. Chunked target caches
may retain allocation capacity while reducing their logical length.

Target and draft random streams are independent. Draft substreams are addressed
by logical output position, so scheduler interleaving or discarded optional
work does not advance the randomness assigned to canonical output.

## Optimistic lookahead

When the target and external assistant use distinct eligible streams, the
scheduler can draft one continuation while target verification is in flight.
This optional branch is promoted only after full acceptance of its parent block
and exact prefix agreement.

If the target bonus equals the branch's first token, that token is consumed as
the bonus and the remaining token/distribution pairs can be reused. The branch
is then extended from its saved assistant frontier to restore the normal
proposal capacity. If the bonus differs, the entire optional branch is
discarded. A terminal bonus also discards the continuation.

Promotion requires:

- an assistant state that can be cloned and discarded independently; and
- draft sampling that is a pure function of logits, explicit history,
  immutable settings, and the supplied position-addressed PRNG state.

External Gemma assistants and the portable history-derived sampler satisfy
these rules. Embedded predictors whose commit advances target-owned state do
not use same-request optimistic lookahead. `PreparedChatGenerationSettings::strategy`
selects standard or Mirostat V2 sampling for ordinary and speculative prepared
chat, with either external or embedded drafting. Batches may mix strategies;
each lane retains its own sampler, penalties, seed, and adaptive state. Mirostat
updates once per committed token using its constrained target distribution,
including accepted proposals, replacements, and bonus tokens. Discarded draft
work does not advance durable adaptive state.

Mirostat does not support exact optimistic promotion, so its lanes skip optional
same-request lookahead even when lookahead is enabled. Ordinary drafting and
verification still run. Standard lanes in a mixed batch remain eligible for
lookahead. Single requests and every lane of a batch validate sampling settings
before backend prompt preparation or execution.

`SpeculativeSchedulerOptions::with_lookahead(false)` disables only the optional
branch; the same drafting, verification, acceptance, cache commit, callback,
and statistics pipeline remains in use. This is the reference setting for
output equivalence and performance comparisons.

By default, per-request lookahead can disable itself after enough resolved
branches when no proposals are reused or discarded proposals outnumber reused
proposals. This adaptive decision changes optional work only, not output.

## Constraints and tools

Each request owns its constraint sampler, tokenizer decoder, protocol parser,
stop matcher, callback, caches, and random roots. Draft and target logits are
masked at their exact logical histories. Drafting stops before a proposal would
cross a grammar-complete boundary.

Target resolution advances a transaction-local copy of constraint and semantic
state. Events become visible only when the matching target cache boundary
commits. Rejection, mismatch, cancellation, or terminal output drops every
remaining branch event.

## Diagnostics

Speculative decoding reports proposal and acceptance counts plus optional work
that was drafted, consumed as a target bonus, reused, or discarded. It also
reports bonus matches/mismatches, adaptive disablement, and time spent drafting
or waiting for verification.

These timings are diagnostic. In particular, time inside the in-flight
verification interval can include host scheduling of assistant work and should
not be interpreted as target kernel time.


## Inspector and controlled sessions

`session.with_controlled(request, options, drive)` lends explicit advancement
with the same request, policy resolution and result as `generate`. The worker
can call `control.step()` or `control.run()` without choosing a drafting path.
A `GenerationStep` retains ordinary records or speculative proposal/verification
evidence. Inspectors borrow `control.speculative()` for proposal-aware snapshots,
interventions and canonical-boundary forecasts, or `control.ordinary()` for the
ordinary typed controller. Returning early cancels and settles unfinished work.
See [execution control](execution-control.md#controlled-speculative-generation).

Controlled sessions also support serial `fork`/`exchange` at exact canonical
boundaries, prospective token forcing and temperature/reseed changes. Target and
draft logits can have separate admitted tensor interventions; their evidence includes
rejected proposals. All paths retain the shared scheduler and acceptance algorithm.
See [controlled speculation](execution-control.md#controlled-speculative-generation)
for scope-wide accounting, supported hooks and snapshot requirements, and the
[Inspector workflow](lm-inspector-execution-control.md#speculative-forks-and-prospective-interventions)
for branch journal reconciliation.

Proposal advancement uses `SpeculativeProposalStep::CausalLogits` or
`TargetMatchingToken`. Separate autoregressive assistants and embedded heads
return causal logits. A masked source returns its selected candidate; the shared
driver attaches target-matching evidence without resampling it. EOS and grammar
prefix completion still end proposal collection through the shared lifecycle.
Target processing and constraints advance only against the accepted canonical
prefix during verification. The same transaction retains or rolls back cache
state before publishing either source's accepted output.

The portable shared-backbone executor uses one ordinary model and one canonical
cache. Its masked source seeds the last committed token that still awaits cache
append, over the retained causal prefix. That seed is excluded from proposals and
included once in verification inputs. Full-attention caches transactionally
truncate rejected K/V; other state mechanisms replay accepted inputs in isolation. Proposal randomness belongs to the request
seed and survives controlled snapshots and restores independently of target draws.
Draft blocks retain their complete admitted geometry when the remaining output
budget shortens proposal collection. The shared physical generation driver uses
this executor and the same controller, sampler, target RNG and commitment hook
as AR. Mode changes wait until verification settles. Native verification observes
the configured completion deadline and quarantines still-live work on timeout;
unsupported cancellation dispositions are rejected before execution.

The MLX shared source uses a single queue and has no eligible optimistic branch.
Lookahead is an optional scheduler optimization; its configured capacities remain
upper bounds. Verification capture records the complete causal transaction at its
physical sequence extent. Individual accepted decisions consume retained causal
rows without repeating that forward or charging another capture invocation.
Snapshots retain the verification evidence, independent proposal RNG, accepted
cursor and canonical cache; restore and fork preserve consumed capture/copy credits.

Prepared requests, forecasts and batch lanes select their initial mode through
`PreparedChatGenerationSettings.decoding_mode`. Absence selects AR. Explicit
`DecodingMode::SelfSpeculation` carries the neutral, validated `SpeculationOptions`;
there is no separate request-level speculation configuration. A retained source
also accepts an initial diffusion mode when its target declares refinement.
Batch lanes agree on one scheduler contract while retaining individual proposal
limits and sampling state. External and embedded sources reject masked refinement
and draft-only adapter options. Their causal proposal probabilities retain
likelihood-ratio verification; shared masked proposals use target matching.

Shared-backbone forecasts retain complete trained canvas geometry independently
of the proposal or output allowance. Proposal workspace evaluates all canvas rows
and prefix-plus-block attention; canonical cache residency excludes transient
block K/V. Verification accounts for proposal overshoot, rollback checkpoints and
configured lookahead copies. Full canvas distribution storage and optional
low-rank activation work remain charged when the output allowance is shortened.
Base and adapter parameters belong to one target catalog and are charged there
once per actual topology replica. Sampling and graph overlap bounds retain their
labeled calibration rather than claiming measured allocator peaks.

Independent causal assistants compose ordinary target refinement through
`TargetRefinement`. Their proposal and verification policy remains causal even
when the target supports masked diffusion. The prepared input supplies a separate
refinement stream; the request seed preserves the target's immutable causal
prefix, aligned assistant prefix and paused numerical state. Refinement advances
one numerical invocation at a time, and causal settlement after finalization
advances both caches before a mode acknowledgement. Model parameters remain with
the existing target and assistant owners.
