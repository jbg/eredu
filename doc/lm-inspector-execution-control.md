# LM Inspector: execution control

Keep native models and sessions on one owning worker. Send pause/cancel requests
through `GenerationControlHandle`; send step, resume, snapshot, restore and branch
commands to that worker. The session exclusively borrows its `LoadedModel`, and
native handles do not need to become `Send` or `Sync`.

Use `eredu::api::LoadedModel<B>` with your backend factory. MLX applications
use `eredu_backend_mlx::MlxBackendFactory` with the same control API. The complete
runnable example is:

```sh
cargo run -p eredu --no-default-features --features mlx \
  --example controlled_generate -- /path/to/local/checkpoint 'Explain gravity briefly.' f32
```

The example uses CPU execution, nonzero temperature, bounded raw-logit capture and
a validated identity intervention. It steps twice, pauses, snapshots, completes a
baseline, verifies an unchanged restore and an unchanged isolated fork, then emits
a reseeded child with prospective zero-logit intervention. JSONL records go to
stdout; agreement and resource usage go to stderr. Supply the checkpoint's exact
intervention dtype (`f32`, `f16` or `bf16`). No downloads are performed.

## Prepare and discover support

Compile retained tokenizer/template sources and call
`prepare_chat(&source, &policy, capacity, &cancellation)`. Build a
`PreparedChatRequest` with the same enforced capacity. Attach borrowed `capture`
and `intervention` declarations; startup admits them against exact prompt
geometry. `CapturePlan::none()` selects ordinary delivery.

Semantic and eligible literal output are explicit policies on that same request.
Use `PreparedChatOutputMode::Text` when the retained template's text capability
permits it. The runnable example above contains source compilation and error
handling; its controlled startup is:

```rust,ignore
let mut request = PreparedChatRequest::new(&chat, settings);
request.capture = Some(&capture);
request.intervention = intervention.as_ref();
request.output_mode = output_mode;
request.stop_sequences = &stops;
let Some(mut session) = model.start_controlled_chat(
    request, trace_limits, control, &mut emit,
)? else {
    return Ok(()); // Cancelled before startup.
};
session.step(&mut emit)?;
```

Read capability reasons from the prepared chat before selecting output policy.
Capture, intervention and trace plans do not replace the request's original
source identity or enforced memory admission.

Text mode consumes the prepared prompt IDs exactly, including the rendered
generation prefix. It retains checkpoint sampling defaults, overrides, seed and
EOS, and emits the existing token records plus `SemanticEvent::TextDelta` and
`Finished`. It decodes incrementally, skips special tokens and matches caller
stops, without interpreting tool calls, reasoning, or profile-specific stops.
Reasoning and tool-like ordinary text can therefore appear literally. Tokenizer
validity still excludes holes and padded logits positions before sampling and
forced-token admission; grammar constraints in semantic mode further restrict
that same domain.

For `TopCandidates` bars, inspect `CaptureCandidates.domain` before interpreting
each candidate's `allowed` flag. With a known domain, render `allowed: false` as
present-but-forbidden: keep its raw score visible and indicate that the token was
never sampleable under that decision's tokenizer/semantic constraints. The flag
is **not a probability adjustment**; raw logits (or a UI softmax of them) are not
the sampler's final probabilities. Both `Original` and `Effective` sources use
the domain before any forced choice. Forcing one token therefore does not mark
every other candidate forbidden.

The summary supplies the allowed count and actual logits vocabulary width;
`constrained` distinguishes semantic restrictions from tokenizer validity alone.
Missing/`None` domain means unknown, including old records and custom samplers
without exact domain provenance. Their default `allowed: true` must not be shown
as confirmed permission. MLX one-row speculative capture provides the same
metadata at each target/draft history; draft membership is separate from target
acceptance.

Text admission rejects every nonempty tool declaration list, including with
`ToolChoice::None`, and rejects `ToolChoice::Required` even with no declarations.
Use a request with no tools and `Auto` or `None`; text mode provides no tool
suppression or tool-output guarantee. This conservative rule preserves requested
tool constraints instead of silently dropping them. Existing `prepare_chat`
thinking admission remains in force. Additionally, explicit `enable_thinking: true`
requires `allow_unparsed_reasoning: true` to select text mode, even if the template
has a recognized reasoning parser. Template defaults remain unchanged. Admission
fails through the existing prepared-chat error convention before native startup.

Check `session.capabilities()`. Stepping and pausing describe the actual loaded
configuration. Configure `enable_snapshots(limits, capacity, copy_limits)` once before
retaining state. This rejects unknown native, grammar or semantic costs. Creating
a snapshot additionally establishes whether complete continuation growth is known
for branching; `capabilities().fork` then reports that result. Every fork still
validates its own plans and budgets.

Snapshot support requires complete native, controller and semantic copy bounds
at the actual boundary. Active and automatic grammars use the same admitted copy
workers as ordinary prepared-chat continuation; a missing bound remains a typed
refusal. Show the returned support reason instead of inferring full snapshot
support from a native KV-copy primitive. Snapshots retain partial Unicode,
caller-stop lookbehind, pending forced choices, decoder state and cumulative
spending. Compatibility includes text versus semantic output policy. See
[execution control](execution-control.md#snapshot-restore-and-fork) for current
capability limits and validation coverage.

## Drive one logical run

`step(callback)` advances at most one committed prediction, resolves native work,
finishes intervention checks and synchronously delivers that prediction's records.
`run(callback)` continues until a sticky pause request or terminal outcome;
`resume(callback)` acknowledges pause and continues. `pause(callback)` retains
partial Unicode, stop lookbehind and protocol state without flushing them.

`GenerationControlHandle::cancel`, `session.cancel` and callback
`ControlFlow::Break(())` permanently cancel. Break closes delivery. They never mean
pause. Callback panics fence the active run. An unresolved native failure remains
owned by the existing backend recovery lifecycle; do not recover by clearing UI
status or importing a native handle into another worker.

Use a bounded channel from the worker to the UI. The record callback is synchronous;
blocking it applies backpressure. If the consumer disconnects, return Break and
release the session on its owning worker. Keep opaque snapshot/branch handles in
a worker-side table with explicit user-visible limits, and drop forgotten entries.

## Reconcile a restore

Index records by `generation.run_id`, `sequence` and `epoch`. Sequences increase
through the run's entire journal; a restore increases epoch and never refunds
transport, capture or copying consumption.

1. Save the `ControlledGenerationSnapshot<B>` handle and its serializable metadata.
2. A snapshot's `output.next_prediction` is the next absolute decision. The most
   recently committed token may still be pending decode input; do not feed it again.
3. On `Restored { output, ... }`, restore the view represented by the original
   run journal strictly before `output.next_sequence`, respecting any earlier
   restore records within that prefix. Discard later visible continuation.
4. Apply subsequent records at their new monotone sequences. Old tokens and text
   are not emitted again. Do not concatenate the abandoned text with the new path.

Snapshots are reusable only within their original exclusive driver tree. Restore
requires the original logical run. A normally completed snapshot stays completed;
restoring an earlier snapshot into a normally completed run can resume that earlier
state. Cancelled and failed runs cannot be revived. Serialized metadata cannot
recreate a handle, admission or compatibility proof.

## Create and switch branches

Call `fork(&snapshot, GenerationBranchOptions { trace_limits, capture_limits,
sampling, intervention }, callback)`. A fresh child starts at the snapshot's absolute
position and keeps its original generation limit. Capture limits must include
the checkpoint's inherited usage. The child has a separate transport budget;
snapshot/branch retention and cumulative copying share the tree's budget.

`BranchStarted` belongs to the child's run and includes its parent snapshot, changes,
inherited capture usage, immutable canonical prompt, generated-token prefix and
exact semantic prefix. Build
the child's independent view from that prefix. It can include the beginning of an
unfinished tool call whose later arguments arrive in child deltas. Partial UTF-8
that was never delivered remains internal to the saved decoder.

`exchange(&mut slot, callback)` activates the slot and puts the previously active
run into it. Update the worker's slot-to-run mapping using `slot.run_id()` and the
active `output_checkpoint().run_id`; a slot is not a permanent branch identity.
Only the active run advances. Its cancellation handle, budgets and lifecycle move
with it. Cancelling one child does not cancel a parent parked in another slot.

Sampling options change future temperature or explicitly reseed. Omitting reseed
preserves exact inherited randomness, penalties and adaptive state. Mirostat cannot
be changed to zero temperature. Intervention replacements are re-admitted with
fresh child session/plan IDs and affect only future scheduled execution. They do
not recompute old KV or recurrent state. Keep absolute schedule positions; use
`prediction_index - parent.output.next_prediction` only as a separate display index.

For an alternative canonical token, activate a branch from a snapshot **before**
that decision, then call `force_next_token(id)`. It validates vocabulary and active
constraints and uses the ordinary commit/decoder path. A nonzero-temperature forced
decision consumes one ordinary RNG draw; greedy consumes none. Mirostat performs
one adaptive update at probability one. The token record reports `forced: true`.
Never replace a committed token by editing its history or decoding and retokenizing
text. `clear_forced_token()` clears an uncommitted choice.

## Determinism and support limits

Unchanged continuations retain exact saved state and RNG. Reproducibility assumes
the same loaded executable, device and deterministic native mechanisms; there is
no promise of identical floating-point output across devices, backend versions,
parallel layouts or nondeterministic kernels. Native CPU conformance covers dense
KV, LFM2 convolution/attention and Qwen3.5 MoE recurrence/attention with an admitted
routing intervention. Facade tests cover semantic history, partial Unicode,
terminal behavior, canonical forcing, sibling isolation and monotone accounting.

Native local-facade tests also verify sampled partial UTF-8 restoration and execute
the complete observed/intervened example on a generated Qwen2 checkpoint. Explicit
text tests use a deliberately unrecognized template, restore and fork partial UTF-8,
and compare greedy/sampled output against ordinary token generation with a padded
logits vocabulary. Portable tests also cover tool/thinking admission, literal
protocol-like text, non-EOS special tokens and caller-stop lookbehind. Run native tests
with `cargo test -p eredu --no-default-features --features mlx --test native_execution_control`.
For an accessible Metal GPU, run
`cargo test -p eredu --test native_execution_control native_metal -- --ignored`.
Metal was unavailable in the development session; CPU-only native tests passed.

Snapshot and branch estimates describe conservative logical storage, including
future mutable growth. They may substantially over-reserve for long tokenizer
continuations and do not cap physical allocator workspaces. Unknown costs reject
admission. Copy failures consume their admitted copying allowance while releasing
unretained state; healthy source snapshots remain reusable. Paged/compressed native
state, partitioned/media/realtime control, persistent snapshots,
cross-process/backend restoration and layer-level pauses are unsupported.


## TTFT and speculative inspection

Read `session.timing().time_to_first_token()` or the `timing` field on controlled
records. These use active generation time, excluding UI pauses and time between
steps. Do not start a separate wall clock around the command loop. The metric
counts the first committed token even when decoding buffers it; `None` means no
commitment. Restoration does not reset it.

Keep speculative resources on the same owning worker, using
`model.with_controlled_prepared_chat_speculative(request, options, |session| { ... })`
(or `with_controlled_managed_plain_text_speculative` for literal text). Inside that closure,
wait for worker commands and call `session.step()` once per requested scheduler
action. Iterate until `None` for uninterrupted completion; returning early cancels
and settles the native work. The existing request's semantic callback still emits
only target-committed conversation events.

Render `step.drafted` as tentative tokens. `step.verification.dispositions` gives
accepted, rejected and discarded results for the original proposal block. Its
committed IDs include the target replacement or bonus, which may not appear among
the proposals. Preserve optimistic `assumed_prefix` and reuse/discard counts in
the timeline. Associate captures by target/draft role, position and step sequence;
a draft capture is not evidence that the token was accepted. The step is a whole
scheduler action, and target commitment is atomic per block.

Use `prepare_speculative_capture(settings, plan)` for one-row `model.logits`
admission. Pass explicitly bounded `trace_limits`, that optional capture admission,
and snapshot limits in `ControlledSpeculativeOptions`. `snapshot_support()` gives
a reason when unavailable. Retain opaque handles in the worker, restore only at a
canonical boundary, then reconcile the UI with `session.token_ids()` and its new
`epoch`. Keep prior timeline records distinguished by epoch. Release unneeded
handles with `release_snapshot`; restoration does not replenish any budget.

Current native snapshots cover external Gemma 4 and Muse Glimmer/DFlash with
supported isolated target state and known semantic costs, plus complete sequential
V3 and V4 sequential/DSpark embedded state across resident/host/disk weights. V4
acceptance covers exact sampler captures, both pooling completion boundaries,
repeated restore and isolated siblings; internal V4 component hooks and distributed
snapshot support remain separate coverage gaps. For internal V3 components,
use `prepare_speculative_activations` and `ControlledSpeculativeOptions.activations`
from run creation; delivered records retain phase, depth and physical sequence
geometry. `readmit_activation_interventions` changes prospective edits at a drained
canonical boundary, and restore reinstates the saved plan without refunding budgets.
Prediction parameter queries and coordinated overlays use the ordinary loaded
parameter APIs; activate an overlay before admitting captures and creating the
controller. See the [component guide](component-validation.md)
for exact-prefix trials and current native evidence. Other incomplete state profiles
and distributed internal collection remain capability gaps. See [execution control](execution-control.md#controlled-speculative-generation)
for the complete boundary and exception list.

### Speculative forks and prospective interventions

Keep opaque branch handles on the same worker as snapshot handles. After prefill,
create a snapshot at a canonical boundary and call `session.fork(&snapshot)`.
`session.exchange(&branch)` activates that child and stores the previous run in the
same slot. Record the returned run ID and canonical prefix before advancing it;
route the request's semantic callback to that active run's journal. Each step also
carries `run_id`, monotone `sequence` and `epoch`. Restore snapshots only into their
own logical run. Forking from another run's snapshot in the same scope is allowed.

```rust,ignore
let snapshot = session.snapshot()?;
let child = session.fork(&snapshot)?;
let active = session.exchange(&child)?;
// Reconcile the Inspector's selected journal with active.run_id/token_ids.
session.override_sampling(SamplingOverride {
    temperature: Some(0.7), reseed: Some(42),
})?;
session.force_next_token(alternative_token)?;
while let Some(step) = session.step()? {
    // Visualize step.drafted, step.verification and step.captures for step.run_id.
}
session.exchange(&child)?; // The parent resumes from where it was paused.
session.release_branch(&child)?;
```

For tensor experiments, prepare one-row capture admission and separate admitted
plans for `SpeculativeCaptureRole::Target` and `Draft` with
`model.prepare_speculative_intervention(&capture, role, plan)`. Pass the capture
admission in `ControlledSpeculativeOptions`, then install those plans using
`session.intervene(vec![target_plan, draft_plan])` before prefill or at a canonical
boundary. Omitted roles have no edits. Ordinary `InterventionPlan` schedules and
slices use absolute generated positions and a `[1, 1, vocabulary]` prediction row.
Raw captures remain before-edit observations; each prediction's intervention
records contain applied/inactive outcomes and bounded before/after evidence.

MLX supports logits edits using the existing native activation primitives. Layer
and routing edits need architecture observer hooks with speculative attribution,
so discovery omits them. Branch exchange uses bounded exact state copies; ensure
copy limits cover both sides of each switch. Capture, trace and TTFT remain scoped
to the shared request, and neither switching nor restoring replenishes budgets.
The external Gemma CPU regression forces a child token, applies conflicting target
and draft logits, observes rejection, replays the edited child, then verifies the
unedited parent still reproduces its original suffix after checkpoint files have
been removed.

## Source ownership and causal row collection

Captured admission separates cumulative frame/tensor H, registered source S,
span-record/neutral-control P and native-control Q=A+C+W. The actual SpanPlan
attachment protects P+Q; its guard follows native work and retained quotes without
keeping full numerical diagnostics. Raw and bound hooks can have different
host-control costs even when their numerical outputs agree.

Causal row assembly binds the actual prepared discovery catalog, target path
owner and accepted capture source. Equal-content independent admissions reject.
Full and Preview(0) retain source identity and physical readout demand. Body hooks
preserve requested StateOnly/LastPosition demand; coupled readout/vocabulary
hooks require Sequence. Preview(0) does not erase source or factory work.

| Family | Causal semantics exercised by neutral fixtures |
| --- | --- |
| LFM2 | Convolution/KV state and absent width-one history; stateless and attention-only local partitions. |
| Nemotron-H | Mamba, attention and dense/routed target rows; target and prediction state remain distinct. |
| Qwen hybrid | Convolution history and recurrent matrix, cached continuations and rank-local state. |
| DeepSeek V3/V4 | Compressed attention, local keys, pending compression and overlap state; readout stream collapse. |
| Kimi Linear | KDA q/k/v histories and recurrent matrix, MLA latent/positional channels and exact path ownership. |
| Muse-Glimmer | Composite target rows, cache-derived masks and post-softcap publication. |
| Gemma4 | Shared-KV publisher/receiver frontiers, sliding history, explicit rotary positions and readout demand. |
| Inkling | Target outer/readout rows, scaled readout, convolution state and cache-owned relative attention. |

Fixtures collect real original/effective callbacks and compare complete state,
uneven continuation chunks and multiple cached decodes. They do not fabricate
callback aliases, seed mutable caches to force parity or infer internal/media
causality from axis names. Native admission, conditional/media row binding and
prediction hooks require their own selected capabilities and sources.

Gemma4 sliding-mask coverage distinguishes contiguous retained tails from
cache-owned shared-consumer requests. Explicit caller masks retain their own
execution behavior and are outside ordinary causal declarations. Width-one
LFM2 has no convolution-history role; width-one Qwen retains its populated
recurrent matrix. Zero mRoPE sections perform no secondary-axis overwrite,
while the positive rotary width determines native frequency storage.

A same-key storage-publication schedule funds all row buffers, registration,
node/output peaks and retained stamp controls, because rows can escape together.
Physical capture storage and adopted backing capacity remain separate. Each
published owner retains source custody through escaped aliases. Retained metadata
or identity alone cannot authorize submission or recreate a consumed grant.

The [conformance guide](prepared-chat-conformance.md) describes public
ordinary/controlled tests. [Native validation](bounded-inference-validation.md)
identifies hardware, stack and build profiles. Neutral family fixtures alone do
not establish native execution, released-checkpoint accuracy or every admission
cross-product.
