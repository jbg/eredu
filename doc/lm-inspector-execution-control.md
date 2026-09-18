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
controller. See the [component guide](component-validation.md#effective-prediction-parameters-and-coordinated-overlays)
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

Captured original admission separates cumulative frame/tensor H, newly registered
source S, span-record/neutral-control P and native-control Q=A+C+W. The same actual
SpanPlan attachment protects P+Q, while its compact guard follows native work and
historical quote objects without retaining full numerical diagnostics. Added
native tests check exact/-1 admission across resident/host/disk, source/plan
binding, historical and certified-work alias lifetime, and a producing work alias
from real controlled capture. Selected-prefill opening inventory and native span
activation remain separate required work; no managed gate changes here. Raw and
bound hooks may have distinct host-control costs despite numerical parity.

The LFM2/LFM2-MoE row companion is bound through the actual prepared catalog,
retained target paths and exact accepted source identity. For three target units
it declares 22 real outer/embedding/readout hooks. Neutral numerical fixtures
compare every declared nonzero value across a cached two-token prefix, an uneven
2/1/2-token continuation and three further decodes, checking all convolution and
KV tensors plus absent state domains after each chunk. Width-one uses no history
slot; missing history for wider kernels still fails. Source tests include Full
and zero Preview, reject equal-but-independent admissions, and reject internal
hooks without causal proof. Physical readout tests separately preserve all body
rows under StateOnly and LastPosition. These are staged conformance tests, not
native or released-checkpoint validation results. Nemotron-H ordinary target row
declarations remain an implementation task requiring its own complete Mamba,
attention, dense/routed and MTP-boundary audit and tests; no inherent noncausal
restriction has been established.

### Nemotron-H declared target rows

Nemotron-H target declarations retain actual `model.layers` invocation paths through the shared ordinary helper: embedding and each outer input/output pair (including effective observations), five readout-input hooks and three vocabulary-score hooks. Four-unit fixtures therefore declare 26 points. The actual prepared discovery/source and immutable path token bind each selected point; a separate equal-valued admission is rejected. Internal attention channels and MTP paths do not gain row-assembly declarations. Preview(0) still binds the same declaration and physical readout requirement.

The numerical fixture collects genuine callbacks; it never synthesizes effective companions. It checks full output/state equivalence after a cached two-token prefix, a five-token continuation split 2/1/2 and three further decodes across mixed/homogeneous Mamba, attention, dense and MoE schedules. Complete convolution/recurrent/KV values and absent fields are checked, including width-one history absence and target/MTP state separation. The prepared TP/PP matrix additionally compares all 18 body callbacks on their actual owners and ordinary final predictions for stateless-leading dense/routed schedules. The scalar selective scan validates recurrence semantics but is not evidence for a concrete accelerator's scan-block rounding, and no native/reference-checkpoint execution is asserted by this source-only increment.

Original capture preparation can seal a same-key finite storage-publication schedule after the exact C publication layout. Its S contribution sums all row buffers/registration/node/output peaks and retained stamp controls because rows can escape together. It joins the original P+Q hold once; physical C and newly adopted backing capacities remain separate. Individual published owners and canonical fixed nodes retain raw original custody through ordinary alias escape. The bounded path does not allocate the outer namespace and has no raw-registration export. Native attachment growth, provider key payloads and complete opening-source collection remain distinct measured obligations; ordinary registry metadata retains its existing accounting classification.

### Muse-Glimmer prepared text evidence

Muse ordinary text now binds actual prepared source identity and retained composite path identity to eighteen real outer/embedding/readout points in the two-layer conformance fixture. Five tests include twelve dense/routed, fixed default/linear/YaRN, tied/untied full-versus-chunk cases after a cached prefix, complete intermediate KV contents, and three repeated decodes. The observer requires actual effective callbacks and preserves the enclosing post-softcap logits publication. Full and Preview(0) retain the same physical readout requirement; Preview(0) does not erase source/factory work. Vision projector/assembly discovery and external assistant availability are unchanged. This source increment does not claim native gate activation, released-checkpoint validation or media/proposal causal equivalence.

### DeepSeek V3 target-row conformance

The V3 ordinary row companion uses actual prepared discovery and retained path identity. Nonzero neutral fixtures cover sixteen dense-prefix/routed, direct/low-rank-query, fixed/YaRN and resident/blockwise compressed-attention configurations. Before any observation path or state is created, the fixture compares every actual parameter name and shape with the architecture parameter description and verifies nonzero finite values. Full-versus-uneven rows and complete mutable compressed state use the existing absolute 2e-4 numerical tolerance; all effective observations must arrive from real traversal callbacks. Full and Preview(0) source binding preserve the original physical readout requirement and reject independent owners or undeclared internal hooks. This increment is causal-row evidence, not released-checkpoint validation, native activation, MTP/proposal or intervention support, or new partition admission.


V4 ordinary target prefill selection now binds actual declared target boundaries
and readout stream-collapse hooks to the retained discovery catalogue and path
owner. Equal independent sources/path owners reject, and Preview(0) still needs
the same physical readout demand. The new neutral fixture checks nonzero local
keys, pending compressor values/gates, pooled values, overlap values/gates and
all cursors at uneven frontiers, including a ratio-128 window completed after a
126-token prefix. Its per-call attention_local_tokens field is checked against
the actual preceding local width plus input width; both saved states execute
the same next decode to prove that scratch width does not alter continuation.
No state payload is seeded, no callback alias is synthesized, and no cumulative
capture accounting, snapshot refund or native activation policy is changed.

### Kimi Linear prepared row/source evidence

Kimi Linear supplies an architecture-issued ordinary causal companion for its
actual target-group paths and shared readout hooks. The cold conformance fixture
prepares genuine artifact sources and discovery, then binds every one of the
eighteen declared points to that exact retained path owner. Full and Preview(0)
selections preserve original source identity and physical readout requirements;
equal but independently prepared owners reject with `Identity`, undeclared
internal component hooks reject with `Undeclared`, and insufficient readout demand
rejects with `Readout`. Six dense/routed fixed/compressed/mixed source cases assert
that binding performs no physical source reads. Preview(0) does not erase the
existing generated-value policy or imply a host-accounting exemption.

The paired equation cases collect actual original and effective callbacks from
prepared traversal and actual readout execution; final model.logits uses the
existing enclosing publication callback. They do not synthesize effective values.
All mutable state, including the KDA q/k/v histories and matrix and MLA's latent
and unrotated positional channels, is compared, and nonzero checks use actual
loaded parameter slots rather than seeded cache buffers. Kernel-one state has
only the recurrent role. No native collector, original admission, managed gateway
or source-publication gate is opened by this family declaration. Seven tests are
staged for central execution; released-checkpoint and native evidence is unchanged.


Original prepared selected fixed/compressed owners now supply their actual
ordinary row semantics for LFM2, Nemotron-H, Qwen hybrid and dense DeepSeek V3.
The same prepared path owner binds discovery, exact capture source and physical
readout: body hooks preserve the requested demand; readout inputs/vocabulary
hooks require Sequence, including Preview(0). Internal hook availability is a
separate family fact; Kimi's fixed/compressed row equivalence is now an explicit
adapter opt-in contingent on central validation of direct and selected tests. Declaration
Vec/String construction remains under original loading/preparation; retained
path capacity uses the existing actual-owner measurement. This adds no per-model
control field, new hold, native publication or capture activation.

### LFM2 NoState/profile consistency

The width-one LFM2 equation now agrees with its declared NoState geometry through
the actual selected stateless and attention-only shells. It neither acquires an
absent Convolution role nor advances a nonexistent fixed frontier. Both LFM2
forward owners derive masks from the actual attention policy inside the local
invocation segment, preserving source/path identity and original physical
readout requirements. No new public semantic fact or native grant is added.

New neutral tests use actual prepared checkpoint/session and partition ownership.
They compare nonzero callbacks and cached predictions across leading/trailing
stateless cuts; existing direct tests assert zero NoState position and compare
all persistent KV/convolution payloads. Wider-kernel missing-role assertions remain
unchanged. These source-only regressions await central execution and make no
new native, released-checkpoint or managed-admission claim.

### Gemma4 direct causal target selection

Gemma4 now declares the ordinary target's actual `model.language_model.layers.*` boundaries and shared embedding/readout paths, with the existing cold causal-row companion. Four text layers contribute 26 declarations; the eight-layer boundary fixture contributes 42. Prepared-source tests bind every declared path under Full and Preview(0), reject independent source/path identities and down-binding of physical readout demand, and verify no additional checkpoint reads. Internal attention/media/merge paths are not inferred from axis names.

The staged direct fixtures compare all declared real callbacks and complete state for dense/sparse, separate/key-as-value, per-layer-input, tied-head and supported partial-rotary variations. Eighteen prepared resident/host/disk TP/PP worlds exercise actual shared-KV receivers and early/shared pipeline cuts. The existing partition session observes complete model.logits and internal readout rows under Sequence demand, then selects the public last-position return. These boundary trials compare every observed row and separately verify the selected public output; they do not claim partition capture admission. Focused rejection tests cover missing publication and a consumer history preceding its submission. Receiver-frontier rejection remains enforced by the existing production check, with numerical receiver ownership covered at every successful boundary; this package adds no direct injected receiver-error test. No source-staged result is an executed native or released-checkpoint result.

### Gemma4 sliding mask regression scope

The Gemma4 history correction leaves the ordinary group-2 causal declarations and all real observation callbacks unchanged. Existing full-versus-chunk comparisons still cover every declared row, complete persistent publisher/receiver state and invocation-local attention history, with exact same-next-decode equality. No observation is synthesized and no state assertion is removed.

New focused tests distinguish contiguous native-style retained tails from cache-owned shared-consumer requests; an uncached publisher/consumer case verifies causal prefix equivalence and window exclusion while honoring explicit rotary embeddings. Explicit caller masks remain outside the ordinary causal declaration but keep their existing execution behavior. Prepared workspace traversal is checked against the actual sliding descriptor and the unchanged full-attention descriptor. No public capture, conditional/media, speculative or partition admission is enabled by this correction.


The selected Kimi successor binds all eighteen actual outer/effective/readout
rows to the session's original prepared source and path owner. Full and
Preview(0) retain exact source identity; equal-content foreign capture sources
reject. BeforeReadout points preserve requested StateOnly/LastPosition output,
while readout-input and vocabulary points require physical Sequence. Actual
callbacks, including effective companions, are collected from shared traversal;
no alias is manufactured. Complete prefix/continuation/decode state and cached
outputs compare across original prepared resident/host/disk executions.

Routed and MTP rejection tests preserve the existing selected constructor policy.
The new architecture fact does not provide native inventory, original funding,
partition binding or managed activation. Four selected tests/168 prepared-session
constructions are staged for central execution, with the direct71 proof an
explicit prerequisite rather than a reported result.


Inkling's ordinary target declaration binds its actual group-2 outer/effective
paths and shared readout points, plus the real read-only `readout.scaled` source.
Full and Preview(0) selections retain exact prepared capture/path identity;
equal-content foreign owners reject. BeforeReadout points preserve the original
requested output, while coupled readout/vocabulary points require Sequence.
Internal attention observations, media assembly and prediction points are not
authorized by these outer-row declarations. No effective callback is fabricated.

The source package stages eight tests covering nonzero complete state and real
callbacks, kernel-one state geometry, original-source binding, direct/prepared
parity, body demand and TP/PP/residency behavior. The cold binding fixture retains
its exact admitted four-position origin; numerical continuation uses a separate
real two-token prefix. The latter's full per-call state proof uses the existing
neutral typed-session snapshot seam, not a new inspection/funding authority.
Generic cache-owned relative attention is not silently substituted for the
resident/sliding fixture. No native opening, grant, gate or controller activation
is added, and no execution result is asserted by this staged documentation.


The Qwen shared width-one correction preserves the existing direct and selected
causal row contracts: only the absent convolution history slot is omitted, while
the recurrent matrix stays populated and advances. Nonzero neutral fixtures add
all-row direct and selected comparisons, original prepared-source/readout checks,
body-only readout, cached continuations and rank-local state comparisons. The
conditional and MTP regressions exercise their real owners without adding new
declarations; media, conditional row binding and native capture activation remain
separate joins. Staged tests have not yet been executed.


Zero mRoPE sections preserve the original three-axis Qwen3-VL coordinate mapping:
secondary sections with no frequencies perform no overwrite, while the first
axis remains the fallback even when its declared section is zero. The full
positive rotary width still determines native frequency and cold host storage.
Explicit distinct-axis scalar oracles, tensor parity, HF/GGUF rejection cases,
cold allocation comparisons, and a gated native measurement test cover this
validation correction. Source-only preparation does not establish conditional
row causality, native readiness, or capture gate activation.


Conditional Qwen causal selection is restricted to the original ordinary-text
origin and exact prepared path owner. All 18 real outer hooks for a two-target-unit
fixture are compared, including effective companions and readout extras.
BeforeReadout selections retain selective readout; readout rows require physical
Sequence output. Zero Preview still binds the same declaration. Foreign source
owners, live foreign runtime tokens, invalidated tokens, changed cached origins
and invocation-only admission reject. Media-required catalog points, modality
merge and prediction units remain outside this ordinary-text declaration.
Complete state checks include KV values/offsets/windows, convolution and recurrent
roles, fixed cursors and the VL Int32 PositionDelta. Kernel-one history and zero
mRoPE section behavior rely on the separately staged correction packages;
central numerical validation remains pending.

### Cold native record fact for future original-span activation

The MLX accepted-record observation distinguishes an empty live child scope
from a record awaiting its existing terminal transition, including records in
durable descendants after an outer scope was sealed. It reads no tensor or
collector inventory and performs no progress, retirement or housekeeping.
Quiescence does not establish that retained allocations are gone or that all
opening sources were collected. Sticky failure/blocked and unavailable states
cannot authorize reuse.

This native-only prerequisite does not wire the remaining-span account marker,
change capture selection or budgets, supply a second grant, or turn a canonical
chunk ticket into general native completion. Original scope association, exact
opening publication, actual remaining-account checks and scope-control quote
composition remain separate obligations of the eventual joined caller.

## Moshi selected frame observations

The selected Moshi numerical frame fixture records the actual `temporal.input`,
two temporal layer outputs, `text_linear.logits`, every executed depth-slice
logit, and separate final `model.logits` callback. Nonzero batch-one/batch-two
values and a changed-source logit counterexample supplement exact comparisons
between ordinary and bounded scheduler driving. Fully forced diagnostics-off
depth tails retain their existing skip behavior; diagnostics-on tails execute
and expose their actual depth logits. Their temporal state, decisions and delayed
history remain equal, while skipped depth caches stay empty after each reset.

This is canonical frame conformance with the existing scalar numerical backend,
not an independent released-model oracle or an ordinary packed-prompt causal
declaration. Original bounded capture accounting, callback intervention, exact
native pending-resource lifetime, true bounded weight residency and distributed
frame proof remain outside this first unit. Central execution is required before
recording these added tests as validated.

### Selected Moshi residency fixture scope

Three additional selected Moshi frame tests compare actual resident, host-payload and disk-recipe construction in 33 numerical sessions. They preserve complete temporal/depth snapshots, delayed coordinate history, observed logits, public outputs and terminal failure state. A single live unit and exact eviction order are checked; host execution adds no source reads and every disk reload increases actual physical read counters. Pinned static modules remain bound. The earlier resident-only scope is extended for these scalar fixtures; TP/PP, PersonaPlex, released-checkpoint/native proof and original finite frame admission remain separate unfinished work.
