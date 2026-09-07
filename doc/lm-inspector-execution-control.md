# LM Inspector: ordinary execution control

Keep native models and sessions on one owning worker. Send pause/cancel requests
through `GenerationControlHandle`; send step, resume, snapshot, restore and branch
commands to that worker. The session exclusively borrows its `LocalModel`, and
native handles do not need to become `Send` or `Sync`.

Use `eredu::api::LocalModel` with the selected backend features. Backend-generic
consumers use `LoadedModel<B>` and the same control API. The complete runnable
example is:

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

Prepare a normal `ChatTemplateRequest`, then call `prepare_observed_chat` with a
bounded `CapturePlan`, or `prepare_intervened_chat` with a validated intervention
request. `CapturePlan::none()` selects ordinary delivery. Pass the result to
`start_controlled_chat(prepared, stops, control, callback)`.

For an unrecognized template, explicitly select
`start_controlled_text(prepared, stops, control, callback)`. Both `LoadedModel<B>`
and `LocalModel` expose it with the same arguments and session return type as their
chat entry point. `start_controlled_chat` remains strict.

```rust
// model: LocalModel (the same selection works with LoadedModel<B>)
let chat = model.prepare_chat(request)?;
let semantic = matches!(chat.semantic_support(), SemanticSupport::Supported);
// A UI can show chat.text_generation_support().unsupported_reason() before starting.
let prepared = match intervention {
    Some(plan) => model.prepare_intervened_chat(&chat, settings, capture, plan, trace_limits)?,
    None => model.prepare_observed_chat(&chat, settings, capture, trace_limits)?,
};
let mut session = if semantic {
    model.start_controlled_chat(prepared, &stops, control, &mut emit)?
} else {
    model.start_controlled_text(prepared, &stops, control, &mut emit)?
};
session.step(&mut emit)?; // Existing session handling stays unchanged.
```

Import `SemanticSupport` from `eredu::runtime::chat`. The
[`LoadedModel::start_controlled_text` rustdoc](../eredu/src/api/control.rs) contains
a complete generic example. Capture, intervention, trace and control types retain
their existing public paths.

Text mode consumes the prepared prompt IDs exactly, including the rendered
generation prefix. It retains checkpoint sampling defaults, overrides, seed and
EOS, and emits the existing token records plus `SemanticEvent::TextDelta` and
`Finished`. It decodes incrementally, skips special tokens and matches caller
stops, without interpreting tool calls, reasoning, or profile-specific stops.
Reasoning and tool-like ordinary text can therefore appear literally. Tokenizer
validity still excludes holes and padded logits positions before sampling and
forced-token admission; grammar constraints in semantic mode further restrict
that same domain.

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
configuration. Configure `enable_snapshots(SnapshotLimits { ... })` once before
retaining state. This rejects unknown native, grammar or semantic costs. Creating
a snapshot additionally establishes whether complete continuation growth is known
for branching; `capabilities().fork` then reports that result. Every fork still
validates its own plans and budgets.

Complete facade snapshots support text mode and the forbidden-tool constraint
mode on a supported tool profile. The runnable example declares a tool and sets
`ToolChoice::None`. Active and automatic llguidance constraints have independent
copying but no complete storage estimate and reject snapshots. A no-tools request
may still use an active semantic grammar. Show the returned support reason in the
UI; do not interpret native KV-copy support as full generation-snapshot support.
Text snapshots retain partial Unicode, caller-stop lookbehind, pending forced
choices and decoder state. Snapshot compatibility includes text versus semantic
mode. Native support and bounded retention/copying admission still apply.

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

1. Save the `LocalGenerationSnapshot` handle and its serializable metadata.
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
state, partitioned/speculative/media/realtime control, persistent snapshots,
cross-process/backend restoration and layer-level pauses are unsupported.
