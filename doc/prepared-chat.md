# Source-funded semantic chat

`LoadedModel::prepare_chat` compiles a request against retained tokenizer and chat
template sources. `start_prepared_chat` consumes that result with an explicit
managed-memory ceiling. It needs ordinary generation support only. The session's
`advance` and `run` methods use the same committed-token and semantic state.

The executable example is `eredu/examples/prepared_chat_generate.rs`. It writes
one JSON semantic event per line, including text, reasoning, incremental tool
arguments and terminal events. Output callbacks receive committed events only.
The example never executes a requested tool.

Create a request file such as:

```json
{
  "messages": [
    {"role": "system", "content": "You are a sensor logging assistant. Use the reading function to save the sensor value the user gives you."},
    {"role": "user", "content": "The sensor value is 17. Please record this reading."}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "reading",
      "description": "Record the requested integer value.",
      "parameters": {
        "type": "object",
        "properties": {"value": {"type": "integer", "enum": [17]}},
        "required": ["value"],
        "additionalProperties": false
      }
    }
  }],
  "tool_choice": "required",
  "max_tokens": 48,
  "prefill_chunk_positions": 128,
  "seed": 0,
  "enable_thinking": false
}
```

Run it against a supported checkpoint, using a capacity suitable for that model
and request. Source compilation consumes the complete tokenizer and selected
template retained by the loaded model:

```sh
cargo run -p eredu --no-default-features --features mlx,metal \
  --example prepared_chat_generate -- CHECKPOINT REQUEST.json 68719476736 metal:0
```

This request passed with both required and automatic tool choice against the
pinned Qwen3.5-0.8B checkpoint at that 64 GiB ceiling; see
[the released-tool record](bounded-followup-released-tools.md) for its revision,
commands and exact events. Other prompts and checkpoints require their own
validation. Source compilation and generation retain their actual funding accounts;
errors retain the accounts needed by their payloads. This ceiling describes
framework-managed allocations. Application input buffers, event copies, output
I/O, allocator caches and unrelated process memory need their own accounting.

To integrate directly, retain the tokenizer/template sources, prepare the chat,
and construct `PreparedChatRequest::new(&chat, settings)`. Set
`settings.inference.managed_memory_capacity_bytes` to the same capacity used to
prepare the chat. Pass a shared cancellation token to preparation and advancement.
A request already cancelled returns `None` before generation starts.

For media, run the appropriate host processor and present its ordered text and
media slots as `HostInputPart` values. Call
`model.prepare_chat_input(&chat, &parts, &cancellation)` once, then move the
returned `OriginalModelInput` into
`request.input = PreparedChatPrompt::Media(input)`. The retained
source binds the input to the chat's tokenizer, template, semantic coordinates,
model execution and preparation account. Foreign or mismatched input is refused.
Chunked prefill consumes that prepared input without reopening its artifacts or
re-encoding the media for each chunk.

For an already tokenized text prefix, set
`request.input = PreparedChatPrompt::TokenIds(&ids)`. The same preparation worker
validates each ID against the retained tokenizer vocabulary and copies the exact
prefix into admitted storage. It does not decode and re-encode that prefix.
The default `PreparedChatPrompt::Rendered` uses the prepared chat's rendered text.

The example accepts those caller-owned arrays in optional `prepared_parts` JSON.
Each part contains `modality`, `kind`, a `payload` with `shape` and typed `values`,
optional `metadata` pairs and `extents`. Values use exactly one key: `u32`, `i32`,
`f32` or `bool`. Shapes, metadata and ordering must come from the selected model's
processor; the example does not invent family-specific geometry. The architecture
validates these inputs before native submission.

Raw capture and intervention declarations can be borrowed in `request.capture`
and `request.intervention`. They are admitted after the paid tokenizer or retained
media layout establishes the actual prompt geometry. Existing original sources
can instead be supplied in `request.options`; overlapping declarations are refused.

Speculation consumes the same preparation through `PreparedChatSpeculativeRequest`.
Supply `chat`, `input`, `settings`, `output_mode`, `skip_special_tokens`, an actual
`drafting` selection, speculative `options`, caller stops, cancellation and the
committed-event callback. `generate_prepared_chat_speculative` and
`with_controlled_prepared_chat_speculative` share this request and preparation.
The latter adds controlled observation/snapshot options and a scoped driver.
The request takes its tokenizer from the prepared chat. `chat.capacity()` reports
the authenticated request ceiling. The comparison example
`eredu/examples/gemma4_speculative_generate.rs` prepares one chat in its consuming
model and uses it for ordinary and speculative generation.

Admitted capture/intervention plans go in `PreparedChatRequest::options`.
`with_capture_observer` observes the shared committed delivery before semantic
events. `snapshot`, `restore_prepared_chat` and `fork_prepared_chat` use the same
session and retain cumulative copy and observation spending. Capture budgets and
snapshot budgets remain distinct from the request's allocation ceiling.
An active session can also call `restore_snapshot` or `fork_snapshot` while
retaining its exclusive model borrow. The latter returns an inactive branch;
`exchange` swaps its complete native, parser, cursor and sampling state
with the active session at a completed boundary. A branch from a different run
is rejected.

Manual advancement also permits `force_next_token(id)`, `clear_forced_token()`
and `pending_forced_token()`. A choice must satisfy the canonical vocabulary and
current grammar. It restricts the next ordinary sampling decision without
advancing model state or RNG during preparation. The next `advance` performs the
usual sampling and commitment; a rejected choice preserves the pending state.

`run_until_paused(&control, emit)` returns the same session at a requested
completed boundary; `resume(&control, emit)` acknowledges the request and continues
it. `status()`, `next_prediction()` and `timing()` describe that same run. Manual
`advance` explicitly advances one decision, including while a pause is pending.
Cancellation takes precedence over pause and preserves ordinary termination.

`preparation_report()` borrows the run's retained admission report; it describes
the original preparation rather than current shared-pool usage. At a healthy
completed boundary, `sampling_state()` remains readable after termination and
after restoring or exchanging a terminal branch. Sampling changes are rejected
after termination, and further advancement emits no duplicate output.

Validation is tracked in `bounded-followup.md`, `execution-control.md` and
`validation/prepared-chat-native-2026-09-18.json`. Portable semantic tests pass.
The selected-executor optimized native artifact passes all seven public cases.
The ordinary tool/capture fixture covers partial and terminal snapshot, restore,
fork and exchange. The authenticated image/tool snapshot fixture passes Required and Auto at
the unchanged 8 GiB limit, including pending branches and partial tool events.
Independent speculative image/tool parity also passes after preserving the
completed ingress source through every prefill chunk. Two-rank tensor and
two-rank pipeline media both pass Required
and Auto through ordinary, manual and recorded execution. Combined TP2/PP2
also passes those policies and execution modes at the same 8 GiB capacity,
including successive requests after completed-session retirement. The combined
image/audio fixture also passes Required and Auto through ordinary, manual and
recorded sessions, using real encoders and deterministic decoder transitions.
TP2 now quotes its actual executor's group submissions while retaining the
resident registry, closing the earlier admission regression at the same cap.
Recorded child branches accept sampling/reseed options while preserving the
parent and cumulative copy spending; direct prepared-chat restore preserves the
saved sampler. Exact scope is recorded in
[public conformance](bounded-followup-public-conformance.md).
The final public build optimizes six Rust packages at level 2 within the
development profile, preserving debug assertions, overflow checks and native
development guards. It uses an explicit 64 MiB Rust test-thread stack. The earlier
default 2 MiB debug test-stack diagnostic overflows in a finite startup
chain; these passes do not establish default-stack execution.
These synthetic fixtures do not establish released-checkpoint numerical accuracy
or application memory and allocator-cache policy.

The pinned released-checkpoint example also passes Required and Auto with a
processed image and an ordinary tool call. The reproducible input generator,
commands and exact scope are in [released image/tool validation](bounded-followup-released-tools.md#authenticated-image-and-tools).

No downstream checkout was used for this work. Its call site and selected
configuration were not inspected; the pinned checkpoint above validates the
framework example only.

Application integration still needs its own memory checks and allocator-cache
policy. Include caller-owned request/media buffers, copied events and retained
application results in those checks; the framework pool covers its admitted
owners, not total process memory. Configure native allocator caches separately
and measure loading, prefill, cached decode, reset and final retirement with the
application's actual retention policy. Wait for native completion before treating
retirement as available capacity. The public example demonstrates the API handoff;
external application call sites and their memory/cache changes are unverified.
