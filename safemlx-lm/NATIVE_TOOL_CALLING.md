# Native tool calling

SafeMLX exposes native tool calling through `LoadedModel::prepare_chat` and the
`generate_prepared_chat*` methods. Applications provide ordinary OpenAI-shaped
JSON Schema function definitions and receive protocol-neutral
`SemanticEvent`s. They do not construct grammars, inspect tokenizer regexes, or
handle a checkpoint's wire parser.

The complete executable example is
[`examples/native_tool_calling.rs`](examples/native_tool_calling.rs). It shows
capability detection, `PreparedChat`, ordinary structured generation,
checkpoint-embedded MTP, an external assistant on CPU with a GPU target,
scheduler options, event accumulation, and finish reasons.

## Capability gating

Stable protocol recognizers verify tokenizer structure and render bounded
synthetic conversations after named-template selection. Reasoning,
visible-text, tool-output parsing, tool-input rendering, and constrained tool
generation are independent capabilities. Architecture, repository name, model
ID, converter, whole-template hash, and protocol-looking source substrings are
never fallback keys.

Recognition fails closed when required tokens are not distinct atomic specials
or when rendered behavior changes. Comments, Jinja refactoring, and unrelated
branches do not affect support when the observed protocol remains equivalent.

```rust
let prepared = model.prepare_chat(request)?;
match prepared.native_tool_support() {
    NativeToolSupport::Supported => {
        eprintln!("profile: {:?}", prepared.format_profile_identity());
    }
    NativeToolSupport::Unsupported { reason } => {
        return Err(reason.clone().into());
    }
}
```

`PreparedChat` owns the rendered prompt and the private executable response
plan. Its public accessors expose diagnostics and tokenizer-level stopping
metadata, but not llguidance values, dialect state, parser state, structural
spellings, or architecture-specific regular expressions.

## Request schemas and generation

Every `prepare_chat` call validates the complete function envelope, function
name, description, and supported JSON Schema subset before rendering. Function
names are 1–64 bytes of ASCII letters, digits, `_`, or `-`; duplicates are
rejected. Parameters must resolve to an object schema. Local non-recursive
`$ref`s, nested objects and arrays, required/optional properties, enums, and
the JSON scalar types are supported. Unsupported keywords, schema
compositions, external or recursive references, malformed bounds, and
additional undeclared fields fail before model execution.

Tokenizer vocabulary analysis and its token trie are built once for each
`LoadedModel`. The request's tools and call limits are compiled on every
`prepare_chat`, so two requests on the same loaded model can safely use
different schemas without rebuilding tokenizer-wide data or sharing mutable
grammar/parser state.

Use one of these cohesive calls:

- `generate_prepared_chat` for ordinary constrained generation.
- `generate_prepared_chat_mtp` for an external assistant.
- `generate_prepared_chat_embedded_mtp` for checkpoint-embedded MTP heads.
- `generate_prepared_chat_mtp_batch` for independent requests interleaved by
  one fair external-assistant scheduler.
- `generate_prepared_chat_embedded_mtp_batch` for independent requests using
  checkpoint-embedded MTP heads.

All paths commit through the same constraint, tokenizer-aware byte decoder,
stop matcher, and semantic event pipeline. Canonical MTP, optimistic lookahead
MTP, and scheduler interleaving may change speculative work and statistics,
but not committed token IDs, semantic events, event order, or finish reason.
Draft and optimistic branches never publish events.

`MtpExecutionStreams::new(target, draft)` supports CPU drafting with a GPU
target, different GPUs, and two distinct streams on the same GPU. A same-GPU
split is an explicit experiment: it enables eligible lookahead but can contend
for the same compute and memory bandwidth. Same-device target/draft handoffs
are backend-ordered with completion events and do not block the host;
cross-device handoffs synchronize before copying. `MtpSchedulerOptions` bounds
in-flight verification and optimistic branches; `.with_lookahead(false)`
provides the canonical equivalence baseline.

## Semantic events and finish reasons

Callbacks receive reasoning deltas, visible text deltas, canonical tool-call
start/argument/end events, and exactly one final event. Accumulate argument
fragments by `index`; do not assume one fragment per call.

```rust
let mut events = Vec::new();
let output = model.generate_prepared_chat(PreparedChatGenerationRequest {
    input: PreparedChatInput::rendered_prompt(&prepared),
    cache: &mut cache,
    sampling_policy: DefaultSampler,
    settings: PreparedChatGenerationSettings::default(),
    caller_stop_sequences: &[],
    stream,
    cancellation: GenerationCancellationToken::new(),
    on_event: |event| events.push(event),
})?;
assert_eq!(
    events.last(),
    Some(&SemanticEvent::Finished {
        reason: output.finish_reason
    })
);
```

Every prepared-chat generator takes an explicit `PreparedChatInput`. Use
`rendered_prompt` for text-only chat. For multimodal chat, replace the complete
checkpoint-rendered media placeholder with processed media and bind that input
to the same prepared chat:

```rust,ignore
let multimodal = model.prepare_chat_input(
    &prepared,
    &[ChatMediaBinding::new(
        "<|vision_start|><|image_pad|><|vision_end|>",
        MediaInput::image_rgb8(image),
    )],
)?;
let output = model.generate_prepared_chat(PreparedChatGenerationRequest {
    input: PreparedChatInput::prepared_model_input(&prepared, &multimodal),
    // cache, sampling_policy, settings, caller_stop_sequences, stream,
    // cancellation, on_event
})?;
```

The placeholder must be the complete envelope emitted by the selected
checkpoint template. SafeMLX validates placeholder count and order, then lets
the architecture processor insert its own boundary tokens and media tensor.

## Cooperative cancellation and backpressure

Every prepared-chat request takes a cloneable `GenerationCancellationToken`;
every MTP batch lane takes its own token. Event callbacks remain synchronous,
so blocking on a bounded channel still pauses generation. A downstream closure
can cancel directly from the callback:

```rust,ignore
let cancellation = GenerationCancellationToken::new();
let cancel_on_close = cancellation.clone();
let output = model.generate_prepared_chat(PreparedChatGenerationRequest {
    input: PreparedChatInput::rendered_prompt(&prepared),
    cache: &mut cache,
    sampling_policy: DefaultSampler,
    settings: PreparedChatGenerationSettings::default(),
    caller_stop_sequences: &[],
    stream,
    cancellation,
    on_event: move |event| {
        // SyncSender::send blocks when the bounded channel is full.
        if event_sender.send(event).is_err() {
            cancel_on_close.cancel();
        }
    },
})?;
```

Cancellation produces `FinishReason::Cancelled` and a final
`SemanticEvent::Finished { reason: Cancelled }`. Returned IDs are exactly the
committed prefix. Cancellation does not flush stop-sequence lookbehind or
protocol-parser buffers and does not synthesize `ToolCallEnd` for an incomplete
call. For MTP, a request with target verification already in flight first
resolves that transaction to the scheduler's safe cache boundary; cancellation
of one lane does not cancel another lane.

Terminal precedence on one committed token is decoded stop sequence, grammar
completion, checkpoint EOS, then maximum tokens. The corresponding
`FinishReason` is `StopSequence`, `GrammarComplete`, `Eos`, `MaxTokens`, or
`Cancelled` when cooperative cancellation wins before a normal terminal
condition.
Incomplete calls never emit `ToolCallEnd`; malformed protocol returns an error.
Profile and caller stop sequences share one overlap-aware matcher and never
leak into visible text.

## Reusable wire formats

Runtime profiles bind recognized protocols to reusable wire-format
implementations rather than model architectures or template artifacts:

- Declarative JSON objects, JSON lists, XML-wrapped JSON, fixed JSON
  envelopes, named JSON arguments, and structural-token JSON objects.
- Gemma structural channels and tool envelopes, selected from tokenizer and
  rendered behavior evidence.
- Inkling message frames with separate reasoning and visible-text channels;
  its native tool-constraint surface remains fail-closed.
- OpenAI Harmony channels and function recipients for GPT-OSS.
- LFM2/LFM2.5 Python-call lists with canonical JSON argument events.

Recognized identities are stable protocol versions:
`xml-tools.v1`, `qwen.xml-tools.reasoning.v1`,
`mistral.json-list-tools.v1`, `mistral.json-list-tools.compact.v1`,
`llama.json-tools.v1`, `llama.python-channel-tools.v1`,
`nemotron.json-list-tools.v1`, `nemotron.json-list-tools.reasoning.v1`,
`gemma.channels.v1`, `inkling.messages.v1`, `harmony.channels.v1`,
`lfm2.python-tools.v1`,
`deepseek.structural-json-tools.v1`, and
`deepseek.structural-json-tools.v2`.

Several unrelated architectures intentionally share one declarative
implementation when their byte-level response protocol is identical. A custom
dialect is reserved for a surface that cannot be expressed declaratively, such
as Harmony channel transitions or LFM2's Python literals. The private dialect
produces the same public `SemanticEvent` vocabulary as every other format.

## Raw and unconstrained generation

Raw prompts and unconstrained sampling remain intentional APIs. The CLI's
`--raw` mode and the existing `generate_input*` iterators bypass
`PreparedChat`; they do not claim native-tool safety or semantic tool events.
Use them for text completion and regression/parity work, not as a fallback
after a native-tool capability failure.
