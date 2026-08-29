# Native tool calling

Eredu native tool calling accepts OpenAI-shaped function definitions and
emits protocol-neutral semantic events. Applications do not need to build a
grammar, inspect tokenizer internals, or parse a checkpoint-specific wire
format.

The sole model-level chat-rendering entry point is `LocalModel::prepare_chat`,
including for chats without tools. It renders the selected checkpoint template,
validates the request, and returns the prompt and its generation metadata.
Applications can encode `PreparedChat::rendered_prompt()` for raw token
generation or pass the prepared chat to one of the `generate_prepared_chat*`
methods. Backend authors can use the generic `LoadedModel<B>` surface instead.
The complete semantic-generation example is
[`eredu/examples/native_tool_calling.rs`](../eredu/examples/native_tool_calling.rs).

## Capability gating

Tool support is behavioral and fail-closed. Eredu selects a template, checks
the tokenizer's required special tokens, renders bounded conversations, and
independently verifies:

- tool-definition rendering;
- reasoning and visible-text framing;
- tool-call parsing;
- argument constraints; and
- stop behavior.

A repository name, model ID, architecture, template hash, or protocol-looking
substring does not grant support. An ordinary chat template can remain usable
for text even when native tool generation is unavailable.

`PreparedChat::native_tool_support()` exposes the result and the recognized
format identity. `PreparedChat` owns the rendered prompt and private executable
response plan; callers do not receive dialect parser or grammar state.

## Function schemas

Every preparation validates the entire tool envelope before model execution.
Function names must be unique, 1–64 bytes long, and contain ASCII letters,
digits, `_`, or `-`. Parameters must resolve to an object schema.

The supported subset includes:

- nested objects and arrays;
- required and optional properties;
- string, number, integer, boolean, and null values;
- enums; and
- local, non-recursive `$ref` values.

External or recursive references, unsupported composition keywords, malformed
bounds, and undeclared additional fields fail during preparation. Tool schemas
are compiled per request, so independent requests on one loaded model can use
different tools without sharing mutable parser state.

## Generation APIs

Choose one cohesive generation call:

- `generate_prepared_chat` for ordinary constrained generation;
- `generate_prepared_chat_speculative` with the opaque `LocalDrafting` loaded from the
  same execution plan; or
- `generate_prepared_chat_speculative_batch` for independently scheduled requests.

The batch API and explicit `SpeculativeDraft` variants are backend-generic
infrastructure. The selected application facade keeps the native drafter and
its failures private.

All paths use the same tokenizer-aware byte decoder, stop matcher, constraint
engine, and semantic event pipeline. Speculation and scheduler interleaving can
change uncommitted work and diagnostics, but not committed tokens, event order,
or finish reason.

Multimodal requests bind processed media to the complete placeholder emitted by
the selected checkpoint template. Eredu validates placeholder count and
order, then lets the architecture processor insert its own boundary tokens and
media tensors. Applications should not manually insert architecture media token
IDs into rendered text.

## Semantic events

Callbacks receive:

- reasoning deltas;
- visible-text deltas;
- tool-call start events;
- argument fragments keyed by call index;
- tool-call end events; and
- exactly one finished event.

Argument fragments may split at arbitrary token boundaries; accumulate them by
call index. An incomplete or malformed call never receives a synthetic end
event.

```rust,ignore
let mut events = Vec::new();
let output = model.generate_prepared_chat(LocalPreparedChatGenerationRequest {
    input: LocalPreparedChatInput::rendered_prompt(&prepared),
    settings: PreparedChatGenerationSettings::default(),
    caller_stop_sequences: &[],
    cancellation: GenerationCancellationToken::new(),
    on_event: |event| events.push(event),
})?;
```

Finish reasons distinguish decoded stop sequences, grammar completion,
checkpoint EOS, maximum tokens, and cancellation. When several conditions
occur on one committed token, precedence is stop sequence, grammar completion,
EOS, then maximum tokens.

## Cancellation and backpressure

Each request has a cloneable `GenerationCancellationToken`; each batch lane has
its own token. Event callbacks are synchronous, so writing to a bounded channel
naturally applies backpressure. If the downstream consumer closes, the callback
can cancel its request.

Cancellation emits a final cancelled event and returns exactly the committed
token prefix. It does not flush hidden stop-sequence lookbehind or partial
parser state, and it does not publish events from speculative branches. A
verification already submitted to a backend is retained until its exact safe
cache boundary, as described in [Cancellation and bounded
execution](cancellation.md).

## Recognized wire-format families

Reusable profiles cover declarative JSON object/list formats, XML-wrapped JSON,
named JSON arguments, Qwen3.6/3.8 tagged parameters, structural-token JSON,
Gemma structural channels, Inkling message frames, OpenAI Harmony channels,
and LFM2-style Python call lists. Multiple model architectures can share a
profile when their observable byte protocol is equivalent.

Qwen3.6 and Qwen3.8 calls contain a tagged function name and one repeated
`<parameter=name>` block per top-level argument. String values are raw text;
all other schema types are JSON encoded. Eredu retains the request's resolved
schemas in the parser plan so ambiguous text such as `true` becomes either the
JSON string `"true"` or boolean `true` according to the selected parameter.
Arguments are emitted to applications as one canonical JSON object. Unknown or
duplicate parameters, wrong types, unsafe names, incomplete tags, and raw
strings containing the unescaped closing delimiter fail closed. Historical
tool calls must use mapping-valued arguments; serialized argument strings are
not accepted for these templates.

Regardless of wire format, the public result is the same semantic event
vocabulary. Unsupported behavior returns a capability error rather than
falling back to unconstrained text.

## Raw generation

Raw prompts and `generate_input*` iterators intentionally bypass prepared chat.
Use them for plain completions and parity work. They do not provide native-tool
constraints or semantic tool events and should not be used as a fallback after
a tool-capability failure.

Maintainers extending the recognized protocol set should follow [Adding a
native tool protocol](tool-protocol-development.md).
