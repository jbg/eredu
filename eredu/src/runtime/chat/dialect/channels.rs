//! Facade projection of the actual dialect onto the shared literal worker.
use super::DeclarativeDialectSpec;
use eredu_text::semantic_channels::{ChannelProgram, DelimitedChannel};
use std::mem::{size_of, size_of_val};
pub(crate) type State = eredu_text::semantic_channels::State<'static>;
pub(crate) type Next = eredu_text::semantic_channels::Next<'static>;
pub(crate) type Step = eredu_text::semantic_channels::Step<'static>;
/// Same ordinary delimiter priority: outer wrapper, call wrapper, JSON wrapper,
/// then an opening JSON object. No family-name or trigger-string inference.
pub(crate) fn tool_delimiter(spec: &DeclarativeDialectSpec) -> &'static str {
    if !spec.output.prefix.is_empty() {
        spec.output.prefix
    } else if !spec.call.prefix.is_empty() {
        spec.call.prefix
    } else {
        spec.json_function
            .map(|f| f.envelope.prefix)
            .filter(|p| !p.is_empty())
            .unwrap_or("{")
    }
}
pub(crate) fn program(spec: &DeclarativeDialectSpec) -> ChannelProgram<'static> {
    ChannelProgram {
        reasoning_channel: spec.reasoning_channel.map(|c| DelimitedChannel {
            prefix: c.prefix,
            suffix: c.suffix,
            prefix_in_prompt: c.prefix_in_prompt,
        }),
        text_channel: spec.text_channel.map(|c| DelimitedChannel {
            prefix: c.prefix,
            suffix: c.suffix,
            prefix_in_prompt: false,
        }),
        tool_delimiter: tool_delimiter(spec),
        tool_is_json: spec.output.prefix.is_empty()
            && spec.call.prefix.is_empty()
            && spec
                .json_function
                .is_some_and(|f| f.envelope.prefix.is_empty()),
    }
}
/// Projects the actual JSON declaration; other payload shapes retain their own
/// required producer rather than being relabelled as JSON objects.
pub(crate) fn json_tools(spec: &DeclarativeDialectSpec) -> Option<eredu_text::semantic_channels::JsonToolProgram<'static>> {
    use eredu_text::semantic_channels::{JsonEnvelope, JsonToolLayout, JsonToolProgram, JsonToolShape};
    let shape = match spec.payload_shape {
        super::DeclarativePayloadShape::JsonObject => JsonToolShape::Object,
        super::DeclarativePayloadShape::JsonList => JsonToolShape::List,
        _ => return None,
    };
    let function = spec.json_function?;
    let envelope = |e: super::ExactEnvelope| JsonEnvelope { prefix: e.prefix, suffix: e.suffix };
    Some(JsonToolProgram { output: envelope(spec.output), call: envelope(spec.call),
        function: envelope(function.envelope), fields: function.fields(), shape, separator: spec.call_separator,
        layout: match spec.parallel_layout { super::ParallelCallLayout::RepeatedEnvelopes => JsonToolLayout::RepeatedEnvelopes,
            super::ParallelCallLayout::SingleEnvelope => JsonToolLayout::SingleEnvelope },
    })
}
pub(crate) fn initial(spec: &DeclarativeDialectSpec) -> State {
    State::initial(&program(spec))
}
pub(crate) fn step(spec: &DeclarativeDialectSpec, state: State, pending: &str) -> Step {
    eredu_text::semantic_channels::step(&program(spec), state, pending)
}
pub(crate) fn control_bytes() -> Option<usize> {
    let parts = [
        eredu_text::semantic_channels::control_bytes()?,
        size_of::<ChannelProgram<'static>>(),
        eredu_text::json_fragments::JsonFieldNames::control_bytes()?,
        size_of::<Option<eredu_text::semantic_channels::JsonToolProgram<'static>>>(),
        size_of::<super::ExactEnvelope>(),
        size_of::<Option<DelimitedChannel<'static>>>(),
        size_of::<(&DeclarativeDialectSpec, State, &str)>(),
        size_of::<Step>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
