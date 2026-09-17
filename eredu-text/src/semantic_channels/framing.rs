//! Exact JSON call framing shared with the ordinary declarative parser.
use super::{JsonToolProgram, JsonToolShape};
use std::mem::{size_of, size_of_val};
/// Fixed branch of the selected JSON envelope worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonFrame {
    /// Before a call prefix or list opener.
    ToolStart,
    /// Before the function-object prefix.
    FunctionStart,
    /// Delegate bytes to the actual object parser.
    Payload,
    /// Before a list item or closing bracket.
    ListItem {
        /// Whether an empty tail is permitted.
        allow_end: bool,
    },
    /// Completed object before its protocol closure.
    AfterPayload,
    /// List function suffix has already been consumed; await separator or end.
    AfterFunctionSuffix,
    /// Between call envelopes or before output closure.
    AfterEnvelope,
    /// Before list-call/output suffixes.
    ToolSuffix,
    /// Return to the selected text/channel worker.
    Outside,
}
/// Identity of exact expected literals; no string allocation or authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonDelimiter {
    /// Call opener, followed by '[' for a list.
    CallPrefix,
    /// Function-object opener.
    FunctionPrefix,
    /// Function-object suffix and optional per-object call suffix.
    FunctionSuffix,
    /// Separator between list calls.
    Separator,
    /// List-call and output suffix.
    ToolSuffix,
}
/// Fixed framing refusal. The actual program remains owned by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum JsonFrameError {
    /// A selected literal differs from the available bytes.
    #[error("exact JSON call delimiter differs from input")]
    Delimiter(JsonDelimiter),
    /// A JSON list ended directly after a call separator.
    #[error("declarative JSON list cannot end after a call separator")]
    ListTail,
    /// Neither a separator nor output suffix matches.
    #[error("expected declarative call separator or output suffix")]
    Separator,
    /// Only framing branches can run through this worker.
    #[error("JSON framing worker received a payload or outside branch")]
    State,
}
/// A failed framing step preserves the prefix already consumed by the ordinary
/// worker before that exact failure (for example a function suffix before a
/// mismatching list separator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{cause}")]
pub struct JsonFrameFailure {
    /// Actual typed refusal.
    #[source]
    pub cause: JsonFrameError,
    /// Bytes already consumed before this refusal.
    pub consumed: usize,
}
impl From<JsonFrameError> for JsonFrameFailure {
    fn from(cause: JsonFrameError) -> Self {
        Self { cause, consumed: 0 }
    }
}
/// One framing action. Completion validation occurs after consume_before and
/// before consume_after, preserving the ordinary failure-prefix ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonFrameStep {
    /// Bytes discarded before any call-completion callback.
    pub consume_before: usize,
    /// Bytes discarded only after successful call completion.
    pub consume_after: usize,
    /// Whether the caller must complete the current tool call.
    pub end_call: bool,
    /// Next shared branch after successful callbacks.
    pub next: JsonFrame,
    /// Stop until more bytes arrive.
    pub wait: bool,
}
impl JsonDelimiter {
    /// Exact two-piece spelling in the ordinary constructor order.
    pub fn literals<'a>(self, program: JsonToolProgram<'a>) -> [&'a str; 2] {
        match self {
            Self::CallPrefix => [
                program.call.prefix,
                if program.shape == JsonToolShape::List {
                    "["
                } else {
                    ""
                },
            ],
            Self::FunctionPrefix => [program.function.prefix, ""],
            Self::FunctionSuffix => [
                program.function.suffix,
                if program.shape == JsonToolShape::Object {
                    program.call.suffix
                } else {
                    ""
                },
            ],
            Self::Separator => [program.separator, ""],
            Self::ToolSuffix => [program.call.suffix, program.output.suffix],
        }
    }
}
fn exact(
    program: JsonToolProgram<'_>,
    delimiter: JsonDelimiter,
    pending: &str,
) -> Result<Option<usize>, JsonFrameError> {
    let parts = delimiter.literals(program);
    let expected = parts[0].bytes().chain(parts[1].bytes());
    let length = parts[0]
        .len()
        .checked_add(parts[1].len())
        .ok_or(JsonFrameError::State)?;
    if pending
        .bytes()
        .zip(expected)
        .any(|(actual, expected)| actual != expected)
    {
        return Err(JsonFrameError::Delimiter(delimiter));
    }
    Ok((pending.len() >= length).then_some(length))
}
impl JsonFrame {
    /// Same object-prefix selection used after a call wrapper or list separator.
    pub fn call_payload(program: JsonToolProgram<'_>) -> Self {
        if program.function.prefix.is_empty() {
            Self::Payload
        } else {
            Self::FunctionStart
        }
    }
    /// Same handoff after the channel worker consumed the selected opener.
    pub fn after_channel(program: JsonToolProgram<'_>) -> Self {
        if !program.output.prefix.is_empty() {
            Self::ToolStart
        } else if !program.call.prefix.is_empty() {
            Self::call_payload(program)
        } else {
            Self::Payload
        }
    }
}
/// Runs the exact ordinary JSON envelope branches, without allocating strings.
pub fn json_frame_step(
    program: JsonToolProgram<'_>,
    state: JsonFrame,
    pending: &str,
) -> Result<JsonFrameStep, JsonFrameFailure> {
    let action = |consume_before, consume_after, end_call, next, wait| JsonFrameStep {
        consume_before,
        consume_after,
        end_call,
        next,
        wait,
    };
    let wait = |consumed| action(consumed, 0, false, state, true);
    match state {
        JsonFrame::ToolStart => {
            if program.shape == JsonToolShape::Object && !program.output.suffix.is_empty() {
                if pending.starts_with(program.output.suffix) {
                    return Ok(action(
                        program.output.suffix.len(),
                        0,
                        false,
                        JsonFrame::Outside,
                        false,
                    ));
                }
                if program.output.suffix.starts_with(pending) {
                    return Ok(wait(0));
                }
            }
            let Some(n) = exact(program, JsonDelimiter::CallPrefix, pending)? else {
                return Ok(wait(0));
            };
            Ok(action(
                n,
                0,
                false,
                if program.shape == JsonToolShape::List {
                    JsonFrame::ListItem { allow_end: true }
                } else {
                    JsonFrame::call_payload(program)
                },
                false,
            ))
        }
        JsonFrame::FunctionStart => {
            let Some(n) = exact(program, JsonDelimiter::FunctionPrefix, pending)? else {
                return Ok(wait(0));
            };
            Ok(action(n, 0, false, JsonFrame::Payload, false))
        }
        JsonFrame::ListItem { allow_end } => {
            if pending.is_empty() {
                return Ok(wait(0));
            }
            if pending.starts_with(']') {
                if !allow_end {
                    return Err(JsonFrameError::ListTail.into());
                }
                Ok(action(1, 0, false, JsonFrame::ToolSuffix, false))
            } else {
                Ok(action(0, 0, false, JsonFrame::call_payload(program), false))
            }
        }
        JsonFrame::AfterPayload => {
            let Some(n) = exact(program, JsonDelimiter::FunctionSuffix, pending)? else {
                return Ok(wait(0));
            };
            if program.shape == JsonToolShape::Object {
                let next = if program.output.prefix.is_empty() && program.separator.is_empty() {
                    JsonFrame::Outside
                } else {
                    JsonFrame::AfterEnvelope
                };
                return Ok(action(n, 0, true, next, false));
            }
            let mut step = json_frame_step(program, JsonFrame::AfterFunctionSuffix, &pending[n..])
                .map_err(|failure| JsonFrameFailure {
                    cause: failure.cause,
                    consumed: n + failure.consumed,
                })?;
            step.consume_before += n;
            Ok(step)
        }
        JsonFrame::AfterFunctionSuffix => {
            if pending.is_empty() {
                return Ok(wait(0));
            }
            if pending.starts_with(']') {
                return Ok(action(0, 1, true, JsonFrame::ToolSuffix, false));
            }
            let Some(separator) = exact(program, JsonDelimiter::Separator, pending)? else {
                return Ok(wait(0));
            };
            Ok(action(
                separator,
                0,
                true,
                JsonFrame::ListItem { allow_end: false },
                false,
            ))
        }
        JsonFrame::AfterEnvelope => {
            if !program.output.suffix.is_empty() && pending.starts_with(program.output.suffix) {
                Ok(action(
                    program.output.suffix.len(),
                    0,
                    false,
                    JsonFrame::Outside,
                    false,
                ))
            } else if (!program.output.suffix.is_empty()
                && program.output.suffix.starts_with(pending))
                || program.separator.starts_with(pending)
            {
                Ok(wait(0))
            } else if pending.starts_with(program.separator) {
                Ok(action(
                    program.separator.len(),
                    0,
                    false,
                    JsonFrame::ToolStart,
                    false,
                ))
            } else {
                Err(JsonFrameError::Separator.into())
            }
        }
        JsonFrame::ToolSuffix => {
            let Some(n) = exact(program, JsonDelimiter::ToolSuffix, pending)? else {
                return Ok(wait(0));
            };
            Ok(action(n, 0, false, JsonFrame::Outside, false))
        }
        JsonFrame::Payload | JsonFrame::Outside => Err(JsonFrameError::State.into()),
    }
}
/// Fixed selected program, state, literal iterators and callback-step controls.
pub fn json_frame_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<[JsonToolProgram<'_>; 2]>(),
        size_of::<[JsonFrame; 2]>(),
        size_of::<[JsonFrameStep; 2]>(),
        size_of::<JsonFrameError>(),
        size_of::<JsonDelimiter>(),
        size_of::<[&str; 2]>(),
        size_of::<std::iter::Chain<std::str::Bytes<'_>, std::str::Bytes<'_>>>(),
        size_of::<
            std::iter::Zip<
                std::str::Bytes<'_>,
                std::iter::Chain<std::str::Bytes<'_>, std::str::Bytes<'_>>,
            >,
        >(),
        size_of::<[Result<JsonFrameStep, JsonFrameFailure>; 2]>(),
        size_of::<JsonFrameFailure>(),
        size_of::<Result<Option<usize>, JsonFrameError>>(),
        size_of::<(usize, usize, bool, bool)>(),
        size_of::<(JsonToolProgram<'_>, JsonFrame, &str)>(),
    ];
    // AfterPayload can enter AfterFunctionSuffix once; all other transitions return.
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)?
        .checked_mul(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        json_fragments::JsonFieldNames,
        semantic_channels::{JsonEnvelope, JsonToolLayout},
    };
    #[test]
    fn list_function_suffix_is_consumed_once_across_splits_and_closure_failure() {
        let program = JsonToolProgram {
            output: JsonEnvelope {
                prefix: "<tools>",
                suffix: "</tools>",
            },
            call: JsonEnvelope {
                prefix: "",
                suffix: "",
            },
            function: JsonEnvelope {
                prefix: "<function>",
                suffix: "</function>",
            },
            fields: JsonFieldNames {
                name: "name",
                arguments: "arguments",
                call_id: None,
            },
            shape: JsonToolShape::List,
            separator: ", ",
            layout: JsonToolLayout::SingleEnvelope,
        };
        for suffix in [", ", "]"] {
            let pending = format!("</function>{suffix}");
            for split in 0..=pending.len() {
                let mut bytes = pending[..split].to_owned();
                let mut state = JsonFrame::AfterPayload;
                let first = json_frame_step(program, state, &bytes).unwrap();
                bytes.drain(..first.consume_before);
                if first.end_call {
                    assert_eq!(split, pending.len());
                    if suffix == "]" {
                        assert_eq!(bytes, "]");
                    }
                    bytes.drain(..first.consume_after);
                    assert!(bytes.is_empty());
                    continue;
                }
                assert!(first.wait);
                state = first.next;
                bytes.push_str(&pending[split..]);
                let next = json_frame_step(program, state, &bytes).unwrap();
                assert!(next.end_call);
                assert!(!next.wait);
                bytes.drain(..next.consume_before);
                // A failed completion callback still owns the closing bracket.
                if suffix == "]" {
                    assert_eq!(bytes, "]");
                    assert_eq!(next.consume_after, 1);
                }
                bytes.drain(..next.consume_after);
                assert!(bytes.is_empty());
            }
        }
        let failure =
            json_frame_step(program, JsonFrame::AfterPayload, "</function>wrong").unwrap_err();
        assert_eq!(failure.consumed, "</function>".len());
        assert_eq!(
            failure.cause,
            JsonFrameError::Delimiter(JsonDelimiter::Separator)
        );
        assert_eq!(
            json_frame_step(program, JsonFrame::ListItem { allow_end: false }, "]")
                .unwrap_err()
                .cause,
            JsonFrameError::ListTail
        );
    }
}
