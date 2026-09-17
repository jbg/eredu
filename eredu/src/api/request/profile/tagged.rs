//! Shared request-history validation with a fixed, source-indexed failure.
use serde_json::Value;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Mapping,
    Name,
    Delimiter,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("tagged history is invalid at message {message}, call {call}, parameter {parameter}")]
pub(crate) struct Failure {
    message: usize,
    call: usize,
    parameter: usize,
    kind: Kind,
}
pub(crate) fn validate(messages: &[Value], suffix: &str) -> Result<(), Failure> {
    for (message_index, message) in messages.iter().enumerate() {
        let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for (call_index, call) in tool_calls.iter().enumerate() {
            let function = call.get("function").unwrap_or(call);
            let Some(arguments) = function.get("arguments") else {
                continue;
            };
            let failure = |parameter, kind| Failure {
                message: message_index,
                call: call_index,
                parameter,
                kind,
            };
            let arguments = arguments
                .as_object()
                .ok_or_else(|| failure(0, Kind::Mapping))?;
            for (parameter, (name, value)) in arguments.iter().enumerate() {
                if name.is_empty() || name.chars().any(|c| matches!(c, '<' | '>' | '\r' | '\n')) {
                    return Err(failure(parameter, Kind::Name));
                }
                if value.as_str().is_some_and(|value| value.contains(suffix)) {
                    return Err(failure(parameter, Kind::Delimiter));
                }
            }
        }
    }
    Ok(())
}
impl Failure {
    /// Ordinary diagnostics read the same still-borrowed immutable input. The
    /// admitted error contains only indices and can outlive every caller string.
    pub(crate) fn ordinary(self, messages: &[Value], profile: &str) -> crate::api::TextModelError {
        let message_index = self.message;
        let call_index = self.call;
        let message = match self.kind {
            Kind::Mapping => format!(
                "messages[{message_index}].tool_calls[{call_index}].function.arguments must be a mapping for {profile} tagged-parameter templates; serialized strings are unsupported"
            ),
            Kind::Name | Kind::Delimiter => {
                let call = &messages[self.message]["tool_calls"][self.call];
                let function = call.get("function").unwrap_or(call);
                let (name, _) = function["arguments"]
                    .as_object()
                    .expect("same rejected immutable history")
                    .iter()
                    .nth(self.parameter)
                    .expect("same rejected parameter");
                match self.kind {
                    Kind::Name => format!(
                        "messages[{message_index}].tool_calls[{call_index}] contains unsafe tagged parameter name {name:?}"
                    ),
                    Kind::Delimiter => format!(
                        "messages[{message_index}].tool_calls[{call_index}] parameter {name:?} contains the unescaped tagged-parameter closing delimiter"
                    ),
                    Kind::Mapping => unreachable!(),
                }
            }
        };
        crate::api::TextModelError::ToolConstraint(message)
    }
}
pub(crate) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Failure>(),
        size_of::<Kind>(),
        size_of::<Result<(), Failure>>(),
        size_of::<(&[Value], &str)>(),
        size_of::<[&Value; 4]>(),
        size_of::<Option<&Vec<Value>>>(),
        size_of::<Option<&Value>>(),
        size_of::<Option<&serde_json::Map<String, Value>>>(),
        size_of::<[usize; 3]>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, Value>>>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, Value>>>(),
        size_of::<std::iter::Enumerate<serde_json::map::Iter<'_>>>(),
        size_of::<(&String, &Value, std::str::Chars<'_>)>(),
        size_of::<(usize, usize)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
