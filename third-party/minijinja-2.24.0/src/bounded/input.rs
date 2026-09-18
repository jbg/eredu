use std::fmt;

/// Borrowed text-only message; the caller retains both string allocations.
#[derive(Clone, Copy, Debug)]
pub struct TextMessage<'a> {
    /// The exact role spelling used by the ordinary template.
    pub role: &'a str,
    /// The exact content, including empty strings, NUL and Unicode scalars.
    pub content: &'a str,
}

/// Fixed rejection before constructing render storage or ordinary Value objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// A message is not an immutable JSON object.
    MessageProfile,
}
impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("chat messages require JSON objects")
    }
}
impl std::error::Error for InputError {}

#[derive(Clone, Copy, Debug)]
enum Source<'a> {
    Text(&'a [TextMessage<'a>]),
    #[cfg(feature = "json")]
    Json(&'a [serde_json::Value]),
    #[cfg(feature = "json")]
    Record(&'a [RecordValue<'a>]),
}

/// Validated readonly access to the actual message slice through both renders.
/// This view performs no allocation and supplies no callback or mutable accessor.
#[derive(Clone, Copy, Debug)]
pub struct Messages<'a> {
    source: Source<'a>,
}
impl<'a> Messages<'a> {
    /// Borrow caller-owned typed strings without creating an owned message list.
    pub fn from_text(messages: &'a [TextMessage<'a>]) -> Self {
        Self {
            source: Source::Text(messages),
        }
    }

    /// Validate and borrow immutable source-owned message records. No object
    /// conversion, allocation, callback or inference grant is performed.
    #[cfg(feature = "json")]
    pub fn from_records(messages: &'a [RecordValue<'a>]) -> Result<Self, InputError> {
        if messages.iter().any(|message| !message.is_object()) {
            return Err(InputError::MessageProfile);
        }
        Ok(Self {
            source: Source::Record(messages),
        })
    }
    #[cfg(feature = "json")]
    pub(in crate::bounded) fn value(self, index: usize) -> Option<record::ReadValue<'a>> {
        match self.source {
            Source::Json(values) => values.get(index).map(record::ReadValue::Json),
            Source::Record(values) => values.get(index).copied().map(record::ReadValue::Record),
            Source::Text(_) => None,
        }
    }
    /// Validate and borrow the facade's actual JSON message representation.
    /// All object members remain available through the shared immutable JSON
    /// operand worker, including tool history, reasoning and structured content.
    #[cfg(feature = "json")]
    pub fn from_json(messages: &'a [serde_json::Value]) -> Result<Self, InputError> {
        for message in messages {
            if !message.is_object() {
                return Err(InputError::MessageProfile);
            }
        }
        Ok(Self {
            source: Source::Json(messages),
        })
    }

    /// Number of messages read by the selected template loop.
    pub fn len(self) -> usize {
        match self.source {
            Source::Text(messages) => messages.len(),
            #[cfg(feature = "json")]
            Source::Json(messages) => messages.len(),
            #[cfg(feature = "json")]
            Source::Record(messages) => messages.len(),
        }
    }

    /// Whether the actual message loop has no iterations.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[cfg(feature = "json")]
    pub(super) fn json_slice(self) -> Option<&'a [serde_json::Value]> {
        match self.source {
            Source::Json(values) => Some(values),
            Source::Text(_) | Source::Record(_) => None,
        }
    }

    /// A loan of the exact caller-owned object, never a reconstructed message.
    /// The render's existing bounded borrow registers own its temporary index.
    #[cfg(feature = "json")]
    pub(super) fn json(self, index: usize) -> Option<&'a serde_json::Value> {
        match self.source {
            Source::Json(messages) => messages.get(index),
            Source::Text(_) | Source::Record(_) => None,
        }
    }

    pub(super) fn get(self, index: usize) -> Option<TextMessage<'a>> {
        match self.source {
            Source::Text(messages) => messages.get(index).copied(),
            #[cfg(feature = "json")]
            Source::Record(messages) => {
                let message = messages.get(index)?;
                if message.length() != Some(2) {
                    return None;
                }
                let RecordValue::Text(role) = message.field("role")? else {
                    return None;
                };
                let RecordValue::Text(content) = message.field("content")? else {
                    return None;
                };
                Some(TextMessage { role, content })
            }
            #[cfg(feature = "json")]
            Source::Json(messages) => messages
                .get(index)
                .and_then(|message| json_message(message).ok()),
        }
    }
}

#[cfg(feature = "json")]
fn json_message(value: &serde_json::Value) -> Result<TextMessage<'_>, InputError> {
    let object = value.as_object().ok_or(InputError::MessageProfile)?;
    if object.len() != 2 {
        return Err(InputError::MessageProfile);
    }
    Ok(TextMessage {
        role: object
            .get("role")
            .and_then(serde_json::Value::as_str)
            .ok_or(InputError::MessageProfile)?,
        content: object
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or(InputError::MessageProfile)?,
    })
}

mod context;
pub(super) use context::{ContextValue, VariableKey};
pub use context::{RenderContext, ScalarBinding, ScalarBindingValue};

#[cfg(feature = "json")]
pub(in crate::bounded) mod record;
#[cfg(feature = "json")]
pub use record::{InputArray, RecordField, RecordFields, RecordValue};
