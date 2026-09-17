//! Same field interpretation for ordinary values and prepared JSON certificates.
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
/// Root category from an actual completed JSON parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonValueKind {
    /// Object value.
    Object,
    /// Array value.
    Array,
    /// Decoded string value.
    String,
    /// Integer or floating numeric value.
    Number,
    /// Boolean value.
    Bool,
    /// Null value.
    Null,
}
impl JsonValueKind {
    /// Classifies an ordinary parsed value without copying it.
    pub fn of(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(_) => Self::Bool,
            serde_json::Value::Number(_) => Self::Number,
            serde_json::Value::String(_) => Self::String,
            serde_json::Value::Array(_) => Self::Array,
            serde_json::Value::Object(_) => Self::Object,
        }
    }
}
/// Selected protocol-owned ID field and exact Unicode scalar length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonCallId<'a> {
    /// Decoded object key.
    pub field: &'a str,
    /// Required Unicode scalar count, when specified by the protocol.
    pub length: Option<usize>,
}
/// Borrowed declaration only. It carries neither source identity nor funding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonFieldNames<'a> {
    /// Name of the function-name string field.
    pub name: &'a str,
    /// Name of the argument-object field.
    pub arguments: &'a str,
    /// Optional protocol ID field.
    pub call_id: Option<JsonCallId<'a>>,
}
/// Semantic role selected in the ordinary name/arguments/ID priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonFieldRole {
    /// Function-name string.
    Name,
    /// Argument object, retained in its exact raw spelling.
    Arguments,
    /// Protocol-owned identifier string.
    CallId,
    /// A syntactically valid field with no selected semantic role.
    Other,
}
/// Fixed field refusal. Descriptors remain with the caller's actual source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum JsonFieldError {
    /// Function-name value is absent or not a string.
    #[error("tool function name must be a string")]
    Name,
    /// Argument value is not an object.
    #[error("tool arguments must be an object")]
    Arguments,
    /// Completed object has no argument field.
    #[error("tool arguments field is missing")]
    MissingArguments,
    /// Protocol identifier is absent or not a string.
    #[error("tool call ID must be a string")]
    CallId,
    /// Protocol identifier has a different Unicode scalar length.
    #[error("tool call ID has the wrong Unicode scalar length")]
    CallIdLength,
}
impl JsonFieldNames<'_> {
    /// Validates one fully parsed field through the same ordinary branch order.
    pub fn inspect(
        self,
        key: &str,
        kind: JsonValueKind,
        text: Option<&str>,
    ) -> Result<JsonFieldRole, JsonFieldError> {
        if key == self.name {
            if kind != JsonValueKind::String || text.is_none() {
                return Err(JsonFieldError::Name);
            }
            Ok(JsonFieldRole::Name)
        } else if key == self.arguments {
            if kind != JsonValueKind::Object {
                return Err(JsonFieldError::Arguments);
            }
            Ok(JsonFieldRole::Arguments)
        } else if self.call_id.is_some_and(|id| id.field == key) {
            let text = text
                .filter(|_| kind == JsonValueKind::String)
                .ok_or(JsonFieldError::CallId)?;
            if self
                .call_id
                .and_then(|id| id.length)
                .is_some_and(|length| text.chars().count() != length)
            {
                return Err(JsonFieldError::CallIdLength);
            }
            Ok(JsonFieldRole::CallId)
        } else {
            Ok(JsonFieldRole::Other)
        }
    }
    /// Same completed-call required-field order as the ordinary parser.
    pub fn complete(
        self,
        name: bool,
        arguments: bool,
        call_id: bool,
    ) -> Result<(), JsonFieldError> {
        if !name {
            return Err(JsonFieldError::Name);
        }
        if !arguments {
            return Err(JsonFieldError::MissingArguments);
        }
        if self.call_id.is_some() && !call_id {
            return Err(JsonFieldError::CallId);
        }
        Ok(())
    }
    /// Fixed loan, traversal, validation and ordinary display frames.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<JsonCallId<'_>>(),
            size_of::<JsonFieldRole>(),
            size_of::<JsonFieldError>(),
            size_of::<Message<'_>>(),
            size_of::<JsonValueKind>(),
            size_of::<(&str, JsonValueKind, Option<&str>)>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<Result<JsonFieldRole, JsonFieldError>>(),
            size_of::<Result<(), JsonFieldError>>(),
            size_of::<(bool, bool, bool)>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
struct Message<'a> {
    cause: JsonFieldError,
    fields: JsonFieldNames<'a>,
}
impl JsonFieldError {
    /// Same ordinary diagnostic spelling, borrowing the actual field declaration.
    /// Formatting allocation, if requested, belongs to the caller.
    pub fn display(self, fields: JsonFieldNames<'_>) -> impl fmt::Display + '_ {
        Message {
            cause: self,
            fields,
        }
    }
}
impl fmt::Display for Message<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cause {
            JsonFieldError::Name => write!(
                f,
                "declarative tool call field {:?} must be a string",
                self.fields.name
            ),
            JsonFieldError::Arguments => write!(
                f,
                "declarative tool call field {:?} must be an object",
                self.fields.arguments
            ),
            JsonFieldError::MissingArguments => write!(
                f,
                "declarative tool call is missing field {:?}",
                self.fields.arguments
            ),
            JsonFieldError::CallId => write!(
                f,
                "declarative tool call field {:?} must be a string",
                self.fields.call_id.map_or("", |id| id.field)
            ),
            JsonFieldError::CallIdLength => {
                let Some(id) = self.fields.call_id else {
                    return f.write_str("declarative tool call has no call ID declaration");
                };
                let Some(length) = id.length else {
                    return f.write_str("declarative tool call has no call ID length declaration");
                };
                write!(
                    f,
                    "declarative tool call field {:?} must contain exactly {} characters",
                    id.field, length
                )
            }
        }
    }
}
