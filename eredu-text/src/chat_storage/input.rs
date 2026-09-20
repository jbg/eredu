//! Borrowed application inputs converted by stock serde only during rendering.
use super::{ChatClockSnapshot, ChatRenderPlanError};
use serde::{Serialize, Serializer, ser::SerializeMap};
use serde_json::{Map, Value};

/// Role/content message borrowed from the request.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TextMessage<'a> {
    /// Role spelling supplied to the template.
    pub role: &'a str,
    /// Complete UTF-8 content.
    pub content: &'a str,
}
/// Invalid effective message input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ChatMessageError {
    /// A message entry is not an object.
    #[error("chat messages require JSON objects")]
    MessageProfile,
}
/// Immutable records used by portable source producers and behavioral probes.
#[derive(Debug, Clone, Copy)]
pub enum ChatRecordValue<'a> {
    /// JSON null.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Borrowed UTF-8 string.
    Text(&'a str),
    /// Ordered sequence.
    Array(&'a [Self]),
    /// Ordered fields; repeated keys retain their first position and last value.
    Object(&'a [ChatRecordField<'a>]),
}
/// One borrowed record field.
#[derive(Debug, Clone, Copy)]
pub struct ChatRecordField<'a> {
    /// Exact field name.
    pub key: &'a str,
    /// Immutable field value.
    pub value: ChatRecordValue<'a>,
}
impl Serialize for ChatRecordValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Text(value) => serializer.serialize_str(value),
            Self::Array(values) => values.serialize(serializer),
            Self::Object(fields) => {
                let mut map = serializer.serialize_map(Some(fields.len()))?;
                for field in *fields {
                    map.serialize_entry(field.key, &field.value)?;
                }
                map.end()
            }
        }
    }
}
/// Borrowed structured sequence supplied to a render request.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
pub enum ChatInputArray<'a> {
    /// JSON values owned by the caller.
    Json(&'a [Value]),
    /// Records owned by a portable producer.
    Record(&'a [ChatRecordValue<'a>]),
}
/// Validated immutable message list.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
enum MessageSource<'a> {
    /// Role/content records.
    Text(&'a [TextMessage<'a>]),
    /// JSON object messages.
    Json(&'a [Value]),
    /// Portable object records.
    Record(&'a [ChatRecordValue<'a>]),
}
/// Validated borrowed messages; mutation and unchecked source construction are private.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(transparent)]
pub struct ChatMessages<'a>(MessageSource<'a>);
impl<'a> ChatMessages<'a> {
    /// Borrows role/content messages.
    pub fn from_text(messages: &'a [TextMessage<'a>]) -> Self {
        Self(MessageSource::Text(messages))
    }
    /// Validates object messages without copying their content.
    pub fn from_json(messages: &'a [Value]) -> Result<Self, ChatMessageError> {
        if messages.iter().any(|message| !message.is_object()) {
            return Err(ChatMessageError::MessageProfile);
        }
        Ok(Self(MessageSource::Json(messages)))
    }
    /// Validates portable object messages.
    pub fn from_records(messages: &'a [ChatRecordValue<'a>]) -> Result<Self, ChatMessageError> {
        if messages
            .iter()
            .any(|message| !matches!(message, ChatRecordValue::Object(_)))
        {
            return Err(ChatMessageError::MessageProfile);
        }
        Ok(Self(MessageSource::Record(messages)))
    }
    /// Number of effective base messages.
    pub fn len(self) -> usize {
        match self.0 {
            MessageSource::Text(x) => x.len(),
            MessageSource::Json(x) => x.len(),
            MessageSource::Record(x) => x.len(),
        }
    }
    /// Whether the message list is empty.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}
/// Request-policy scalar overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ChatScalarBindingValue<'a> {
    /// Boolean control.
    Bool(bool),
    /// Borrowed text control.
    Text(&'a str),
}
/// One binding applied after default and caller variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChatScalarBinding<'a> {
    /// Variable name.
    pub name: &'a str,
    /// Replacement value; later repeated names win.
    pub value: ChatScalarBindingValue<'a>,
}
/// Borrowed request context. Base values are overridden by defaults, caller
/// variables, then scalar policy bindings. The same clock serves both variants.
#[derive(Debug, Clone, Copy)]
pub struct ChatRenderContext<'a> {
    messages: ChatMessages<'a>,
    tools: Option<ChatInputArray<'a>>,
    defaults: Option<&'a Map<String, Value>>,
    overrides: Option<&'a Map<String, Value>>,
    scalars: &'a [ChatScalarBinding<'a>],
    pub(super) clock: Option<ChatClockSnapshot>,
}
impl<'a> ChatRenderContext<'a> {
    /// Uses empty tools/documents and no variable overlays.
    pub fn from_messages(messages: ChatMessages<'a>) -> Self {
        Self {
            messages,
            tools: None,
            defaults: None,
            overrides: None,
            scalars: &[],
            clock: None,
        }
    }
    /// Validates effective messages; unused base messages need not be objects.
    pub fn from_json(
        messages: &'a [Value],
        defaults: Option<&'a Map<String, Value>>,
        overrides: Option<&'a Map<String, Value>>,
    ) -> Result<Self, ChatMessageError> {
        let replacement = overrides
            .and_then(|m| m.get("messages"))
            .or_else(|| defaults.and_then(|m| m.get("messages")));
        let effective = match replacement {
            Some(Value::Array(values)) => ChatMessages::from_json(values)?,
            Some(_) => ChatMessages::from_text(&[]),
            None => ChatMessages::from_json(messages)?,
        };
        Self::from_messages(effective).with_variables(defaults, overrides)
    }
    /// Supplies base tool declarations, below default and caller precedence.
    pub fn with_tools(mut self, tools: ChatInputArray<'a>) -> Self {
        self.tools = Some(tools);
        self
    }
    /// Replaces the default and caller variable maps.
    pub fn with_variables(
        mut self,
        defaults: Option<&'a Map<String, Value>>,
        overrides: Option<&'a Map<String, Value>>,
    ) -> Result<Self, ChatMessageError> {
        if let Some(value) = overrides
            .and_then(|m| m.get("messages"))
            .or_else(|| defaults.and_then(|m| m.get("messages")))
        {
            self.messages = match value {
                Value::Array(values) => ChatMessages::from_json(values)?,
                _ => ChatMessages::from_text(&[]),
            };
        }
        self.defaults = defaults;
        self.overrides = overrides;
        Ok(self)
    }
    /// Applies request-policy scalars above both variable maps.
    pub fn with_scalar_overrides(mut self, scalars: &'a [ChatScalarBinding<'a>]) -> Self {
        self.scalars = scalars;
        self
    }
    /// Uses a retained clock observation for all renders of this request.
    pub fn with_clock(mut self, clock: ChatClockSnapshot) -> Self {
        self.clock = Some(clock);
        self
    }
    /// Whether variables and tool declarations leave the base context unchanged.
    pub fn is_plain(self) -> bool {
        self.tools.is_none()
            && self.scalars.is_empty()
            && self.defaults.is_none_or(Map::is_empty)
            && self.overrides.is_none_or(Map::is_empty)
    }
    /// Effective base messages; overlays still determine template lookup.
    pub fn messages(self) -> ChatMessages<'a> {
        self.messages
    }

    pub(super) fn input_bytes(
        self,
        max_bytes: usize,
        max_depth: usize,
    ) -> Result<usize, ChatRenderPlanError> {
        fn json(value: &Value, depth: usize) -> Result<(), ChatRenderPlanError> {
            if depth == 0 {
                return Err(ChatRenderPlanError::InputDepth);
            }
            match value {
                Value::Array(values) => {
                    for value in values {
                        json(value, depth - 1)?;
                    }
                }
                Value::Object(values) => {
                    for value in values.values() {
                        json(value, depth - 1)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        fn record(value: &ChatRecordValue<'_>, depth: usize) -> Result<(), ChatRenderPlanError> {
            if depth == 0 {
                return Err(ChatRenderPlanError::InputDepth);
            }
            match value {
                ChatRecordValue::Array(values) => {
                    for value in *values {
                        record(value, depth - 1)?;
                    }
                }
                ChatRecordValue::Object(fields) => {
                    for field in *fields {
                        record(&field.value, depth - 1)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        let array = |values: ChatInputArray<'_>| -> Result<(), ChatRenderPlanError> {
            match values {
                ChatInputArray::Json(values) => {
                    for value in values {
                        json(value, max_depth)?;
                    }
                }
                ChatInputArray::Record(values) => {
                    for value in values {
                        record(value, max_depth)?;
                    }
                }
            }
            Ok(())
        };
        match self.messages.0 {
            MessageSource::Json(v) => array(ChatInputArray::Json(v))?,
            MessageSource::Record(v) => array(ChatInputArray::Record(v))?,
            _ => {}
        }
        if let Some(tools) = self.tools {
            array(tools)?;
        }
        for map in [self.defaults, self.overrides].into_iter().flatten() {
            for value in map.values() {
                json(value, max_depth)?;
            }
        }
        struct Count {
            bytes: usize,
            limit: usize,
        }
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.bytes = self
                    .bytes
                    .checked_add(bytes.len())
                    .ok_or(std::io::ErrorKind::FileTooLarge)?;
                if self.bytes > self.limit {
                    return Err(std::io::ErrorKind::FileTooLarge.into());
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count {
            bytes: 0,
            limit: max_bytes,
        };
        serde_json::to_writer(
            &mut count,
            &(
                self.messages,
                self.tools,
                self.defaults,
                self.overrides,
                self.scalars,
            ),
        )
        .map_err(|_| ChatRenderPlanError::InputBytes)?;
        Ok(count.bytes)
    }
    pub(super) fn values(self, generation: bool) -> Result<Map<String, Value>, serde_json::Error> {
        let mut values = Map::new();
        values.insert("messages".into(), serde_json::to_value(self.messages)?);
        values.insert(
            "tools".into(),
            match self.tools {
                Some(tools) => serde_json::to_value(tools)?,
                None => Value::Array(Vec::new()),
            },
        );
        values.insert("documents".into(), Value::Array(Vec::new()));
        values.insert("add_generation_prompt".into(), Value::Bool(generation));
        for map in [self.defaults, self.overrides].into_iter().flatten() {
            values.extend(map.clone());
        }
        for binding in self.scalars {
            values.insert(binding.name.into(), serde_json::to_value(binding.value)?);
        }
        Ok(values)
    }
}
