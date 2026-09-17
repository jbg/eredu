//! Actual immutable external bindings, without a merged map or input adoption.
use super::*;
use crate::value::primitive::scalar::Scalar;

/// Pointer-free address in one borrowed immutable map. No caller reference can
/// escape inside a retained render/error buffer.
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) struct VariableKey {
    layer: u8,
    index: usize,
}
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) enum ContextValue<'a> {
    Text(VariableKey, &'a str),
    Scalar(Scalar),
    Messages,
    EmptySequence,
    #[cfg(feature = "chat-clock")]
    Clock,
    #[cfg(feature = "json")]
    Structured(usize, &'a serde_json::Value),
    #[cfg(feature = "json")]
    Records(usize, record::ReadValue<'a>),
    Undefined,
    Unsupported,
}
/// Borrowed scalar replacement produced by request policy without a JSON map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarBindingValue<'a> {
    /// Exact Boolean control.
    Bool(bool),
    /// Immutable scalar spelling retained by the caller through rendering.
    Text(&'a str),
}
/// One lexical binding. Later entries win, matching repeated map insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarBinding<'a> {
    /// Exact variable name, with no protocol or object interpretation.
    pub name: &'a str,
    /// The scalar value used for this render only.
    pub value: ScalarBindingValue<'a>,
}

/// Borrowed external context used by both measurement and admitted execution.
/// Defaults replace base keys, and caller values replace defaults. It does not
/// certify structured operations or provide any mutable callback/owner escape.
#[derive(Clone, Copy, Debug)]
pub struct RenderContext<'a> {
    scalars: &'a [ScalarBinding<'a>],
    #[cfg(feature = "json")]
    tools: Option<&'a [RecordValue<'a>]>,
    #[cfg(feature = "chat-clock")]
    clock: Option<crate::bounded::clock::Snapshot>,
    messages: Messages<'a>,
    #[cfg(feature = "json")]
    defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[cfg(feature = "json")]
    overrides: Option<&'a serde_json::Map<String, serde_json::Value>>,
}
impl<'a> RenderContext<'a> {
    /// Existing role/content input with empty tool/document base variables.
    pub fn from_messages(messages: Messages<'a>) -> Self {
        Self {
            scalars: &[],
            #[cfg(feature = "json")]
            tools: None,
            #[cfg(feature = "chat-clock")]
            clock: None,
            messages,
            #[cfg(feature = "json")]
            defaults: None,
            #[cfg(feature = "json")]
            overrides: None,
        }
    }
    /// Borrow the actual facade maps in ordinary precedence. Replacement
    /// messages are validated instead of unused base messages.
    #[cfg(feature = "json")]
    pub fn from_json(
        messages: &'a [serde_json::Value],
        defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
        overrides: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Self, InputError> {
        let replacement = overrides
            .and_then(|v| v.get("messages"))
            .or_else(|| defaults.and_then(|v| v.get("messages")));
        let messages = match replacement {
            Some(serde_json::Value::Array(values)) => Messages::from_json(values)?,
            Some(_) => Messages::from_text(&[]),
            None => Messages::from_json(messages)?,
        };
        Ok(Self {
            scalars: &[],
            #[cfg(feature = "json")]
            tools: None,
            #[cfg(feature = "chat-clock")]
            clock: None,
            messages,
            defaults,
            overrides,
        })
    }
    /// Borrow the exact tool declarations supplied by a source producer.
    /// Default/caller variable replacement keeps ordinary precedence.
    #[cfg(feature = "json")]
    pub fn with_record_tools(mut self, tools: &'a [RecordValue<'a>]) -> Self {
        self.tools = Some(tools);
        self
    }
    /// Add actual scalar bindings to already borrowed typed messages.
    #[cfg(feature = "json")]
    pub fn with_variables(
        mut self,
        defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
        overrides: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Self, InputError> {
        if let Some(value) = overrides
            .and_then(|v| v.get("messages"))
            .or_else(|| defaults.and_then(|v| v.get("messages")))
        {
            self.messages = match value {
                serde_json::Value::Array(values) => Messages::from_json(values)?,
                _ => Messages::from_text(&[]),
            };
        }
        self.defaults = defaults;
        self.overrides = overrides;
        Ok(self)
    }
    /// Borrow policy-produced scalars above both caller and default maps.
    /// This replaces the previous overlay, allocates nothing, and grants no
    /// object/callback access. Each text address remains a lexical source index.
    pub fn with_scalar_overrides(mut self, scalars: &'a [ScalarBinding<'a>]) -> Self {
        self.scalars = scalars;
        self
    }
    /// Retain an immutable environmental timestamp for the built-in function.
    #[cfg(feature = "chat-clock")]
    pub fn with_clock(mut self, clock: crate::bounded::clock::Snapshot) -> Self {
        self.clock = Some(clock);
        self
    }
    #[cfg(feature = "chat-clock")]
    pub(in crate::bounded) fn clock(self) -> Option<crate::bounded::clock::Snapshot> {
        self.clock
    }
    /// True only when no explicit map can alter any base lookup.
    pub fn is_plain(self) -> bool {
        #[cfg(feature = "json")]
        if self.tools.is_some() {
            return false;
        }
        if !self.scalars.is_empty() {
            return false;
        }
        #[cfg(feature = "json")]
        {
            return self.defaults.is_none_or(|v| v.is_empty())
                && self.overrides.is_none_or(|v| v.is_empty());
        }
        #[cfg(not(feature = "json"))]
        {
            true
        }
    }
    /// Exact effective message source. Context lookup still determines whether
    /// a reserved-key scalar replacement is used instead of this list.
    pub fn messages(self) -> Messages<'a> {
        self.messages
    }
    #[cfg(feature = "json")]
    fn binding(self, name: &str) -> Option<(VariableKey, &'a serde_json::Value)> {
        for (layer, map) in [(1, self.overrides), (0, self.defaults)] {
            if let Some(map) = map {
                for (index, (key, value)) in map.iter().enumerate() {
                    if key.as_str() == name {
                        return Some((VariableKey { layer, index }, value));
                    }
                }
            }
        }
        None
    }
    pub(in crate::bounded) fn lookup(self, name: &str, generation: bool) -> ContextValue<'a> {
        for (index, binding) in self.scalars.iter().enumerate().rev() {
            if binding.name == name {
                return match binding.value {
                    ScalarBindingValue::Bool(value) => ContextValue::Scalar(Scalar::Bool(value)),
                    ScalarBindingValue::Text(text) => {
                        ContextValue::Text(VariableKey { layer: 2, index }, text)
                    }
                };
            }
        }
        #[cfg(feature = "json")]
        if let Some((key, value)) = self.binding(name) {
            return match value {
                serde_json::Value::String(text) => ContextValue::Text(key, text),
                serde_json::Value::Bool(value) => ContextValue::Scalar(Scalar::Bool(*value)),
                serde_json::Value::Null => ContextValue::Scalar(Scalar::None),
                serde_json::Value::Number(value) => {
                    ContextValue::Scalar(if let Some(value) = value.as_u64() {
                        Scalar::U64(value)
                    } else if let Some(value) = value.as_i64() {
                        Scalar::I64(value)
                    } else if let Some(value) = value.as_f64() {
                        Scalar::F64(value)
                    } else {
                        return ContextValue::Unsupported;
                    })
                }
                serde_json::Value::Array(_) if name == "messages" => ContextValue::Messages,
                serde_json::Value::Array(values) => ContextValue::Structured(values.len(), value),
                serde_json::Value::Object(values) => ContextValue::Structured(values.len(), value),
            };
        }
        match name {
            #[cfg(feature = "chat-clock")]
            "strftime_now" if self.clock.is_some() => ContextValue::Clock,
            "messages" => ContextValue::Messages,
            "add_generation_prompt" => ContextValue::Scalar(Scalar::Bool(generation)),
            #[cfg(feature = "json")]
            "tools" if self.tools.is_some() => {
                let values = self.tools.expect("checked source");
                ContextValue::Records(
                    values.len(),
                    record::ReadValue::Record(RecordValue::Array(values)),
                )
            }
            "tools" | "documents" => ContextValue::EmptySequence,
            _ => ContextValue::Undefined,
        }
    }
    pub(in crate::bounded) fn text(self, key: VariableKey) -> Option<&'a str> {
        if key.layer == 2 {
            return match self.scalars.get(key.index)?.value {
                ScalarBindingValue::Text(text) => Some(text),
                ScalarBindingValue::Bool(_) => None,
            };
        }
        #[cfg(feature = "json")]
        {
            let map = if key.layer == 1 {
                self.overrides?
            } else {
                self.defaults?
            };
            return map.iter().nth(key.index)?.1.as_str();
        }
        #[cfg(not(feature = "json"))]
        {
            let _ = key;
            None
        }
    }
    pub(in crate::bounded) fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let parts = [
            size_of::<Self>(),
            size_of::<VariableKey>(),
            size_of::<ScalarBinding<'_>>(),
            size_of::<ScalarBindingValue<'_>>(),
            size_of::<std::iter::Rev<std::iter::Enumerate<std::slice::Iter<'_, ScalarBinding<'_>>>>>(
            ),
            size_of::<Option<(usize, &ScalarBinding<'_>)>>(),
            size_of::<ContextValue<'_>>(),
            size_of::<Result<Self, InputError>>(),
            size_of::<Option<(&str, &str)>>(),
            size_of::<(&Self, &str, bool)>(),
            size_of::<Scalar>(),
        ];
        let mut total = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        #[cfg(feature = "json")]
        {
            total = total
                .checked_add(size_of::<serde_json::map::Iter<'_>>())?
                .checked_add(size_of::<std::iter::Enumerate<serde_json::map::Iter<'_>>>())?
                .checked_add(size_of::<Option<(usize, (&String, &serde_json::Value))>>())?
                .checked_add(size_of::<(u8, usize, &str, &serde_json::Value)>())?
                .checked_add(size_of::<Option<(VariableKey, &serde_json::Value)>>())?
                .checked_add(size_of::<
                    [(u8, Option<&serde_json::Map<String, serde_json::Value>>); 2],
                >())?;
        }
        Some(total)
    }
}
