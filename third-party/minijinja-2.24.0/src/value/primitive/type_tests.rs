//! Built-in test decisions shared by dynamic values and borrowed closed values.
//! Facts are queried lazily: ordinary custom-object representation and iterator
//! callbacks run only for the same test that queried them before.
use crate::value::{Value, ValueKind, ValueRepr};

#[derive(Clone, Copy, Debug)]
pub(crate) enum TypeTest {
    Boolean,
    Number,
    Integer,
    Float,
    String,
    Sequence,
    Mapping,
    Iterable,
    True,
    False,
}
pub(crate) trait Facts {
    fn kind(&self) -> ValueKind;
    fn integer(&self) -> bool;
    fn float(&self) -> bool;
    fn boolean(&self) -> Option<bool>;
    fn iterable(&self) -> bool;
}
impl Facts for Value {
    fn kind(&self) -> ValueKind {
        Value::kind(self)
    }
    fn integer(&self) -> bool {
        self.is_integer()
    }
    fn float(&self) -> bool {
        matches!(self.0, ValueRepr::F64(_))
    }
    fn boolean(&self) -> Option<bool> {
        if let ValueRepr::Bool(v) = self.0 {
            Some(v)
        } else {
            None
        }
    }
    fn iterable(&self) -> bool {
        self.try_iter().is_ok()
    }
}
impl TypeTest {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        if !cfg!(feature = "builtins") {
            return None;
        }
        Some(match name {
            "boolean" => Self::Boolean,
            "number" => Self::Number,
            "integer" => Self::Integer,
            "float" => Self::Float,
            "string" => Self::String,
            "sequence" => Self::Sequence,
            "mapping" => Self::Mapping,
            "iterable" => Self::Iterable,
            "true" => Self::True,
            "false" => Self::False,
            _ => return None,
        })
    }
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::String => "string",
            Self::Sequence => "sequence",
            Self::Mapping => "mapping",
            Self::Iterable => "iterable",
            Self::True => "true",
            Self::False => "false",
        }
    }
    pub(crate) fn check(self, value: &impl Facts) -> bool {
        match self {
            Self::Boolean => value.kind() == ValueKind::Bool,
            Self::Number => matches!(value.kind(), ValueKind::Number),
            Self::Integer => value.integer(),
            Self::Float => value.float(),
            Self::String => matches!(value.kind(), ValueKind::String),
            Self::Sequence => matches!(value.kind(), ValueKind::Seq),
            Self::Mapping => matches!(value.kind(), ValueKind::Map),
            Self::Iterable => value.iterable(),
            Self::True => value.boolean() == Some(true),
            Self::False => value.boolean() == Some(false),
        }
    }
}
