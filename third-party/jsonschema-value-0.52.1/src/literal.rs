//! Immutable schema literals with source-visible owned container capacities.
//! Construction and diagnostic conversion are cold; validation borrows these
//! values through the same equality worker used for ordinary serde values.
use serde_json::{Map, Number, Value};

/// A completed literal declaration. Object members retain the source's key
/// iteration order; constructors never sort, reparse or normalize numbers.
#[derive(Clone, Debug)]
pub enum Literal {
    /// JSON null.
    Null,
    /// JSON Boolean.
    Bool(bool),
    /// The exact compiled schema number.
    Number(Number),
    /// Owned UTF-8 text.
    String(String),
    /// Array elements in their original order.
    Array(Vec<Literal>),
    /// Unique members in their source map's iteration order.
    Object(Vec<(String, Literal)>),
}
impl Literal {
    /// Copy a source literal during the caller's paid/cold construction.
    #[must_use]
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(*value),
            Value::Number(value) => Self::Number(value.clone()),
            Value::String(value) => Self::String(value.clone()),
            Value::Array(values) => Self::Array(values.iter().map(Self::from_value).collect()),
            Value::Object(values) => Self::from_object(values),
        }
    }

    /// Copy object members directly, without an intermediate serde map.
    #[must_use]
    pub fn from_object(values: &Map<String, Value>) -> Self {
        Self::Object(values.iter().map(|(key, value)| (key.clone(), Self::from_value(value))).collect())
    }

    /// Reconstruct ordinary diagnostic metadata. This allocating conversion is
    /// not an original-validation operation or an execution authority.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.clone()),
            Self::Array(values) => Value::Array(values.iter().map(Self::to_value).collect()),
            Self::Object(values) => Value::Object(values.iter().map(|(key, value)| (key.clone(), value.to_value())).collect()),
        }
    }
}

/// Borrowed expected-value shape shared by both stored representations.
pub(crate) enum View<'a, E: Expected + 'a> {
    Null,
    Bool(bool),
    Number(&'a Number),
    String(&'a str),
    Array(&'a [E]),
    Object(E::Members<'a>),
}
pub(crate) trait Expected: Sized {
    type Members<'a>: ExactSizeIterator<Item = (&'a str, &'a Self)> where Self: 'a;
    fn view(&self) -> View<'_, Self>;
}

fn json_member<'a>((key, value): (&'a String, &'a Value)) -> (&'a str, &'a Value) {
    (key.as_str(), value)
}
fn literal_member((key, value): &(String, Literal)) -> (&str, &Literal) {
    (key.as_str(), value)
}
impl Expected for Value {
    type Members<'a> = std::iter::Map<serde_json::map::Iter<'a>, fn((&'a String, &'a Value)) -> (&'a str, &'a Value)>;
    fn view(&self) -> View<'_, Self> {
        match self {
            Self::Null => View::Null,
            Self::Bool(value) => View::Bool(*value),
            Self::Number(value) => View::Number(value),
            Self::String(value) => View::String(value),
            Self::Array(values) => View::Array(values),
            Self::Object(values) => View::Object(values.iter().map(json_member as fn(_) -> _)),
        }
    }
}
impl Expected for Literal {
    type Members<'a> = std::iter::Map<std::slice::Iter<'a, (String, Literal)>, fn(&'a (String, Literal)) -> (&'a str, &'a Literal)>;
    fn view(&self) -> View<'_, Self> {
        match self {
            Self::Null => View::Null,
            Self::Bool(value) => View::Bool(*value),
            Self::Number(value) => View::Number(value),
            Self::String(value) => View::String(value),
            Self::Array(values) => View::Array(values),
            Self::Object(values) => View::Object(values.iter().map(literal_member as fn(_) -> _)),
        }
    }
}
