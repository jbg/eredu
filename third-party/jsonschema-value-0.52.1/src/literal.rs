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
        Self::from_value_with_allocations(value, &serde_json::allocation::Unenforced)
            .expect("ordinary literal allocation")
    }
    /// Copy the same literal with reservations before each reached allocation.
    pub fn from_value_with_allocations(
        value: &Value,
        allocations: &dyn serde_json::allocation::Allocation,
    ) -> Result<Self, serde_json::allocation::AllocationError> {
        let allocator = serde_json::allocation::Allocator::new(allocations);
        Ok(match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(*value),
            Value::Number(value) => Self::Number(value.try_clone_with_allocations(allocations)?),
            Value::String(value) => Self::String(allocator.copy_string(value)?),
            Value::Array(values) => {
                let mut output = Vec::new();
                allocator.grow(&mut output, values.len())?;
                for value in values {
                    output.push(Self::from_value_with_allocations(value, allocations)?);
                }
                Self::Array(output)
            }
            Value::Object(values) => Self::from_object_with_allocations(values, allocations)?,
        })
    }
    /// Copy object members directly, without an intermediate serde map.
    #[must_use]
    pub fn from_object(values: &Map<String, Value>) -> Self {
        Self::from_object_with_allocations(values, &serde_json::allocation::Unenforced)
            .expect("ordinary object literal allocation")
    }
    /// Copy object members through the same producer with explicit allocation funding.
    pub fn from_object_with_allocations(
        values: &Map<String, Value>,
        allocations: &dyn serde_json::allocation::Allocation,
    ) -> Result<Self, serde_json::allocation::AllocationError> {
        let allocator = serde_json::allocation::Allocator::new(allocations);
        let mut output = Vec::new();
        allocator.grow(&mut output, values.len())?;
        for (key, value) in values {
            output.push((
                allocator.copy_string(key)?,
                Self::from_value_with_allocations(value, allocations)?,
            ));
        }
        Ok(Self::Object(output))
    }

    /// Reconstruct diagnostic metadata through the same source producer.
    #[must_use]
    pub fn to_value(&self) -> Value {
        self.to_value_with_allocations(&serde_json::allocation::Unenforced)
            .expect("ordinary literal diagnostic conversion")
    }
    /// Reconstruct the actual diagnostic JSON tree with prospective allocation.
    pub fn to_value_with_allocations(
        &self,
        allocations: &dyn serde_json::allocation::Allocation,
    ) -> Result<Value, serde_json::allocation::AllocationError> {
        let allocator = serde_json::allocation::Allocator::new(allocations);
        Ok(match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.try_clone_with_allocations(allocations)?),
            Self::String(value) => Value::String(allocator.copy_string(value)?),
            Self::Array(values) => {
                let mut output = Vec::new();
                allocator.grow(&mut output, values.len())?;
                for value in values {
                    output.push(value.to_value_with_allocations(allocations)?);
                }
                Value::Array(output)
            }
            Self::Object(values) => {
                let mut output = Map::new();
                for (key, value) in values {
                    output.try_insert_with_allocations(
                        allocator.copy_string(key)?,
                        value.to_value_with_allocations(allocations)?,
                        allocations,
                    )?;
                }
                Value::Object(output)
            }
        })
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
    type Members<'a>: ExactSizeIterator<Item = (&'a str, &'a Self)>
    where
        Self: 'a;
    fn view(&self) -> View<'_, Self>;
}

fn json_member<'a>((key, value): (&'a String, &'a Value)) -> (&'a str, &'a Value) {
    (key.as_str(), value)
}
fn literal_member((key, value): &(String, Literal)) -> (&str, &Literal) {
    (key.as_str(), value)
}
impl Expected for Value {
    type Members<'a> = std::iter::Map<
        serde_json::map::Iter<'a>,
        fn((&'a String, &'a Value)) -> (&'a str, &'a Value),
    >;
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
    type Members<'a> = std::iter::Map<
        std::slice::Iter<'a, (String, Literal)>,
        fn(&'a (String, Literal)) -> (&'a str, &'a Literal),
    >;
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
