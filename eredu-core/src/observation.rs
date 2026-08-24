//! Portable, explicitly requested execution observations.

use std::collections::{btree_map::Entry, BTreeMap};

use serde::{Deserialize, Serialize};

/// Materialized row-major tensor values.
///
/// Backends may normalize native storage such as F16 or BF16 to F32 when
/// crossing the observation boundary. This type describes the host values,
/// not checkpoint storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dtype", content = "values", rename_all = "snake_case")]
pub enum TensorObservationData {
    /// IEEE F32 values.
    F32(Vec<f32>),
    /// Signed 64-bit values.
    I64(Vec<i64>),
    /// Unsigned 64-bit values.
    U64(Vec<u64>),
    /// Boolean values.
    Bool(Vec<bool>),
}

impl TensorObservationData {
    /// Number of materialized values.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    /// Whether no values are present.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One complete host-materialized tensor observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorObservation {
    shape: Vec<usize>,
    data: TensorObservationData,
}

impl TensorObservation {
    /// Validates a tensor shape and its row-major values.
    pub fn new(shape: Vec<usize>, data: TensorObservationData) -> Result<Self, ObservationError> {
        let elements = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(ObservationError::ShapeOverflow)
        })?;
        if elements != data.len() {
            return Err(ObservationError::ElementCount {
                shape,
                expected: elements,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Logical row-major shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Materialized row-major values.
    pub const fn data(&self) -> &TensorObservationData {
        &self.data
    }

    /// Consumes this observation into its shape and values.
    pub fn into_parts(self) -> (Vec<usize>, TensorObservationData) {
        (self.shape, self.data)
    }
}

/// One portable observation value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ObservationValue {
    /// Materialized tensor.
    Tensor(TensorObservation),
    /// Floating-point scalar, including timings and ratios.
    Float(f64),
    /// Signed integer scalar.
    Integer(i64),
    /// Unsigned integer scalar.
    Unsigned(u64),
    /// Boolean scalar.
    Boolean(bool),
    /// Textual identity, label, or diagnostic.
    Text(String),
}

/// Deterministically ordered, path-addressed observations from one operation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservationSet {
    values: BTreeMap<String, ObservationValue>,
}

impl ObservationSet {
    /// Creates an empty set.
    pub const fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Inserts one uniquely named observation.
    pub fn insert(
        &mut self,
        path: impl Into<String>,
        value: ObservationValue,
    ) -> Result<(), ObservationError> {
        let path = path.into();
        if path.is_empty() {
            return Err(ObservationError::EmptyPath);
        }
        match self.values.entry(path) {
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
            Entry::Occupied(entry) => {
                return Err(ObservationError::DuplicatePath(entry.key().clone()));
            }
        }
        Ok(())
    }

    /// Looks up an observation by its stable path.
    pub fn get(&self, path: &str) -> Option<&ObservationValue> {
        self.values.get(path)
    }

    /// Iterates in stable path order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ObservationValue)> {
        self.values
            .iter()
            .map(|(path, value)| (path.as_str(), value))
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no observations are present.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Adds a prefix to every path, preserving deterministic order.
    pub fn prefixed(self, prefix: &str) -> Result<Self, ObservationError> {
        if prefix.is_empty() {
            return Ok(self);
        }
        let mut output = Self::new();
        for (path, value) in self.values {
            output.insert(format!("{prefix}.{path}"), value)?;
        }
        Ok(output)
    }

    /// Extends this set, rejecting path collisions.
    pub fn extend(&mut self, other: Self) -> Result<(), ObservationError> {
        for (path, value) in other.values {
            self.insert(path, value)?;
        }
        Ok(())
    }
}

/// One activation-path selector.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "match", content = "path", rename_all = "snake_case")]
pub enum ObservationSelector {
    /// Select exactly one stable path.
    Exact(String),
    /// Select a path and all descendants separated by `.`.
    Prefix(String),
}

impl ObservationSelector {
    /// Returns whether this selector accepts `path`.
    pub fn matches(&self, path: &str) -> bool {
        match self {
            Self::Exact(expected) => path == expected,
            Self::Prefix(prefix) => {
                path == prefix
                    || path
                        .strip_prefix(prefix)
                        .is_some_and(|suffix| suffix.starts_with('.'))
            }
        }
    }
}

/// Explicit selection for an instrumented execution pass.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationRequest {
    selectors: Vec<ObservationSelector>,
}

impl ObservationRequest {
    /// Selects every observation point reached by the operation.
    pub const fn all() -> Self {
        Self {
            selectors: Vec::new(),
        }
    }

    /// Selects the supplied exact paths or prefixes.
    pub fn selected(selectors: impl IntoIterator<Item = ObservationSelector>) -> Self {
        Self {
            selectors: selectors.into_iter().collect(),
        }
    }

    /// Returns whether a named point is requested.
    pub fn matches(&self, path: &str) -> bool {
        self.selectors.is_empty() || self.selectors.iter().any(|selector| selector.matches(path))
    }

    /// Requested selectors; empty means all points.
    pub fn selectors(&self) -> &[ObservationSelector] {
        &self.selectors
    }
}

/// A completed instrumented operation and its portable observations.
#[derive(Debug)]
pub struct InspectedOutput<O> {
    /// Ordinary backend output from the operation.
    pub output: O,
    /// Requested host-materialized observations.
    pub observations: ObservationSet,
}

/// Invalid portable observation data.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ObservationError {
    /// Tensor element-count multiplication overflowed.
    #[error("tensor observation shape element count overflowed")]
    ShapeOverflow,
    /// Tensor shape and host values disagree.
    #[error(
        "tensor observation shape {shape:?} requires {expected} values, but received {actual}"
    )]
    ElementCount {
        /// Logical shape.
        shape: Vec<usize>,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Observation paths must be nonempty.
    #[error("observation path must not be empty")]
    EmptyPath,
    /// Observation paths are unique within one operation.
    #[error("duplicate observation path {0:?}")]
    DuplicatePath(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_shape_and_values_must_agree() {
        let tensor = TensorObservation::new(
            vec![2, 2],
            TensorObservationData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        assert_eq!(tensor.shape(), [2, 2]);
        assert!(matches!(tensor.data(), TensorObservationData::F32(_)));
        assert!(matches!(
            TensorObservation::new(vec![2], TensorObservationData::I64(vec![1])),
            Err(ObservationError::ElementCount { .. })
        ));
    }

    #[test]
    fn selectors_and_sets_are_stable_and_collision_safe() {
        let request = ObservationRequest::selected([
            ObservationSelector::Exact("model.logits".into()),
            ObservationSelector::Prefix("model.layers.2".into()),
        ]);
        assert!(request.matches("model.logits"));
        assert!(request.matches("model.layers.2.output"));
        assert!(!request.matches("model.layers.20.output"));

        let mut set = ObservationSet::new();
        set.insert("model.logits", ObservationValue::Unsigned(3))
            .unwrap();
        assert_eq!(
            set.insert("model.logits", ObservationValue::Unsigned(4)),
            Err(ObservationError::DuplicatePath("model.logits".into()))
        );
    }
}
