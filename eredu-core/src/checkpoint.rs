//! Neutral checkpoint tensor catalog contracts.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Portable tensor element type.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDtype {
    /// Boolean.
    Bool,
    /// IEEE f32.
    F32,
    /// IEEE f16.
    F16,
    /// Brain float 16.
    Bf16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE f64.
    F64,
    /// Complex number represented by two IEEE f32 values.
    Complex64,
    /// Backend-independent encoded/quantized storage.
    Encoded(String),
}

/// Location of bytes within one checkpoint artifact member.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorStorage {
    /// Logical file/member name.
    pub member: String,
    /// Byte offset from the member start.
    pub offset: u64,
    /// Stored byte length.
    pub length: u64,
}

/// Tensor descriptor without a materialized runtime array.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Canonical checkpoint name.
    pub name: String,
    /// Row-major logical shape.
    pub shape: Vec<usize>,
    /// Logical or encoded dtype.
    pub dtype: TensorDtype,
    /// Optional source location.
    pub storage: Option<TensorStorage>,
}

/// Validated name-indexed tensor catalog.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorCatalog {
    tensors: BTreeMap<String, TensorDescriptor>,
}

impl TensorCatalog {
    /// Validates unique names and non-zero dimensions. An empty shape is a
    /// valid rank-zero scalar with one element.
    pub fn new(tensors: impl IntoIterator<Item = TensorDescriptor>) -> Result<Self, CatalogError> {
        let mut map = BTreeMap::new();
        for tensor in tensors {
            if tensor.name.trim().is_empty() {
                return Err(CatalogError::EmptyName);
            }
            if tensor.shape.contains(&0) {
                return Err(CatalogError::InvalidShape(tensor.name));
            }
            let name = tensor.name.clone();
            if map.insert(name.clone(), tensor).is_some() {
                return Err(CatalogError::Duplicate(name));
            }
        }
        Ok(Self { tensors: map })
    }
    /// Looks up a descriptor by canonical name.
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }
    /// Iterates over descriptors in deterministic name order.
    pub fn descriptors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }
    /// Number of cataloged tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Tensor catalog validation error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CatalogError {
    /// Tensor name is empty.
    #[error("checkpoint tensor name must not be empty")]
    EmptyName,
    /// Tensor name is duplicated.
    #[error("duplicate checkpoint tensor {0}")]
    Duplicate(String),
    /// Shape contains a zero dimension.
    #[error("checkpoint tensor {0} has an invalid shape")]
    InvalidShape(String),
}
