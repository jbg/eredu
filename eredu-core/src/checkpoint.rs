//! Neutral checkpoint tensor catalog contracts.

use serde::{Deserialize, Serialize};
use std::{any::Any, collections::BTreeMap, fmt, sync::Arc};

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

/// Validated immutable name-indexed tensor catalog. Clones share its storage.
#[derive(Debug, Clone)]
pub struct TensorCatalog(Arc<CatalogData>);
#[derive(Debug)]
struct CatalogData {
    tensors: BTreeMap<String, TensorDescriptor>,
    _custody: Option<Arc<dyn CatalogCustody>>,
}
trait CatalogCustody: Any + fmt::Debug + Send + Sync {}
impl<C: Any + fmt::Debug + Send + Sync> CatalogCustody for C {}
impl PartialEq for TensorCatalog {
    fn eq(&self, other: &Self) -> bool {
        self.0.tensors == other.0.tensors
    }
}
impl Eq for TensorCatalog {}
impl Serialize for TensorCatalog {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> {
            tensors: &'a BTreeMap<String, TensorDescriptor>,
        }
        Fields {
            tensors: &self.0.tensors,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for TensorCatalog {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            tensors: BTreeMap<String, TensorDescriptor>,
        }
        let fields = Fields::deserialize(deserializer)?;
        Ok(Self(Arc::new(CatalogData {
            tensors: fields.tensors,
            _custody: None,
        })))
    }
}
impl TensorCatalog {
    /// Validates unique names and non-zero dimensions. An empty shape is a
    /// valid rank-zero scalar with one element.
    pub fn new(tensors: impl IntoIterator<Item = TensorDescriptor>) -> Result<Self, CatalogError> {
        Self::construct(tensors, None)
    }
    /// Constructs after the caller's metadata admission and retains its custody
    /// across catalog clones. Serialization includes values, not policy custody.
    /// On construction failure this argument is dropped; callers that need
    /// custody retained with an error must keep a separate shared owner.
    pub fn with_custody<C: Any + fmt::Debug + Send + Sync>(
        tensors: impl IntoIterator<Item = TensorDescriptor>,
        custody: C,
    ) -> Result<Self, CatalogError> {
        Self::construct(tensors, Some(Arc::new(custody)))
    }
    fn construct(
        tensors: impl IntoIterator<Item = TensorDescriptor>,
        custody: Option<Arc<dyn CatalogCustody>>,
    ) -> Result<Self, CatalogError> {
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
        Ok(Self(Arc::new(CatalogData {
            tensors: map,
            _custody: custody,
        })))
    }
    /// Looks up a descriptor by canonical name.
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.0.tensors.get(name)
    }
    /// Iterates over descriptors in deterministic name order.
    pub fn descriptors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.0.tensors.values()
    }
    /// Number of cataloged tensors.
    pub fn len(&self) -> usize {
        self.0.tensors.len()
    }
    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.0.tensors.is_empty()
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

#[cfg(test)]
mod tests;
