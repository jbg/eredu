//! MLX materialization of backend-neutral checkpoint tensor leases.
//!
//! A [`crate::backend::runtime::checkpoint::store::WeightLease`] pins the bytes backing a safetensors
//! view. Materialization returns an owning completion guard that retains the
//! view through asynchronous MLX evaluation. Callers may order a compatible
//! consumer stream without blocking the host, or synchronize the exact
//! materialization before taking its independently owned output.

use eredu_checkpoint::store::{StoreError, TensorMetadata, TensorSelection};
use eredu_checkpoint::{
    gguf_store::GgufLease as NeutralGgufLease,
    store::{
        CheckpointLease, EncodedTensorLease, MemoryLease as NeutralMemoryLease,
        SafetensorsLease as NeutralSafetensorsLease,
    },
    StoredDtype,
};

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};

use safemlx::{ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Event, Stream};
use safetensors::tensor::{Dtype, TensorView};

use super::gguf::GgufTensor;

#[cfg(test)]
use super::gguf::GgufCheckpoint;
#[cfg(test)]
use eredu_checkpoint::gguf_store::{
    GgufPhysicalSelection, GgufWeightStore as NeutralGgufWeightStore,
};
#[cfg(test)]
use eredu_checkpoint::store::{
    CheckpointSource, ReadPolicy as WeightReadPolicy, SafetensorsWeightStore, TensorReadRequest,
    WeightStoreBackend, WeightStoreDiagnostics,
};
#[cfg(test)]
use eredu_gguf::TensorSelection as GgufTensorSelection;

pub(super) fn safetensors_dtype(
    key: &str,
    value: &StoredDtype,
) -> Result<Dtype, CheckpointMaterializationError> {
    match value {
        StoredDtype::Bool => Ok(Dtype::BOOL),
        StoredDtype::U8 => Ok(Dtype::U8),
        StoredDtype::I8 => Ok(Dtype::I8),
        StoredDtype::I16 => Ok(Dtype::I16),
        StoredDtype::U16 => Ok(Dtype::U16),
        StoredDtype::F16 => Ok(Dtype::F16),
        StoredDtype::BF16 => Ok(Dtype::BF16),
        StoredDtype::I32 => Ok(Dtype::I32),
        StoredDtype::U32 => Ok(Dtype::U32),
        StoredDtype::F32 => Ok(Dtype::F32),
        StoredDtype::F64 => Ok(Dtype::F64),
        StoredDtype::I64 => Ok(Dtype::I64),
        StoredDtype::U64 => Ok(Dtype::U64),
        StoredDtype::C64 => Err(CheckpointMaterializationError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
        StoredDtype::F8E4M3 => Ok(Dtype::F8_E4M3),
        StoredDtype::F4 => Ok(Dtype::F4),
        StoredDtype::F8E8M0 => Ok(Dtype::F8_E8M0),
        StoredDtype::F8E5M2 => Err(CheckpointMaterializationError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
        StoredDtype::Other(_) => Err(CheckpointMaterializationError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
    }
}

/// Failures while lowering a neutral checkpoint lease into an MLX array.
///
/// Catalog, selection, mapping, and checkpoint I/O failures retain their
/// backend-neutral [`StoreError`] value in [`Self::Store`].
#[derive(Debug, thiserror::Error)]
pub enum CheckpointMaterializationError {
    /// Backend-neutral catalog, selection, mapping, or checkpoint I/O failure.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The stored encoding cannot be materialized by MLX.
    #[error("stored dtype {dtype:?} for tensor {key:?} is unsupported")]
    UnsupportedStoredDtype {
        /// Tensor key.
        key: String,
        /// Unsupported on-disk encoding.
        dtype: StoredDtype,
    },
    /// A materialization shape, element count, byte size, or MLX dimension overflowed.
    #[error("MLX checkpoint materialization overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: String,
    },
    /// A retained encoded lease could not provide a valid tensor view.
    #[error("invalid encoded checkpoint tensor {key:?} from {path}: {message}", path = .path.display())]
    InvalidEncodedTensor {
        /// Logical tensor key.
        key: String,
        /// Encoded payload path or a synthetic in-memory identity.
        path: PathBuf,
        /// Tensor-view validation detail.
        message: String,
    },
    /// Encoded tensor-to-MLX conversion failed.
    #[error("failed to convert checkpoint tensor {key:?}: {source}")]
    MlxConversion {
        /// Tensor key.
        key: String,
        /// Conversion error.
        #[source]
        source: safemlx::error::ConversionError,
    },
    /// An MLX selection, copy, or evaluation operation failed.
    #[error("MLX {operation} failed for tensor {key:?}: {source}")]
    Mlx {
        /// Tensor key.
        key: String,
        /// Operation being performed.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// MLX's converted-GGUF lease cache was poisoned by a prior panic.
    #[error("MLX converted-GGUF cache state is unavailable")]
    StatePoisoned,
    /// Portable GGUF payload conversion to MLX failed.
    #[error("failed to convert GGUF tensor {key:?} to MLX: {message}")]
    GgufConversion {
        /// Requested logical tensor.
        key: String,
        /// Backend failure detail.
        message: String,
    },
}

#[derive(Debug)]
struct CachedGgufGroup {
    arrays: Vec<(String, Array)>,
}

/// MLX stream and lease-coalescing state used for neutral parameter realization.
#[derive(Debug, Clone)]
pub struct MlxParameterMaterializationContext {
    source_stream: Stream,
    execution_stream: Stream,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
}

impl MlxParameterMaterializationContext {
    /// Creates a reusable materialization context for one source/execution stream pair.
    pub fn new(source_stream: &Stream, execution_stream: &Stream) -> Self {
        Self {
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            converted_groups: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Host/source stream used to create and transform checkpoint arrays.
    pub const fn source_stream(&self) -> &Stream {
        &self.source_stream
    }

    /// Destination stream used for the final execution weight.
    pub const fn execution_stream(&self) -> &Stream {
        &self.execution_stream
    }

    /// Wraps a neutral checkpoint lease for MLX materialization.
    pub fn weight_lease(
        &self,
        lease: CheckpointLease,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        WeightLease::from_checkpoint_lease(lease, Arc::clone(&self.converted_groups))
    }
}

#[cfg(test)]
pub(crate) mod test_support;

mod leases;
mod materialization;

pub use leases::WeightLease;
pub use materialization::{PendingWeightMaterialization, WeightMaterialization};

#[cfg(test)]
use leases::WeightLeaseSource;

#[cfg(test)]
mod tests;
