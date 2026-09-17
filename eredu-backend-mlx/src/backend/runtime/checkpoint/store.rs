//! MLX materialization of backend-neutral checkpoint tensor leases.
//!
//! A [`crate::backend::runtime::checkpoint::store::WeightLease`] pins the bytes backing a safetensors
//! view. Materialization returns an owning completion guard that retains the
//! view through asynchronous MLX evaluation. Callers may order a compatible
//! consumer stream without blocking the host, or synchronize the exact
//! materialization before taking its independently owned output.

use eredu_checkpoint::store::{StoreError, TensorMetadata, TensorSelection};
use eredu_checkpoint::{
    StoredDtype,
    gguf_store::GgufLease as NeutralGgufLease,
    store::{
        CheckpointLease, EncodedTensorLease, MemoryLease as NeutralMemoryLease,
        SafetensorsLease as NeutralSafetensorsLease,
    },
};

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};

use safemlx::{Array, Event, Stream, ops::indexing::TryIndexOp, transforms::async_eval_with_event};
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
    /// Actual immutable host-copy cause and independent source/control custody.
    #[error(transparent)]
    PreparedGgufHostCopy(#[from] PreparedGgufHostCopyFailure),
    /// Original prepared GGUF failure retaining destinations, same source box and custody.
    #[error(transparent)]
    PreparedGguf(#[from] PreparedGgufMaterializationFailure),
    /// Actual admitted raw storage/source failure, retaining its allocation owner.
    #[error("{0}")]
    PreparedGgufAdmitted(#[from] PreparedGgufAdmittedFailure),
    /// Prepared acquisition failure retaining its actual source/destination and custody.
    #[error(transparent)]
    PreparedAcquisition(#[from] PreparedSourceAcquisitionFailure),
    /// Original native fixed cause, without a formatted key/error shell.
    #[error("original checkpoint operation: {0}")]
    OriginalNative(#[source] safemlx::error::Exception),
    /// Explicit original operation did not match its registered current role.
    #[error("original checkpoint operation domain mismatch")]
    OriginalOperationDomain,
    /// A selected prepared payload cannot hold the actual operation inputs.
    #[error("original {family} payload needs {required} rows but prepared {capacity}")]
    OriginalPayloadCapacity {
        /// Concrete destination family.
        family: &'static str,
        /// Actual caller-provided population.
        required: usize,
        /// Prepared final vector capacity.
        capacity: usize,
    },
    /// A prepared typed operation bank is exhausted; no fallback is permitted.
    #[error("original {family} operation storage exhausted after {prepared} slots")]
    OriginalOperationCapacity {
        /// Concrete prepared operation family.
        family: &'static str,
        /// Actual number of slots prepared in the consumed bank.
        prepared: usize,
    },

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

mod cache;
mod cache_context;
pub(crate) use cache_context::{CacheHandle, CacheInitializationError};

#[derive(Debug)]
struct CachedGgufGroup {
    arrays: CachedGgufArrays,
}

// Supplied metadata stays with the same weak-cache group; no owning names are
// reconstructed. Values retire before the final name-allocation receipts.
#[derive(Debug)]
enum CachedGgufArrays {
    Ordinary(Vec<(String, Array)>),
    Supplied {
        values: [Option<Array>; 3],
        names: eredu_gguf::StoredOutputNames<
            crate::backend::runtime::execution::generic::gguf_host_typed::supplied::AdmittedFamily,
        >,
    },
}
impl CachedGgufArrays {
    fn get(&self, name: &str) -> Option<&Array> {
        match self {
            Self::Ordinary(rows) => rows
                .iter()
                .find_map(|(key, value)| (key == name).then_some(value)),
            Self::Supplied { values, names } => names
                .iter()
                .position(|key| key == name)
                .and_then(|index| values[index].as_ref()),
        }
    }
}

/// MLX stream and lease-coalescing state used for neutral parameter realization.
#[derive(Debug, Clone)]
pub struct MlxParameterMaterializationContext {
    source_stream: Stream,
    execution_stream: Stream,
    converted_groups: CacheHandle,
}

impl MlxParameterMaterializationContext {
    /// Creates a reusable materialization context for one source/execution stream pair.
    pub fn new(source_stream: &Stream, execution_stream: &Stream) -> Self {
        let converted_groups = CacheHandle::ordinary();
        Self {
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            converted_groups,
        }
    }

    /// Consume an already prepared cache owner and owned stream wrappers.
    /// This constructor allocates nothing and never promotes an ordinary cache.
    /// Stream-handle and selected-source coverage remain separate obligations.
    pub(crate) fn with_cache(
        source_stream: Stream,
        execution_stream: Stream,
        converted_groups: CacheHandle,
    ) -> Self {
        Self {
            source_stream,
            execution_stream,
            converted_groups,
        }
    }

    pub(crate) fn cache_handle(&self) -> CacheHandle {
        self.converted_groups.clone()
    }

    pub(crate) fn matches_cache(&self, cache: &CacheHandle) -> bool {
        self.converted_groups.same(cache)
    }

    /// Once-only shared cache/context owner, excluding its dynamic rows and
    /// streams. This is existing storage, never another per-miss charge.
    pub(crate) fn cache_context_storage_bytes()
    -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        cache::context_storage_bytes()
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
        WeightLease::from_checkpoint_lease(lease, self.converted_groups.clone())
    }
}

#[cfg(test)]
pub(crate) mod test_support;

mod leases;
mod materialization;

pub use leases::WeightLease;
#[cfg(test)]
pub(crate) use materialization::OriginalGgufMissFixture;
pub use materialization::{
    GgufHostCopyCause, PendingWeightMaterialization, PreparedGgufAdmittedFailure,
    PreparedGgufHostCopyFailure, PreparedGgufMaterializationFailure, WeightMaterialization,
};

#[cfg(test)]
use leases::WeightLeaseSource;

#[cfg(test)]
mod tests;

pub(crate) use materialization::{
    PreparedMaterializationObservation, PreparedPendingWeight, PreparedWeightMaterialization,
};

pub(crate) use materialization::{MaterializationPayloadShape, OriginalMaterializationSlots};

mod acquisition;
pub use acquisition::PreparedSourceAcquisitionFailure;
pub(crate) use acquisition::PreparedSourceAcquisitions;

mod prepared_streams;
mod source_stream;
pub use prepared_streams::PreparedMaterializationStreamError;
pub(crate) use prepared_streams::{
    ManagerMaterializationContext, PreparedMaterializationStreams,
    prepare_one_materialization_stream, prepare_materialization_stream_from_plan,
};
pub use source_stream::{
    MaterializationSourceStreamError, MaterializationSourceWorkerError,
    PreparedMaterializationSourceStream, PreparedMaterializationSourceWorker,
};
/// Synchronous borrowed view; it owns no wrapper or callback and never escapes
/// into a native worker. Public context/Stream clone semantics remain intact.
#[derive(Clone, Copy)]
pub(crate) struct MaterializationView<'a> {
    source: &'a Stream,
    execution: &'a Stream,
    converted_groups: &'a CacheHandle,
}
impl<'a> From<&'a MlxParameterMaterializationContext> for MaterializationView<'a> {
    fn from(context: &'a MlxParameterMaterializationContext) -> Self {
        Self {
            source: &context.source_stream,
            execution: &context.execution_stream,
            converted_groups: &context.converted_groups,
        }
    }
}
impl<'a> From<&MaterializationView<'a>> for MaterializationView<'a> {
    fn from(view: &MaterializationView<'a>) -> Self {
        *view
    }
}
impl<'a> MaterializationView<'a> {
    pub(crate) fn source_stream(self) -> &'a Stream {
        self.source
    }
    pub(crate) fn execution_stream(self) -> &'a Stream {
        self.execution
    }
    pub(crate) fn weight_lease(
        self,
        lease: CheckpointLease,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        WeightLease::from_checkpoint_lease(lease, self.converted_groups.clone())
    }
}
