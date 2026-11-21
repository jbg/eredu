//! Exact geometry and operations for native append-stream storage.

use eredu_nn::{Error, Tensor, TensorElementType};
use std::ops::Range;

/// Exact lane count and finite storage limits bound before native state allocation.
/// Layer indices are local to the selected state layout. Byte limits are per lane;
/// admission must account for all lanes and all replicas of a binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendStreamBinding {
    /// Local state layer owning this stream.
    pub layer: usize,
    /// Number of independent sequence lanes.
    pub lanes: u32,
    /// Exact parameter-free record geometry.
    pub spec: AppendStreamSpec,
    /// Finite retained-history and operator-read limits.
    pub limits: AppendStreamLimits,
    /// Resident payload allowance for one lane.
    pub payload_bytes: u64,
    /// Compact read and tail-copy allowance for one lane.
    pub scratch_bytes: u64,
    /// Resident chunk catalog allowance for one lane.
    pub catalog_bytes: u64,
}

/// Physical storage provided for an admitted append stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendStreamBacking {
    /// Device-resident immutable chunks.
    Resident,
    /// Shared paged cache storage.
    Paged,
}

/// Exact identity, scalar representation and record width declared by an architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendStreamSpec {
    /// Stable slot within a layer and sequence lane.
    pub slot: u32,
    /// Number of scalar fields in one immutable record.
    pub width: i32,
    /// Exact storage type; integer columns are never promoted through floating point.
    pub element: TensorElementType,
}
impl AppendStreamSpec {
    /// Validates geometry and returns exact logical storage bytes per record.
    pub fn record_bytes(&self) -> Result<u64, AppendStreamError> {
        let bytes = match self.element {
            TensorElementType::Bool | TensorElementType::I8 | TensorElementType::U8 => 1,
            TensorElementType::F16
            | TensorElementType::Bf16
            | TensorElementType::I16
            | TensorElementType::U16 => 2,
            TensorElementType::F32 | TensorElementType::I32 | TensorElementType::U32 => 4,
            TensorElementType::F64
            | TensorElementType::I64
            | TensorElementType::U64
            | TensorElementType::Complex64 => 8,
        };
        if self.width <= 0 {
            return Err(AppendStreamError::Geometry);
        }
        Ok(self.width as u64 * bytes)
    }
}
/// Mechanism limits selected independently of stream identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendStreamLimits {
    /// Maximum immutable record count retained by this mechanism.
    pub entries: usize,
    /// Entries per physical chunk, including the mutable tail.
    pub page_entries: usize,
    /// Largest compact range returned to an operator.
    pub read_entries: usize,
}
impl AppendStreamLimits {
    /// Checks addressable, nonzero logical and physical bounds.
    pub fn validate(&self) -> Result<(), AppendStreamError> {
        if [self.entries, self.page_entries, self.read_entries]
            .into_iter()
            .any(|n| n == 0 || n > i32::MAX as usize)
        {
            return Err(AppendStreamError::Geometry);
        }
        Ok(())
    }

    /// Tail replacement and compact range copies can coexist with their sources.
    pub fn scratch_bytes(&self, spec: &AppendStreamSpec) -> Result<u64, AppendStreamError> {
        self.validate()?;
        (self.page_entries as u64 + self.read_entries as u64)
            .checked_mul(spec.record_bytes()?)
            .and_then(|n| n.checked_mul(2))
            .ok_or(AppendStreamError::Geometry)
    }
}
/// Typed admission, frontier and payload failures.
#[derive(Debug, thiserror::Error)]
pub enum AppendStreamError {
    /// Invalid dimensions, type or chunking policy.
    #[error("append stream geometry or scalar representation differs from its declaration")]
    Geometry,
    /// Exact append frontier did not match retained state.
    #[error("append stream frontier {actual} differs from expected {expected}")]
    Frontier {
        /// Caller frontier.
        expected: usize,
        /// Retained frontier.
        actual: usize,
    },
    /// A requested range is outside retained history.
    #[error("append stream range is outside retained history")]
    Range,
    /// Required storage or range exceeds the selected bound.
    #[error("append stream needs {required} {resource}, admitted {limit}")]
    Budget {
        /// Bounded resource.
        resource: &'static str,
        /// Required amount.
        required: u64,
        /// Admitted amount.
        limit: u64,
    },
    /// Backend range transfer, tensor construction or completion failure.
    #[error(transparent)]
    Tensor(#[from] Error),
}
/// Storage contract used by architecture-owned streaming algorithms.
///
/// Implementations may retain resident chunks or page immutable records through
/// a shared residency manager. Reads return only the requested compact records;
/// they must not gather the complete stream. Append/frontier updates are atomic
/// on synchronous failure. The execution owner retains native work until completion
/// and checkpoints all streams together with ordinary attention and fixed state.
pub trait AppendOnlyStream<T: Tensor> {
    /// Exact declaration retained at construction.
    fn specification(&self) -> &AppendStreamSpec;
    /// Records already committed to this stream.
    fn len(&self) -> usize;
    /// Whether the stream contains no records.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Appends `[entries,width]` at the specified exact frontier.
    fn append(
        &mut self,
        frontier: usize,
        values: T,
        context: &T::Context,
    ) -> Result<(), AppendStreamError>;
    /// Returns compact `[range.len(),width]` storage, within the read bound.
    fn read(&mut self, range: Range<usize>, context: &T::Context) -> Result<T, AppendStreamError>;
}
