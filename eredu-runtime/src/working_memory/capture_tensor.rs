//! One fresh F32 host destination; capture/native execution remains separate.
use super::{WorkingMemoryError, WorkingMemoryPool, funding::CaptureTensorCustody};
use eredu_core::{
    ObservationError, SharedTensorObservation, TensorObservation, TensorObservationData,
    capture::CaptureTensorGeometry,
};
use std::{fmt, mem::size_of};

mod parent_account;
pub(in crate::working_memory) mod prefill;
mod transfer;
pub(super) use transfer::allocate_scheduled;
pub use transfer::{CaptureTensorTransferFinishError, PreparedCaptureTensorTransfer};

/// Limits on this independent host destination, never an allocation declaration.
#[derive(Debug, Clone, Copy)]
pub struct CaptureTensorLimits {
    /// Complete domain ceiling, including all existing sources and accounts.
    pub capacity_bytes: u64,
    /// Optional additional ceiling on this exact closed construction envelope.
    pub application_memory_budget_bytes: Option<u64>,
}
impl CaptureTensorLimits {
    /// Select a domain ceiling without an additional application limit.
    pub const fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            application_memory_budget_bytes: None,
        }
    }
}

/// Checked allocation program derived from one actual admitted output geometry.
/// Source/native backing and transfer scratch are intentionally not included.
#[derive(Debug)]
pub struct CaptureTensorHostPlan<'a> {
    geometry: CaptureTensorGeometry<'a>,
    retained: u64,
    peak: u64,
}
impl<'a> CaptureTensorHostPlan<'a> {
    /// Price shape/data, final shared controls and explicit closed construction moves.
    /// No payload is copied or allocated and no scope is acquired here.
    pub fn prepare(geometry: CaptureTensorGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let allocation = |count: usize, width: usize| {
            count
                .checked_mul(width)
                .filter(|bytes| *bytes <= isize::MAX as usize)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(WorkingMemoryError::Overflow)
        };
        let shape = allocation(geometry.shape().len(), size_of::<usize>())?;
        let data = allocation(geometry.elements(), size_of::<f32>())?;
        let inline = u64::try_from(size_of::<TensorObservation>())
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let retained = inline
            .checked_add(shape)
            .and_then(|n| n.checked_add(data))
            .ok_or(WorkingMemoryError::Overflow)?;
        let shared_controls =
            SharedTensorObservation::retained_control_bytes::<CaptureTensorCustody>()
                .ok_or(WorkingMemoryError::Overflow)?;
        // The final Arc already contains the retained DTO, so replace that
        // inline term instead of counting it twice. Keep the existing two DTO
        // moves and shape/scalar fill values. The same original P covers the
        // exact Arc and concrete custody Box until their deallocation; payload
        // diagnostics remain DTO/shape/data only. Allocator bookkeeping and
        // compiler frames are not measured by these concrete layout facts.
        let shape_moves = if geometry.shape().is_empty() {
            0
        } else {
            2 * size_of::<usize>() as u64
        };
        let scalar_moves = if geometry.elements() == 0 {
            0
        } else {
            2 * size_of::<f32>() as u64
        };
        let peak = shape
            .checked_add(data)
            .and_then(|n| n.checked_add(shared_controls))
            .and_then(|n| n.checked_add(inline.checked_mul(2)?))
            .and_then(|n| n.checked_add(shape_moves))
            .and_then(|n| n.checked_add(scalar_moves))
            .and_then(|n| {
                n.checked_add(
                    u64::try_from(CaptureTensorGeometry::preparation_control_bytes()?).ok()?,
                )
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            geometry,
            retained,
            peak,
        })
    }
    /// The actual immutable selection/phase geometry borrowed by this plan.
    pub fn geometry(&self) -> &CaptureTensorGeometry<'a> {
        &self.geometry
    }
    /// Actual retained DTO, shape and data capacities.
    pub fn retained_payload_bytes(&self) -> u64 {
        self.retained
    }
    /// Protected payload, final shared controls and explicit construction/fill moves.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
}

/// Typed rejection before the host destination buffers exist.
#[derive(Debug, thiserror::Error)]
pub enum CaptureTensorConstructionError {
    /// Existing domain policy/accounting rejected this exact construction.
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    /// Exact P does not fit the application's independent destination limit.
    #[error("capture tensor needs {required_bytes} bytes; application limit is {budget_bytes}")]
    ApplicationBudgetExceeded {
        /// Exact closed construction envelope.
        required_bytes: u64,
        /// Caller-selected upper limit.
        budget_bytes: u64,
    },
}

impl WorkingMemoryPool {
    /// Reserve and allocate one bounded host transfer destination.
    ///
    /// The source is actual admitted geometry, not a caller byte declaration or
    /// arbitrary allocation callback. This does not spend capture quota, read a
    /// source tensor, authorize transfer, or enable managed instrumentation.
    /// A native caller independently validates the source and funds all work
    /// before pushing resulting scalars. Copying an existing host observation
    /// with source credit requires a separate source-owning operation.
    pub fn prepare_capture_tensor<'a>(
        &self,
        plan: CaptureTensorHostPlan<'a>,
        limits: CaptureTensorLimits,
    ) -> Result<PreparedCaptureTensor<'a>, CaptureTensorConstructionError> {
        if let Some(budget_bytes) = limits.application_memory_budget_bytes {
            if plan.peak > budget_bytes {
                return Err(CaptureTensorConstructionError::ApplicationBudgetExceeded {
                    required_bytes: plan.peak,
                    budget_bytes,
                });
            }
        }
        let custody = self.open_capture_tensor_account(&plan, limits.capacity_bytes)?;
        allocate(plan, custody)
    }
}

// Shared closed worker for independent and original-parent funding. A parent
// may become fenced after the hold commit; recheck before either buffer exists.
pub(super) fn allocate<'a>(
    plan: CaptureTensorHostPlan<'a>,
    custody: CaptureTensorCustody,
) -> Result<PreparedCaptureTensor<'a>, CaptureTensorConstructionError> {
    custody.validate()?;
    #[cfg(test)]
    tests::before_allocate();
    // P exists before either requested-capacity vector. Pinned Vec source
    // establishes exact capacity for these non-ZST element types; no push
    // reaches capacity and no reallocation/DTO copy occurs at finish.
    let mut shape = Vec::with_capacity(plan.geometry.shape().len());
    for &dimension in plan.geometry.shape() {
        shape.push(dimension);
    }
    let data = Vec::with_capacity(plan.geometry.elements());
    Ok(PreparedCaptureTensor {
        shape,
        data,
        plan,
        custody,
    })
}

/// Partial destination with no mutable slice/Vec/raw scope escape.
/// Values precede the final private host custody on ordinary drop and unwind.
#[must_use = "finish this exact destination or retire its partial payload"]
pub struct PreparedCaptureTensor<'a> {
    shape: Vec<usize>,
    data: Vec<f32>,
    plan: CaptureTensorHostPlan<'a>,
    custody: CaptureTensorCustody,
}
impl PreparedCaptureTensor<'_> {
    /// Fixed destination shape from the immutable admitted selection.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Required number of destination scalars.
    pub fn len(&self) -> usize {
        self.plan.geometry.elements()
    }
    /// Whether the selected tensor contains no scalars.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Scalars already installed in the one fixed allocation.
    pub fn initialized_count(&self) -> usize {
        self.data.len()
    }
    /// Write only a previously initialized destination slot. This is used by
    /// the shared disjoint-prefix copier; it cannot grow or complete a partial
    /// buffer and grants no native allocation/completion authority.
    pub(in crate::working_memory) fn replace_initialized_f32(&mut self,index:usize,value:f32)->Result<(),WorkingMemoryError> {
        self.custody.validate()?;
        if self.data.len()!=self.len(){return Err(WorkingMemoryError::IdentityMismatch);}
        let slot=self.data.get_mut(index).ok_or(WorkingMemoryError::IdentityMismatch)?;
        *slot=value;Ok(())
    }
    /// Full original closed construction hold, not a native byte allowance.
    pub fn protected_bytes(&self) -> u64 {
        self.plan.peak
    }
    /// Copy one already-owned scalar without growing a buffer or creating native
    /// work. The native caller still owns its independent source/transfer custody.
    pub fn push_f32(&mut self, value: f32) -> Result<(), WorkingMemoryError> {
        self.custody.validate()?;
        if self.data.len() == self.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.data.push(value);
        Ok(())
    }
}
impl<'a> PreparedCaptureTensor<'a> {
    /// Complete the existing host payload without copying buffers or certifying
    /// native completion. Missing values retain the same funded partial owner.
    pub fn finish(self) -> Result<SharedTensorObservation, CaptureTensorFinishError<'a>> {
        if let Err(error) = self.custody.validate() {
            return Err(CaptureTensorFinishError(FinishFailure::Rejected {
                builder: self,
                error,
            }));
        }
        if self.data.len() != self.len() {
            return Err(CaptureTensorFinishError(FinishFailure::Incomplete(self)));
        }
        let Self {
            shape,
            data,
            plan,
            custody,
        } = self;
        let observation = match TensorObservation::new(shape, TensorObservationData::F32(data)) {
            Ok(observation) => observation,
            Err(error) => {
                return Err(CaptureTensorFinishError(FinishFailure::Invalid {
                    error,
                    _custody: custody,
                }));
            }
        };
        debug_assert_eq!(observation.retained_payload_bytes(), Some(plan.retained));
        Ok(SharedTensorObservation::retain(observation, custody))
    }
}
impl fmt::Debug for PreparedCaptureTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCaptureTensor")
            .field("shape", &self.shape)
            .field("initialized", &self.data.len())
            .field("protected_bytes", &self.plan.peak)
            .finish_non_exhaustive()
    }
}

enum FinishFailure<'a> {
    Incomplete(PreparedCaptureTensor<'a>),
    Rejected {
        builder: PreparedCaptureTensor<'a>,
        error: WorkingMemoryError,
    },
    // Defensive structural failure retains any shape owned by the original
    // error before the final custody. No owning error/payload extraction exists.
    Invalid {
        error: ObservationError,
        _custody: CaptureTensorCustody,
    },
}
/// An incomplete/invalid result retains all surviving host payload and custody.
pub struct CaptureTensorFinishError<'a>(FinishFailure<'a>);
impl<'a> CaptureTensorFinishError<'a> {
    /// Recover only an incomplete builder with the same fixed buffer and hold.
    pub fn into_builder(self) -> Result<PreparedCaptureTensor<'a>, Self> {
        match self.0 {
            FinishFailure::Incomplete(builder) => Ok(builder),
            other => Err(Self(other)),
        }
    }

    // Only scheduled terminal conversion calls this. Its separate shared H
    // custody stays live while partial buffers retire and any error-owned shape
    // moves to the final opaque error. No raw payload escapes or gets cloned.
    pub(super) fn into_terminal(self) -> TerminalFailure {
        match self.0 {
            FinishFailure::Incomplete(builder) => TerminalFailure::Incomplete {
                initialized: builder.initialized_count(),
                expected: builder.len(),
            },
            FinishFailure::Rejected { error, .. } => TerminalFailure::Rejected(error),
            FinishFailure::Invalid { error, .. } => TerminalFailure::Invalid(error),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum TerminalFailure {
    #[error("capture tensor has {initialized} of {expected} values")]
    Incomplete { initialized: usize, expected: usize },
    #[error(transparent)]
    Rejected(WorkingMemoryError),
    #[error(transparent)]
    Invalid(ObservationError),
}
impl fmt::Debug for CaptureTensorFinishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            FinishFailure::Incomplete(builder) => f
                .debug_tuple("IncompleteCaptureTensor")
                .field(builder)
                .finish(),
            FinishFailure::Rejected { builder, error } => f
                .debug_struct("RejectedCaptureTensor")
                .field("builder", builder)
                .field("error", error)
                .finish(),
            FinishFailure::Invalid { error, .. } => {
                f.debug_tuple("InvalidCaptureTensor").field(error).finish()
            }
        }
    }
}
impl fmt::Display for CaptureTensorFinishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            FinishFailure::Incomplete(builder) => write!(
                f,
                "capture tensor has {} of {} values",
                builder.initialized_count(),
                builder.len()
            ),
            FinishFailure::Rejected { error, .. } => fmt::Display::fmt(error, f),
            FinishFailure::Invalid { error, .. } => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for CaptureTensorFinishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            FinishFailure::Invalid { error, .. } => Some(error),
            FinishFailure::Rejected { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
