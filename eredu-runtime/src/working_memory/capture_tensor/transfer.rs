//! Original-account host construction bound to a caller-owned native scope.
use super::*;
use crate::working_memory::{
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkingMemoryReservation,
    WorkingMemoryStorage,
};

impl WorkingMemoryFundingRun {
    /// Constructs only the host destination, while binding its parent and every
    /// provided source origin to the same active native scope under one lock.
    /// Source mapping/completeness and the quoted native operation remain the
    /// backend's closed-program contract; storage handles alone cannot prove it.
    ///
    /// All source pins, including prior scope pins, travel with `native` through
    /// settlement or quarantine. Constructor failure restores the prior bundle;
    /// after success, execution failure must keep the enlarged bundle and actual
    /// native recovery payload until explicit completion. No native permission
    /// or automatic certification is supplied by this host-only constructor.
    pub fn prepare_capture_tensor_with_source<'a, 's, K: Clone + Ord + Send + Sync + 'static>(
        &self,
        reservation: &WorkingMemoryReservation,
        native: &'s mut WorkingMemoryFundingScope,
        plan: CaptureTensorHostPlan<'a>,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<PreparedCaptureTensorTransfer<'a, 's, K>, CaptureTensorConstructionError> {
        let (custody, rollback) =
            self.hold_capture_tensor_with_source(reservation, native, &plan, &complete_source)?;
        let builder = allocate(plan, custody)?;
        let native = rollback.commit();
        Ok(PreparedCaptureTensorTransfer {
            builder,
            source: complete_source,
            native,
        })
    }
}

/// Fixed host buffers, temporary source witness and an exclusive borrow of the
/// exact native scope. A sibling scope in the same account cannot substitute
/// for it, and the caller cannot certify it while this builder/error is alive.
/// No raw buffer, custody or
/// numerical authority export exists. The final immutable owner retains only
/// host custody; native source pins remain on the caller's native scope.
pub struct PreparedCaptureTensorTransfer<'a, 's, K: Ord + Send + 'static> {
    builder: PreparedCaptureTensor<'a>,
    source: WorkingMemoryStorage<K>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> PreparedCaptureTensorTransfer<'_, '_, K> {
    /// Rechecks parent liveness, exact active native account and source origins.
    /// Call before native work and again after settling before host iteration.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Fixed output shape from the admitted selection.
    pub fn shape(&self) -> &[usize] {
        self.builder.shape()
    }
    /// Fixed scalar count.
    pub fn len(&self) -> usize {
        self.builder.len()
    }
    /// Whether this destination has zero values.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }
    /// Number of initialized values.
    pub fn initialized_count(&self) -> usize {
        self.builder.initialized_count()
    }
    /// Protected host envelope, with no native allowance implied.
    pub fn protected_bytes(&self) -> u64 {
        self.builder.protected_bytes()
    }
    /// Fill one settled scalar without allocation. Caller still owns native
    /// recovery; this method does not establish that the scalar was settled.
    pub fn push_f32(&mut self, value: f32) -> Result<(), WorkingMemoryError> {
        self.validate()?;
        self.builder.push_f32(value)
    }
    /// Copies one unsigned scalar into the same fixed, source-bound destination.
    pub fn push_u64(&mut self, value: u64) -> Result<(), WorkingMemoryError> {
        self.validate()?;
        self.builder.push_u64(value)
    }
}
impl<'a, 's, K: Ord + Send + 'static> PreparedCaptureTensorTransfer<'a, 's, K> {
    /// Finish host custody only; temporary source witness retires independently
    /// of the native scope's complete pins. Failed finish owns partial payload.
    pub fn finish(
        self,
    ) -> Result<SharedTensorObservation, CaptureTensorTransferFinishError<'a, 's, K>> {
        if let Err(error) = self.validate() {
            return Err(CaptureTensorTransferFinishError(
                TransferFailure::Rejected {
                    builder: self,
                    error,
                },
            ));
        }
        if self.initialized_count() != self.len() {
            return Err(CaptureTensorTransferFinishError(
                TransferFailure::Incomplete(self),
            ));
        }
        self.builder
            .finish()
            .map_err(|error| CaptureTensorTransferFinishError(TransferFailure::Host(error)))
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for PreparedCaptureTensorTransfer<'_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PreparedCaptureTensorTransfer")
            .field(&self.builder)
            .finish()
    }
}
enum TransferFailure<'a, 's, K: Ord + Send + 'static> {
    Incomplete(PreparedCaptureTensorTransfer<'a, 's, K>),
    Rejected {
        builder: PreparedCaptureTensorTransfer<'a, 's, K>,
        error: WorkingMemoryError,
    },
    Host(CaptureTensorFinishError<'a>),
}
/// Retains failed destination and its funding; only healthy incomplete fill can
/// resume. No owning raw observation export or native certification is provided.
pub struct CaptureTensorTransferFinishError<'a, 's, K: Ord + Send + 'static>(
    TransferFailure<'a, 's, K>,
);
impl<'a, 's, K: Ord + Send + 'static> CaptureTensorTransferFinishError<'a, 's, K> {
    /// Recovers an incomplete, otherwise accepted transfer builder.
    pub fn into_builder(self) -> Result<PreparedCaptureTensorTransfer<'a, 's, K>, Self> {
        match self.0 {
            TransferFailure::Incomplete(builder) => Ok(builder),
            other => Err(Self(other)),
        }
    }

    pub(in crate::working_memory) fn into_terminal(self) -> TerminalFailure {
        match self.0 {
            TransferFailure::Incomplete(builder) => TerminalFailure::Incomplete {
                initialized: builder.initialized_count(),
                expected: builder.len(),
            },
            TransferFailure::Rejected { error, .. } => TerminalFailure::Rejected(error),
            TransferFailure::Host(error) => error.into_terminal(),
        }
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for CaptureTensorTransferFinishError<'_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<K: Ord + Send + 'static> fmt::Display for CaptureTensorTransferFinishError<'_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            TransferFailure::Incomplete(builder) => write!(
                f,
                "capture transfer has {} of {} values",
                builder.initialized_count(),
                builder.len()
            ),
            TransferFailure::Rejected { builder, error } => write!(
                f,
                "capture transfer with {} values rejected: {error}",
                builder.initialized_count()
            ),
            TransferFailure::Host(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl<K: Ord + Send + 'static> std::error::Error for CaptureTensorTransferFinishError<'_, '_, K> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            TransferFailure::Rejected { error, .. } => Some(error),
            TransferFailure::Host(error) => std::error::Error::source(error),
            _ => None,
        }
    }
}

// Scheduled caller already owns the single total-H hold. Reuse the same exact
// allocation and transfer representation, including its exclusive native borrow.
pub(in crate::working_memory) fn allocate_scheduled<
    'a,
    's,
    K: Clone + Ord + Send + Sync + 'static,
>(
    plan: CaptureTensorHostPlan<'a>,
    custody: CaptureTensorCustody,
    native: &'s mut WorkingMemoryFundingScope,
    complete_source: WorkingMemoryStorage<K>,
    segment: Option<&'s mut crate::working_memory::CaptureSourceSegment>,
) -> Result<PreparedCaptureTensorTransfer<'a, 's, K>, CaptureTensorConstructionError> {
    let rollback = match segment {
        Some(segment) => custody.bind_segment_source(native, segment, &complete_source)?,
        None => custody.bind_scheduled_source(native, &complete_source)?,
    };
    #[cfg(test)]
    crate::working_memory::capture_run::tests::before_transfer_allocation();
    let builder = allocate(plan, custody)?;
    let native = rollback.commit();
    Ok(PreparedCaptureTensorTransfer {
        builder,
        source: complete_source,
        native,
    })
}
