//! Closed tensor/candidate record/frame storage within an original request account.
use super::{
    WorkingMemoryError, WorkingMemoryFundingRun, WorkingMemoryReservation,
    funding::CaptureTensorCustody,
};
use eredu_core::{
    ObservationDtype, ObservationValueType, SharedTensorObservation, TensorObservationData,
    capture::*, checkpoint::TensorDtype,
};
use std::{fmt, mem::size_of};

mod builder;
pub(in crate::working_memory) mod interventions;
pub(super) use builder::allocate;
mod plan;
// Only the frame holds partition provenance. Independently shared tensors retain
// their existing numerical custody; neither owner retains its enclosing frame.
pub(super) type CaptureFrameCustody = (
    CaptureTensorCustody,
    Option<eredu_nn::workspace::HostMetadataFunding>,
);
pub use builder::{CaptureStepFinishError, PreparedCaptureDelivery, PreparedCaptureStep};
pub use builder::{PendingCaptureDelivery, PendingCaptureDeliveryError};
pub use plan::CaptureStepHostPlan;

/// A checked frame construction or state-transition rejection.
#[derive(Debug, thiserror::Error)]
pub enum CaptureStepError {
    /// The original parent account cannot cover or continue construction.
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    /// Paid partition provenance destinations could not be constructed.
    #[error(transparent)]
    PartitionStorage(#[from] eredu_nn::Error),
    /// Partition provenance differs from this frame or was already installed.
    #[error("capture partition evidence differs from selection {index}")]
    PartitionState { index: usize },
    /// The immutable admitted raw tensor geometry rejected this coordinate.
    #[error(transparent)]
    Geometry(#[from] CaptureTensorGeometryError),
    /// The admitted capture contract rejected geometry or logical accounting.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// The shared finite-summary payload validation rejected this value.
    #[error(transparent)]
    Summary(#[from] crate::capture::CaptureSummaryError),
    /// The shared fixed-edge histogram validation rejected this value.
    #[error(transparent)]
    Histogram(#[from] crate::capture::CaptureHistogramError),
    /// Only the implemented unpartitioned floating capture records are accepted.
    #[error("unsupported capture frame selection {index}")]
    UnsupportedSelection {
        /// Original selection index.
        index: usize,
    },
    /// The indexed record is absent, skipped, or has already been completed.
    #[error("capture frame record {index} is not pending")]
    RecordNotPending {
        /// Original selection index.
        index: usize,
    },
    /// The supplied tensor/dtype differs from this closed selection geometry.
    #[error("capture frame tensor differs from selection {index}")]
    TensorMismatch {
        /// Original selection index.
        index: usize,
    },
    /// The actual successful record exceeds its already charged wire bound.
    #[error("capture frame record {index} exceeds its encoded reservation")]
    EncodedSize {
        /// Original selection index.
        index: usize,
    },
    /// Completion metadata is invalid; it never provides transaction authority.
    #[error("invalid capture frame completion metadata")]
    InvalidCompletion,
    /// This source needs an additional original edit/evidence producer.
    #[error("original intervention destination is unavailable for operation {index}")]
    UnsupportedIntervention { index: usize },
    /// This attributed intervention is absent, inactive, repeated or unfinished.
    #[error("intervention record {index} is not in the required state")]
    InterventionState { index: usize },
}

// Actual sidecar payload allocated before record mutation. Each buffer can move
// into its corresponding frame field exactly once, without growth.
struct RecordBuffers {
    routed: Option<crate::working_memory::capture_run::RoutedInvocationTarget>,
    source: Vec<u64>,
    selected: Vec<u64>,
    diagnostic: Vec<u8>,
}

impl WorkingMemoryFundingRun {
    /// Allocates one closed frame from this original request's protected reserve.
    ///
    /// The opaque reservation must belong to this exact run. The plan supplies
    /// concrete construction geometry, not caller-declared bytes. Every buffer
    /// is protected before allocation, and native adoption cannot spend the hold.
    /// The enclosing worker must already price the admitted source plan, every
    /// simultaneously surviving frame/tensor, and native work in its original
    /// quote, authenticate invocation/provenance, and spend logical capture quota.
    /// This method grants none of those authorities or native completion.
    pub fn prepare_capture_step<'a>(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: CaptureStepHostPlan<'a>,
    ) -> Result<PreparedCaptureStep<'a>, CaptureStepError> {
        let custody = self.hold_capture_step(reservation, &plan)?;
        builder::allocate(plan, custody)
    }
}

#[cfg(test)]
mod tests;
