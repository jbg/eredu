//! Closed scheduling authorities consumed by the shared prefill traversal.
use super::*;
use crate::working_memory::SpeculativePrefillScheduleAuthority;

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::working_memory::InferenceRequest {}
    impl Sealed for crate::working_memory::SpeculativePrefillScheduleAuthority {}
}

/// A retained admitted scheduler owner. Implementations are closed to the two
/// runtime account kinds; descriptive traces cannot implement native authority.
pub trait PrefillSchedulingAuthority: sealed::Sealed + Clone {
    #[doc(hidden)]
    fn geometry(&self) -> InferenceGeometry;
    #[doc(hidden)]
    fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError>;
    #[doc(hidden)]
    fn validate_same_request(&self, other: &Self) -> Result<(), WorkingMemoryError>;
    #[doc(hidden)]
    fn begin_prefill(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError>;
    #[doc(hidden)]
    fn inference_request(&self) -> Option<&InferenceRequest>;
    #[doc(hidden)]
    fn speculative_schedule(&self) -> Option<&SpeculativePrefillScheduleAuthority>;
}
impl PrefillSchedulingAuthority for InferenceRequest {
    fn geometry(&self) -> InferenceGeometry {
        self.geometry()
    }
    fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.validate(execution, geometry)
    }
    fn validate_same_request(&self, other: &Self) -> Result<(), WorkingMemoryError> {
        self.validate_same_request(other)
    }
    fn begin_prefill(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.begin_prefill(execution, geometry)
    }
    fn inference_request(&self) -> Option<&InferenceRequest> {
        Some(self)
    }
    fn speculative_schedule(&self) -> Option<&SpeculativePrefillScheduleAuthority> {
        None
    }
}
impl PrefillSchedulingAuthority for SpeculativePrefillScheduleAuthority {
    fn geometry(&self) -> InferenceGeometry {
        self.geometry()
    }
    fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.validate(execution, geometry)
    }
    fn validate_same_request(&self, other: &Self) -> Result<(), WorkingMemoryError> {
        if self.same_schedule(other) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    fn begin_prefill(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.begin(execution, geometry)
    }
    fn inference_request(&self) -> Option<&InferenceRequest> {
        None
    }
    fn speculative_schedule(&self) -> Option<&SpeculativePrefillScheduleAuthority> {
        Some(self)
    }
}

impl<T, C: Completion, R: PrefillSchedulingAuthority> PrefillDriver<T, C, R> {
    /// Uses the same traversal while retaining the selected admitted account.
    /// A speculative scheduling account supplies no native allocation permission.
    pub(crate) fn new_scheduled(
        execution: &InferenceExecutionIdentity,
        reservation: R,
        geometry: InferenceGeometry,
        cancellation: GenerationCancellationToken,
    ) -> Result<Self, WorkingMemoryError> {
        reservation.begin_prefill(execution, geometry)?;
        Ok(Self {
            geometry,
            reservation,
            cancellation,
            next: 0,
            pending: None,
            failed: false,
            cancelled: false,
        })
    }
}

impl<T, C: Completion> PrefillDriver<T, C, SpeculativePrefillScheduleAuthority> {
    /// Starts the shared driver under an admitted speculative schedule issuer.
    /// Executors must independently admit and retain every native occurrence.
    pub fn new_speculative_schedule(
        execution: &InferenceExecutionIdentity,
        authority: SpeculativePrefillScheduleAuthority,
        cancellation: GenerationCancellationToken,
    ) -> Result<Self, WorkingMemoryError> {
        let geometry = authority.geometry();
        Self::new_scheduled(execution, authority, geometry, cancellation)
    }
}
