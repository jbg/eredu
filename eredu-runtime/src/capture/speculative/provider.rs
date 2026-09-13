//! Retained partition resources for the ordinary speculative capture owner.
use super::*;
use crate::capture::partition::*;
use eredu_core::Completion;
use std::sync::Arc;

/// Composes native primitives with exact retained layout and transport owners.
/// This provider does not select architecture membership or allocate native
/// values. Its observer retains prepaid source/receipt work for a whole phase.
pub struct PartitionCaptureBackendProvider<P, T, L, N, F> {
    provider: P,
    transport: Arc<T>,
    layout: Arc<L>,
    identity: PartitionCaptureIdentity,
    limits: PartitionCaptureReceiptLimits,
    estimate: N,
    map_partition_error: F,
}

impl<P, T, L, N, F> PartitionCaptureBackendProvider<P, T, L, N, F> {
    /// Retains already selected resources. `identity` must describe their loaded
    /// setup and parameter version. It binds before the first invocation, while
    /// the first actual forward epoch is claimed by shared preparation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: P,
        transport: Arc<T>,
        layout: Arc<L>,
        identity: PartitionCaptureIdentity,
        limits: PartitionCaptureReceiptLimits,
        estimate: N,
        map_partition_error: F,
    ) -> Self {
        Self {
            provider,
            transport,
            layout,
            identity,
            limits,
            estimate,
            map_partition_error,
        }
    }
}

impl<P, T, L, N, F> CaptureBackendProvider for PartitionCaptureBackendProvider<P, T, L, N, F>
where
    P: CaptureBackendProvider,
    T: PartitionCaptureHookTransport,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
    L: PartitionCaptureLayout + crate::intervention::PartitionActivationLayout,
    N: FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    F: Fn(PartitionCaptureObserverError<P::Error>) -> CaptureExecutionError<P::Error>,
{
    type Tensor = P::Tensor;
    type Error = P::Error;
    type Backend<'a>
        = P::Backend<'a>
    where
        Self: 'a;

    fn backend(&mut self) -> Self::Backend<'_> {
        self.provider.backend()
    }

    fn bind_session(&self, session: &mut CaptureSession) -> Result<(), CaptureError> {
        session.ensure_partition_capture(self.identity.clone())
    }

    fn with_observer<E>(
        &mut self,
        session: &mut CaptureSession,
        map_error: impl Fn(CaptureExecutionError<Self::Error>) -> E,
        routed_error: &dyn Fn(eredu_nn::Error) -> eredu_nn::Error,
        operation: &mut dyn FnMut(
            &mut dyn crate::ActivationObserver<Self::Tensor, E>,
        ) -> Result<(), E>,
    ) -> Result<(), E> {
        let prediction = session.prediction;
        let mut limits = self.limits;
        limits.max_record_bytes = limits
            .max_record_bytes
            .min(session.plan().plan().limits.per_step.encoded_bytes);
        let map_partition = &self.map_partition_error;
        let mut observer = PartitionCaptureObserver::for_step(
            session,
            self.provider.backend(),
            &*self.transport,
            &*self.layout,
            prediction,
            limits,
            &mut self.estimate,
            |error| map_error(map_partition(error)),
        )
        .with_session_identity(self.identity.clone())
        .with_interventions()
        .with_routed_error_handler(routed_error);
        operation(&mut observer)
    }
}
