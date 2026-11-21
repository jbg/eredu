//! Composes admitted capture with native tensor and transport mechanisms.
use std::sync::Arc;

use eredu_core::{
    capture::{
        CaptureError, CaptureSelection, PartitionCaptureNativeEstimate, ResolvedCaptureSlice,
        SpeculativeCapturePrimitives,
    },
    intervention::{InterventionEstimator, InterventionMechanisms},
    speculative::{AdmittedSpeculativeActivations, SpeculativeControlError},
    Completion, SpeculativeRequestId,
};

use crate::inspection::SpeculativeActivationObserver;

use super::{
    partition::{PartitionCaptureHookTransport, PartitionCaptureObserverError},
    CaptureBackendProvider, CaptureExecutionError, SpeculativeCaptureObserver,
    SpeculativePartitionBinding,
};

/// Native transforms and retained transport used by the portable speculative collector.
pub trait SpeculativeCaptureMechanisms {
    /// Exact native queue selected for an invocation.
    type Queue;
    /// Associates an already selected single queue with a portable invocation.
    fn single_context(queue: &Self::Queue) -> Self::Context<'_>;
    /// Checks that the native queues realize the admitted target and draft placement.
    fn bind_context<'a>(
        target: &'a Self::Queue,
        draft: &'a Self::Queue,
        topology: eredu_core::SpeculativeExecutionTopology,
    ) -> Result<Self::Context<'a>, Self::Error>;

    /// Exact native target placement for this invocation.
    type Context<'a>;
    /// Native tensor and intervention primitives.
    type Provider: CaptureBackendProvider + 'static;
    /// Transport for selected partition fragments.
    type Transport: PartitionCaptureHookTransport<
            Error: Send + Sync + 'static,
            Completion: Completion<Error: Send + Sync + 'static>,
        > + eredu_core::run_preparation::TextPreparationTransport<Error: Send + Sync + 'static>
        + 'static;
    /// Error at the model invocation boundary.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Borrows the binding already selected by portable installation.
    fn partition_binding<'a>(
        context: &Self::Context<'a>,
    ) -> Option<&'a SpeculativePartitionBinding<Self::Transport>>;

    /// Attaches the transport and layout selected by the portable batch driver.
    fn attach_partition<'a>(
        context: Self::Context<'a>,
        binding: Option<&'a SpeculativePartitionBinding<Self::Transport>>,
    ) -> Self::Context<'a>;

    /// Coordinates scheduler readiness over the retained portable binding.
    fn coordinate_speculative_step(
        context: &Self::Context<'_>,
        local: Vec<eredu_core::SpeculativeScheduleState>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        match Self::partition_binding(context) {
            Some(binding) => binding.coordinate_speculative_step(local),
            None => Ok(local),
        }
    }

    /// Coordinates request commands without advancing the executable.
    fn coordinate_decoding_control(
        context: &Self::Context<'_>,
        local: eredu_core::decoding::DecodingCoordination,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        match Self::partition_binding(context) {
            Some(binding) => binding.coordinate_decoding_control(local),
            None => Ok(local),
        }
    }

    /// Agrees preparation outcomes before scheduler work proceeds.
    fn agree_text_preparation(
        context: &Self::Context<'_>,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        match Self::partition_binding(context) {
            Some(binding) => binding.agree_text_preparation(stage, status),
            None => Ok(match status {
                S::Ready => O::Ready,
                S::Cancelled => O::Cancelled,
                S::Failed => O::Rejected { rank: 0 },
            }),
        }
    }

    /// Opens the native transformations for the selected placement and transport.
    fn provider(context: &Self::Context<'_>, transport: Option<&Self::Transport>)
        -> Self::Provider;

    /// Side-effect-free native capability facts.
    fn intervention_mechanisms() -> InterventionMechanisms;

    /// Native cost model for intervention work.
    fn intervention_estimator() -> Arc<dyn InterventionEstimator>;

    /// Native allocation and transfer bounds for a selected fragment.
    fn estimate_partition(
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<PartitionCaptureNativeEstimate, CaptureError>;

    /// Translates one native or transport failure into the collector's error domain.
    fn partition_error(
        error: PartitionCaptureObserverError<<Self::Provider as CaptureBackendProvider>::Error>,
    ) -> CaptureExecutionError<<Self::Provider as CaptureBackendProvider>::Error>;

    /// Translates one settled capture failure at the model boundary.
    fn invocation_error(
        error: &CaptureExecutionError<<Self::Provider as CaptureBackendProvider>::Error>,
    ) -> Self::Error;
}

/// Portable assignment and observer derived from capture primitives.
pub type SpeculativeContextFor<'a, P> = crate::speculative::SpeculativeExecutionContext<
    'a,
    <P as SpeculativeCapturePrimitives>::Queues,
    <P as SpeculativeCapturePrimitives>::Transport,
    dyn crate::speculative::autoregressive::SpeculativeTargetObserver<
            <P as SpeculativeCapturePrimitives>::InvocationTensor,
            <P as SpeculativeCapturePrimitives>::ObserverError,
            <P as SpeculativeCapturePrimitives>::Scores,
        > + 'a,
>;

/// Native tensor provider over the exact queue and transport opened by portable binding.
pub struct SpeculativeCaptureProvider<P: SpeculativeCapturePrimitives> {
    queue: <P::Queues as eredu_core::speculative::SpeculativeQueueMechanisms>::Queue,
    transport: Option<P::Transport>,
}

impl<P: SpeculativeCapturePrimitives> SpeculativeCaptureProvider<P> {
    /// Retains the native resources selected for this observer.
    pub fn new(
        queue: <P::Queues as eredu_core::speculative::SpeculativeQueueMechanisms>::Queue,
        transport: Option<P::Transport>,
    ) -> Self {
        Self { queue, transport }
    }
}

impl<P: SpeculativeCapturePrimitives> CaptureBackendProvider for SpeculativeCaptureProvider<P> {
    type Tensor = P::Tensor;
    type Error = P::Error;
    type Backend<'a> = P::Backend<'a>;

    fn backend(&mut self) -> Self::Backend<'_> {
        P::capture_backend(&self.queue, self.transport.as_ref())
    }
}

impl<P> SpeculativeCaptureMechanisms for P
where
    P: SpeculativeCapturePrimitives,
    P::Transport: PartitionCaptureHookTransport<
            Error: Send + Sync + 'static,
            Completion: Completion<Error: Send + Sync + 'static>,
        > + eredu_core::run_preparation::TextPreparationTransport<Error: Send + Sync + 'static>,
{
    type Queue = <P::Queues as eredu_core::speculative::SpeculativeQueueMechanisms>::Queue;

    fn single_context(queue: &Self::Queue) -> Self::Context<'_> {
        crate::speculative::SpeculativeExecutionContext::single(queue)
    }

    fn bind_context<'a>(
        target: &'a Self::Queue,
        draft: &'a Self::Queue,
        topology: eredu_core::SpeculativeExecutionTopology,
    ) -> Result<Self::Context<'a>, Self::Error> {
        crate::speculative::SpeculativeExecutionContext::bind(target, draft, topology)
    }

    type Context<'a> = SpeculativeContextFor<'a, P>;
    type Provider = SpeculativeCaptureProvider<P>;
    type Transport = P::Transport;
    type Error = P::InvocationError;

    fn partition_binding<'a>(
        context: &Self::Context<'a>,
    ) -> Option<&'a SpeculativePartitionBinding<Self::Transport>> {
        context.capture_binding()
    }

    fn attach_partition<'a>(
        context: Self::Context<'a>,
        binding: Option<&'a SpeculativePartitionBinding<Self::Transport>>,
    ) -> Self::Context<'a> {
        context.with_capture_binding(binding)
    }

    fn provider(
        context: &Self::Context<'_>,
        transport: Option<&Self::Transport>,
    ) -> Self::Provider {
        SpeculativeCaptureProvider::new(context.target().clone(), transport.cloned())
    }

    fn intervention_mechanisms() -> InterventionMechanisms {
        P::intervention_mechanisms()
    }

    fn intervention_estimator() -> Arc<dyn InterventionEstimator> {
        P::intervention_estimator()
    }

    fn estimate_partition(
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<PartitionCaptureNativeEstimate, CaptureError> {
        Ok(PartitionCaptureNativeEstimate {
            capture: P::capture_workspace().estimate(shape, selection, slice)?,
            generated_creation_bytes: P::generated_creation_bytes(shape)?,
        })
    }

    fn partition_error(
        error: PartitionCaptureObserverError<P::Error>,
    ) -> CaptureExecutionError<P::Error> {
        use crate::capture::partition::PartitionCaptureExchangeError;
        match error {
            PartitionCaptureObserverError::Capture(error) => error,
            PartitionCaptureObserverError::Exchange(PartitionCaptureExchangeError::Capture(
                error,
            )) => CaptureExecutionError::Admission(error),
            error => CaptureExecutionError::Backend(P::observation_failure(
                eredu_core::BackendFailure::from_error(error),
            )),
        }
    }

    fn invocation_error(error: &CaptureExecutionError<P::Error>) -> Self::Error {
        P::invocation_failure(error)
    }
}

pub(crate) type ActivationObserverFor<M> = Box<
    dyn SpeculativeActivationObserver<
        <<M as SpeculativeCaptureMechanisms>::Provider as CaptureBackendProvider>::Tensor,
        <M as SpeculativeCaptureMechanisms>::Error,
    >,
>;

pub(crate) fn bind_speculative_observer<M: SpeculativeCaptureMechanisms + 'static>(
    plan: &AdmittedSpeculativeActivations,
    request: SpeculativeRequestId,
    context: M::Context<'_>,
) -> Result<Option<ActivationObserverFor<M>>, SpeculativeControlError> {
    if plan.is_empty() {
        return Ok(None);
    }
    let estimator = M::intervention_estimator();
    if let Some(partition) = M::partition_binding(&context) {
        return partition.observer(
            plan,
            request,
            &M::intervention_mechanisms(),
            M::provider(&context, Some(partition.transport())),
            M::estimate_partition,
            M::partition_error,
            M::invocation_error,
            estimator,
        );
    }
    Ok(SpeculativeCaptureObserver::from_admitted(
        plan,
        M::provider(&context, None),
        M::invocation_error,
        request,
        estimator,
    )?
    .map(|observer| Box::new(observer) as ActivationObserverFor<M>))
}
