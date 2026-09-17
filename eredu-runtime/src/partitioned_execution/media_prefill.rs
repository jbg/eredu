//! Prepared media passes reuse the unchanged selected partition schedule.
use super::*;
use crate::media_prefill::{
    CompositePrefillCut, MediaIngressError, MediaInvocation, MediaTextExecutionStrategy,
    PrefillIngressArchitecture, PreparedMediaPrefill,
};

/// Architecture/provider preparation for the existing partitioned pass type.
/// Only the selected composite executor implements this media protocol.
pub trait PartitionedMediaGroupExecutor<A, B, S, G, R, I>:
    PartitionedGroupExecutor<A, B, S, G, R, I>
where
    B: CommunicationBackend + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Borrows the exact runtime invocation; no replacement source or scheduler.
    fn begin_media<'a, 'source>(
        &mut self,
        invocation: &'a mut MediaInvocation<'source, A, B, S>,
        state: &mut S,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Pass<'a, 'a>, A::Error>
    where
        'source: 'a,
        S: 'a;
}

impl<A, B, S, Resident, Bounded, E, G, R, I, T, U, V>
    MediaTextExecutionStrategy<A, B, S, Resident, Bounded>
    for PartitionedTextExecution<E, G, R, I, T, U, V>
where
    B: CommunicationBackend
        + crate::TerminalCommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    Resident: LayerwisePolicy<B, A::Unit>,
    Bounded: LayerwisePolicy<B, A::Unit, Error = Resident::Error>,
    E: PartitionedMediaGroupExecutor<A, B, S, G, R, I>
        + crate::parameter_operations::LayeredParameterOwner<B, S, Architecture = A>,
    E::Policy: LayerwisePolicy<B, A::Unit, Error = Resident::Error>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    A::Error: std::fmt::Display,
    Resident::Error: std::fmt::Display,
{
    fn prepare_media_cut(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
    ) -> Result<CompositePrefillCut, A::Error> {
        let (architecture, _) = runtime
            .executor
            .parameter_parts_ref()
            .ok_or_else(|| A::ingress_error(MediaIngressError::ForeignGraph))?;
        architecture.validate_ingress_plan(plan)?;
        let mut cut = CompositePrefillCut::new(
            architecture.execution_graph()?,
            architecture.primary_execution_group(),
        )
        .map_err(A::ingress_error)?;
        cut.local_ingress_owner = runtime
            .plan
            .drivers
            .get(cut.primary)
            .and_then(Option::as_ref)
            .is_some_and(|driver| driver.range().start == 0);
        Ok(cut)
    }
    fn forward_media_span<O>(
        &mut self,
        runtime: &mut Self::Runtime,
        source: &mut PreparedMediaPrefill<A, B, S>,
        chunk: &crate::prefill::PrefillChunk,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, Resident::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        // Ordinary borrowed paths authenticate the local replicated runtime.
        // Partitioned capture retains its separate local-path/transport contract.
        if paths.is_some() {
            return Err(ReplicatedTextSessionError::PreparedObservation(
                crate::PreparedSessionObservationError::Unavailable,
            ));
        }
        runtime.execution_rejection_agreed = false;
        let initial = source
            .start(chunk)
            .map_err(|e| ReplicatedTextSessionError::Architecture(A::ingress_error(e)))?;
        let mut invocation = MediaInvocation {
            source,
            span: chunk,
            initial,
        };
        let pass = runtime
            .executor
            .begin_media(&mut invocation, state, demand, context)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        run_partition_pass::<A, B, S, Resident, Bounded, E, G, R, I, T, U, V, O>(
            runtime, pass, state, context, observer,
        )
    }
}
