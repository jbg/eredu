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
        paths: Option<&'a crate::PreparedLayeredObservationPaths>,
    ) -> Result<Self::Pass<'a, 'a>, A::Error>
    where
        'source: 'a,
        S: 'a;
}

impl<A, B, S, Bounded, E, G, R, I, T, U, V>
    MediaTextExecutionStrategy<A, B, S, E::Policy, Bounded>
    for PartitionedTextExecution<E, G, R, I, T, U, V>
where
    B: CommunicationBackend
        + crate::TerminalCommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    Bounded: LayerwisePolicy<B, A::Unit, Error = <E::Policy as LayerwisePolicy<B, A::Unit>>::Error>,
    E: PartitionedMediaGroupExecutor<A, B, S, G, R, I>
        + crate::parameter_operations::LayeredParameterOwner<B, S, Architecture = A>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    A::Error: std::fmt::Display,
    <E::Policy as LayerwisePolicy<B, A::Unit>>::Error: std::fmt::Display,
{
    fn prepare_media_cut(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
    ) -> Result<CompositePrefillCut, A::Error> {
        let (architecture, _) = runtime
            .executor
            .parameter_parts_ref()
            .ok_or_else(|| A::ingress_error(MediaIngressError::ForeignGraph, None))?;
        architecture.validate_ingress_plan(plan, None)?;
        let mut cut = CompositePrefillCut::new(
            architecture.execution_graph()?.into_owned(),
            architecture.primary_execution_group(),
        )
        .map_err(|cause| A::ingress_error(cause, None))?;
        cut.local_ingress_owner = runtime
            .plan
            .drivers
            .get(cut.primary)
            .and_then(Option::as_ref)
            .is_some_and(|driver| driver.range().start == 0);
        Ok(cut)
    }
    fn prepare_media_cut_with_metadata(
        runtime: &Self::Runtime, plan: &A::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<CompositePrefillCut, eredu_nn::Error>
    where A: PrefillIngressArchitecture<B, S, Error = eredu_nn::Error>,
    {
        context.charge_metadata(std::mem::size_of::<(
            &Self::Runtime, &A::IngressPlan, CompositePrefillCut,
            Result<CompositePrefillCut, eredu_nn::Error>,
        )>())?;
        let (architecture, _) = runtime.executor.parameter_parts_ref()
            .ok_or_else(|| context.metadata_source(MediaIngressError::ForeignGraph))?;
        architecture.validate_ingress_plan(plan, Some(context))?;
        let graph = architecture.execution_graph()?.into_owned_with_metadata(context)?;
        let mut cut = CompositePrefillCut::new_with_metadata(graph, architecture.primary_execution_group(), context)?;
        cut.local_ingress_owner = runtime.plan.drivers.get(cut.primary)
            .and_then(Option::as_ref).is_some_and(|driver| driver.range().start == 0);
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
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if let Some(paths)=paths {
            runtime.executor.validate_observation_paths(paths).map_err(|cause|
                crate::replicated_session::observation_paths::map_prepared(cause, |never|match never {}))?;
        }
        runtime.execution_rejection_agreed = false;
        let initial = source
            .start(chunk)
            .map_err(|e| ReplicatedTextSessionError::Architecture(A::ingress_error(e, None)))?;
        let mut invocation = MediaInvocation {
            source,
            span: chunk,
            initial,
        };
        let pass = runtime
            .executor
            .begin_media(&mut invocation, state, demand, context, paths)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        run_partition_pass::<A, B, S, E::Policy, Bounded, E, G, R, I, T, U, V, O>(
            runtime, pass, state, context, observer,
        )
    }
}
