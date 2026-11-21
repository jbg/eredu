//! Portable construction of component pipelines from exact rank resources.

use eredu_core::checkpoint_mechanisms::unit_binding::ComponentPreparationScope as _;

use eredu_core::communication::{pipeline::ComponentPipelineMechanisms, CommunicationAuthority};
use std::borrow::Borrow;

use eredu_core::communication::{
    BroadcastBackend, CommunicationTensorMetadata, FailureAgreementBackend,
    PartitionTensorAllocator, PipelineActivationDtype, PointToPointBackend, RankCommunication,
};
use eredu_core::{
    BackendFailure, BoundedCompletionOutcome, CollectiveGroupId, DistributedCommitEpoch,
    DistributedCommitOutcome,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    BoundComponentPipeline, DenseDiskStreamReport, DistributedExecutionPhase, ExecutionUnitAddress,
    ExpertPass, LayerWeightResidency, LayerwisePolicy, NoopObserver, OpaqueBoundaryTransport,
    OpaqueFailureAgreement, OpaqueOutputPublisher, PartitionCommunication,
    PartitionedExecutionPlan, PartitionedTextExecution, PartitionedTextRuntime,
    PipelinePartitionArchitecture, PipelinePartitionExecutor, ReplicatedTextExecutionStrategy,
    ResidencyReport, RuntimeState,
};

/// Admitted rank facts consumed by the portable component constructor.
pub struct ComponentPipelinePlan {
    pub(super) addresses: Vec<ExecutionUnitAddress>,
    pub(super) plan: PartitionedExecutionPlan,
    pub(super) activation_dtype: PipelineActivationDtype,
    pub(super) tensor_group: Option<CollectiveGroupId>,
    pub(super) residency: LayerWeightResidency,
}

/// Portable executor and retained communication for a prepared component.
type ComponentPipelineRuntime<A, B, S, P, F, G, R, I> = PartitionedTextRuntime<
    A,
    B,
    S,
    PipelinePartitionExecutor<A, B, S, P, F>,
    G,
    R,
    I,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// Admitted policy, transport, and execution queues assembled by the portable driver.
pub struct ComponentPipelineResources<C: RankCommunication, P> {
    pub(super) policy: P,
    pub(super) communication: PartitionCommunication<C::Backend, C::Group, C::Route, C::Metadata>,
    pub(super) parallel: Option<C::Parallel>,
    pub(super) executor: <C::Backend as eredu_core::SubmissionBackend>::OwnedExecutor,
}

pub(super) fn assemble_component_pipeline<A, B, S, C, P, F, O>(
    native: (F, O),
    architecture: A,
    selected: ComponentPipelinePlan,
    resources: ComponentPipelineResources<C, P>,
) -> Result<ComponentPipeline<A, B, S, P, F, C::Group, C::Route, C::Metadata, O>, BackendFailure>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend,
    B::ParallelContext: Sized,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    S: RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    A::Error: From<eredu_nn::Error>,
    P: LayerwisePolicy<B, A::Unit> + Clone,
    F: PartitionTensorAllocator<B>,
    O: ComponentPipelineMechanisms<B>,
{
    let (allocator, mechanisms) = native;
    let executor = PipelinePartitionExecutor::new(
        architecture,
        resources.policy.clone(),
        selected.addresses,
        resources.parallel,
        allocator,
        selected.activation_dtype,
    )
    .map_err(BackendFailure::from_error)?
    .with_local_agreement(selected.tensor_group);
    let bounded = !selected.residency.is_fully_resident();
    let authority = resources.communication.authority();
    let completion_wait = resources
        .communication
        .manifest()
        .completion_policy()
        .ok_or_else(|| contract("component pipeline requires a bounded completion policy"))?
        .bounded_wait();
    let runtime = PartitionedTextRuntime::new(
        selected.plan,
        executor,
        resources.communication,
        resources.executor,
        OpaqueBoundaryTransport,
        OpaqueOutputPublisher,
        OpaqueFailureAgreement,
        selected.residency.execution_residency(),
        bounded,
    )
    .map_err(BackendFailure::from_error)?;
    Ok(ComponentPipeline {
        runtime,
        policy: resources.policy,
        mechanisms,
        authority,
        completion_wait,
        failed: false,
    })
}

type Strategy<A, B, S, P, F, G, R, I> = PartitionedTextExecution<
    PipelinePartitionExecutor<A, B, S, P, F>,
    G,
    R,
    I,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// Portable invocation, completion, and commit driver for a bound component rank.
pub struct ComponentPipeline<A, B, S, P, F, G, R, I, O>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend,
    B::ParallelContext: Sized,
    S: RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    A::Error: From<eredu_nn::Error>,
    P: LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    O: ComponentPipelineMechanisms<B>,
{
    runtime: ComponentPipelineRuntime<A, B, S, P, F, G, R, I>,
    policy: P,
    mechanisms: O,
    authority: CommunicationAuthority,
    completion_wait: eredu_core::BoundedCompletionWait,
    failed: bool,
}

impl<A, B, S, P, F, G, R, I, O> ComponentPipeline<A, B, S, P, F, G, R, I, O>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend,
    B::ParallelContext: Sized,
    S: RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    P: LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    O: ComponentPipelineMechanisms<B>,
{
    fn forward_inner<T>(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &B::Executor,
        prepare_output: impl FnOnce(B::Tensor) -> Result<(T, Vec<B::Tensor>), BackendFailure>,
    ) -> Result<(T, DistributedCommitOutcome), BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<B::Tensor>,
    {
        if !<Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::agree_distributed_descriptor(
            &mut self.runtime,
            DistributedExecutionPhase::InputPreparation,
            true,
            descriptor,
            stream,
        )
        .map_err(BackendFailure::from_error)?
        {
            return Err(contract("component pipeline invocation descriptors differ"));
        }
        let executed = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::forward_with_observer(
            &mut Strategy::<A, B, S, P, F, G, R, I>::new(),
            &mut self.runtime,
            input,
            state,
            ExpertPass::Prefill,
            stream,
            &mut NoopObserver,
        );
        let agreement = self.agree(
            DistributedExecutionPhase::Execution,
            executed.is_ok(),
            stream,
        );
        let (output, _) = executed.map_err(BackendFailure::from_error)?;
        agreement?;
        let published = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::publish_observed_output(&mut self.runtime, output, stream);
        let agreement = self.agree(
            DistributedExecutionPhase::OutputPublication,
            published.is_ok(),
            stream,
        );
        let output = published.map_err(BackendFailure::from_error)?;
        agreement?;
        let prepared = prepare_output(output.clone());
        let agreement = self.agree(
            DistributedExecutionPhase::OutputObservation,
            prepared.is_ok(),
            stream,
        );
        let (prepared, output_roots) = prepared?;
        agreement?;
        let mut roots = vec![output];
        state.visit_retained_tensors(&mut |tensor| roots.push(tensor.clone()));
        roots.extend(output_roots);
        let completed =
            self.mechanisms
                .complete_outputs(roots, &self.authority, self.completion_wait, stream);
        let agreement = self.agree(
            DistributedExecutionPhase::MechanismCompletion,
            matches!(completed, Ok(BoundedCompletionOutcome::Completed)),
            stream,
        );
        match completed? {
            BoundedCompletionOutcome::Completed => {}
            _ => {
                return Err(contract(
                    "component pipeline completion deadline expired; native resources remain retained",
                ));
            }
        }
        agreement?;
        let outcome = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::commit_after_completion(&mut self.runtime, epoch, stream);
        Ok((prepared, outcome))
    }

    fn agree(
        &mut self,
        phase: DistributedExecutionPhase,
        success: bool,
        stream: &B::Executor,
    ) -> Result<(), BackendFailure> {
        let agreed = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::agree_distributed_phase(&mut self.runtime, phase, success, stream)
        .map_err(BackendFailure::from_error)?;
        if !agreed {
            return Err(contract(
                "component pipeline rank rejected the component transition",
            ));
        }
        Ok(())
    }
}
fn contract(message: &'static str) -> BackendFailure {
    BackendFailure::from_error(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    ))
}

impl<A, B, S, P, F, G, R, I, O> BoundComponentPipeline<A, B, S>
    for ComponentPipeline<A, B, S, P, F, G, R, I, O>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend,
    B::ParallelContext: Sized,
    S: RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    P: LayerwisePolicy<B, A::Unit, Error: std::error::Error + Send + Sync + 'static>
        + crate::UnitResidencyTelemetry<Error: std::error::Error + Send + Sync + 'static>,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    O: ComponentPipelineMechanisms<B>,
{
    fn architecture(&self) -> &A {
        self.runtime.executor().architecture()
    }

    fn agree(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &B::Executor,
    ) -> Result<(), BackendFailure> {
        if self.agree_application(descriptor, ready, context)? {
            Ok(())
        } else {
            Err(contract(
                "component pipeline rank rejected the application decision",
            ))
        }
    }

    fn agree_application(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &B::Executor,
    ) -> Result<bool, BackendFailure> {
        <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::agree_distributed_descriptor(
            &mut self.runtime,
            DistributedExecutionPhase::InputPreparation,
            ready,
            descriptor,
            context,
        )
        .map_err(BackendFailure::from_error)
    }

    fn invoke<T>(
        &mut self,
        input: A::Input<'_>,
        roots: Vec<B::Tensor>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &B::Executor,
        project: impl FnOnce(B::Tensor) -> Result<(T, Vec<B::Tensor>), BackendFailure>,
    ) -> Result<(T, DistributedCommitOutcome), BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<B::Tensor>,
    {
        if self.failed {
            return Err(contract(
                "component pipeline requires recovery after a failed invocation",
            ));
        }
        let mut roots = roots;
        state.visit_retained_tensors(&mut |tensor| roots.push(tensor.clone()));
        let result = self.mechanisms.begin(roots).and_then(|scope| {
            scope.finish(self.forward_inner(input, state, descriptor, epoch, context, project))
        });
        self.failed = self.authority.ensure_active().is_err()
            || matches!(&result, Ok((_, outcome)) if !matches!(outcome, DistributedCommitOutcome::Committed(_)));
        result
    }

    fn commit_candidate(
        &mut self,
        epoch: DistributedCommitEpoch,
        stream: &B::Executor,
    ) -> Result<(), BackendFailure> {
        let outcome = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
            A,
            B,
            S,
            P,
            P,
        >>::commit_after_completion(&mut self.runtime, epoch, stream);
        if outcome == DistributedCommitOutcome::Committed(epoch) {
            Ok(())
        } else {
            self.failed = true;
            Err(contract(
                "component pipeline application commit did not complete",
            ))
        }
    }

    fn publish_readout(
        &mut self,
        roots: Vec<B::Tensor>,
        stream: &B::Executor,
        project: impl FnOnce(&mut A, Option<&B::ParallelContext>) -> Result<B::Tensor, BackendFailure>,
    ) -> Result<B::Tensor, BackendFailure> {
        let scope = self.mechanisms.begin(roots)?;
        let result = (|| {
            let (architecture, parallel) = self.runtime.executor_mut().readout_resources();
            let local = project(architecture, parallel);
            let agreement = self.agree(DistributedExecutionPhase::Execution, local.is_ok(), stream);
            let local = local?;
            agreement?;
            let output = <Strategy<A, B, S, P, F, G, R, I> as ReplicatedTextExecutionStrategy<
                A,
                B,
                S,
                P,
                P,
            >>::publish_observed_output(&mut self.runtime, local, stream)
            .map_err(BackendFailure::from_error);
            let agreement = self.agree(
                DistributedExecutionPhase::OutputPublication,
                output.is_ok(),
                stream,
            );
            let output = output?;
            agreement?;
            Ok(output)
        })();
        scope.finish(result)
    }

    fn settle_values<'a>(
        values: impl Iterator<Item = &'a B::Tensor>,
    ) -> Result<(), BackendFailure> {
        O::settle_values(values)
    }

    fn residency(&self) -> Result<ResidencyReport, BackendFailure> {
        self.policy
            .residency_report()
            .map_err(BackendFailure::from_error)
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.policy
            .dense_stream_report()
            .map_err(BackendFailure::from_error)
    }
}
