//! Native completion and binding for architecture-selected component pipelines.

use super::generic::{
    architecture_execution_layout, prepare_layerwise_policy_from_addressed_bindings,
    MlxSelectedLayerwisePolicy,
};
use crate::{
    backend::{
        distributed::MlxDistributedSession,
        nn::shared::{MlxCommunicationTensorMetadata, MlxNeuralBackend},
        runtime::distributed::{
            boundary::MlxPartitionTensorAllocator, completion::MlxCommunicationCompletion,
            topology::CommunicationRouteRealization, Group,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    BackendFailure, BoundedCompletion, BoundedCompletionOutcome, CollectiveGroupId,
    DistributedCommitEpoch, DistributedCommitOutcome,
};
use eredu_nn::Parameterized;
use eredu_runtime::{
    CommunicationManifest, DenseDiskStreamReport, DistributedExecutionPhase, ExecutionUnitAddress,
    ExpertPass, LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata, NoopObserver,
    OpaqueBoundaryTransport, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionCommunicationAuthority, PartitionedExecutionPlan, PartitionedTextExecution,
    PartitionedTextRuntime, PipelineActivationDtype, ReplicatedTextExecutionStrategy,
    ResidencyReport, RuntimeState, WeightBinding,
};
use eredu_runtime::{PipelinePartitionArchitecture, PipelinePartitionExecutor};
use safemlx::{Array, Stream};
use std::marker::PhantomData;

type Policy<A, S> =
    MlxSelectedLayerwisePolicy<<A as LayeredArchitecture<MlxNeuralBackend, S>>::Unit, ()>;
type Executor<A, S> =
    PipelinePartitionExecutor<A, MlxNeuralBackend, S, Policy<A, S>, MlxPartitionTensorAllocator>;
type Strategy<A, S> = PartitionedTextExecution<
    Executor<A, S>,
    Group,
    CommunicationRouteRealization,
    MlxCommunicationTensorMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Runtime<A, S> = PartitionedTextRuntime<
    A,
    MlxNeuralBackend,
    S,
    Policy<A, S>,
    Executor<A, S>,
    Group,
    CommunicationRouteRealization,
    MlxCommunicationTensorMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// Exact selected execution, residency and communication ownership for one rank.
pub(crate) struct MlxComponentPipeline<A, S>
where
    S: RuntimeState<MlxNeuralBackend>,
    A: PipelinePartitionArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
{
    runtime: Runtime<A, S>,
    policy: Policy<A, S>,
    authority: PartitionCommunicationAuthority,
    groups: Vec<Group>,
    completion_wait: eredu_core::BoundedCompletionWait,
    failed: bool,
}

impl<A, S> MlxComponentPipeline<A, S>
where
    S: RuntimeState<MlxNeuralBackend> + 'static,
    A: PipelinePartitionArchitecture<MlxNeuralBackend, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    A::Unit: Parameterized<MlxTensor> + 'static,
{
    /// Native realization consumes an already validated portable selection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mut architecture: A,
        source: SharedCheckpointSource,
        units: Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
        world: &safemlx::distributed::Group,
        residency: LayerWeightResidency,
        weights_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<(Self, LayerwiseModelMetadata), BackendFailure> {
        validate_component_communication(&manifest, tensor_group)?;
        let completion_wait = manifest
            .completion_policy()
            .ok_or_else(|| contract("component pipeline requires a bounded completion policy"))?
            .bounded_wait();
        let distributed = MlxDistributedSession::from_manifest(&manifest, world, execution_stream)
            .map_err(BackendFailure::from_error)?;
        let groups = manifest
            .groups()
            .iter()
            .filter_map(|group| distributed.selected_group(group.id()).cloned())
            .collect();
        let (communication, parallel, communication_stream) = distributed
            .into_component_communication(manifest, tensor_group)
            .map_err(BackendFailure::from_error)?;
        let authority = communication.authority();
        let addresses = units
            .iter()
            .map(|(address, _)| *address)
            .collect::<Vec<_>>();
        let layout = architecture_execution_layout::<_, S>(&architecture)
            .map_err(BackendFailure::from_error)?
            .select(addresses.iter().copied())
            .map_err(BackendFailure::from_error)?;
        detached_preparation(
            Vec::<Array>::new(),
            || {
                let (policy, metadata) = prepare_layerwise_policy_from_addressed_bindings(
                    source,
                    &mut architecture,
                    (),
                    PhantomData::<S>,
                    residency,
                    execution_stream,
                    weights_stream,
                    |_| false,
                    vec![],
                    units,
                )
                .map_err(BackendFailure::from_error)?;
                let policy = if residency.is_fully_resident() {
                    let resident = policy
                        .into_resident(&architecture, execution_stream, PhantomData::<S>)
                        .map_err(BackendFailure::from_error)?;
                    Policy::<A, S>::resident(resident, &layout, &addresses)
                } else {
                    Policy::<A, S>::bounded(policy, &layout, &addresses)
                }
                .map_err(BackendFailure::from_error)?;
                let executor = Executor::<A, S>::new(
                    architecture,
                    policy.clone(),
                    addresses,
                    parallel,
                    MlxPartitionTensorAllocator,
                    activation_dtype,
                )
                .map_err(BackendFailure::from_error)?
                .with_local_agreement(tensor_group);
                let runtime = Runtime::<A, S>::new(
                    plan,
                    executor,
                    communication,
                    communication_stream,
                    OpaqueBoundaryTransport,
                    OpaqueOutputPublisher,
                    OpaqueFailureAgreement,
                    residency.execution_residency(),
                    (!residency.is_fully_resident()).then(|| policy.clone()),
                )
                .map_err(BackendFailure::from_error)?;
                Ok((
                    Self {
                        runtime,
                        policy,
                        authority,
                        groups,
                        completion_wait,
                        failed: false,
                    },
                    metadata,
                ))
            },
            BackendFailure::from_error,
        )
    }
    /// Retained architecture for parameter-free state lifecycle operations.
    pub(crate) fn architecture(&self) -> &A {
        self.runtime.executor().architecture()
    }

    pub(crate) fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.policy
            .residency_report()
            .map_err(BackendFailure::from_error)
    }
    pub(crate) fn dense_stream_report(
        &self,
    ) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.policy
            .dense_stream_report()
            .map_err(BackendFailure::from_error)
    }

    /// Publishes within the selected rank group, completes exact output/state
    /// roots, and agrees the caller's epoch. External publication remains with
    /// the enclosing session, which owns the unpublished candidate state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<R, F, O>(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
        mut input_roots: Vec<Array>,
        state_roots: F,
        prepare_output: O,
    ) -> Result<(R, DistributedCommitOutcome), BackendFailure>
    where
        F: Fn(&S) -> Vec<Array>,
        O: FnOnce(&MlxTensor) -> Result<(R, Vec<Array>), BackendFailure>,
    {
        if self.failed {
            return Err(contract(
                "component pipeline requires recovery after a failed invocation",
            ));
        }
        input_roots.extend(state_roots(state));
        let result = detached_preparation(
            input_roots,
            || {
                self.forward_inner(
                    input,
                    state,
                    descriptor,
                    epoch,
                    stream,
                    &state_roots,
                    prepare_output,
                )
            },
            BackendFailure::from_error,
        );
        self.failed = self.authority.ensure_active().is_err()
            || matches!(&result,Ok((_,outcome)) if !matches!(outcome,DistributedCommitOutcome::Committed(_)));
        result
    }
    fn forward_inner<R, F, O>(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
        state_roots: &F,
        prepare_output: O,
    ) -> Result<(R, DistributedCommitOutcome), BackendFailure>
    where
        F: Fn(&S) -> Vec<Array>,
        O: FnOnce(&MlxTensor) -> Result<(R, Vec<Array>), BackendFailure>,
    {
        if !<Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
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
        let executed = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
        >>::forward_with_observer(
            &mut Strategy::<A, S>::new(),
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
        let published = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
        >>::publish_observed_output(&mut self.runtime, output, stream);
        let agreement = self.agree(
            DistributedExecutionPhase::OutputPublication,
            published.is_ok(),
            stream,
        );
        let output = published.map_err(BackendFailure::from_error)?;
        agreement?;
        let prepared = prepare_output(&output);
        let agreement = self.agree(
            DistributedExecutionPhase::OutputObservation,
            prepared.is_ok(),
            stream,
        );
        let (prepared, output_roots) = prepared?;
        agreement?;
        let arrays = std::iter::once(output.as_array().clone())
            .chain(state_roots(state))
            .chain(output_roots)
            .collect::<Vec<_>>();
        let completed = MlxCommunicationCompletion::submit(
            arrays.iter(),
            arrays.clone(),
            vec![],
            self.groups.clone(),
            vec![],
            vec![stream.clone()],
        )
        .map(|completion| {
            completion.with_authority(
                self.authority.clone(),
                eredu_runtime::CommunicationOperation::FailureAgreement,
                DistributedExecutionPhase::MechanismCompletion,
            )
        })
        .and_then(|completion| completion.wait_bounded(self.completion_wait))
        .map_err(|error| {
            self.authority.completion_failure(
                error,
                eredu_runtime::CommunicationOperation::FailureAgreement,
                DistributedExecutionPhase::MechanismCompletion,
                None,
            )
        });
        let agreement = self.agree(
            DistributedExecutionPhase::MechanismCompletion,
            matches!(completed, Ok(BoundedCompletionOutcome::Completed)),
            stream,
        );
        match completed.map_err(BackendFailure::from_error)? {
            BoundedCompletionOutcome::Completed => {}
            _ => return Err(contract(
                "component pipeline completion deadline expired; native resources remain retained",
            )),
        }
        agreement?;
        let outcome = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
        >>::commit_after_completion(&mut self.runtime, epoch, stream);
        Ok((prepared, outcome))
    }

    fn agree(
        &mut self,
        phase: DistributedExecutionPhase,
        success: bool,
        stream: &Stream,
    ) -> Result<(), BackendFailure> {
        let agreed = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
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

/// Checks standalone stage-cohort requirements without native realization.
pub(crate) fn validate_component_communication(
    manifest: &CommunicationManifest,
    tensor_group: Option<CollectiveGroupId>,
) -> Result<(), BackendFailure> {
    if let Some(group) = tensor_group {
        eredu_runtime::validate_pipeline_local_agreement(manifest, group)
            .map_err(BackendFailure::from_error)?;
        let selected = manifest
            .groups()
            .iter()
            .find(|selected| selected.id() == group)
            .expect("validated pipeline cohort exists");
        if !crate::backend::runtime::distributed::independent_status_members(
            manifest.world_size(),
            selected.members(),
        ) {
            return Err(contract(
                    "component cohort status agreement requires unsupported inactive-stage participation",
                ));
        }
    }
    if manifest.completion_policy().is_none() {
        return Err(contract(
            "component pipeline requires a bounded completion policy",
        ));
    }
    Ok(())
}
