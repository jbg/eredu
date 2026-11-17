//! Native completion and binding for architecture-selected component pipelines.

use super::generic::{
    prepare_layerwise_policy_from_addressed_bindings, MlxSelectedLayerwisePolicy,
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
use eredu_core::{
    BackendFailure, BoundedCompletion, BoundedCompletionOutcome, CollectiveGroupId,
    DistributedCommitEpoch, DistributedCommitOutcome,
};
use eredu_nn::Parameterized;
use eredu_runtime::{
    CommunicationManifest, DenseDiskStreamReport, DistributedExecutionPhase, ExecutionUnitAddress,
    ExpertPass, LayeredArchitecture, LayerwiseModelMetadata, NoopObserver, OpaqueBoundaryTransport,
    OpaqueFailureAgreement, OpaqueOutputPublisher, PartitionCommunicationAuthority,
    PartitionedExecutionPlan, PartitionedTextExecution, PartitionedTextRuntime,
    PipelineActivationDtype, ReplicatedTextExecutionStrategy, ResidencyReport, RuntimeState,
};
use eredu_runtime::{PipelinePartitionArchitecture, PipelinePartitionExecutor};
use safemlx::{Array, Stream};
use std::marker::PhantomData;

type Policy<A, S> =
    MlxSelectedLayerwisePolicy<<A as LayeredArchitecture<MlxNeuralBackend, S>>::Unit, ()>;
type Executor<A, S, P = Policy<A, S>> =
    PipelinePartitionExecutor<A, MlxNeuralBackend, S, P, MlxPartitionTensorAllocator>;
type Strategy<A, S> = PartitionedTextExecution<
    Executor<A, S>,
    Group,
    CommunicationRouteRealization,
    MlxCommunicationTensorMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Runtime<A, S, P = Policy<A, S>> = PartitionedTextRuntime<
    A,
    MlxNeuralBackend,
    S,
    P,
    Executor<A, S, P>,
    Group,
    CommunicationRouteRealization,
    MlxCommunicationTensorMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// Realizes the ordinary portable partition driver with MLX transport and allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_pipeline_runtime<A, S, P>(
    architecture: A,
    policy: P,
    addresses: Vec<ExecutionUnitAddress>,
    plan: PartitionedExecutionPlan,
    activation_dtype: PipelineActivationDtype,
    communication: eredu_runtime::PartitionCommunication<
        MlxNeuralBackend,
        Group,
        CommunicationRouteRealization,
        MlxCommunicationTensorMetadata,
    >,
    parallel: Option<Group>,
    communication_stream: Stream,
    local_agreement: Option<CollectiveGroupId>,
    residency: eredu_runtime::ExecutionResidency,
    bounded_policy: Option<P>,
) -> Result<Runtime<A, S, P>, BackendFailure>
where
    S: RuntimeState<MlxNeuralBackend>,
    A: PipelinePartitionArchitecture<MlxNeuralBackend, S>,
    A::Error: From<eredu_nn::Error>,
    P: eredu_runtime::LayerwisePolicy<MlxNeuralBackend, A::Unit>,
{
    let executor = Executor::<A, S, P>::new(
        architecture,
        policy,
        addresses,
        parallel,
        MlxPartitionTensorAllocator,
        activation_dtype,
    )
    .map_err(BackendFailure::from_error)?
    .with_local_agreement(local_agreement);
    Runtime::<A, S, P>::new(
        plan,
        executor,
        communication,
        communication_stream,
        OpaqueBoundaryTransport,
        OpaqueOutputPublisher,
        OpaqueFailureAgreement,
        residency,
        bounded_policy,
    )
    .map_err(BackendFailure::from_error)
}

pub(crate) trait PipelineCommunication {
    fn session(
        &self,
        manifest: &CommunicationManifest,
        stream: &Stream,
    ) -> Result<MlxDistributedSession, BackendFailure>;
}

impl PipelineCommunication for safemlx::distributed::Group {
    fn session(
        &self,
        manifest: &CommunicationManifest,
        stream: &Stream,
    ) -> Result<MlxDistributedSession, BackendFailure> {
        MlxDistributedSession::from_manifest(manifest, self, stream)
            .map_err(BackendFailure::from_error)
    }
}

impl PipelineCommunication for MlxDistributedSession {
    fn session(
        &self,
        _: &CommunicationManifest,
        _: &Stream,
    ) -> Result<MlxDistributedSession, BackendFailure> {
        Ok(self.clone())
    }
}

/// Exact selected execution, residency and communication ownership for one rank.
pub struct MlxComponentPipeline<A, S>
where
    S: RuntimeState<MlxNeuralBackend>,
    A: PipelinePartitionArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
{
    runtime: Runtime<A, S>,
    policy: Policy<A, S>,
    // Retains the native world for standalone component execution.
    _distributed: MlxDistributedSession,
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
    /// Realizes a retained plan with either a native world or an established session.
    pub(crate) fn bind_prepared(
        prepared: eredu_runtime::PreparedComponentPipeline<A>,
        communication: &impl PipelineCommunication,
        weights_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<(Self, LayerwiseModelMetadata), BackendFailure> {
        let distributed = communication.session(&prepared.manifest, execution_stream)?;
        let eredu_runtime::PreparedComponentPipeline {
            mut architecture,
            source,
            invocations,
            pinned,
            units,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
            residency,
        } = prepared;
        validate_component_communication(&manifest, tensor_group)?;
        let completion_wait = manifest
            .completion_policy()
            .ok_or_else(|| contract("component pipeline requires a bounded completion policy"))?
            .bounded_wait();
        let groups = manifest
            .groups()
            .iter()
            .filter_map(|group| distributed.selected_group(group.id()).cloned())
            .collect();
        let (communication, parallel, communication_stream) = distributed
            .clone()
            .into_component_communication(manifest, tensor_group)
            .map_err(BackendFailure::from_error)?;
        let authority = communication.authority();
        let addresses = invocations.logical_addresses().collect::<Vec<_>>();
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
                    invocations,
                    pinned,
                    units,
                )
                .map_err(BackendFailure::from_error)?;
                let policy = if residency.is_fully_resident() {
                    let resident = policy
                        .into_resident(&architecture, execution_stream, PhantomData::<S>)
                        .map_err(BackendFailure::from_error)?;
                    Policy::<A, S>::resident(resident)
                } else {
                    Policy::<A, S>::bounded(policy)
                };
                let runtime = bind_pipeline_runtime(
                    architecture,
                    policy.clone(),
                    addresses,
                    plan,
                    activation_dtype,
                    communication,
                    parallel,
                    communication_stream,
                    tensor_group,
                    residency.execution_residency(),
                    (!residency.is_fully_resident()).then(|| policy.clone()),
                )?;
                Ok((
                    Self {
                        runtime,
                        policy,
                        _distributed: distributed,
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

    fn forward_inner<R>(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
        prepare_output: impl FnOnce(MlxTensor) -> Result<(R, Vec<MlxTensor>), BackendFailure>,
    ) -> Result<(R, DistributedCommitOutcome), BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<MlxTensor>,
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
        let prepared = prepare_output(output.clone());
        let agreement = self.agree(
            DistributedExecutionPhase::OutputObservation,
            prepared.is_ok(),
            stream,
        );
        let (prepared, output_roots) = prepared?;
        agreement?;
        let arrays = std::iter::once(output.as_array().clone())
            .chain(retained_arrays(state))
            .chain(
                output_roots
                    .into_iter()
                    .map(|tensor| tensor.as_array().clone()),
            )
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
            _ => {
                return Err(contract(
                    "component pipeline completion deadline expired; native resources remain retained",
                ));
            }
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

fn retained_arrays(state: &impl eredu_nn::RetainedTensorRoots<MlxTensor>) -> Vec<Array> {
    let mut roots = Vec::new();
    state.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
    roots
}

impl<A, S> eredu_runtime::BoundComponentPipeline<A, MlxNeuralBackend, S>
    for MlxComponentPipeline<A, S>
where
    S: RuntimeState<MlxNeuralBackend> + 'static,
    A: PipelinePartitionArchitecture<MlxNeuralBackend, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    A::Unit: Parameterized<MlxTensor> + 'static,
{
    fn architecture(&self) -> &A {
        self.runtime.executor().architecture()
    }

    fn agree(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &Stream,
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
        context: &Stream,
    ) -> Result<bool, BackendFailure> {
        <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
        >>::agree_distributed_descriptor(
            &mut self.runtime,
            DistributedExecutionPhase::InputPreparation,
            ready,
            descriptor,
            context,
        )
        .map_err(BackendFailure::from_error)
    }

    fn invoke<R>(
        &mut self,
        input: A::Input<'_>,
        roots: Vec<MlxTensor>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &Stream,
        project: impl FnOnce(MlxTensor) -> Result<(R, Vec<MlxTensor>), BackendFailure>,
    ) -> Result<(R, DistributedCommitOutcome), BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<MlxTensor>,
    {
        if self.failed {
            return Err(contract(
                "component pipeline requires recovery after a failed invocation",
            ));
        }
        let roots = roots
            .into_iter()
            .map(|tensor| tensor.as_array().clone())
            .chain(retained_arrays(state))
            .collect::<Vec<_>>();
        let result = detached_preparation(
            roots,
            || self.forward_inner(input, state, descriptor, epoch, context, project),
            BackendFailure::from_error,
        );
        self.failed = self.authority.ensure_active().is_err()
            || matches!(&result, Ok((_, outcome)) if !matches!(outcome, DistributedCommitOutcome::Committed(_)));
        result
    }

    fn commit_candidate(
        &mut self,
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(), BackendFailure> {
        let outcome = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            Policy<A, S>,
            Policy<A, S>,
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
        roots: Vec<MlxTensor>,
        stream: &Stream,
        project: impl FnOnce(&mut A, Option<&Group>) -> Result<MlxTensor, BackendFailure>,
    ) -> Result<MlxTensor, BackendFailure> {
        detached_preparation(
            roots
                .into_iter()
                .map(|tensor| tensor.as_array().clone())
                .collect::<Vec<_>>(),
            || {
                let (architecture, parallel) = self.runtime.executor_mut().readout_resources();
                let local = project(architecture, parallel);
                let agreement =
                    self.agree(DistributedExecutionPhase::Execution, local.is_ok(), stream);
                let local = local?;
                agreement?;
                let output = <Strategy<A, S> as ReplicatedTextExecutionStrategy<
                    A,
                    MlxNeuralBackend,
                    S,
                    Policy<A, S>,
                    Policy<A, S>,
                >>::publish_observed_output(
                    &mut self.runtime, local, stream
                )
                .map_err(BackendFailure::from_error);
                let agreement = self.agree(
                    DistributedExecutionPhase::OutputPublication,
                    output.is_ok(),
                    stream,
                );
                let output = output?;
                agreement?;
                Ok(output)
            },
            BackendFailure::from_error,
        )
    }

    fn settle_values<'a>(
        values: impl Iterator<Item = &'a MlxTensor>,
    ) -> Result<(), BackendFailure> {
        safemlx::transforms::eval(values.map(MlxTensor::as_array))
            .map_err(BackendFailure::from_error)
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
