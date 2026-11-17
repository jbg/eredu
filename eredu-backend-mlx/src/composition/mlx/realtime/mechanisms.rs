//! MLX resource binding and completion for portable realtime execution.

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::{MlxHybridState, MlxJointState, MlxJointStateBranch, MlxKeyValueState},
            generation::MlxSamplingBackend,
            realtime::MlxRealtimeChannelObserver,
        },
    },
    native::{
        ExecutionContext, MlxComponentBinder, MlxPrepublicationFrame, MlxRealtimeBinding,
        MlxRealtimeExecution, MlxRealtimeExecutionContext, MlxReplicatedCoordinator,
        SharedRealtimeExecution,
    },
    MlxBackendFactory, MlxTensor,
};
use eredu_core::{
    scheduler::SemanticStateTransaction, BackendFailure, DevicePlan, RealtimeInputFrame,
    RealtimeOutputFrame,
};
use eredu_nn::{RetainedTensorRoots, Tensor};
use eredu_runtime::{
    BeginRealtimeSubmission, CategoricalFrameBranch, CategoricalRealtimeMechanisms,
    PreparedRealtimeBinding, PrepublicationRealtimeFrame, RealtimeBackendFactory,
    RealtimeBindingMechanisms, RealtimeFrameHostObserver, RealtimeFramePlan, RealtimeHostFrame,
    RealtimeHostOutputProjection, RealtimeScheduleState, RealtimeSubmissionBranch,
    SpeechRealtimeMechanisms,
};

use super::MlxRealtimeSupport;

impl CategoricalRealtimeMechanisms for MlxRealtimeExecutionContext {
    type Neural = MlxNeuralBackend;
    type State = MlxKeyValueState;
    type Execution = MlxRealtimeExecution;
    type BindingError = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<
        Self::Neural,
        Self::State,
    >>::Error;
    type Group = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<
        Self::Neural,
        Self::State,
    >>::Group;
    type Route = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<
        Self::Neural,
        Self::State,
    >>::Route;
    type TensorMetadata = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<
        Self::Neural,
        Self::State,
    >>::TensorMetadata;
    type Binding<'a> = MlxRealtimeBinding<'a>;
    type Frame<P>
        = MlxPrepublicationFrame<P>
    where
        P: RealtimeHostOutputProjection<
            RealtimeHostFrame,
            Output = RealtimeOutputFrame,
            Error: Send + Sync,
        >;

    fn bind<P, E, F>(&self, prepared: P) -> Result<E, BackendFailure>
    where
        for<'a> P: PreparedRealtimeBinding<
            Self::Neural,
            Self::State,
            Self::Binding<'a>,
            Execution = E,
            Error = F,
        >,
        F: std::error::Error + Send + Sync + 'static,
    {
        self.bind_realtime(prepared, None)
            .map_err(BackendFailure::from_error)
    }

    fn validate_execution(&self, execution: &Self::Execution) -> Result<(), BackendFailure> {
        self.validate_execution(execution)
    }

    fn submit<P>(
        &self,
        execution: &Self::Execution,
        plan: RealtimeFramePlan<P>,
        input: &RealtimeInputFrame,
        branch: &mut CategoricalFrameBranch<Self>,
    ) -> Result<Self::Frame<P>, BackendFailure>
    where
        P: RealtimeHostOutputProjection<
            RealtimeHostFrame,
            Output = RealtimeOutputFrame,
            Error: Send + Sync,
        >,
    {
        self.submit_realtime(execution, plan, input, branch)
            .map_err(BackendFailure::from_error)
    }

    fn into_host_output<P>(frame: Self::Frame<P>) -> Result<RealtimeOutputFrame, BackendFailure>
    where
        P: RealtimeHostOutputProjection<
            RealtimeHostFrame,
            Output = RealtimeOutputFrame,
            Error: Send + Sync,
        >,
    {
        frame.into_host_output().map_err(BackendFailure::from_error)
    }
}

impl SpeechRealtimeMechanisms for MlxRealtimeExecutionContext {
    type Neural = MlxNeuralBackend;
    type Backbone = MlxHybridState;
    type Speech = MlxKeyValueState;
    type SpeechLayer =
        <MlxKeyValueState as eredu_runtime::LayerRuntimeState<Self::Neural>>::LayerState;
    type Joint = MlxJointState;
    type JointBranch = <MlxJointState as SemanticStateTransaction>::Branch;
    type Components = MlxComponentBinder;
    type Cohort = MlxReplicatedCoordinator;
    type Sampling = MlxSamplingBackend;
    type Owner<C: 'static> = SharedRealtimeExecution<C>;
    type Observer = MlxRealtimeChannelObserver;

    fn components(&self) -> Self::Components {
        self.component_binder()
    }

    fn own<C: 'static>(&self, components: C) -> Self::Owner<C> {
        SharedRealtimeExecution::new(components, self.stream())
    }

    fn share<C: 'static>(owner: &Self::Owner<C>) -> Self::Owner<C> {
        owner.share()
    }

    fn is_busy<C: 'static>(owner: &Self::Owner<C>) -> bool {
        owner.is_busy()
    }

    fn observer<C: 'static>(owner: &Self::Owner<C>) -> Self::Observer {
        MlxRealtimeChannelObserver::new(owner.stream())
    }

    fn submit<C, S, Q, P, A, I, O, H>(
        owner: &Self::Owner<C>,
        branch: &mut RealtimeSubmissionBranch<Self, S, Q, P, A>,
        input: &I,
        joint: impl Fn(&S) -> &MlxJointStateBranch,
        visit: impl for<'a> Fn(&'a S, &mut dyn FnMut(&'a MlxTensor)),
        observer: H,
        invoke: impl FnOnce(
            &mut C,
            &mut RealtimeSubmissionBranch<Self, S, Q, P, A>,
            &<MlxTensor as Tensor>::Context,
            &mut BeginRealtimeSubmission<'_, RealtimeSubmissionBranch<Self, S, Q, P, A>>,
        ) -> Result<O, BackendFailure>,
    ) -> Result<PrepublicationRealtimeFrame<O, Self::Completion, H>, BackendFailure>
    where
        C: 'static,
        Q: RealtimeScheduleState,
        I: RetainedTensorRoots<MlxTensor> + ?Sized,
        O: RetainedTensorRoots<MlxTensor>,
        H: RealtimeFrameHostObserver<O>,
    {
        owner.submit(
            branch,
            input,
            |branch| {
                let mut roots = joint(branch.model_state())
                    .retained_arrays()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                visit(branch.model_state(), &mut |tensor| {
                    roots.push(tensor.as_array().clone())
                });
                roots.extend(
                    branch
                        .random_state()
                        .map(|random| random.as_array().clone()),
                );
                roots
            },
            |branch| joint(branch.model_state()).retention(),
            observer,
            invoke,
        )
    }
}

impl RealtimeBackendFactory for MlxBackendFactory {
    type Support = MlxRealtimeSupport;
    type Mechanisms = MlxRealtimeExecutionContext;

    fn realtime_support(&self) -> Self::Support {
        MlxRealtimeSupport::new(false)
    }

    fn media_graph_support(&self) -> impl eredu_runtime::RealtimeMechanismSupport {
        MlxRealtimeSupport::new(crate::native::MlxRingTransport::is_available())
    }

    fn open_realtime(&self, device: &DevicePlan) -> Result<Self::Mechanisms, BackendFailure> {
        let weights = ExecutionContext::cpu().map_err(BackendFailure::from_error)?;
        let execution = self.execution_context(device)?;
        Ok(MlxRealtimeExecutionContext::new(
            execution.stream(),
            weights.stream(),
        ))
    }
}

#[cfg(feature = "audio")]
impl eredu_architectures::processor_execution::MediaGraphMechanisms
    for MlxRealtimeExecutionContext
{
    type Neural = MlxNeuralBackend;
    type Transport =
        <crate::native::MlxRingTransport as eredu_runtime::PipelineTransportProvider>::Transport;
    type Graph = MlxComponentBinder;

    fn graph(&self) -> Self::Graph {
        self.component_binder()
    }

    fn media_lowering(
        &self,
    ) -> impl eredu_architectures::processor_execution::MediaInputLowering<MlxTensor> {
        crate::native::MlxMediaInputLowering
    }

    fn pipeline_transport(
        &self,
    ) -> impl eredu_runtime::PipelineTransportProvider<Transport = Self::Transport> {
        crate::native::MlxRingTransport
    }
}
