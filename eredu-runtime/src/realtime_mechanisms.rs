//! Native mechanism contracts for backend-selected realtime binding.

use eredu_core::{
    scheduler::{SemanticStateTransaction, TransitionOutput},
    BackendFailure, Completion, DevicePlan, RealtimeDecisionDiagnostics,
    RealtimeFrameScheduleState, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend,
    RetainedTensorRoots, Tensor,
};

use crate::{
    execution_control::{ForkStateEstimate, IsolatedStateFork, SnapshotValue},
    CommunicationBackend, CompletedRealtimeFrame, ComponentPreparation, DenseSamplingBackend,
    GenerationSampler, LayerRuntimeState, LayeredTensorCohort, LocalComponentMechanisms,
    ParameterUnitMechanisms, PreparedRealtimeBinding, PrepublicationRealtimeFrame,
    RealtimeBindingMechanisms, RealtimeFrameHostObserver, RealtimeFramePlan,
    RealtimeGenerationBranch, RealtimeHostChannelValues, RealtimeHostOutputProjection,
    RealtimeJointStateParts, RealtimeLoadSupport, RealtimePayloadBranch, RealtimeScheduleState,
    RealtimeSessionBranch, RealtimeStartupMechanisms, RealtimeStartupPreparation,
    RealtimeStateGroup, RealtimeStateMaterializer, ResettableRuntimeState, RuntimeLayerState,
    RuntimeStateComponents, SelectedRealtimeStateRealization, SelectedStateRealization,
    SubmissionBackend,
};

/// Supplies cold support and opens native resources for portable realtime binding.
pub trait RealtimeBackendFactory {
    /// Capability predicates available before native resources are opened.
    type Support: RealtimeLoadSupport;
    /// Native mechanisms for categorical and coupled speech sessions.
    type Mechanisms: CategoricalRealtimeMechanisms + SpeechRealtimeMechanisms;

    /// Reports exact, side-effect-free support for local realtime loading.
    fn realtime_support(&self) -> Self::Support;

    /// Opens native execution and weight resources on the selected device.
    fn open_realtime(&self, device: &DevicePlan) -> Result<Self::Mechanisms, BackendFailure>;
}

/// Host values observed before a realtime frame can commit.
pub type RealtimeHostFrame =
    CompletedRealtimeFrame<RealtimeHostChannelValues, RealtimeDecisionDiagnostics>;

type FrameTensor<M> =
    <<M as CategoricalRealtimeMechanisms>::Neural as eredu_nn::NeuralBackend>::Tensor;

/// Unpublished generic scheduler state passed to categorical submission.
pub type CategoricalFrameBranch<M> = RealtimeSessionBranch<
    RealtimePayloadBranch<
        <<M as CategoricalRealtimeMechanisms>::State as SemanticStateTransaction>::Branch,
        FrameTensor<M>,
    >,
    RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
>;

/// Native mechanisms for a local categorical frame scheduler.
pub trait CategoricalRealtimeMechanisms:
    RealtimeStartupMechanisms<Random: Clone>
    + RealtimeStateMaterializer<SelectedRealtimeStateRealization, Self::State>
    + Sized
    + 'static
{
    /// Neural primitives used to construct selected components.
    type Neural: CommunicationBackend<
            Executor = <FrameTensor<Self> as Tensor>::Context,
            ParallelContext: Sized,
        > + DistributedNeuralBackend;
    /// Transactional decode state.
    type State: LayerRuntimeState<Self::Neural, LayerState: AttentionCache<FrameTensor<Self>>>
        + ResettableRuntimeState<Self::Neural>
        + SemanticStateTransaction<Error: Send + Sync + 'static>
        + 'static;
    /// Retained native executor.
    type Execution;
    /// Failure reported by the binding context.
    type BindingError: std::error::Error + Send + Sync + 'static;
    /// Retained communication group.
    type Group: std::borrow::Borrow<<Self::Neural as CommunicationBackend>::CommunicationGroup>
        + 'static;
    /// Realized communication route.
    type Route: std::borrow::Borrow<<Self::Neural as CommunicationBackend>::CommunicationRoute>
        + 'static;
    /// Native tensor metadata for transfer admission.
    type TensorMetadata: crate::CommunicationTensorMetadata<Self::Neural> + 'static;
    /// Binding context for architecture-owned preparations.
    type Binding<'a>: RealtimeBindingMechanisms<
        Self::Neural,
        Self::State,
        Execution = Self::Execution,
        Sampling: crate::SamplingBackend<Error: std::fmt::Display>,
        Error = Self::BindingError,
        Group = Self::Group,
        Route = Self::Route,
        TensorMetadata = Self::TensorMetadata,
    >;
    /// Native transition that gates scheduler publication on host observation.
    type Frame<P>: TransitionOutput
    where
        P: RealtimeHostOutputProjection<
            RealtimeHostFrame,
            Output = RealtimeOutputFrame,
            Error: Send + Sync,
        >;

    /// Binds any admitted preparation expressed using the native binding context.
    fn bind<P, E, F>(&self, prepared: P) -> Result<E, BackendFailure>
    where
        for<'a> P: PreparedRealtimeBinding<
            Self::Neural,
            Self::State,
            Self::Binding<'a>,
            Execution = E,
            Error = F,
        >,
        F: std::error::Error + Send + Sync + 'static;

    /// Checks native execution identity before allocating session state.
    fn validate_execution(&self, execution: &Self::Execution) -> Result<(), BackendFailure>;

    /// Submits the portable frame plan on an unpublished branch.
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
        >;

    /// Consumes the observed host frame after the scheduler commits its branch.
    fn into_host_output<P>(frame: Self::Frame<P>) -> Result<RealtimeOutputFrame, BackendFailure>
    where
        P: RealtimeHostOutputProjection<
            RealtimeHostFrame,
            Output = RealtimeOutputFrame,
            Error: Send + Sync,
        >;
}

type SpeechTensor<M> = <<M as SpeechRealtimeMechanisms>::Neural as NeuralBackend>::Tensor;
type Context<M> = <SpeechTensor<M> as Tensor>::Context;
/// Unpublished realtime generation and native completion during submission.
pub type RealtimeSubmissionBranch<M, S, Q, P, A> = RealtimeGenerationBranch<
    S,
    Q,
    P,
    A,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
>;

/// Begins exact native submission after portable invocation preflight.
pub type BeginRealtimeSubmission<'a, B> = dyn FnMut(&B) -> Result<(), BackendFailure> + 'a;

/// Native components, state, and submission ownership for a coupled PCM session.
pub trait SpeechRealtimeMechanisms:
    RealtimeStartupMechanisms<
        Random: Clone + SnapshotValue + 'static,
        Completion: Completion<Error: Send + Sync + 'static> + 'static,
    > + RealtimeStateMaterializer<SelectedStateRealization, Self::Backbone>
    + RealtimeStateMaterializer<SelectedStateRealization, Self::Speech>
    + RealtimeStateGroup<Self::Backbone, Self::Speech, State = Self::Joint>
    + Sized
    + 'static
{
    /// Tensor, neural, and submission operators.
    type Neural: GroupedNeuralBackend
        + DistributedNeuralBackend
        + NeuralBackend<Tensor: SnapshotValue>
        + SubmissionBackend<Executor = Context<Self>>;
    /// Backbone cache owner.
    type Backbone: LayerRuntimeState<
            Self::Neural,
            LayerState: AttentionCache<SpeechTensor<Self>> + RuntimeStateComponents<Self::Neural>,
        > + 'static;
    /// Independent speech cache owner.
    type Speech;
    /// One layer in the speech cache.
    type SpeechLayer: AttentionCache<SpeechTensor<Self>> + RuntimeLayerState<Self::Neural>;
    /// Joint transaction and copy-budget owner.
    type Joint: SemanticStateTransaction<Branch = Self::JointBranch, Error: Send + Sync + 'static>
        + IsolatedStateFork
        + ForkStateEstimate
        + 'static;
    /// Unpublished joint candidate.
    type JointBranch: RealtimeJointStateParts<Left = Self::Backbone, Right = [Self::SpeechLayer]>;
    /// Shared native parameter and component binding.
    type Components: ParameterUnitMechanisms<Self::Neural>
        + LocalComponentMechanisms<Self::Neural, Self::Backbone>
        + ComponentPreparation<Context = Context<Self>>
        + 'static;
    /// Local protocol's cohort mechanism.
    type Cohort: LayeredTensorCohort<Self::Neural> + 'static;
    /// Native sampling operations.
    type Sampling: DenseSamplingBackend<
        Logits = SpeechTensor<Self>,
        Token = SpeechTensor<Self>,
        Context = Context<Self>,
        RandomState = Self::Random,
        Error: std::error::Error + Send + Sync + 'static,
    >;
    /// Exclusive invocation owner shared by descendants.
    type Owner<C: 'static>: RealtimeStartupPreparation<C, SpeechTensor<Self>>;
    /// Native-to-host conversion used by architecture output validation.
    type Observer: RealtimeFrameHostObserver<
            CompletedRealtimeFrame<SpeechTensor<Self>, SpeechTensor<Self>>,
            Output = RealtimeHostFrame,
            Error: Send + Sync,
        > + 'static;

    /// Opens the common component binder.
    fn components(&self) -> Self::Components;
    /// Adopts already bound components into their submission owner.
    fn own<C: 'static>(&self, components: C) -> Self::Owner<C>;
    /// Shares immutable execution and its availability gate.
    fn share<C: 'static>(owner: &Self::Owner<C>) -> Self::Owner<C>;
    /// Whether a sibling invocation currently owns the gate.
    fn is_busy<C: 'static>(owner: &Self::Owner<C>) -> bool;
    /// Constructs the native observer for one submission.
    fn observer<C: 'static>(owner: &Self::Owner<C>) -> Self::Observer;

    /// Retains the joint branch, other tensors, RNG and input until native completion.
    fn submit<C, S, Q, P, A, I, O, H>(
        owner: &Self::Owner<C>,
        branch: &mut RealtimeSubmissionBranch<Self, S, Q, P, A>,
        input: &I,
        joint: impl Fn(&S) -> &Self::JointBranch,
        visit: impl for<'a> Fn(&'a S, &mut dyn FnMut(&'a SpeechTensor<Self>)),
        observer: H,
        invoke: impl FnOnce(
            &mut C,
            &mut RealtimeSubmissionBranch<Self, S, Q, P, A>,
            &Context<Self>,
            &mut BeginRealtimeSubmission<'_, RealtimeSubmissionBranch<Self, S, Q, P, A>>,
        ) -> Result<O, BackendFailure>,
    ) -> Result<PrepublicationRealtimeFrame<O, Self::Completion, H>, BackendFailure>
    where
        C: 'static,
        Q: RealtimeScheduleState,
        I: RetainedTensorRoots<SpeechTensor<Self>> + ?Sized,
        O: RetainedTensorRoots<SpeechTensor<Self>>,
        H: RealtimeFrameHostObserver<O>;
}
