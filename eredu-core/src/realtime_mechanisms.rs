//! Native submission, state and tensor contracts for realtime execution.

use std::ops::Deref;

use eredu_nn::{NeuralBackend, Tensor};

use crate::{BackendFailure, Completion};

mod support;
pub use support::{
    EmptyRealtimeIdentity, RealtimeIdentity, RealtimeLoadSupport, RealtimeMechanism,
    RealtimeMechanismFacts, RealtimeMechanismSupport,
};

/// Native contexts for frame execution and parameter materialization.
pub struct RealtimeContexts<C> {
    /// Queue context for the selected execution device.
    pub execution: C,
    /// Queue context for host checkpoint materialization.
    pub weights: C,
}

/// Native context creation and cold capability queries for realtime binding.
pub trait RealtimeContextFactory {
    /// Native queue context.
    type Context;
    /// Local categorical and speech support facts.
    type Support;
    /// Finite media graph support facts.
    type MediaGraphSupport;
    /// Native mechanisms over an exact pair of contexts.
    type Mechanisms: From<RealtimeContexts<Self::Context>>;

    /// Queries local support without allocating native resources.
    fn realtime_support(&self) -> Self::Support;
    /// Queries finite media graph support without allocating native resources.
    fn media_graph_support(&self) -> Self::MediaGraphSupport;
    /// Opens one execution context on the admitted device.
    fn open_realtime_execution(
        &self,
        device: &crate::DevicePlan,
    ) -> Result<Self::Context, BackendFailure>;
    /// Opens one host context for weight materialization.
    fn open_realtime_host(&self) -> Result<Self::Context, BackendFailure>;
}

/// Host readback of the exact tensor selected by the portable frame driver.
pub trait RealtimeTensorReadback {
    /// Native tensor evaluated on the selected context.
    type Tensor: Tensor;

    /// Reads integer values without narrowing overflow or floating conversion.
    fn read_i32(&self, value: &Self::Tensor) -> Result<Vec<i32>, BackendFailure>;

    /// Reads values as row-major F32 on the selected context.
    fn read_f32(&self, value: &Self::Tensor) -> Result<Vec<f32>, BackendFailure>;
}

/// Native queue, recovery, and teardown for installed realtime execution.
pub trait RealtimeQueue: Sized + 'static {
    /// Native tensor roots submitted on this queue.
    type Tensor: Tensor;
    /// Host transfer on this exact queue.
    type Readback: RealtimeTensorReadback<Tensor = Self::Tensor> + 'static;
    /// Exact native completion.
    type Completion: Completion<Error: Send + Sync + 'static>;
    /// Native resources for an invocation that has begun. Dropping a live scope
    /// fences its owner and retains pending work until terminal proof or teardown.
    type Scope;
    /// Native retirement wrapper that can defer destruction of components.
    type Retired<T: 'static>: Deref<Target = T>;

    /// Borrows the tensor context for this execution.
    fn context(&self) -> &<Self::Tensor as Tensor>::Context;
    /// Retains the queue context for readback after frame completion.
    fn readback(&self) -> Self::Readback;
    /// Arranges for safe destruction of installed components.
    fn retire<T: 'static>(value: T) -> Self::Retired<T>;
    /// Reaps scopes that have reached native terminal proof.
    fn reclaim();
    /// Drains retired ordinary owners after a shared execution handle is dropped.
    fn release_retired();
    /// Enters native recovery, retaining initial roots and portable authority.
    /// A rejection leaves no live work and releases the owner.
    fn begin(
        &self,
        owner: impl RealtimeSubmissionOwner + 'static,
        initial: &[Self::Tensor],
    ) -> Result<Self::Scope, BackendFailure>;
    /// Submits the exact roots needed to prove safe completion.
    fn finish(
        scope: Self::Scope,
        roots: Vec<Self::Tensor>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, BackendFailure>>;
    /// Reports the exact retained resource count.
    fn retained_resources(completion: &Self::Completion) -> usize;
    /// Seals already-started work after an execution failure.
    fn abandon(scope: Self::Scope, roots: Vec<Self::Tensor>) -> Self::Completion;
    /// Settles native bookkeeping after synchronous startup.
    fn finish_startup() -> Result<(), BackendFailure>;
}

/// Move-only ownership held by native recovery through completion or teardown.
pub trait RealtimeSubmissionOwner {
    /// Permanently rejects further invocations on the shared execution.
    fn poison(&self);
}

/// Failure before completion creation or after native work already began.
pub enum RealtimeCompletionCreationError<C, E> {
    /// No native work escaped the failed completion operation.
    BeforeSubmission(E),
    /// Native work began and its exact quarantine completion must be retained.
    AfterSubmission {
        /// Original native submission failure.
        error: E,
        /// Exact completion retaining every possibly live resource.
        completion: C,
    },
}

impl<C, E> RealtimeCompletionCreationError<C, E> {
    /// Reports a failure proven to precede native submission.
    pub const fn before_submission(error: E) -> Self {
        Self::BeforeSubmission(error)
    }

    /// Reports a failure after submission while preserving quarantine ownership.
    pub const fn after_submission(error: E, completion: C) -> Self {
        Self::AfterSubmission { error, completion }
    }
}

/// Disjoint access to two parts of one unpublished state transaction.
pub trait RealtimeJointStateParts {
    /// First state part.
    type Left: ?Sized;
    /// Second state part.
    type Right: ?Sized;

    /// Borrows the parts while the branch keeps transaction ownership.
    fn parts_mut(&mut self) -> (&mut Self::Left, &mut Self::Right);
}

/// Native RNG and exact completion types for realtime generation.
pub trait RealtimeStartupMechanisms {
    /// Explicit random state adopted by the generation transaction.
    type Random;
    /// Exact completion attached to candidate execution.
    type Completion: Completion;

    /// Realizes an explicitly selected random seed.
    fn realize_random(&self, seed: u64) -> Result<Self::Random, BackendFailure>;
}

/// Narrow backend mechanism for copying one validated host token matrix.
pub trait RealtimeHostTokenMaterializer {
    /// Opaque/native token tensor.
    type Tensor;
    /// Materialization failure.
    type Error;

    /// Copies one already validated row-major i32 matrix.
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Host materialization capability for mixed categorical and dense channels.
pub trait RealtimeHostFrameMaterializer: RealtimeHostTokenMaterializer {
    /// Copies a validated finite f32 frame with its leading batch dimension.
    fn materialize_f32(
        &mut self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Narrow tensor mechanisms needed to populate and resolve scheduled slots.
pub trait RealtimeFrameTensorMechanisms {
    /// Opaque/native token payload.
    type Tensor: Clone;
    /// Mechanism failure.
    type Error;

    /// Creates one batch-by-one column containing an architecture-selected token.
    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error>;
}

/// Native mechanisms for binding exact realtime resources.
pub trait RealtimeBindingMechanisms<B>
where
    B: crate::communication::CommunicationBackend<
        Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    B::ParallelContext: Sized,
{
    /// Native queue retained with the installed executor.
    type Queue: crate::realtime_mechanisms::RealtimeQueue<Tensor = B::Tensor>;
    /// Retains native streams and collective resources for this binding.
    fn execution_queue(&self) -> Self::Queue;

    /// Native binding error.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Retained communication group.
    type Group: std::borrow::Borrow<B::CommunicationGroup>;
    /// Realized communication route.
    type Route: std::borrow::Borrow<B::CommunicationRoute>;
    /// Native tensor metadata used in transfer admission.
    type TensorMetadata: crate::communication::CommunicationTensorMetadata<B>;
    /// Numerical implementation selected for the bound executor.
    type Sampling;
    /// Native checkpoint materialization and placement.
    type Materializer;

    /// Borrowed context for architecture module construction.
    fn context(&self) -> &B::Executor;

    /// Native control-plane operations for the selected world.
    type CommunicationSetup: crate::communication::CommunicationSetupTransport<Error = Self::Error>;

    /// Borrows the selected native world before portable agreement.
    fn open_communication_setup(&self) -> Result<Self::CommunicationSetup, Self::Error>;

    /// Exact native resources for the admitted rank.
    type Rank: crate::communication::RankCommunication<
        Backend = B,
        Group = Self::Group,
        Route = Self::Route,
        Metadata = Self::TensorMetadata,
        Parallel = B::ParallelContext,
        Error = Self::Error,
    >;

    /// Realizes native resources from the agreed world binding.
    fn realize_rank(
        &self,
        prepared: crate::communication::WorldCommunicationBinding<Self::CommunicationSetup>,
    ) -> Result<Self::Rank, Self::Error>;

    /// Supplies native parameter operations for the selected execution streams.
    fn materializer(&self) -> Self::Materializer;
}

type CategoricalTensor<M> = <<M as CategoricalRealtimeNative>::Neural as NeuralBackend>::Tensor;
type CategoricalContext<M> = <CategoricalTensor<M> as Tensor>::Context;
type SpeechTensor<M> = <<M as SpeechRealtimeNative>::Neural as NeuralBackend>::Tensor;
type SpeechContext<M> = <SpeechTensor<M> as Tensor>::Context;

/// Native mechanisms for a local categorical frame scheduler.
pub trait CategoricalRealtimeNative:
    RealtimeStartupMechanisms<Random: Clone, Completion: Clone>
    + RealtimeRandomRoot<CategoricalTensor<Self>>
    + for<'a> RealtimeBindingTarget<
        Self::Rank,
        Self::FrameQueue,
        Binding<'a>: RealtimeBindingMechanisms<
            Self::Neural,
            Materializer = Self::Materializer,
            Queue = Self::FrameQueue,
            Sampling = Self::FrameSampling,
            Error = Self::BindingError,
            Group = Self::Group,
            Route = Self::Route,
            TensorMetadata = Self::TensorMetadata,
        >,
    > + Sized
    + 'static
{
    /// Neural primitives used to construct selected components.
    type Neural: crate::communication::CommunicationBackend<
            Executor = <CategoricalTensor<Self> as Tensor>::Context,
            ParallelContext: Sized,
        > + eredu_nn::DistributedNeuralBackend
        + crate::checkpoint_mechanisms::CheckpointParameterBackend
        + crate::backend::LocalComponentStorage<
            CacheStorage: crate::cache::mechanisms::ResidentKeyValueStorage
                              + crate::cache::mechanisms::PagedKeyValueStorage
                              + 'static,
        >;
    /// Native queue and exact recovery for the selected categorical model.
    type FrameQueue: RealtimeQueue<Tensor = CategoricalTensor<Self>, Completion = Self::Completion>;
    /// Failure reported by the binding context.
    type BindingError: std::error::Error + Send + Sync + From<BackendFailure> + 'static;
    /// Retained communication group.
    type Group: std::borrow::Borrow<
            <Self::Neural as crate::communication::CommunicationBackend>::CommunicationGroup,
        > + 'static;
    /// Realized communication route.
    type Route: std::borrow::Borrow<
            <Self::Neural as crate::communication::CommunicationBackend>::CommunicationRoute,
        > + 'static;
    /// Native tensor metadata for transfer admission.
    type TensorMetadata: crate::communication::CommunicationTensorMetadata<Self::Neural> + 'static;
    /// Exact parameter resources composed by portable realtime construction.
    type Materializer: crate::checkpoint_mechanisms::unit_binding::NativeParameterUnitBinding<
            Self::Neural,
            UnitError = Self::BindingError,
            Parameters: crate::checkpoint_mechanisms::ExactParameterMaterializer<Self::Neural>,
        > + 'static;
    /// Native rank association for a selected multi-device execution.
    type Rank;

    /// Numeric mechanisms used by the portable decision driver.
    type FrameSampling: crate::sampling::SamplingBackend<
            Token = CategoricalTensor<Self>,
            Logits = CategoricalTensor<Self>,
            Context = CategoricalContext<Self>,
            RandomState = Self::Random,
            Error: std::error::Error + Send + Sync + 'static,
        > + 'static;
    /// Failure supplied by native frame mechanisms.
    type FrameError: std::error::Error + Send + Sync + 'static;
    /// Native host-token conversion and tensor creation.
    type FrameTensors<'a>: RealtimeHostTokenMaterializer<Tensor = CategoricalTensor<Self>, Error = Self::FrameError>
        + RealtimeFrameTensorMechanisms<Tensor = CategoricalTensor<Self>, Error = Self::FrameError>;

    /// Prepares tensor operations for the installed frame context.
    fn frame_tensors(queue: &Self::FrameQueue) -> Self::FrameTensors<'_>;
}

/// Native components, state, and submission ownership for a coupled PCM session.
pub trait SpeechRealtimeNative:
    RealtimeStartupMechanisms<
        Random: Clone + crate::execution_control::SnapshotValue + 'static,
        Completion: Completion<Error: Send + Sync + 'static> + Clone + 'static,
    > + RealtimeRandomRoot<SpeechTensor<Self>>
    + Sized
    + 'static
{
    /// Tensor, neural, and submission operators.
    type Neural: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + NeuralBackend<Tensor: crate::execution_control::SnapshotValue>
        + crate::SubmissionBackend<Executor = SpeechContext<Self>>;
    /// Native tensor, cache transfer, and copy submission primitives.
    type CacheBackend: crate::cache::mechanisms::ResidentKeyValueStorage
        + crate::cache::mechanisms::PagedKeyValueStorage
        + crate::cache::mechanisms::CompressedCacheStorage
        + crate::cache::mechanisms::CacheCopySubmission<Value = SpeechTensor<Self>>;
    /// Borrows the queue selected for state copying.
    fn state_copy_queue(
        &self,
    ) -> <Self::CacheBackend as crate::cache::mechanisms::LiveCacheBackend>::Queue;
    /// Shared native parameter and component binding.
    type Components: crate::checkpoint_mechanisms::unit_binding::NativeParameterUnitBinding<Self::Neural>
        + crate::checkpoint_mechanisms::unit_binding::ComponentPreparation<
            Context = SpeechContext<Self>,
        > + 'static;
    /// Native sampling operations.
    type Sampling: crate::sampling::DenseSamplingBackend<
        Logits = SpeechTensor<Self>,
        Token = SpeechTensor<Self>,
        Context = SpeechContext<Self>,
        RandomState = Self::Random,
        Error: std::error::Error + Send + Sync + 'static,
    >;
    /// Native queue and exact recovery for the coupled session.
    type Queue: RealtimeQueue<Tensor = SpeechTensor<Self>, Completion = Self::Completion>;
    /// Opens the common component binder.
    fn components(&self) -> Self::Components;
    /// Retains native resources for the selected execution queue.
    fn speech_queue(&self) -> Self::Queue;
}

/// Native target identity and materialization lifetime for an exact realtime queue.
pub trait RealtimeBindingTarget<Rank, Queue: RealtimeQueue>: 'static {
    /// Native handles borrowed during checkpoint and state materialization.
    type Binding<'a>;

    /// World size and global rank of the supplied local native rank.
    fn native_rank_identity(rank: &Rank) -> (usize, usize);

    /// World size and global rank of the retained native communication group.
    fn world_identity(&self) -> Option<(usize, usize)>;

    /// Supplies optional observation and initialization of native allocator defaults.
    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn crate::capability::NativeAllocatorCacheDefaults> {
        None
    }

    /// Acquires materialization authority on the admitted native target.
    fn open_binding(&self, native_rank: Option<&Rank>)
        -> Result<Self::Binding<'_>, BackendFailure>;

    /// Settles native materialization bookkeeping at the end of binding.
    fn finish_binding(&self) -> Result<(), BackendFailure>;

    /// Checks native execution identity before allocating session state.
    fn validate_execution(&self, queue: &Queue) -> Result<(), BackendFailure>;
}

/// Native tensor storage of a realtime random stream.
pub trait RealtimeRandomRoot<T>: RealtimeStartupMechanisms {
    /// Retains the native value backing the supplied random state.
    fn random_root(random: &Self::Random) -> T;
}

/// Queue recovery and numerical sampling for a realtime graph.
pub trait RealtimeGraphResources {
    /// Neural backend used by the bound queues and tensors.
    type Backend: crate::backend::LocalComponentStorage;
    /// Native sampling on the graph's tensor representation.
    type Sampling: crate::sampling::DenseSamplingBackend<
        Logits = <Self::Backend as eredu_nn::NeuralBackend>::Tensor,
        Token = <Self::Backend as eredu_nn::NeuralBackend>::Tensor,
        Context = <<Self::Backend as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        Error: std::error::Error + Send + Sync + 'static,
    >;

    /// Selected device context.
    fn graph_context(
        &self,
    ) -> &<<Self::Backend as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context;

    /// Opens native retirement tracking for forthcoming submissions.
    fn open_input_recovery() -> Result<impl Sized, BackendFailure>;

    /// Retains the supplied lifetime authority through native recovery.
    fn hold_fork_retention(
        retention: crate::execution_control::SnapshotRetention,
    ) -> Result<impl Sized, BackendFailure>;
}

/// Physical resources for one media-conditioned graph.
pub trait MediaGraphNative {
    /// Neural and submission operations used by the portable graph.
    type Neural: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + crate::SubmissionBackend<
            Executor = <<Self::Neural as NeuralBackend>::Tensor as Tensor>::Context,
        > + 'static;
    /// Native weight, queue and startup resources.
    type Graph;
    /// Provider of the selected native world.
    type World: crate::communication::pipeline::NativePipelineWorld;

    /// Borrows exact resources from the selected execution context.
    fn graph(&self) -> Self::Graph;

    /// Supplies tensor conversions for the portable processor.
    fn media_lowering(
        &self,
    ) -> impl crate::input::MediaInputLowering<<Self::Neural as NeuralBackend>::Tensor>;

    /// Supplies the native world initializer.
    fn pipeline_world(&self) -> Self::World;
}
