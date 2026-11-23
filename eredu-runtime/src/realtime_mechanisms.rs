//! Native mechanism contracts for backend-selected realtime binding.

use eredu_core::realtime_mechanisms::{
    CategoricalRealtimeNative, RealtimeCompletionCreationError, RealtimeContextFactory,
    RealtimeContexts, RealtimeQueue, RealtimeStartupMechanisms, RealtimeTensorReadback,
    SpeechRealtimeNative,
};

use eredu_core::{
    scheduler::SemanticStateTransaction, BackendFailure, DevicePlan, RealtimeDecisionDiagnostics,
    RealtimeFrameScheduleState, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
};
use eredu_nn::{NeuralBackend, RetainedTensorRoots, Tensor};

use crate::{
    execute_realtime_frame, InstalledRealtimeExecution, JointRealtimeBranch, JointRealtimeState,
    MappedRealtimeFrameHostObserver, RealtimeCompletionMechanism, RealtimeFrameResources,
    SharedRealtimeExecution,
};

use crate::{
    CompletedRealtimeFrame, GenerationSampler, LocalComponentMechanisms, PreparedRealtimeBinding,
    PrepublicationRealtimeFrame, RealtimeFrameChannel, RealtimeFrameHostObserver,
    RealtimeFramePlan, RealtimeGenerationBranch, RealtimeHostChannelValues,
    RealtimeHostOutputProjection, RealtimeLoadSupport, RealtimePayloadBranch,
    RealtimeScheduleState, RealtimeSessionBranch,
};

/// Opens exact native contexts and binds them for the selected portable realtime route.
pub trait RealtimeBackendFactory:
    RealtimeContextFactory<
    Support: RealtimeLoadSupport,
    MediaGraphSupport: crate::RealtimeMechanismSupport,
    Mechanisms: CategoricalRealtimeNative + SpeechRealtimeMechanisms,
>
{
    /// Opens the host materialization context and the selected execution context.
    fn open_realtime(&self, device: &DevicePlan) -> Result<Self::Mechanisms, BackendFailure> {
        let weights = self.open_realtime_host()?;
        let execution = self.open_realtime_execution(device)?;
        Ok(RealtimeContexts { execution, weights }.into())
    }
}

impl<F> RealtimeBackendFactory for F where
    F: RealtimeContextFactory<
        Support: RealtimeLoadSupport,
        MediaGraphSupport: crate::RealtimeMechanismSupport,
        Mechanisms: CategoricalRealtimeNative + SpeechRealtimeMechanisms,
    >
{
}

/// Host values observed before a realtime frame can commit.
pub type RealtimeHostFrame =
    CompletedRealtimeFrame<RealtimeHostChannelValues, RealtimeDecisionDiagnostics>;

/// Portable projection of completed native values into declared host channels.
pub struct RealtimeChannelObserver<R>(R);

impl<R> RealtimeChannelObserver<R> {
    /// Binds tensor readback for the selected execution context.
    pub fn new(readback: R) -> Self {
        Self(readback)
    }
}

/// Failure observing a completed realtime channel or diagnostic.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeChannelObservationError {
    /// Native tensor read failed.
    #[error(transparent)]
    Readback(#[from] BackendFailure),
    /// Channel geometry or element type disagrees with the declaration.
    #[error("{0}")]
    Channel(String),
    /// Portable frame validation failed.
    #[error(transparent)]
    Frame(#[from] crate::RealtimeCompletedFrameError),
    /// Logit diagnostics have invalid geometry.
    #[error(transparent)]
    Diagnostic(#[from] eredu_core::observation::ObservationError),
}

impl<R: RealtimeTensorReadback>
    RealtimeFrameHostObserver<CompletedRealtimeFrame<R::Tensor, R::Tensor>>
    for RealtimeChannelObserver<R>
{
    type Output = RealtimeHostFrame;
    type Error = RealtimeChannelObservationError;

    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<R::Tensor, R::Tensor>,
    ) -> Result<Self::Output, Self::Error> {
        let values = self.observe_channels(frame, frame.values())?;
        let aligned = self.observe_channels(frame, frame.aligned())?;
        let diagnostics = frame
            .diagnostics()
            .iter()
            .enumerate()
            .map(|(prediction, logits)| {
                Ok(RealtimeDecisionDiagnostics::new(
                    prediction,
                    realtime_shape(logits)?,
                    self.0.read_f32(logits)?,
                )?)
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;
        Ok(CompletedRealtimeFrame::new(
            frame.channels().clone(),
            frame.batch(),
            values,
            aligned,
            frame.decision_channels().to_vec(),
            diagnostics,
        )?)
    }
}

impl<R: RealtimeTensorReadback> RealtimeChannelObserver<R> {
    fn observe_channels(
        &self,
        frame: &CompletedRealtimeFrame<R::Tensor, R::Tensor>,
        channels: &[RealtimeFrameChannel<R::Tensor>],
    ) -> Result<Vec<RealtimeFrameChannel<RealtimeHostChannelValues>>, RealtimeChannelObservationError>
    {
        use eredu_core::RealtimeChannelPayload;
        use eredu_nn::TensorElementType;

        channels
            .iter()
            .map(|value| {
                let declaration = frame.channels().channel(value.channel()).ok_or(
                    crate::RealtimeCompletedFrameError::UnknownChannel(value.channel()),
                )?;
                let dimensions = match &declaration.payload {
                    RealtimeChannelPayload::Categorical { width, .. } => {
                        std::slice::from_ref(width)
                    }
                    RealtimeChannelPayload::Dense { shape } => shape,
                };
                let expected = std::iter::once(frame.batch()).chain(dimensions.iter().copied());
                let tensor = value.payload();
                let actual = realtime_shape(tensor)?;
                if actual.iter().copied().ne(expected.clone()) {
                    let expected: Vec<_> = expected.collect();
                    return Err(RealtimeChannelObservationError::Channel(format!(
                        "realtime channel {:?} has shape {actual:?}, expected {expected:?}",
                        value.channel()
                    )));
                }
                let payload = match &declaration.payload {
                    RealtimeChannelPayload::Categorical { .. } => {
                        RealtimeHostChannelValues::Categorical(self.0.read_i32(tensor)?)
                    }
                    RealtimeChannelPayload::Dense { .. } => {
                        if !matches!(
                            tensor.element_type(),
                            Some(
                                TensorElementType::F16
                                    | TensorElementType::Bf16
                                    | TensorElementType::F32
                                    | TensorElementType::F64
                            )
                        ) {
                            return Err(RealtimeChannelObservationError::Channel(format!(
                                "realtime dense channel {:?} requires floating-point values",
                                value.channel()
                            )));
                        }
                        RealtimeHostChannelValues::Dense(self.0.read_f32(tensor)?)
                    }
                };
                Ok(RealtimeFrameChannel::new(
                    value.channel(),
                    value.coordinate(),
                    payload,
                ))
            })
            .collect()
    }
}

fn realtime_shape(tensor: &impl Tensor) -> Result<Vec<usize>, RealtimeChannelObservationError> {
    tensor
        .shape()
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| {
                RealtimeChannelObservationError::Channel(
                    "negative realtime tensor dimension".into(),
                )
            })
        })
        .collect()
}

type FrameTensor<M> = <<M as CategoricalRealtimeNative>::Neural as eredu_nn::NeuralBackend>::Tensor;

/// Unpublished generic scheduler state passed to categorical submission.
pub type CategoricalFrameBranch<M> = RealtimeSessionBranch<
    RealtimePayloadBranch<
        <CategoricalRealtimeStateFor<M> as SemanticStateTransaction>::Branch,
        FrameTensor<M>,
    >,
    RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
>;

type FrameState<M> = <CategoricalRealtimeStateFor<M> as SemanticStateTransaction>::Branch;
type FrameCompletion<M> = <M as RealtimeStartupMechanisms>::Completion;
type RawFrame<M> = CompletedRealtimeFrame<FrameTensor<M>, FrameTensor<M>>;
type FrameResources<'a, M> = RealtimeFrameResources<
    'a,
    FrameTensor<M>,
    FrameState<M>,
    Box<dyn RetainedTensorRoots<FrameTensor<M>>>,
>;
/// Portable installed categorical executor over the selected native queue.
pub type CategoricalRealtimeExecution<M> = InstalledRealtimeExecution<
    <M as CategoricalRealtimeNative>::FrameSampling,
    CategoricalRealtimeStateFor<M>,
    <M as CategoricalRealtimeNative>::FrameQueue,
>;

/// Submitted realtime output observed and projected before scheduler publication.
pub type CategoricalRealtimeFrame<M, P> = PrepublicationRealtimeFrame<
    RawFrame<M>,
    FrameCompletion<M>,
    MappedRealtimeFrameHostObserver<
        RealtimeChannelObserver<
            <<M as CategoricalRealtimeNative>::FrameQueue as RealtimeQueue>::Readback,
        >,
        P,
    >,
>;

/// Portable categorical cache and transaction owner derived from native storage.
pub type CategoricalRealtimeStateFor<M> = crate::state::KeyValueState<
    <<M as CategoricalRealtimeNative>::Neural as eredu_core::backend::LocalComponentStorage>::CacheStorage,
>;

#[derive(Debug, thiserror::Error)]
#[error("native rank or world group does not match selected realtime rank {rank} of {world_size}")]
struct RealtimeBindingTargetError {
    world_size: usize,
    rank: usize,
}

/// Constructs an admitted categorical realtime model on its exact native target.
pub fn bind_prepared_realtime<M, P, E, F>(
    mechanisms: &M,
    prepared: P,
    native_rank: Option<M::Rank>,
) -> Result<E, BackendFailure>
where
    M: CategoricalRealtimeNative,
    for<'a> P: PreparedRealtimeBinding<
        M::Neural,
        CategoricalRealtimeStateFor<M>,
        M::Binding<'a>,
        Execution = E,
        Error = F,
    >,
    F: std::error::Error + Send + Sync + 'static,
{
    let selected = prepared.realization();
    let world_size = selected.topology().world_size();
    let target = (world_size, selected.rank());
    let world_matches = mechanisms.world_identity() == (world_size > 1).then_some(target);
    let rank_matches = native_rank.as_ref().map_or(world_size == 1, |rank| {
        M::native_rank_identity(rank) == target
    });
    if !world_matches || !rank_matches {
        return Err(BackendFailure::from_error(RealtimeBindingTargetError {
            world_size,
            rank: selected.rank(),
        }));
    }
    let binding = mechanisms.open_binding(native_rank.as_ref())?;
    crate::automatic_support::initialize_allocator_defaults(mechanisms.allocator_cache_defaults())?;
    let result = prepared.bind(&binding).map_err(BackendFailure::from_error);
    drop(binding);
    let finish = mechanisms.finish_binding();
    let execution = result?;
    finish?;
    Ok(execution)
}

/// Executes one admitted categorical frame through native tensor and completion mechanisms.
pub fn submit_categorical_realtime_frame<M, P>(
    mechanisms: &M,
    execution: &CategoricalRealtimeExecution<M>,
    plan: RealtimeFramePlan<P>,
    input: &RealtimeInputFrame,
    branch: &mut CategoricalFrameBranch<M>,
) -> Result<CategoricalRealtimeFrame<M, P>, BackendFailure>
where
    M: CategoricalRealtimeNative,
    M::Completion: Clone,
    P: RealtimeHostOutputProjection<
        RealtimeHostFrame,
        Output = RealtimeOutputFrame,
        Error: Send + Sync,
    >,
{
    execution.require_healthy()?;
    let queue = execution.queue();
    mechanisms.validate_execution(queue)?;
    let (ingress, decisions, projection) = plan.into_parts();
    let payload = branch
        .payload_contract()
        .map_err(BackendFailure::from_error)?;
    let mut guard = CategoricalSubmissionGuard::<M> {
        branch,
        submission: CategoricalSubmission {
            execution,
            scope: None,
        },
    };
    let mut host = M::frame_tensors(queue);
    let mut tensors = M::frame_tensors(queue);
    let mut executor = execution;
    let submitted = execute_realtime_frame::<M::FrameSampling, _, _, _, _, _, _, _, _, _>(
        &ingress,
        &payload,
        input,
        guard.branch.generation_mut(),
        &decisions,
        &mut host,
        &mut tensors,
        &mut executor,
        &mut guard.submission,
        queue.context(),
    )
    .map_err(BackendFailure::from_error)?;
    Ok(
        PrepublicationRealtimeFrame::new(submitted, RealtimeChannelObserver::new(queue.readback()))
            .project_host_output(projection),
    )
}

struct CategoricalSubmissionGuard<'a, M: CategoricalRealtimeNative> {
    branch: &'a mut CategoricalFrameBranch<M>,
    submission: CategoricalSubmission<'a, M>,
}

impl<M: CategoricalRealtimeNative> Drop for CategoricalSubmissionGuard<'_, M> {
    fn drop(&mut self) {
        let branch = self.branch.generation_mut();
        let state = branch.model_state();
        if let Some(scope) = self.submission.scope.take() {
            let mut roots = Vec::new();
            state
                .model_state()
                .visit_retained_tensors(&mut |value| roots.push(value.clone()));
            roots.extend(state.payload_history().retained_values().cloned());
            roots.extend(branch.random_state().and_then(M::random_root));
            let completion = M::FrameQueue::abandon(scope, roots);
            let _ = branch.attach_submission_completion(completion);
        }
    }
}

struct CategoricalSubmission<'a, M: CategoricalRealtimeNative> {
    execution: &'a CategoricalRealtimeExecution<M>,
    scope: Option<<M::FrameQueue as RealtimeQueue>::Scope>,
}

impl<M: CategoricalRealtimeNative> RealtimeCompletionMechanism<FrameResources<'_, M>>
    for CategoricalSubmission<'_, M>
{
    type Error = BackendFailure;
    type Completion = M::Completion;

    fn begin_submission(&mut self) -> Result<(), Self::Error> {
        self.scope = Some(self.execution.begin()?);
        Ok(())
    }

    fn complete(
        &mut self,
        resources: FrameResources<'_, M>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
    {
        let scope = self.scope.take().expect("admitted categorical submission");
        let mut roots = Vec::new();
        resources
            .model_state
            .visit_retained_tensors(&mut |value| roots.push(value.clone()));
        roots.extend(resources.input.retained_payloads().cloned());
        roots.extend(resources.output.retained_payloads().cloned());
        roots.extend(resources.output.diagnostics().iter().cloned());
        roots.extend(resources.payload_history.retained_values().cloned());
        if let Some(execution) = resources.execution {
            execution.visit_retained_tensors(&mut |value| roots.push(value.clone()));
        }
        M::FrameQueue::finish(scope, roots)
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        M::FrameQueue::retained_resources(completion)
    }
}

type SpeechTensor<M> = <<M as SpeechRealtimeNative>::Neural as NeuralBackend>::Tensor;
type Context<M> = <SpeechTensor<M> as Tensor>::Context;
/// Portable transactional owner of the speech and backbone caches.
pub type SpeechRealtimeState<M> = JointRealtimeState<<M as SpeechRealtimeNative>::CacheBackend>;

/// Unpublished candidate of the coupled speech state.
pub type SpeechRealtimeStateBranch<M> =
    JointRealtimeBranch<<M as SpeechRealtimeNative>::CacheBackend>;
/// Unpublished realtime generation and native completion during submission.
pub type SpeechRealtimeSubmissionBranch<M, S, Q, P, A> = RealtimeGenerationBranch<
    S,
    Q,
    P,
    A,
    eredu_core::sampling::host_random::HostRandomState,
    <<M as SpeechRealtimeNative>::Queue as RealtimeQueue>::Completion,
>;

/// Begins exact native submission after portable invocation preflight.
pub type BeginRealtimeSubmission<'a, B> = dyn FnMut(&B) -> Result<(), BackendFailure> + 'a;

/// Portable speech composition over native cache and component resources.
pub trait SpeechRealtimeMechanisms:
    SpeechRealtimeNative<
    Components: LocalComponentMechanisms<
        Self::Neural,
        crate::state::HybridState<Self::CacheBackend>,
    >,
>
{
}

impl<M> SpeechRealtimeMechanisms for M
where
    M: SpeechRealtimeNative,
    M::Components: LocalComponentMechanisms<M::Neural, crate::state::HybridState<M::CacheBackend>>,
{
}

/// Executes a coupled frame and attaches completion before exposing its host observer.
#[allow(clippy::too_many_arguments)]
pub fn submit_speech_realtime_frame<M, C, S, Q, P, A, I, O, H>(
    owner: &SharedRealtimeExecution<C, M::Queue>,
    branch: &mut SpeechRealtimeSubmissionBranch<M, S, Q, P, A>,
    input: &I,
    joint: impl Fn(&S) -> &SpeechRealtimeStateBranch<M>,
    visit: impl for<'a> Fn(&'a S, &mut dyn FnMut(&'a SpeechTensor<M>)),
    observer: H,
    invoke: impl FnOnce(
        &mut C,
        &mut SpeechRealtimeSubmissionBranch<M, S, Q, P, A>,
        &Context<M>,
        &mut BeginRealtimeSubmission<'_, SpeechRealtimeSubmissionBranch<M, S, Q, P, A>>,
    ) -> Result<O, BackendFailure>,
) -> Result<
    PrepublicationRealtimeFrame<O, <M::Queue as RealtimeQueue>::Completion, H>,
    BackendFailure,
>
where
    M: SpeechRealtimeMechanisms,
    C: 'static,
    Q: RealtimeScheduleState,
    I: RetainedTensorRoots<SpeechTensor<M>> + ?Sized,
    O: RetainedTensorRoots<SpeechTensor<M>>,
    H: RealtimeFrameHostObserver<O>,
{
    use crate::{submit_realtime_output, RealtimeCompletionAttachmentError};

    if branch.has_submission_completion() {
        return Err(BackendFailure::from_error(
            RealtimeCompletionAttachmentError::AlreadyAttached,
        ));
    }
    let mut components = owner.open_components()?;
    let context = owner.queue().context();
    let gather_roots = |branch: &SpeechRealtimeSubmissionBranch<M, S, Q, P, A>| {
        let (backbone, speech) = joint(branch.model_state()).parts();
        let mut roots = Vec::new();
        let mut retain = |value: &SpeechTensor<M>| roots.push(value.clone());
        backbone.visit_retained_tensors(&mut retain);
        speech.visit_retained_tensors(&mut retain);
        visit(branch.model_state(), &mut |tensor| {
            roots.push(tensor.clone())
        });
        roots
    };
    let mut guard = SpeechSubmissionGuard {
        branch,
        scope: None,
        abandon: |branch: &mut SpeechRealtimeSubmissionBranch<M, S, Q, P, A>, scope| {
            let completion = M::Queue::abandon(scope, gather_roots(branch));
            let _ = branch.attach_submission_completion(completion);
        },
    };
    let output = invoke(&mut components, guard.branch, context, &mut |branch| {
        let mut initial = gather_roots(branch);
        input.visit_retained_tensors(&mut |tensor| initial.push(tensor.clone()));
        guard.scope = Some(owner.begin(joint(branch.model_state()).retention(), &initial)?);
        Ok(())
    })?;
    let mut final_roots = gather_roots(guard.branch);
    output.visit_retained_tensors(&mut |tensor| final_roots.push(tensor.clone()));
    let scope = guard.scope.take().ok_or_else(|| {
        BackendFailure::from_error(std::io::Error::other(
            "realtime speech invocation returned without a submission",
        ))
    })?;
    let submitted = submit_realtime_output(guard.branch, output, |_, _| {
        let completion = M::Queue::finish(scope, final_roots)?;
        let retained = M::Queue::retained_resources(&completion);
        Ok((completion, retained))
    })
    .map_err(|error| match error {
        crate::RealtimeOutputSubmissionError::BeforeSubmission(error)
        | crate::RealtimeOutputSubmissionError::AfterSubmission(error) => error,
        crate::RealtimeOutputSubmissionError::Attachment(error) => {
            BackendFailure::from_error(error)
        }
    })?;
    Ok(PrepublicationRealtimeFrame::new(submitted, observer))
}

struct SpeechSubmissionGuard<'a, B, S, F: FnMut(&mut B, S)> {
    branch: &'a mut B,
    scope: Option<S>,
    abandon: F,
}

impl<B, S, F: FnMut(&mut B, S)> Drop for SpeechSubmissionGuard<'_, B, S, F> {
    fn drop(&mut self) {
        if let Some(scope) = self.scope.take() {
            (self.abandon)(self.branch, scope);
        }
    }
}
