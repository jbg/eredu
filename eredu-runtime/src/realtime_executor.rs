//! Mandatory family-blind coordination of one complete realtime frame.
//!
//! This module joins portable ingress, schedule interpretation, ordered model
//! decisions, delayed payload history, and exact completion attachment. Model
//! equations and native tensor operations remain behind narrow injected traits.

use crate::host_metadata::funded_vec;
use std::{cell::RefCell, sync::Arc};

use eredu_core::{
    scheduler::{DistributedTransitionOutput, TransitionOutput, WorkDescriptor},
    Completion, RealtimeInputFrame,
};

use crate::{
    complete_realtime_frame, prepare_realtime_frame, CompletedRealtimeFrame,
    MaterializedRealtimeInput, RealtimeCompletionAttachmentError, RealtimeFrameInterpretationError,
    RealtimeFrameTensorMechanisms, RealtimeGenerationBranch, RealtimeHostTokenMaterializer,
    RealtimeIngressContract, RealtimeIngressError, RealtimePayloadBranch, RealtimePayloadContract,
    RealtimePayloadContractError, RealtimePayloadHistory, Sampler, SamplingBackend,
    SequentialDecisionDriver, SequentialDecisionError, SequentialDecisionPlan,
    SequentialDecisionPlanError,
};

mod host_source;
pub use host_source::RealtimeCoordinatorHostSource;

/// Architecture-owned execution of one already interpreted realtime frame.
///
/// Implementations receive only canonical temporal tensors and the existing
/// ordered decision driver. They do not validate host frames, advance the
/// portable schedule, resolve delayed coordinates, or publish state.
pub trait PreparedRealtimeFrameExecutor<B, S, M>
where
    B: SamplingBackend,
    S: Sampler<B>,
{
    /// Architecture execution failure.
    type Error;
    /// Architecture execution resources retained until exact completion exists.
    type Retained;

    /// Validates the already prepared canonical transition before model work
    /// or completion. Input materialization and interpretation have occurred;
    /// rejection retains the coordinator's existing unpublished-branch semantics.
    fn validate_transition(
        &self,
        _transition: &eredu_core::RealtimeFrameTransition,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Mutates only the unpublished model-state branch and resolves every
    /// decision through `driver` in architecture order.
    fn execute(
        &mut self,
        model_state: &mut M,
        temporal: &[B::Token],
        driver: &mut SequentialDecisionDriver<B, S>,
        context: &B::Context,
    ) -> Result<Self::Retained, Self::Error>;
}

/// Narrow exact-completion mechanism for a fully interpreted frame.
///
/// The mechanism may retain output, history, and model-state native resources,
/// but it cannot inspect portable frame source/target semantics.
pub trait RealtimeFrameCompletionMechanism<T, M, R> {
    /// Exact completion object attached to the generation branch.
    type Completion;
    /// Native submission or retention failure.
    type Error;

    /// Submits or records all work required by the completed unpublished frame.
    fn complete(
        &mut self,
        input: MaterializedRealtimeInput<T>,
        output: &CompletedRealtimeFrame<T, T>,
        model_state: &M,
        payload_history: &RealtimePayloadHistory<T>,
        execution: Option<R>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>;

    /// Reports the exact number of resources explicitly owned by a completion.
    fn retained_resources(&self, _completion: &Self::Completion) -> usize {
        0
    }
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

/// One completed native frame paired with the exact completion tracked by the scheduler.
pub struct SubmittedRealtimeFrame<T, C> {
    frame: CompletedRealtimeFrame<T, T>,
    completion: C,
    retained_resources: usize,
}

impl<T, C> SubmittedRealtimeFrame<T, C> {
    /// Returns native output values before portable host observation.
    pub const fn frame(&self) -> &CompletedRealtimeFrame<T, T> {
        &self.frame
    }

    /// Returns the exact completion handle shared with the unpublished state branch.
    pub const fn completion(&self) -> &C {
        &self.completion
    }

    /// Consumes the scheduler output into native frame values and exact completion.
    pub fn into_parts(self) -> (CompletedRealtimeFrame<T, T>, C) {
        (self.frame, self.completion)
    }
}

impl<T, C> TransitionOutput for SubmittedRealtimeFrame<T, C>
where
    C: Completion,
{
    type Error = C::Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.completion.is_complete()
    }

    fn retained_resources(&self) -> usize {
        self.retained_resources
    }
}

/// Family-blind conversion of one completed native frame into its host representation.
///
/// The observer is invoked only after the exact native completion reports ready and
/// successfully validates through [`Completion::wait`]. Implementations may perform
/// tensor-to-host conversion and portable output validation, but cannot publish session state.
pub trait RealtimeFrameHostObserver<T> {
    /// Portable or otherwise host-owned output cached before session publication.
    type Output;
    /// Host conversion or output validation failure.
    type Error: std::error::Error + 'static;

    /// Observes one exactly completed native frame.
    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<T, T>,
    ) -> Result<Self::Output, Self::Error>;
}

enum HostObservationState<H: RealtimeFrameHostObserver<T>, T> {
    Pending(H),
    Ready(H::Output),
    Failed(Arc<H::Error>),
}

/// Scheduler transition which gates publication on exact completion and host observation.
///
/// Core schedulers call [`TransitionOutput::is_complete`] before committing the associated
/// semantic-state branch. This wrapper returns `true` only after the native completion has
/// validated and the family-blind observer has produced and cached the host output. Therefore an
/// observation failure follows the scheduler's ordinary failed-transition path and cannot publish
/// the branch. The cached output is consumed only from the transition returned after commit.
pub struct PrepublicationRealtimeFrame<T, C, H>
where
    H: RealtimeFrameHostObserver<T>,
{
    submitted: SubmittedRealtimeFrame<T, C>,
    observation: RefCell<HostObservationState<H, T>>,
}

impl<T, C, H> PrepublicationRealtimeFrame<T, C, H>
where
    H: RealtimeFrameHostObserver<T>,
{
    /// Binds one submitted native frame to its mandatory prepublication observer.
    pub fn new(submitted: SubmittedRealtimeFrame<T, C>, observer: H) -> Self {
        Self {
            submitted,
            observation: RefCell::new(HostObservationState::Pending(observer)),
        }
    }

    /// Consumes the cached host output after the scheduler returns this committed transition.
    ///
    /// Calling this before [`TransitionOutput::is_complete`] succeeds reports that observation is
    /// still pending. A failed observation retains its original error for stable diagnostics.
    pub fn into_host_output(self) -> Result<H::Output, RealtimeHostOutputUnavailable<H::Error>> {
        match self.observation.into_inner() {
            HostObservationState::Pending(_) => Err(RealtimeHostOutputUnavailable::Pending),
            HostObservationState::Ready(output) => Ok(output),
            HostObservationState::Failed(error) => {
                Err(RealtimeHostOutputUnavailable::Observation(error))
            }
        }
    }
}

impl<T, C, H> TransitionOutput for PrepublicationRealtimeFrame<T, C, H>
where
    C: Completion,
    C::Error: 'static,
    H: RealtimeFrameHostObserver<T>,
{
    type Error = RealtimePrepublicationError<C::Error, H::Error>;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        {
            let observation = self.observation.borrow();
            match &*observation {
                HostObservationState::Ready(_) => return Ok(true),
                HostObservationState::Failed(error) => {
                    return Err(RealtimePrepublicationError::Observation(Arc::clone(error)))
                }
                HostObservationState::Pending(_) => {}
            }
        }

        if !self
            .submitted
            .completion()
            .is_complete()
            .map_err(RealtimePrepublicationError::Completion)?
        {
            return Ok(false);
        }
        self.submitted
            .completion()
            .wait()
            .map_err(RealtimePrepublicationError::Completion)?;

        let mut observation = self.observation.borrow_mut();
        let HostObservationState::Pending(observer) = &mut *observation else {
            unreachable!("prepublication observation state was checked before completion wait")
        };
        match observer.observe(self.submitted.frame()) {
            Ok(output) => {
                *observation = HostObservationState::Ready(output);
                Ok(true)
            }
            Err(error) => {
                let error = Arc::new(error);
                *observation = HostObservationState::Failed(Arc::clone(&error));
                Err(RealtimePrepublicationError::Observation(error))
            }
        }
    }

    fn retained_resources(&self) -> usize {
        self.submitted.retained_resources()
    }
}

impl<T, C, H> DistributedTransitionOutput for PrepublicationRealtimeFrame<T, C, H>
where
    C: Completion,
    C::Error: 'static,
    H: RealtimeFrameHostObserver<T>,
    H::Output: WorkDescriptor,
{
    fn distributed_output_words(&self) -> Option<usize> {
        match &*self.observation.borrow() {
            HostObservationState::Ready(observed) => observed.descriptor_words(),
            HostObservationState::Pending(_) | HostObservationState::Failed(_) => None,
        }
    }
    fn encode_distributed_output(&self, output: &mut Vec<u32>) -> Result<(), String> {
        match &*self.observation.borrow() {
            HostObservationState::Ready(observed) => observed
                .encode_descriptor(output)
                .map_err(|error| error.to_string()),
            HostObservationState::Pending(_) => {
                Err("realtime output observation is still pending".into())
            }
            HostObservationState::Failed(error) => {
                Err(format!("realtime output observation failed: {error}"))
            }
        }
    }
}

/// Exact-completion or host-observation failure before session publication.
#[derive(Debug)]
pub enum RealtimePrepublicationError<C, O> {
    /// Native completion readiness or validation failed.
    Completion(C),
    /// Host conversion or output validation failed and was cached exactly once.
    Observation(Arc<O>),
}

impl<C, O> std::fmt::Display for RealtimePrepublicationError<C, O>
where
    C: std::error::Error,
    O: std::error::Error,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completion(error) => {
                write!(formatter, "realtime exact completion failed: {error}")
            }
            Self::Observation(error) => {
                write!(formatter, "realtime host observation failed: {error}")
            }
        }
    }
}

impl<C, O> std::error::Error for RealtimePrepublicationError<C, O>
where
    C: std::error::Error + 'static,
    O: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Completion(error) => Some(error),
            Self::Observation(error) => Some(error.as_ref()),
        }
    }
}

/// A prepublication frame was consumed before a successful host observation was available.
#[derive(Debug)]
pub enum RealtimeHostOutputUnavailable<O> {
    /// Exact completion and host observation have not succeeded yet.
    Pending,
    /// Host observation failed; the original cached failure is retained.
    Observation(Arc<O>),
}

impl<O> std::fmt::Display for RealtimeHostOutputUnavailable<O>
where
    O: std::error::Error,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => formatter.write_str("realtime host output is not observed yet"),
            Self::Observation(error) => {
                write!(formatter, "realtime host observation failed: {error}")
            }
        }
    }
}

impl<O> std::error::Error for RealtimeHostOutputUnavailable<O>
where
    O: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pending => None,
            Self::Observation(error) => Some(error.as_ref()),
        }
    }
}

/// Portable text-then-depth decision controls for one realtime executor.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeDecisionExecution {
    allow_fully_forced_tail_skip: bool,
}

impl RealtimeDecisionExecution {
    /// Selects whether an architecture-proven fully forced tail may be omitted.
    pub const fn new(allow_fully_forced_tail_skip: bool) -> Self {
        Self {
            allow_fully_forced_tail_skip,
        }
    }

    /// Whether an architecture-proven fully forced tail may omit model units.
    pub const fn allows_fully_forced_tail_skip(&self) -> bool {
        self.allow_fully_forced_tail_skip
    }
}

/// Coordinates one complete realtime frame on an already unpublished branch.
///
/// Host validation precedes opaque materialization. Schedule and payload-history
/// changes are staged locally until preparation, model decisions, output
/// completion, and exact native completion creation all succeed. Model state is
/// necessarily mutated through its caller-owned transaction branch. Therefore,
/// on any returned error the caller **must discard the entire branch**; the fair
/// scheduler's submission path already enforces that rule.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn execute_realtime_frame<B, S, M, T, C, H, F, E, K>(
    contract: &RealtimeIngressContract,
    payload_contract: &RealtimePayloadContract,
    frame: &RealtimeInputFrame,
    branch: &mut RealtimeGenerationBranch<RealtimePayloadBranch<M, T>, S, B::RandomState, C>,
    decisions: &RealtimeDecisionExecution,
    host_materializer: &mut H,
    tensor_mechanisms: &mut F,
    model_executor: &mut E,
    completion_mechanism: &mut K,
    context: &B::Context,
) -> Result<
    SubmittedRealtimeFrame<T, C>,
    RealtimeFrameCoordinatorError<H::Error, F::Error, E::Error, B::Error, K::Error>,
>
where
    B: SamplingBackend<Token = T, Logits = T>,
    B::RandomState: Clone,
    C: Completion + Clone,
    S: Sampler<B> + Clone,
    T: Clone,
    H: RealtimeHostTokenMaterializer<Tensor = T>,
    F: RealtimeFrameTensorMechanisms<Tensor = T>,
    E: PreparedRealtimeFrameExecutor<B, S, M>,
    K: RealtimeFrameCompletionMechanism<T, M, E::Retained, Completion = C>,
{
    let (model,history,schedule,sampling,samplers,random,completion)=branch.frame_parts_mut();
    let view=RealtimeFrameExecutionView::new(model,history,schedule,sampling,
        samplers,random.as_ref(),completion.is_some());
    let mut updates=RealtimeFrameExecutionUpdates::new();
    let result=execute_realtime_frame_view::<B,S,M,T,C,H,F,E,K>(contract,payload_contract,
        frame,view,&mut updates,decisions,host_materializer,tensor_mechanisms,
        model_executor,completion_mechanism,context);
    // Sampling advances at the same point as the ordinary driver; a failed
    // submitted frame still publishes its exact quarantine completion to the
    // unpublished branch before the caller discards that branch.
    if let Some((next_samplers,next_random))=updates.sampling {
        *samplers=next_samplers; *random=next_random;
    }
    if let Some(next)=updates.schedule { *schedule=next; }
    if let Some(next)=updates.history { *history=next; }
    if let Some(next)=updates.completion { *completion=Some(next); }
    result
}

/// Borrowed source of one invocation of the common frame coordinator.
///
/// Metadata adapters project only model state, tensors and random state. The
/// actual portable schedule and sampler controls remain borrowed. This view
/// grants neither scheduler publication nor native submission authority.
pub struct RealtimeFrameExecutionView<'a,M,T,S,R> {
    model:&'a mut M,
    history:&'a RealtimePayloadHistory<T>,
    schedule:&'a eredu_core::RealtimeFrameScheduleState,
    sampling:eredu_core::RealtimeSampling,
    samplers:&'a [S],
    random:Option<&'a R>,
    submitted:bool,
}
impl<'a,M,T,S,R> RealtimeFrameExecutionView<'a,M,T,S,R> {
    /// Borrows exact existing sources and the explicit submission state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(model:&'a mut M,history:&'a RealtimePayloadHistory<T>,
        schedule:&'a eredu_core::RealtimeFrameScheduleState,sampling:eredu_core::RealtimeSampling,
        samplers:&'a [S],random:Option<&'a R>,submitted:bool)->Self {
        Self {model,history,schedule,sampling,samplers,random,submitted}
    }
}

/// Transaction-local effects of the common coordinator. Only its successful
/// publication points populate these fields; a completion created after a
/// native error is retained independently of schedule/history publication.
pub struct RealtimeFrameExecutionUpdates<T,S,R,C> {
    history:Option<RealtimePayloadHistory<T>>,
    schedule:Option<eredu_core::RealtimeFrameScheduleState>,
    sampling:Option<(Vec<S>,Option<R>)>,
    completion:Option<C>,
}
impl<T,S,R,C> RealtimeFrameExecutionUpdates<T,S,R,C> {
    /// Creates an empty destination, without cloning any source.
    pub const fn new()->Self { Self {history:None,schedule:None,sampling:None,completion:None} }
    /// Published payload source, available only after successful completion creation.
    pub fn payload_history(&self)->Option<&RealtimePayloadHistory<T>> { self.history.as_ref() }
    /// Advanced random/sampler owners from a completed decision driver.
    pub fn sampling_state(&self)->Option<(&[S],Option<&R>)> {
        self.sampling.as_ref().map(|(samplers,random)|(samplers.as_slice(),random.as_ref()))
    }
    /// Completion retained even when native submission returns an error.
    pub fn completion(&self)->Option<&C> { self.completion.as_ref() }
}
impl<T,S,R,C> Default for RealtimeFrameExecutionUpdates<T,S,R,C> {
    fn default()->Self { Self::new() }
}

/// Runs the same ordinary coordinator from borrowed exact source fields.
/// Callers must retain `updates` and discard the unpublished model branch on
/// failure, including any completion placed in `updates` after native work.
#[allow(clippy::too_many_arguments,clippy::type_complexity)]
pub fn execute_realtime_frame_view<B,S,M,T,C,H,F,E,K>(
    contract:&RealtimeIngressContract,payload_contract:&RealtimePayloadContract,
    frame:&RealtimeInputFrame,view:RealtimeFrameExecutionView<'_,M,T,S,B::RandomState>,
    updates:&mut RealtimeFrameExecutionUpdates<T,S,B::RandomState,C>,
    decisions:&RealtimeDecisionExecution,host_materializer:&mut H,tensor_mechanisms:&mut F,
    model_executor:&mut E,completion_mechanism:&mut K,context:&B::Context,
)->Result<SubmittedRealtimeFrame<T,C>,RealtimeFrameCoordinatorError<H::Error,F::Error,E::Error,B::Error,K::Error>>
where B:SamplingBackend<Token=T,Logits=T>,B::RandomState:Clone,C:Completion+Clone,
    S:Sampler<B>+Clone,T:Clone,H:RealtimeHostTokenMaterializer<Tensor=T>,
    F:RealtimeFrameTensorMechanisms<Tensor=T>,E:PreparedRealtimeFrameExecutor<B,S,M>,
    K:RealtimeFrameCompletionMechanism<T,M,E::Retained,Completion=C> {
    if view.submitted || updates.completion.is_some() || updates.history.is_some()
        || updates.schedule.is_some() || updates.sampling.is_some() {
        return Err(RealtimeFrameCoordinatorError::CompletionAttachment(
            RealtimeCompletionAttachmentError::AlreadyAttached,
        ));
    }
    view.schedule
        .validate_schedule(contract.schedule())
        .map_err(|error| {
            RealtimeFrameCoordinatorError::Interpretation(
                RealtimeFrameInterpretationError::Schedule(error),
            )
        })?;
    if payload_contract.schedule() != contract.schedule() {
        return Err(RealtimeFrameCoordinatorError::PayloadContract(
            RealtimePayloadContractError::ScheduleMismatch,
        ));
    }
    if payload_contract.batch().get() != frame.batch() {
        return Err(RealtimeFrameCoordinatorError::PayloadContract(
            RealtimePayloadContractError::BatchMismatch,
        ));
    }
    if payload_contract.text_domain() != contract.text_domain() {
        return Err(RealtimeFrameCoordinatorError::PayloadContract(
            RealtimePayloadContractError::TextDomainMismatch,
        ));
    }
    if payload_contract.audio_domain() != contract.audio_domain() {
        return Err(RealtimeFrameCoordinatorError::PayloadContract(
            RealtimePayloadContractError::AudioDomainMismatch,
        ));
    }
    view.history.validate_contract(payload_contract).map_err(|error| {
        RealtimeFrameCoordinatorError::Interpretation(RealtimeFrameInterpretationError::History(error))
    })?;
    if let Some(funding)=view.history.host_funding() {
        funding.reserve_metadata(host_source::coordinator_header_bytes::<B,S,M,T,C,H,F,E,K>()
            .ok_or(RealtimeFrameCoordinatorError::HostMetadata(eredu_core::HostMetadataFundingError::Overflow))?)
            .map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
    }
    // Borrowed validation and the selected source admission precede every
    // coordinator-owned policy/history clone and opaque tensor constructor.
    let validated = contract.inspect(frame).map_err(RealtimeFrameCoordinatorError::Ingress)?;
    let input = validated.materialize(host_materializer)
        .map_err(RealtimeFrameCoordinatorError::Materialization)?;
    let mut payload_history = crate::realtime_interpreter::clone_frame_history(view.history,tensor_mechanisms)
        .map_err(RealtimeFrameCoordinatorError::Interpretation)?;
    payload_history.bind_or_validate_contract(payload_contract).map_err(|error| {
        RealtimeFrameCoordinatorError::Interpretation(RealtimeFrameInterpretationError::History(error))
    })?;

    let schedule = contract.schedule();
    let mut schedule_state = match payload_history.host_funding() {
        Some(funding)=>view.schedule.try_clone_with_host_source(funding).map_err(|cause|
            RealtimeFrameCoordinatorError::Interpretation(RealtimeFrameInterpretationError::Schedule(
                eredu_core::RealtimeScheduleError::HostMetadata(cause))))?,
        None=>view.schedule.clone(),
    };
    let prepared = prepare_realtime_frame(
        schedule,
        &mut schedule_state,
        &mut payload_history,
        &input,
        tensor_mechanisms,
    )
    .map_err(RealtimeFrameCoordinatorError::Interpretation)?;

    model_executor
        .validate_transition(prepared.transition())
        .map_err(RealtimeFrameCoordinatorError::Model)?;

    let (completed, execution_retained) = if prepared.transition().model_call_required() {
        let funding=payload_history.host_funding();
        let mut directives=funded_vec(prepared.directives().len(),funding)
            .map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
        for directive in prepared.directives() {directives.push(match directive {
            crate::PredictionDirective::Sample=>crate::PredictionDirective::Sample,
            crate::PredictionDirective::Force(value)=>crate::PredictionDirective::Force(
                crate::realtime_interpreter::clone_frame_tensor(value,funding,tensor_mechanisms)
                    .map_err(RealtimeFrameCoordinatorError::Interpretation)?),
        });}
        let plan = SequentialDecisionPlan::from_directives(directives,
            prepared.retains_diagnostics(),decisions.allow_fully_forced_tail_skip)
            .map_err(RealtimeFrameCoordinatorError::DecisionPlan)?;
        let sampling=view.sampling;
        let mut temperatures=funded_vec(plan.len(),funding).map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
        temperatures.push(sampling.text_temperature());
        temperatures.extend(std::iter::repeat_n(sampling.audio_temperature(),schedule.depth_audio_codebooks()));
        let mut samplers=funded_vec(view.samplers.len(),funding).map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
        for value in view.samplers {
            samplers.push(match funding {
                Some(funding)=>value.clone_with_host_source(funding).map_err(RealtimeFrameCoordinatorError::HostMetadata)?,
                None=>value.clone(),
            });
        }
        let random=match (view.random,funding) {
            (Some(value),Some(funding))=>Some(B::clone_random_with_host_source(value,funding,context)
                .map_err(RealtimeFrameCoordinatorError::RandomCopy)?),
            (Some(value),None)=>Some(value.clone()),(None,_)=>None,
        };
        let mut driver=match funding {
            Some(funding)=>SequentialDecisionDriver::<B,S>::new_with_host_source(plan,samplers,temperatures,random,funding),
            None=>SequentialDecisionDriver::<B,S>::new(plan,samplers,temperatures,random),
        }.map_err(RealtimeFrameCoordinatorError::DecisionPlan)?;
        let execution_retained = model_executor
            .execute(
                &mut *view.model,
                prepared.temporal(),
                &mut driver,
                context,
            )
            .map_err(RealtimeFrameCoordinatorError::Model)?;
        driver
            .finish()
            .map_err(RealtimeFrameCoordinatorError::Decision)?;
        let mut resolved=funded_vec(driver.decisions().len(),funding)
            .map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
        for decision in driver.decisions() {resolved.push(crate::realtime_interpreter::clone_frame_tensor(
            decision.token(),funding,tensor_mechanisms).map_err(RealtimeFrameCoordinatorError::Interpretation)?);}
        let mut diagnostics=funded_vec(driver.diagnostics().len(),funding)
            .map_err(RealtimeFrameCoordinatorError::HostMetadata)?;
        for diagnostic in driver.diagnostics() {diagnostics.push(crate::realtime_interpreter::clone_frame_tensor(
            diagnostic.logits(),funding,tensor_mechanisms).map_err(RealtimeFrameCoordinatorError::Interpretation)?);}
        let completed = complete_realtime_frame(
            schedule,
            &mut payload_history,
            prepared,
            resolved,
            diagnostics,
            tensor_mechanisms,
        )
        .map_err(RealtimeFrameCoordinatorError::Interpretation)?;
        updates.sampling=Some(driver.finish_into_sampling_state()
            .map_err(RealtimeFrameCoordinatorError::Decision)?);
        (completed, Some(execution_retained))
    } else {
        (
            complete_realtime_frame(
                schedule,
                &mut payload_history,
                prepared,
                Vec::new(),
                Vec::new(),
                tensor_mechanisms,
            )
            .map_err(RealtimeFrameCoordinatorError::Interpretation)?,
            None,
        )
    };

    let completion = match completion_mechanism.complete(
        input,
        &completed,
        &*view.model,
        &payload_history,
        execution_retained,
    ) {
        Ok(completion) => completion,
        Err(RealtimeCompletionCreationError::BeforeSubmission(error)) => {
            return Err(RealtimeFrameCoordinatorError::Completion(error));
        }
        Err(RealtimeCompletionCreationError::AfterSubmission { error, completion }) => {
            updates.completion=Some(completion);
            return Err(RealtimeFrameCoordinatorError::CompletionAfterSubmission(
                error,
            ));
        }
    };
    let retained_resources = completion_mechanism.retained_resources(&completion);

    updates.schedule=Some(schedule_state);
    updates.history=Some(payload_history);
    updates.completion=Some(completion.clone());
    Ok(SubmittedRealtimeFrame {
        frame: completed,
        completion,
        retained_resources,
    })
}

/// Failure while coordinating an unpublished complete realtime frame.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeFrameCoordinatorError<H, F, E, B, K> {
    /// The admitted host account refused a concrete destination.
    #[error(transparent)]
    HostMetadata(eredu_core::HostMetadataFundingError),
    /// The backend's actual random-owner copy source failed.
    #[error("realtime random-state copy failed")]
    RandomCopy(#[source] eredu_core::BackendFailure),
    /// Session payload identity did not match ingress or accepted batch.
    #[error(transparent)]
    PayloadContract(RealtimePayloadContractError),
    /// Portable host validation failed before opaque materialization.
    #[error(transparent)]
    Ingress(RealtimeIngressError),
    /// Validated host payload materialization failed.
    #[error("realtime host payload materialization failed")]
    Materialization(H),
    /// Schedule, history, or tensor interpretation failed.
    #[error("realtime frame interpretation failed")]
    Interpretation(RealtimeFrameInterpretationError<F>),
    /// Decision policy or branch sampling-state cardinality was invalid.
    #[error(transparent)]
    DecisionPlan(SequentialDecisionPlanError),
    /// Ordered sampling or decision completion failed.
    #[error("realtime ordered decision failed")]
    Decision(SequentialDecisionError<B>),
    /// Typed architecture execution failed.
    #[error("realtime prepared model execution failed")]
    Model(E),
    /// Exact completion creation or resource retention failed.
    #[error("realtime exact completion creation failed")]
    Completion(K),
    /// Native completion creation failed after work began; the branch owns quarantine evidence.
    #[error("realtime exact completion creation failed after native submission")]
    CompletionAfterSubmission(K),
    /// The caller supplied a branch which already represented a submission.
    #[error(transparent)]
    CompletionAttachment(RealtimeCompletionAttachmentError),
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible, rc::Rc, time::Instant};

    use eredu_core::{
        scheduler::{RequestId, Scheduler, SchedulerLimits, SemanticStateTransaction},
        Completion, RealtimeFrameConvention, RealtimeSpeechConfig, TokenFilter,
    };

    use super::*;
    use crate::{
        PenaltyConfig, RealtimeGenerationState, RealtimePayloadGeneration,
        RealtimePayloadOwnerIdentity, RealtimePayloadState, TokenDomain,
    };

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Matrix {
        values: Vec<i32>,
        shape: [usize; 2],
        // Declared after values: expiry witnesses retirement of this input's Vec.
        lifetime: Option<Rc<()>>,
    }

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("test mechanism failed")]
    struct MechanismError;

    #[derive(Default)]
    struct Mechanisms {
        calls: usize,
        track: bool,
        lifetimes: Vec<std::rc::Weak<()>>,
    }
    impl Mechanisms {
        fn matrix(&mut self, values: Vec<i32>, shape: [usize; 2]) -> Matrix {
            let lifetime = self.track.then(|| {
                let owner = Rc::new(());
                self.lifetimes.push(Rc::downgrade(&owner));
                owner
            });
            Matrix {
                values,
                shape,
                lifetime,
            }
        }
    }

    impl RealtimeHostTokenMaterializer for Mechanisms {
        type Tensor = Matrix;
        type Error = MechanismError;

        fn materialize_i32(
            &mut self,
            values: &[i32],
            shape: [usize; 2],
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(self.matrix(values.to_vec(), shape))
        }
    }

    impl RealtimeFrameTensorMechanisms for Mechanisms {
        type Tensor = Matrix;
        type Error = MechanismError;

        fn column(
            &mut self,
            matrix: &Self::Tensor,
            column: usize,
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            let columns = matrix.shape[1];
            Ok(self.matrix(
                (0..matrix.shape[0])
                    .map(|row| matrix.values[row * columns + column])
                    .collect(),
                [matrix.shape[0], 1],
            ))
        }

        fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(self.matrix(vec![token; batch], [batch, 1]))
        }

        fn stack_columns(
            &mut self,
            columns: &[Self::Tensor],
            batch: usize,
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            let values = (0..batch)
                .flat_map(|row| columns.iter().map(move |column| column.values[row]))
                .collect();
            Ok(self.matrix(values, [batch, columns.len()]))
        }
    }

    struct TestBackend;

    impl SamplingBackend for TestBackend {
        type Logits = Matrix;
        type Token = Matrix;
        type RandomState = usize;
        type Context = ();
        type Error = MechanismError;

        fn error(_message: String) -> Self::Error {
            MechanismError
        }

        fn validate_token(
            token: &Self::Token,
            domain: TokenDomain,
            _context: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            token
                .values
                .iter()
                .all(|value| {
                    usize::try_from(*value).is_ok_and(|value| value < domain.cardinality())
                })
                .then(|| token.clone())
                .ok_or(MechanismError)
        }

        fn scale_temperature(
            logits: &Self::Logits,
            _temperature: f32,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits.clone())
        }

        fn apply_penalties(
            logits: &Self::Logits,
            _history: &[u32],
            _penalties: PenaltyConfig,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits.clone())
        }

        fn apply_top_k(
            logits: Self::Logits,
            _top_k: i32,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_top_p(
            logits: Self::Logits,
            _top_p: f32,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_min_p(
            logits: Self::Logits,
            _min_p: f32,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_token_filter(
            logits: &Self::Logits,
            _filter: &TokenFilter,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits.clone())
        }

        fn apply_mirostat(
            logits: &Self::Logits,
            _history: &[u32],
            _penalties: PenaltyConfig,
            _tau: f32,
            _eta: f32,
            _context: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits.clone())
        }

        fn sample_raw(
            logits: &Self::Logits,
            _temperature: f32,
            random: Option<&mut Self::RandomState>,
            _context: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            if let Some(random) = random {
                *random += 1;
            }
            Ok(logits.clone())
        }

        fn sample_processed(
            logits: &Self::Logits,
            temperature: f32,
            random: Option<&mut Self::RandomState>,
            context: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            Self::sample_raw(logits, temperature, random, context)
        }

        fn token_id(token: &Self::Token, _context: &Self::Context) -> Result<u32, Self::Error> {
            token
                .values
                .first()
                .copied()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(MechanismError)
        }

        fn token_probability(
            _logits: &Self::Logits,
            _token: u32,
            _context: &Self::Context,
        ) -> Result<f32, Self::Error> {
            Ok(1.0)
        }
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct TestSampler;

    impl Sampler<TestBackend> for TestSampler {
        fn sample(
            &mut self,
            logits: &Matrix,
            temperature: f32,
            random: Option<&mut usize>,
            context: &(),
        ) -> Result<Matrix, MechanismError> {
            TestBackend::sample_raw(logits, temperature, random, context)
        }
    }

    #[derive(Debug, Clone)]
    struct ModelState {
        executions: usize,
        discards: Rc<Cell<usize>>,
    }

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("test model-state transaction failed")]
    struct ModelStateError;

    impl SemanticStateTransaction for ModelState {
        type Branch = Self;
        type Error = ModelStateError;

        fn branch(&self) -> Result<Self::Branch, Self::Error> {
            Ok(self.clone())
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
            *self = branch;
            Ok(())
        }

        fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
            branch.discards.set(branch.discards.get() + 1);
            Ok(())
        }
    }

    #[derive(Debug, Clone, Default)]
    struct TestCompletion {
        waits: Rc<Cell<usize>>,
    }

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("test completion failed")]
    struct CompletionError;

    impl Completion for TestCompletion {
        type Error = CompletionError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            self.waits.set(self.waits.get() + 1);
            Ok(())
        }
    }

    struct Executor {
        fail: bool,
        calls: usize,
    }

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("test model execution failed")]
    struct ExecutionError;

    impl PreparedRealtimeFrameExecutor<TestBackend, TestSampler, ModelState> for Executor {
        type Error = ExecutionError;
        type Retained = Matrix;

        fn execute(
            &mut self,
            model_state: &mut ModelState,
            _temporal: &[Matrix],
            driver: &mut SequentialDecisionDriver<TestBackend, TestSampler>,
            _context: &(),
        ) -> Result<Self::Retained, Self::Error> {
            self.calls += 1;
            model_state.executions += 1;
            if self.fail {
                return Err(ExecutionError);
            }
            for prediction in 0..driver.plan().len() {
                let value = if prediction == 0 { 2 } else { 6 };
                let domain = TokenDomain::new(if prediction == 0 { 10 } else { 9 });
                driver
                    .resolve(
                        prediction,
                        &Matrix {
                            values: vec![value],
                            shape: [1, 1],
                            lifetime: None,
                        },
                        domain,
                        &(),
                    )
                    .map_err(|_| ExecutionError)?;
            }
            Ok(Matrix {
                values: vec![7],
                shape: [1, 1],
                lifetime: None,
            })
        }
    }

    #[derive(Default)]
    struct CompletionMechanism {
        calls: usize,
        retained_calls: usize,
        fail_after_submission: bool,
        waits: Rc<Cell<usize>>,
    }

    impl RealtimeFrameCompletionMechanism<Matrix, ModelState, Matrix> for CompletionMechanism {
        type Completion = TestCompletion;
        type Error = CompletionError;

        fn complete(
            &mut self,
            _input: MaterializedRealtimeInput<Matrix>,
            _output: &CompletedRealtimeFrame<Matrix, Matrix>,
            _model_state: &ModelState,
            _payload_history: &RealtimePayloadHistory<Matrix>,
            execution: Option<Matrix>,
        ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
        {
            self.calls += 1;
            if let Some(execution) = execution {
                self.retained_calls += 1;
                assert_eq!(execution.values, vec![7]);
            }
            let completion = TestCompletion {
                waits: self.waits.clone(),
            };
            if self.fail_after_submission {
                return Err(RealtimeCompletionCreationError::after_submission(
                    CompletionError,
                    completion,
                ));
            }
            Ok(completion)
        }
    }

    #[derive(Clone)]
    struct ControlledCompletion {
        ready: Rc<Cell<bool>>,
        fail_wait: bool,
        waits: Rc<Cell<usize>>,
    }

    impl Completion for ControlledCompletion {
        type Error = CompletionError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(self.ready.get())
        }

        fn wait(&self) -> Result<(), Self::Error> {
            self.waits.set(self.waits.get() + 1);
            if self.fail_wait {
                Err(CompletionError)
            } else {
                Ok(())
            }
        }
    }

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("test host observation failed")]
    struct ObservationError;

    struct Observer {
        calls: Rc<Cell<usize>>,
        fail: bool,
    }

    impl RealtimeFrameHostObserver<Matrix> for Observer {
        type Output = Vec<i32>;
        type Error = ObservationError;

        fn observe(
            &mut self,
            frame: &CompletedRealtimeFrame<Matrix, Matrix>,
        ) -> Result<Self::Output, Self::Error> {
            self.calls.set(self.calls.get() + 1);
            if self.fail {
                Err(ObservationError)
            } else {
                Ok(frame.text().values.clone())
            }
        }
    }

    #[derive(Clone)]
    struct PublicationState {
        value: usize,
        published: Rc<Cell<usize>>,
        discards: Rc<Cell<usize>>,
    }

    impl SemanticStateTransaction for PublicationState {
        type Branch = Self;
        type Error = Infallible;

        fn branch(&self) -> Result<Self::Branch, Self::Error> {
            Ok(self.clone())
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
            branch.published.set(branch.value);
            *self = branch;
            Ok(())
        }

        fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
            branch.discards.set(branch.discards.get() + 1);
            Ok(())
        }
    }

    type Generation = RealtimeGenerationState<
        RealtimePayloadState<ModelState, Matrix>,
        TestSampler,
        usize,
        TestCompletion,
    >;

    fn schedule(convention: RealtimeFrameConvention) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(2, 1, 1, 1, 9, 8, convention, vec![0, 0, 1]).unwrap()
    }

    fn state(schedule: &RealtimeSpeechConfig, discards: Rc<Cell<usize>>) -> Generation {
        let payload = RealtimePayloadState::new(
            ModelState {
                executions: 0,
                discards,
            },
            RealtimePayloadHistory::new(schedule.clone()),
            schedule,
        )
        .unwrap();
        Generation::new(
            payload,
            schedule.clone(),
            eredu_core::RealtimeSampling::greedy(),
            vec![TestSampler, TestSampler],
            Some(0),
        )
        .unwrap()
    }

    fn contract(schedule: &RealtimeSpeechConfig) -> RealtimeIngressContract {
        RealtimeIngressContract::new(schedule.clone(), TokenDomain::new(10), TokenDomain::new(9))
            .unwrap()
    }

    fn payload_contract(schedule: &RealtimeSpeechConfig) -> RealtimePayloadContract {
        RealtimePayloadContract::new(
            schedule.clone(),
            1,
            TokenDomain::new(10),
            TokenDomain::new(9),
            RealtimePayloadGeneration::new(1).unwrap(),
            RealtimePayloadOwnerIdentity::new(1).unwrap(),
        )
        .unwrap()
    }

    fn frame() -> RealtimeInputFrame {
        RealtimeInputFrame::new(1, vec![4])
            .with_forced_text(vec![3])
            .with_partially_forced_generated_audio(vec![5], vec![false])
    }

    fn submitted_test_frame<C>(completion: C) -> SubmittedRealtimeFrame<Matrix, C> {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let state = state(&schedule, Rc::new(Cell::new(0)));
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: false,
            calls: 0,
        };
        let mut completion_mechanism = CompletionMechanism::default();
        let submitted = execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
            &contract(&schedule),
            &payload_contract(&schedule),
            &frame(),
            &mut branch,
            &RealtimeDecisionExecution::new(true),
            &mut host,
            &mut tensors,
            &mut executor,
            &mut completion_mechanism,
            &(),
        )
        .unwrap();
        let SubmittedRealtimeFrame {
            frame,
            retained_resources,
            ..
        } = submitted;
        SubmittedRealtimeFrame {
            frame,
            completion,
            retained_resources,
        }
    }

    #[test]
    fn complete_frame_changes_only_the_unpublished_branch_until_commit() {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let discards = Rc::new(Cell::new(0));
        let mut state = state(&schedule, discards);
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: false,
            calls: 0,
        };
        let mut completion = CompletionMechanism::default();

        let output = execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
            &contract(&schedule),
            &payload_contract(&schedule),
            &frame(),
            &mut branch,
            &RealtimeDecisionExecution::new(true),
            &mut host,
            &mut tensors,
            &mut executor,
            &mut completion,
            &(),
        )
        .unwrap();

        assert_eq!(output.frame().text().values, vec![3]);
        assert_eq!(output.frame().sampled_audio().values, vec![6]);
        assert_eq!(executor.calls, 1);
        assert_eq!(completion.calls, 1);
        assert_eq!(completion.retained_calls, 1);
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.model_state().model_state().executions, 0);
        assert!(state.model_state().payload_history().is_empty());

        state.commit_branch(branch).unwrap();
        assert_eq!(state.schedule_state().frontier(), 1);
        assert_eq!(state.model_state().model_state().executions, 1);
        assert!(!state.model_state().payload_history().is_empty());
        assert_eq!(state.random_state(), Some(&1));
    }

    #[test]
    fn model_failure_requires_discard_and_never_changes_canonical_state() {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let discards = Rc::new(Cell::new(0));
        let state = state(&schedule, discards.clone());
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: true,
            calls: 0,
        };
        let mut completion = CompletionMechanism::default();

        assert!(matches!(
            execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
                &contract(&schedule),
                &payload_contract(&schedule),
                &frame(),
                &mut branch,
                &RealtimeDecisionExecution::new(true),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            ),
            Err(RealtimeFrameCoordinatorError::Model(ExecutionError))
        ));
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.model_state().model_state().executions, 0);
        assert!(state.model_state().payload_history().is_empty());
        assert_eq!(completion.calls, 0);

        Generation::discard_branch(branch).unwrap();
        assert_eq!(discards.get(), 1);
    }

    #[test]
    fn post_submission_completion_failure_is_quarantined_until_discard_waits() {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let discards = Rc::new(Cell::new(0));
        let state = state(&schedule, discards.clone());
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: false,
            calls: 0,
        };
        let waits = Rc::new(Cell::new(0));
        let mut completion = CompletionMechanism {
            fail_after_submission: true,
            waits: waits.clone(),
            ..CompletionMechanism::default()
        };

        assert!(matches!(
            execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
                &contract(&schedule),
                &payload_contract(&schedule),
                &frame(),
                &mut branch,
                &RealtimeDecisionExecution::new(true),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            ),
            Err(RealtimeFrameCoordinatorError::CompletionAfterSubmission(
                CompletionError
            ))
        ));
        assert!(branch.has_submission_completion());
        assert_eq!(waits.get(), 0);
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.model_state().model_state().executions, 0);

        Generation::discard_branch(branch).unwrap();
        assert_eq!(waits.get(), 1);
        assert_eq!(discards.get(), 1);
    }

    #[test]
    fn initialization_bypasses_the_nonempty_decision_plan_and_model() {
        let schedule = schedule(RealtimeFrameConvention::AbsoluteDelayedSlots);
        let discards = Rc::new(Cell::new(0));
        let state = state(&schedule, discards);
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: true,
            calls: 0,
        };
        let mut completion = CompletionMechanism::default();

        let output = execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
            &contract(&schedule),
            &payload_contract(&schedule),
            &frame(),
            &mut branch,
            &RealtimeDecisionExecution::new(true),
            &mut host,
            &mut tensors,
            &mut executor,
            &mut completion,
            &(),
        )
        .unwrap();

        assert_eq!(executor.calls, 0);
        assert_eq!(completion.calls, 1);
        assert_eq!(completion.retained_calls, 0);
        assert_eq!(output.frame().text().values, vec![9]);
        assert_eq!(branch.schedule_state().frontier(), 1);
    }

    #[test]
    fn branch_and_schedule_preflight_precede_every_native_mechanism() {
        let active_schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let state = state(&active_schedule, Rc::new(Cell::new(0)));
        let mut branch = state.branch().unwrap();
        branch
            .attach_submission_completion(TestCompletion::default())
            .unwrap();
        let mut host = Mechanisms::default();
        let mut tensors = Mechanisms::default();
        let mut executor = Executor {
            fail: false,
            calls: 0,
        };
        let mut completion = CompletionMechanism::default();

        assert!(matches!(
            execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
                &contract(&active_schedule),
                &payload_contract(&active_schedule),
                &frame(),
                &mut branch,
                &RealtimeDecisionExecution::new(true),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            ),
            Err(RealtimeFrameCoordinatorError::CompletionAttachment(
                RealtimeCompletionAttachmentError::AlreadyAttached
            ))
        ));
        assert_eq!(host.calls, 0);
        assert_eq!(tensors.calls, 0);
        assert_eq!(executor.calls, 0);
        assert_eq!(completion.calls, 0);

        let mut branch = state.branch().unwrap();
        let mismatched = schedule(RealtimeFrameConvention::AbsoluteDelayedSlots);
        assert!(matches!(
            execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
                &contract(&mismatched),
                &payload_contract(&mismatched),
                &frame(),
                &mut branch,
                &RealtimeDecisionExecution::new(true),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            ),
            Err(RealtimeFrameCoordinatorError::Interpretation(
                RealtimeFrameInterpretationError::Schedule(_)
            ))
        ));
        assert_eq!(host.calls, 0);
        assert_eq!(tensors.calls, 0);
        assert_eq!(executor.calls, 0);
        assert_eq!(completion.calls, 0);
    }

    #[test]
    fn prepublication_observation_waits_for_exact_completion_and_is_cached_once() {
        let ready = Rc::new(Cell::new(false));
        let waits = Rc::new(Cell::new(0));
        let observer_calls = Rc::new(Cell::new(0));
        let transition = PrepublicationRealtimeFrame::new(
            submitted_test_frame(ControlledCompletion {
                ready: ready.clone(),
                fail_wait: false,
                waits: waits.clone(),
            }),
            Observer {
                calls: observer_calls.clone(),
                fail: false,
            },
        );

        assert!(!transition.is_complete().unwrap());
        assert_eq!(waits.get(), 0);
        assert_eq!(observer_calls.get(), 0);

        ready.set(true);
        assert!(transition.is_complete().unwrap());
        assert!(transition.is_complete().unwrap());
        assert_eq!(waits.get(), 1);
        assert_eq!(observer_calls.get(), 1);
        assert_eq!(transition.into_host_output().unwrap(), vec![3]);
    }

    #[test]
    fn completion_and_observation_failures_never_expose_host_output() {
        let observer_calls = Rc::new(Cell::new(0));
        let completion_failure = PrepublicationRealtimeFrame::new(
            submitted_test_frame(ControlledCompletion {
                ready: Rc::new(Cell::new(true)),
                fail_wait: true,
                waits: Rc::new(Cell::new(0)),
            }),
            Observer {
                calls: observer_calls.clone(),
                fail: false,
            },
        );
        assert!(matches!(
            completion_failure.is_complete(),
            Err(RealtimePrepublicationError::Completion(CompletionError))
        ));
        assert_eq!(observer_calls.get(), 0);
        assert!(matches!(
            completion_failure.into_host_output(),
            Err(RealtimeHostOutputUnavailable::Pending)
        ));

        let observer_calls = Rc::new(Cell::new(0));
        let observation_failure = PrepublicationRealtimeFrame::new(
            submitted_test_frame(ControlledCompletion {
                ready: Rc::new(Cell::new(true)),
                fail_wait: false,
                waits: Rc::new(Cell::new(0)),
            }),
            Observer {
                calls: observer_calls.clone(),
                fail: true,
            },
        );
        assert!(matches!(
            observation_failure.is_complete(),
            Err(RealtimePrepublicationError::Observation(_))
        ));
        assert!(matches!(
            observation_failure.is_complete(),
            Err(RealtimePrepublicationError::Observation(_))
        ));
        assert_eq!(observer_calls.get(), 1);
        assert!(matches!(
            observation_failure.into_host_output(),
            Err(RealtimeHostOutputUnavailable::Observation(_))
        ));
    }

    #[test]
    fn scheduler_commits_only_after_host_observation_succeeds() {
        let limits = SchedulerLimits::new(1, 1).unwrap();
        let request = RequestId::new(7);
        let published = Rc::new(Cell::new(0));
        let discards = Rc::new(Cell::new(0));
        let mut scheduler = Scheduler::new(limits).unwrap();
        scheduler
            .register(
                request,
                PublicationState {
                    value: 0,
                    published: published.clone(),
                    discards: discards.clone(),
                },
            )
            .unwrap();
        scheduler.enqueue(request, frame()).unwrap();
        let observer_calls = Rc::new(Cell::new(0));
        let progress = scheduler
            .run_local_turn(Instant::now(), |_, _, branch| {
                branch.value += 1;
                Ok::<_, Infallible>(PrepublicationRealtimeFrame::new(
                    submitted_test_frame(ControlledCompletion {
                        ready: Rc::new(Cell::new(true)),
                        fail_wait: false,
                        waits: Rc::new(Cell::new(0)),
                    }),
                    Observer {
                        calls: observer_calls.clone(),
                        fail: true,
                    },
                ))
            })
            .unwrap();
        assert!(progress.committed.is_empty());
        assert_eq!(progress.failed.len(), 1);
        assert_eq!(published.get(), 0);
        assert_eq!(discards.get(), 1);
        assert_eq!(observer_calls.get(), 1);

        let request = RequestId::new(8);
        let published = Rc::new(Cell::new(0));
        let discards = Rc::new(Cell::new(0));
        let mut scheduler = Scheduler::new(limits).unwrap();
        scheduler
            .register(
                request,
                PublicationState {
                    value: 0,
                    published: published.clone(),
                    discards: discards.clone(),
                },
            )
            .unwrap();
        scheduler.enqueue(request, frame()).unwrap();
        let observer_calls = Rc::new(Cell::new(0));
        let mut progress = scheduler
            .run_local_turn(Instant::now(), |_, _, branch| {
                branch.value += 1;
                Ok::<_, Infallible>(PrepublicationRealtimeFrame::new(
                    submitted_test_frame(ControlledCompletion {
                        ready: Rc::new(Cell::new(true)),
                        fail_wait: false,
                        waits: Rc::new(Cell::new(0)),
                    }),
                    Observer {
                        calls: observer_calls.clone(),
                        fail: false,
                    },
                ))
            })
            .unwrap();
        assert!(progress.failed.is_empty());
        assert_eq!(progress.committed.len(), 1);
        assert_eq!(published.get(), 1);
        assert_eq!(discards.get(), 0);
        assert_eq!(observer_calls.get(), 1);
        let (_, _, transition) = progress.committed.pop().unwrap();
        assert_eq!(transition.into_host_output().unwrap(), vec![3]);
    }
    #[test]
    fn required_transition_rejection_retires_prepared_inputs_without_model_or_commit() {
        struct Required(Executor);
        impl PreparedRealtimeFrameExecutor<TestBackend, TestSampler, ModelState> for Required {
            type Error = ExecutionError;
            type Retained = Matrix;
            fn validate_transition(
                &self,
                transition: &eredu_core::RealtimeFrameTransition,
            ) -> Result<(), ExecutionError> {
                transition
                    .model_call_required()
                    .then_some(())
                    .ok_or(ExecutionError)
            }
            fn execute(
                &mut self,
                state: &mut ModelState,
                temporal: &[Matrix],
                driver: &mut SequentialDecisionDriver<TestBackend, TestSampler>,
                context: &(),
            ) -> Result<Matrix, ExecutionError> {
                self.0.execute(state, temporal, driver, context)
            }
        }
        let schedule = schedule(RealtimeFrameConvention::AbsoluteDelayedSlots);
        let discards = Rc::new(Cell::new(0));
        let state = state(&schedule, discards.clone());
        let mut branch = state.branch().unwrap();
        let mut host = Mechanisms {
            track: true,
            ..Default::default()
        };
        let mut tensors = Mechanisms {
            track: true,
            ..Default::default()
        };
        let mut executor = Required(Executor {
            fail: false,
            calls: 0,
        });
        let mut completion = CompletionMechanism::default();
        let result = execute_realtime_frame::<TestBackend, _, _, _, _, _, _, _, _>(
            &contract(&schedule),
            &payload_contract(&schedule),
            &frame(),
            &mut branch,
            &RealtimeDecisionExecution::new(true),
            &mut host,
            &mut tensors,
            &mut executor,
            &mut completion,
            &(),
        );
        assert!(matches!(
            result,
            Err(RealtimeFrameCoordinatorError::Model(ExecutionError))
        ));
        assert!(
            host.calls > 0 && tensors.calls > 0,
            "rejection follows actual input preparation"
        );
        assert!(!host.lifetimes.is_empty() && !tensors.lifetimes.is_empty());
        assert!(host
            .lifetimes
            .iter()
            .chain(&tensors.lifetimes)
            .all(|weak| weak.upgrade().is_none()));
        assert_eq!(executor.0.calls, 0);
        assert_eq!(completion.calls, 0);
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(branch.schedule_state().frontier(), 0);
        assert_eq!(state.model_state().model_state().executions, 0);
        assert_eq!(branch.model_state().model_state().executions, 0);
        assert!(state.model_state().payload_history().is_empty());
        assert!(branch.model_state().payload_history().is_empty());
        assert_eq!(state.random_state(), Some(&0));
        assert_eq!(branch.random_state(), Some(&0));
        assert!(!branch.has_submission_completion());
        Generation::discard_branch(branch).unwrap();
        assert_eq!(discards.get(), 1);
    }
}
