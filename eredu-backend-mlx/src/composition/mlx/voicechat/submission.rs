//! Coupled native submission through shared completion and host publication.

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

use eredu_architectures::voicechat::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    component_residency::VoiceChatResidencyReport,
    execution::{ExecutionError, ExecutionOutput, Invocation},
    observation::VoiceChatHostObserver,
    perception_rnnt::PerceptionRnntLimits,
    schedule::ScheduleState,
    silence::{PreparedSilenceConditioning, SilenceConditioningError},
    state::StateBranch,
    tts_mog::TtsError,
    tts_prompt::PreparedTtsPrompt,
};
use eredu_core::{backend::Completion, BackendFailure};
use eredu_nn::{RetainedTensorRoots, TensorElementType};
use eredu_runtime::{
    submit_realtime_output, PrepublicationRealtimeFrame, RealtimeGenerationBranch,
    RealtimeOutputSubmissionError,
};
use safemlx::{Array, Stream};

use super::{MlxJointStateBranch, MlxTensor, MlxVoiceChatComponents};
use crate::backend::{
    ordinary_retirement::{self, OrdinaryRetirement},
    random::RandomState,
    runtime::{generation::MlxSamplingBackend, realtime::MlxRealtimeChannelObserver},
    submission_recovery,
};
use crate::composition::mlx::realtime::{
    MlxRealtimeCompletion, MlxRealtimeExecutionResources, MlxRealtimeSubmissionScope,
};

/// Unpublished shared generation branch consumed by one native invocation.
pub type MlxVoiceChatBranch = RealtimeGenerationBranch<
    StateBranch<MlxTensor, MlxJointStateBranch>,
    ScheduleState,
    AgentSamplingPolicy,
    AgentSampler,
    RandomState,
    MlxRealtimeCompletion,
>;

/// Native output gated by exact completion and architecture-owned host validation.
pub type MlxVoiceChatPrepublication = PrepublicationRealtimeFrame<
    ExecutionOutput<MlxTensor>,
    MlxRealtimeCompletion,
    VoiceChatHostObserver<MlxRealtimeChannelObserver>,
>;

/// Portable failures from native coupled submission; native sources remain erased.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatSubmissionError {
    /// Startup silence policy or perception preparation failed.
    #[error(transparent)]
    Silence(#[from] SilenceConditioningError),
    /// A previous invocation still owns this execution or has permanently failed.
    #[error("{0}")]
    Admission(&'static str),
    /// Architecture preflight or coupled equations failed.
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    /// Speech component assembly rejected its retained geometry.
    #[error(transparent)]
    Speech(#[from] TtsError),
    /// Native submission or completion failed, preserving the original cause.
    #[error("VoiceChat native submission failed: {0}")]
    Backend(#[source] BackendFailure),
    /// Shared output attachment rejected this candidate.
    #[error(transparent)]
    Attachment(#[from] eredu_runtime::RealtimeCompletionAttachmentError),
}

struct ExecutionPayload {
    components: RefCell<MlxVoiceChatComponents>,
    stream: Stream,
    poisoned: Cell<bool>,
    active: Cell<bool>,
    _cache_cleanup: crate::backend::ordinary_retirement::TrimCacheOnRetirement,
}

/// Selected native execution whose invocations use the ordinary shared scheduler.
///
/// This low-level adapter owns weights and submission lifetimes. Callers retain
/// session admission, generation state, explicit invocations and conversational
/// policy. Returned host output is deliverable only after its branch commits.
pub struct MlxVoiceChatExecution {
    payload: Arc<OrdinaryRetirement<ExecutionPayload>>,
}

impl MlxVoiceChatExecution {
    /// Adopts materialized components and their execution stream.
    pub fn new(components: MlxVoiceChatComponents, stream: &Stream) -> Self {
        Self {
            payload: Arc::new(OrdinaryRetirement::new(ExecutionPayload {
                components: RefCell::new(components),
                stream: stream.clone(),
                poisoned: Cell::new(false),
                active: Cell::new(false),
                _cache_cleanup: crate::backend::ordinary_retirement::TrimCacheOnRetirement,
            })),
        }
    }

    /// Shares immutable model ownership and its native invocation gate.
    ///
    /// Every handle uses the same active/poison state, so preparation and
    /// invocation remain serialized until exact completion permits release.
    /// Generation state, neural caches, and RNG are not copied by this operation.
    pub fn share(&self) -> Self {
        Self {
            payload: self.payload.clone(),
        }
    }

    /// Reports outstanding shared execution before a sibling session submits.
    /// Poisoned execution follows the ordinary typed failure path instead of
    /// reporting permanent backpressure.
    pub fn is_busy(&self) -> bool {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        !self.payload.poisoned.get() && self.payload.active.get()
    }

    /// Admitted parameter-pool allowances and construction reserve for this execution.
    /// Live state, activations, and native allocator overhead remain separate budgets.
    pub fn residency_report(&self) -> VoiceChatResidencyReport {
        *self.payload.components.borrow().residency_report()
    }

    /// Prepares the architecture-declared tool silence and establishes native completion.
    /// No live input, RNNT hypothesis, generation state or scheduler position is advanced.
    /// The returned tensor may be retained as the facade's prepared silence input.
    pub fn prepare_silence(
        &mut self,
        dtype: TensorElementType,
    ) -> Result<MlxTensor, VoiceChatSubmissionError> {
        PreparedSilenceConditioning::validate_dtype(dtype)?;
        self.prepare(
            |components, stream| Ok(components.prepare_silence(dtype, stream)?),
            |silence, visitor| visitor(silence),
        )
    }

    /// Prepares fixed-Aria startup and completes every retained prompt tensor.
    /// The prompt can initialize independent session branches without advancing
    /// live input, neural caches, generation state, or RNG.
    pub fn prepare_prompt(
        &mut self,
        batch: i32,
        max_token_positions: usize,
    ) -> Result<PreparedTtsPrompt<MlxTensor>, VoiceChatSubmissionError> {
        self.prepare(
            |components, stream| {
                Ok(components.prepare_prompt(batch, max_token_positions, stream)?)
            },
            |prompt, visitor| prompt.visit_retained_tensors(visitor),
        )
    }

    fn prepare<T>(
        &mut self,
        operation: impl FnOnce(
            &mut MlxVoiceChatComponents,
            &Stream,
        ) -> Result<T, VoiceChatSubmissionError>,
        visit_roots: impl FnOnce(&T, &mut dyn FnMut(&MlxTensor)),
    ) -> Result<T, VoiceChatSubmissionError> {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.require_available()?;
        let mut components = self.payload.components.borrow_mut();
        self.payload.active.set(true);
        let owner: Arc<dyn MlxRealtimeExecutionResources> =
            Arc::new(FrameOwner(OrdinaryRetirement::new(FramePayload {
                execution: self.payload.clone(),
                _neural: (),
                _inputs: Vec::new(),
            })));
        let (output, completion) = submit_preparation(
            owner,
            || operation(&mut components, &self.payload.stream),
            visit_roots,
        )?;
        completion.wait().map_err(|error| {
            VoiceChatSubmissionError::Backend(BackendFailure::from_error(error))
        })?;
        // Fixed prompt/silence preparation is a one-use construction region.
        // Release settled scope owners before discarding its unused buffers,
        // with no component borrow held during ordinary retirement.
        drop(completion);
        drop(components);
        crate::trim_allocator_cache().map_err(|error| {
            VoiceChatSubmissionError::Backend(BackendFailure::from_error(error))
        })?;
        Ok(output)
    }

    fn require_available(&self) -> Result<(), VoiceChatSubmissionError> {
        if self.payload.poisoned.get() {
            return Err(VoiceChatSubmissionError::Admission(
                "VoiceChat execution is poisoned",
            ));
        }
        if self.payload.active.get() {
            return Err(VoiceChatSubmissionError::Admission(
                "VoiceChat execution retains an outstanding invocation",
            ));
        }
        Ok(())
    }

    /// Executes one admitted candidate and returns the shared prepublication gate.
    /// `NeedsInput` and other preflight errors submit nothing and permit retry.
    /// Once execution starts, a failed candidate carries failed completion and
    /// this execution is poisoned; neither state nor output can be published.
    pub fn submit(
        &mut self,
        branch: &mut MlxVoiceChatBranch,
        invocation: &Invocation<'_, MlxTensor>,
        input_limits: PerceptionRnntLimits,
        max_character_cells: usize,
    ) -> Result<MlxVoiceChatPrepublication, VoiceChatSubmissionError> {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.require_available()?;
        let mut components = self.payload.components.borrow_mut();
        let mut execution = components.execution(input_limits, max_character_cells)?;
        execution.preflight::<MlxSamplingBackend, _, _, _>(invocation, branch, None)?;
        // Retain the execution and independent joint lifetime authority before
        // entering native work. Recovery may outlive both caller and candidate.
        let mut roots = branch_roots(branch);
        invocation.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
        self.payload.active.set(true);
        let owner: Arc<dyn MlxRealtimeExecutionResources> =
            Arc::new(FrameOwner(OrdinaryRetirement::new(FramePayload {
                execution: self.payload.clone(),
                _neural: branch.model_state().neural.retention(),
                _inputs: roots.clone(),
            })));
        submit_invocation(branch, owner, roots, &self.payload.stream, |branch| {
            execution.execute::<MlxSamplingBackend, _, _, _>(
                invocation,
                branch,
                None,
                &self.payload.stream,
            )
        })
    }
}

fn submit_preparation<T>(
    owner: Arc<dyn MlxRealtimeExecutionResources>,
    operation: impl FnOnce() -> Result<T, VoiceChatSubmissionError>,
    visit_roots: impl FnOnce(&T, &mut dyn FnMut(&MlxTensor)),
) -> Result<(T, MlxRealtimeCompletion), VoiceChatSubmissionError> {
    let scope = MlxRealtimeSubmissionScope::begin(Some(owner))
        .map_err(|error| VoiceChatSubmissionError::Backend(BackendFailure::from_error(error)))?;
    // Failure seals token-validation roots as well as retaining the execution
    // owner through eager work, including during unwinding.
    let mut guard = PreparationGuard(Some(scope));
    let output = operation()?;
    let mut roots = Vec::new();
    visit_roots(&output, &mut |tensor| roots.push(tensor.as_array().clone()));
    let scope = guard.0.take().expect("live preparation scope");
    let completion = scope.finish(roots).map_err(|failure| {
        let error = match failure {
            eredu_runtime::RealtimeCompletionCreationError::BeforeSubmission(error)
            | eredu_runtime::RealtimeCompletionCreationError::AfterSubmission { error, .. } => {
                error
            }
        };
        VoiceChatSubmissionError::Backend(BackendFailure::from_error(error))
    })?;
    Ok((output, completion))
}

struct PreparationGuard(Option<MlxRealtimeSubmissionScope>);

impl Drop for PreparationGuard {
    fn drop(&mut self) {
        if let Some(scope) = self.0.take() {
            drop(scope.fail(Vec::new()));
        }
    }
}

struct FramePayload<N> {
    execution: Arc<OrdinaryRetirement<ExecutionPayload>>,
    _neural: N,
    _inputs: Vec<Array>,
}
impl<N> Drop for FramePayload<N> {
    fn drop(&mut self) {
        self.execution.active.set(false);
    }
}
struct FrameOwner<N: 'static>(OrdinaryRetirement<FramePayload<N>>);
impl<N: 'static> MlxRealtimeExecutionResources for FrameOwner<N> {
    fn poison(&self) {
        self.0.execution.poisoned.set(true);
    }
}

fn branch_roots(branch: &MlxVoiceChatBranch) -> Vec<Array> {
    let mut roots = branch
        .model_state()
        .neural
        .retained_arrays()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    branch
        .model_state()
        .visit_component_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
    roots.extend(
        branch
            .random_state()
            .map(|random| random.as_array().clone()),
    );
    roots
}

struct InvocationGuard<'a> {
    branch: &'a mut MlxVoiceChatBranch,
    scope: Option<MlxRealtimeSubmissionScope>,
}
impl InvocationGuard<'_> {
    fn fail(&mut self) {
        if let Some(scope) = self.scope.take() {
            let completion = scope.fail(branch_roots(self.branch));
            // Exclusive branch access and preflight guarantee no prior attachment.
            // Recovery independently retains resources even during unwinding.
            let _ = self.branch.attach_submission_completion(completion);
        }
    }
}
impl Drop for InvocationGuard<'_> {
    fn drop(&mut self) {
        self.fail();
    }
}

fn submit_invocation(
    branch: &mut MlxVoiceChatBranch,
    owner: Arc<dyn MlxRealtimeExecutionResources>,
    retained_inputs: Vec<Array>,
    stream: &Stream,
    execute: impl FnOnce(&mut MlxVoiceChatBranch) -> Result<ExecutionOutput<MlxTensor>, ExecutionError>,
) -> Result<MlxVoiceChatPrepublication, VoiceChatSubmissionError> {
    let scope = MlxRealtimeSubmissionScope::begin(Some(owner))
        .map_err(|error| VoiceChatSubmissionError::Backend(BackendFailure::from_error(error)))?;
    let mut guard = InvocationGuard {
        branch,
        scope: Some(scope),
    };
    let output = execute(guard.branch)?;
    let mut roots = retained_inputs;
    roots.extend(branch_roots(guard.branch));
    output.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
    let scope = guard.scope.take().expect("live invocation scope");
    let submitted = submit_realtime_output(guard.branch, output, |_, _| {
        scope.finish(roots).map(|completion| {
            let count = completion.retained_resources();
            (completion, count)
        })
    })
    .map_err(|error| match error {
        RealtimeOutputSubmissionError::BeforeSubmission(error)
        | RealtimeOutputSubmissionError::AfterSubmission(error) => {
            VoiceChatSubmissionError::Backend(BackendFailure::from_error(error))
        }
        RealtimeOutputSubmissionError::Attachment(error) => {
            VoiceChatSubmissionError::Attachment(error)
        }
    })?;
    Ok(PrepublicationRealtimeFrame::new(
        submitted,
        VoiceChatHostObserver::new(MlxRealtimeChannelObserver::new(stream)),
    ))
}

#[cfg(all(test, feature = "audio"))]
mod tests;
