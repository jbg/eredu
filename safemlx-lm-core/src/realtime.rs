//! Backend-generic realtime token-session execution and scheduling.

use crate::{
    backend::{Completion, Submission},
    scheduler::{
        RequestId, RequestStatus, Scheduler, SchedulerCapabilities, SchedulerError,
        SchedulerLimits, SchedulerReport, SemanticStateTransaction, TransitionOutput,
        WorkDescriptor, WorkId,
    },
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path, time::Instant};

/// Static codec-token geometry shared by every session of one realtime model.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct RealtimeSpeechConfig {
    total_audio_codebooks: usize,
    input_audio_codebooks: usize,
    generated_audio_codebooks: usize,
    depth_audio_codebooks: usize,
    text_padding_token: i32,
    audio_padding_token: i32,
    audio_delays: Vec<usize>,
}

impl<'de> Deserialize<'de> for RealtimeSpeechConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            total_audio_codebooks: usize,
            input_audio_codebooks: usize,
            generated_audio_codebooks: usize,
            depth_audio_codebooks: usize,
            text_padding_token: i32,
            audio_padding_token: i32,
            audio_delays: Vec<usize>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(
            raw.total_audio_codebooks,
            raw.input_audio_codebooks,
            raw.generated_audio_codebooks,
            raw.depth_audio_codebooks,
            raw.text_padding_token,
            raw.audio_padding_token,
            raw.audio_delays,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl RealtimeSpeechConfig {
    /// Creates and validates portable realtime codec geometry.
    pub fn new(
        total_audio_codebooks: usize,
        input_audio_codebooks: usize,
        generated_audio_codebooks: usize,
        depth_audio_codebooks: usize,
        text_padding_token: i32,
        audio_padding_token: i32,
        audio_delays: Vec<usize>,
    ) -> Result<Self, RealtimeConfigError> {
        if total_audio_codebooks == 0
            || input_audio_codebooks == 0
            || generated_audio_codebooks == 0
            || depth_audio_codebooks == 0
        {
            return Err(RealtimeConfigError::EmptyCodebookGeometry);
        }
        if input_audio_codebooks + generated_audio_codebooks != total_audio_codebooks {
            return Err(RealtimeConfigError::CodebookPartition {
                total: total_audio_codebooks,
                input: input_audio_codebooks,
                generated: generated_audio_codebooks,
            });
        }
        if audio_delays.len() != total_audio_codebooks {
            return Err(RealtimeConfigError::DelayCount {
                expected: total_audio_codebooks,
                actual: audio_delays.len(),
            });
        }
        Ok(Self {
            total_audio_codebooks,
            input_audio_codebooks,
            generated_audio_codebooks,
            depth_audio_codebooks,
            text_padding_token,
            audio_padding_token,
            audio_delays,
        })
    }

    /// Total number of temporal-model audio codebooks.
    pub const fn total_audio_codebooks(&self) -> usize {
        self.total_audio_codebooks
    }
    /// Number of live input-side codebooks per frame.
    pub const fn input_audio_codebooks(&self) -> usize {
        self.input_audio_codebooks
    }
    /// Number of generated-side codebooks per frame.
    pub const fn generated_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks
    }
    /// Number of depth-transformer codebooks per frame.
    pub const fn depth_audio_codebooks(&self) -> usize {
        self.depth_audio_codebooks
    }
    /// Text token used before sampled text is available.
    pub const fn text_padding_token(&self) -> i32 {
        self.text_padding_token
    }
    /// Audio token used while delayed streams warm up.
    pub const fn audio_padding_token(&self) -> i32 {
        self.audio_padding_token
    }
    /// Per-codebook delays excluding the leading text delay.
    pub fn audio_delays(&self) -> &[usize] {
        &self.audio_delays
    }
    /// Largest audio delay in frames.
    pub fn max_audio_delay(&self) -> usize {
        self.audio_delays.iter().copied().max().unwrap_or(0)
    }
}

/// Invalid portable realtime configuration.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RealtimeConfigError {
    /// Every realtime codebook dimension must be nonzero.
    #[error("realtime codebook geometry must be nonzero")]
    EmptyCodebookGeometry,
    /// Input and generated codebooks must partition the temporal codebooks.
    #[error(
        "realtime input ({input}) and generated ({generated}) codebooks do not partition total {total}"
    )]
    CodebookPartition {
        /// Total temporal codebooks.
        total: usize,
        /// Input codebooks.
        input: usize,
        /// Generated codebooks.
        generated: usize,
    },
    /// The delay schedule must describe every temporal audio codebook.
    #[error("realtime delay schedule has {actual} entries, expected {expected}")]
    DelayCount {
        /// Expected delay count.
        expected: usize,
        /// Actual delay count.
        actual: usize,
    },
    /// Sampling temperatures must be finite and nonnegative.
    #[error(
        "realtime sampling temperatures must be finite and non-negative, got text={text} audio={audio}"
    )]
    SamplingTemperature {
        /// Invalid text temperature.
        text: f32,
        /// Invalid audio temperature.
        audio: f32,
    },
}

/// Portable sampling controls for one realtime request.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RealtimeSampling {
    text_temperature: f32,
    audio_temperature: f32,
    seed: u64,
}

impl<'de> Deserialize<'de> for RealtimeSampling {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            text_temperature: f32,
            audio_temperature: f32,
            seed: u64,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.text_temperature, raw.audio_temperature, raw.seed)
            .map_err(serde::de::Error::custom)
    }
}

impl RealtimeSampling {
    /// Creates validated request-local controls.
    pub fn new(
        text_temperature: f32,
        audio_temperature: f32,
        seed: u64,
    ) -> Result<Self, RealtimeConfigError> {
        if !text_temperature.is_finite()
            || text_temperature < 0.0
            || !audio_temperature.is_finite()
            || audio_temperature < 0.0
        {
            return Err(RealtimeConfigError::SamplingTemperature {
                text: text_temperature,
                audio: audio_temperature,
            });
        }
        Ok(Self {
            text_temperature,
            audio_temperature,
            seed,
        })
    }

    /// Deterministic greedy sampling.
    pub const fn greedy() -> Self {
        Self {
            text_temperature: 0.0,
            audio_temperature: 0.0,
            seed: 0,
        }
    }
    /// Text sampling temperature.
    pub const fn text_temperature(self) -> f32 {
        self.text_temperature
    }
    /// Audio sampling temperature.
    pub const fn audio_temperature(self) -> f32 {
        self.audio_temperature
    }
    /// Deterministic root seed interpreted by the selected backend.
    pub const fn seed(self) -> u64 {
        self.seed
    }
    /// Whether either stream requires stochastic sampling.
    pub const fn is_stochastic(self) -> bool {
        self.text_temperature != 0.0 || self.audio_temperature != 0.0
    }
}

impl Default for RealtimeSampling {
    fn default() -> Self {
        Self::greedy()
    }
}

/// High-level contract implemented once per realtime execution backend.
///
/// Codec frames, generated outputs, cache/session state, model values, and
/// completions are opaque associated types. Core schedules complete realtime
/// steps and never models tensor operations or exposes native streams.
pub trait RealtimeBackend {
    /// Backend-owned loaded realtime model.
    type Model;
    /// Stable identity used to reject cross-model session handoff.
    type ModelIdentity: Clone + Debug + Eq;
    /// Backend-owned encoded frame or prompt transition.
    type Input: WorkDescriptor;
    /// Backend-owned generated text/audio frame.
    type Output;
    /// Request-local cache, delayed streams, sampler, and random state.
    type Session: SemanticStateTransaction<Branch = Self::Session, Error = Self::Error>;
    /// Exact completion retaining submitted input/output resources.
    type Completion: Completion<Error = Self::Error>;
    /// Structured backend failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable backend name used for scheduler capability telemetry.
    fn name(&self) -> &str;
    /// Returns the complete model identity.
    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity;
    /// Describes the first material difference between two model identities.
    fn model_identity_mismatch(
        &self,
        expected: &Self::ModelIdentity,
        actual: &Self::ModelIdentity,
    ) -> Option<String> {
        (expected != actual).then(|| "model identity".into())
    }
    /// Returns portable codec geometry.
    fn speech_config(&self, model: &Self::Model) -> RealtimeSpeechConfig;
    /// Creates one request-local session.
    fn create_session(
        &self,
        model: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error>;
    /// Validates a released session before attaching it to this model.
    fn validate_session(
        &self,
        model: &Self::Model,
        session: &Self::Session,
    ) -> Result<(), Self::Error>;
    /// Validates one backend-owned input frame against model geometry.
    fn validate_input(&self, model: &Self::Model, input: &Self::Input) -> Result<(), Self::Error>;
    /// Returns the stable batch dimension of a validated input.
    fn input_batch_size(&self, input: &Self::Input) -> usize;
    /// Replaces request-local sampling and randomness state.
    fn set_sampling(
        &self,
        session: &mut Self::Session,
        sampling: RealtimeSampling,
    ) -> Result<(), Self::Error>;
    /// Submits one complete temporal/depth transition.
    fn submit_step(
        &self,
        model: &mut Self::Model,
        session: &mut Self::Session,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error>;
    /// Number of backend resources explicitly retained by a completion.
    fn retained_resources(&self, _completion: &Self::Completion) -> usize {
        0
    }
}

/// Model preparation contract for a complete realtime backend.
///
/// Artifact interpretation and materialization belong to the selected backend;
/// callers use the same loading function regardless of the concrete runtime.
pub trait RealtimeModelLoadingBackend: RealtimeBackend + Sized {
    /// Backend-specific materialization policy.
    type LoadOptions;

    /// Prepares one backend-owned realtime model from an artifact directory.
    fn prepare_realtime_model(
        &self,
        artifact: &Path,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error>;
}

/// Loads a realtime model on the selected backend using default load policy.
pub fn load_realtime_model<B>(
    backend: B,
    artifact: impl AsRef<Path>,
) -> Result<RealtimeModel<B>, B::Error>
where
    B: RealtimeModelLoadingBackend,
    B::LoadOptions: Default,
{
    load_realtime_model_with_options(backend, artifact, B::LoadOptions::default())
}

/// Loads a realtime model on the selected backend using explicit load policy.
pub fn load_realtime_model_with_options<B: RealtimeModelLoadingBackend>(
    backend: B,
    artifact: impl AsRef<Path>,
    options: B::LoadOptions,
) -> Result<RealtimeModel<B>, B::Error> {
    let model = backend.prepare_realtime_model(artifact.as_ref(), options)?;
    Ok(RealtimeModel::new(backend, model))
}

/// Selected realtime backend and its loaded model.
pub struct RealtimeModel<B: RealtimeBackend> {
    backend: B,
    model: B::Model,
}

impl<B: RealtimeBackend> RealtimeModel<B> {
    /// Binds one loaded model to its execution backend.
    pub const fn new(backend: B, model: B::Model) -> Self {
        Self { backend, model }
    }
    /// Borrows the selected backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }
    /// Borrows the backend-owned model.
    pub const fn model(&self) -> &B::Model {
        &self.model
    }
    /// Mutably borrows the backend-owned model.
    pub fn model_mut(&mut self) -> &mut B::Model {
        &mut self.model
    }
    /// Portable codec-token geometry.
    pub fn speech_config(&self) -> RealtimeSpeechConfig {
        self.backend.speech_config(&self.model)
    }
    /// Consumes the runtime into backend and model values.
    pub fn into_parts(self) -> (B, B::Model) {
        (self.backend, self.model)
    }
}

/// Request-local realtime state released from a scheduler.
pub struct RealtimeSession<B: RealtimeBackend> {
    model_identity: B::ModelIdentity,
    state: B::Session,
    batch_size: Option<usize>,
}

impl<B: RealtimeBackend> RealtimeSession<B> {
    /// Borrows backend-owned request state.
    pub const fn state(&self) -> &B::Session {
        &self.state
    }
    /// Mutably borrows backend-owned request state.
    pub fn state_mut(&mut self) -> &mut B::Session {
        &mut self.state
    }
    /// Committed batch dimension, when at least one frame was accepted.
    pub const fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }
}

impl<B: RealtimeBackend> SemanticStateTransaction for RealtimeSession<B> {
    type Branch = Self;
    type Error = B::Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(Self {
            model_identity: self.model_identity.clone(),
            state: self.state.branch()?,
            batch_size: self.batch_size,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.state.commit_branch(branch.state)?;
        self.batch_size = branch.batch_size;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        B::Session::discard_branch(branch.state)
    }
}

struct RealtimeTransition<B: RealtimeBackend> {
    backend_name: String,
    retained_resources: usize,
    output: B::Output,
    completion: B::Completion,
}

impl<B: RealtimeBackend> TransitionOutput for RealtimeTransition<B> {
    type Error = B::Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.completion.is_complete()
    }
    fn backend_name(&self) -> Option<String> {
        Some(self.backend_name.clone())
    }
    fn retained_resources(&self) -> usize {
        self.retained_resources
    }
}

/// One committed realtime transition and its scheduler identity.
pub struct RealtimeCompletedStep<O> {
    work: WorkId,
    output: O,
}

impl<O> RealtimeCompletedStep<O> {
    /// Scheduler-assigned work identity.
    pub const fn work(&self) -> WorkId {
        self.work
    }
    /// Borrows the backend-owned generated frame.
    pub const fn output(&self) -> &O {
        &self.output
    }
    /// Consumes this completion.
    pub fn into_parts(self) -> (WorkId, O) {
        (self.work, self.output)
    }
}

/// Realtime coordination failure with structured backend context.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeError<E: std::error::Error + 'static> {
    /// Selected backend rejected model, session, input, or execution.
    #[error("realtime backend failed: {0}")]
    Backend(#[source] E),
    /// Generic scheduler lifecycle or capacity failure.
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    /// A runtime or released session belongs to a different model.
    #[error("realtime model {component} does not match the scheduler model")]
    ModelMismatch {
        /// Backend-defined identity component that differs.
        component: String,
    },
    /// A request changed its batch size after its first accepted frame.
    #[error("realtime request {request} changed batch size from {expected} to {actual}")]
    BatchSize {
        /// Request identity.
        request: u64,
        /// Committed batch size.
        expected: usize,
        /// New input batch size.
        actual: usize,
    },
    /// A bounded drain must permit at least one frame.
    #[error("realtime scheduler frame bound must be positive")]
    EmptyRunBound,
    /// Sampling cannot change while accepted frames still use the prior state.
    #[error("realtime request {request} has {queued} queued frames; drain or cancel them before changing sampling")]
    SamplingWhileQueued {
        /// Request identity.
        request: u64,
        /// Accepted queued frames.
        queued: usize,
    },
    /// At least one submitted transition failed asynchronously.
    #[error("realtime work {work:?} failed asynchronously: {message}")]
    Asynchronous {
        /// Failed work identity.
        work: WorkId,
        /// Scheduler-provided failure context.
        message: String,
    },
}

/// Fair bounded realtime scheduler generic over the selected backend.
pub struct RealtimeScheduler<B: RealtimeBackend> {
    model_identity: B::ModelIdentity,
    scheduler: Scheduler<B::Input, RealtimeSession<B>, RealtimeTransition<B>>,
}

impl<B: RealtimeBackend> RealtimeScheduler<B> {
    /// Binds an empty scheduler to one selected model.
    pub fn new(
        model: &RealtimeModel<B>,
        limits: SchedulerLimits,
    ) -> Result<Self, RealtimeError<B::Error>> {
        Ok(Self {
            model_identity: model.backend.model_identity(&model.model),
            scheduler: Scheduler::new(limits)?,
        })
    }

    fn validate_model(&self, model: &RealtimeModel<B>) -> Result<(), RealtimeError<B::Error>> {
        let actual = model.backend.model_identity(&model.model);
        if let Some(component) = model
            .backend
            .model_identity_mismatch(&self.model_identity, &actual)
        {
            return Err(RealtimeError::ModelMismatch { component });
        }
        Ok(())
    }

    /// Registers a request with fresh backend-owned state.
    pub fn register_request(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        let state = model
            .backend
            .create_session(&model.model, sampling)
            .map_err(RealtimeError::Backend)?;
        self.scheduler.register(
            request,
            RealtimeSession {
                model_identity: self.model_identity.clone(),
                state,
                batch_size: None,
            },
        )?;
        Ok(())
    }

    /// Registers a previously released request session.
    pub fn register_request_with_session(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        session: RealtimeSession<B>,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        if let Some(component) = model
            .backend
            .model_identity_mismatch(&self.model_identity, &session.model_identity)
        {
            return Err(RealtimeError::ModelMismatch { component });
        }
        model
            .backend
            .validate_session(&model.model, &session.state)
            .map_err(RealtimeError::Backend)?;
        self.scheduler.register(request, session)?;
        Ok(())
    }

    /// Enqueues one encoded or forced frame.
    pub fn enqueue(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        input: B::Input,
    ) -> Result<WorkId, RealtimeError<B::Error>> {
        self.enqueue_with_deadline(model, request, input, None)
    }

    /// Enqueues one frame with an optional absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        input: B::Input,
        deadline: Option<Instant>,
    ) -> Result<WorkId, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        model
            .backend
            .validate_input(&model.model, &input)
            .map_err(RealtimeError::Backend)?;
        let batch = model.backend.input_batch_size(&input);
        self.validate_batch(request, batch)?;
        let work = self
            .scheduler
            .enqueue_with_deadline(request, input, deadline)?;
        self.scheduler
            .request_state_mut(request)?
            .batch_size
            .get_or_insert(batch);
        Ok(work)
    }

    /// Atomically enqueues ordered frames.
    pub fn enqueue_batch(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        inputs: Vec<B::Input>,
    ) -> Result<Vec<WorkId>, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        let mut expected = self
            .scheduler
            .request_state(request)
            .ok_or(SchedulerError::UnknownRequest(request))?
            .batch_size;
        for input in &inputs {
            model
                .backend
                .validate_input(&model.model, input)
                .map_err(RealtimeError::Backend)?;
            let actual = model.backend.input_batch_size(input);
            if let Some(expected) = expected {
                if actual != expected {
                    return Err(RealtimeError::BatchSize {
                        request: request.value(),
                        expected,
                        actual,
                    });
                }
            } else {
                expected = Some(actual);
            }
        }
        let work = self.scheduler.enqueue_batch(request, inputs)?;
        if let Some(batch) = expected {
            self.scheduler
                .request_state_mut(request)?
                .batch_size
                .get_or_insert(batch);
        }
        Ok(work)
    }

    fn validate_batch(
        &self,
        request: RequestId,
        actual: usize,
    ) -> Result<(), RealtimeError<B::Error>> {
        let state = self
            .scheduler
            .request_state(request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        if let Some(expected) = state.batch_size {
            if expected != actual {
                return Err(RealtimeError::BatchSize {
                    request: request.value(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Advances one unbounded fair scheduling turn.
    pub fn run_queued(
        &mut self,
        model: &mut RealtimeModel<B>,
    ) -> Result<Vec<RealtimeCompletedStep<B::Output>>, RealtimeError<B::Error>> {
        self.run_bounded(model, usize::MAX)
    }

    /// Advances at most `max_frames` fair-ordered transitions.
    pub fn run_bounded(
        &mut self,
        model: &mut RealtimeModel<B>,
        max_frames: usize,
    ) -> Result<Vec<RealtimeCompletedStep<B::Output>>, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        if max_frames == 0 {
            return Err(RealtimeError::EmptyRunBound);
        }
        let now = Instant::now();
        let mut progress = self.scheduler.poll_completions(now);
        self.scheduler.prepare_bounded(max_frames, now)?;
        let backend_name = model.backend.name().to_owned();
        let backend = &model.backend;
        let backend_model = &mut model.model;
        progress.newly_submitted = self.scheduler.submit_prepared(
            now,
            |_, input, session| -> Result<RealtimeTransition<B>, B::Error> {
                let submission = backend.submit_step(backend_model, &mut session.state, input)?;
                let retained_resources = backend.retained_resources(&submission.completion);
                Ok(RealtimeTransition {
                    backend_name: backend_name.clone(),
                    retained_resources,
                    output: submission.output,
                    completion: submission.completion,
                })
            },
        )?;
        let completed = self.scheduler.poll_completions(now);
        progress.committed.extend(completed.committed);
        progress.failed.extend(completed.failed);
        if let Some((work, failure)) = progress.failed.first() {
            return Err(RealtimeError::Asynchronous {
                work: *work,
                message: failure.to_string(),
            });
        }
        Ok(progress
            .committed
            .into_iter()
            .map(|(work, _, transition)| RealtimeCompletedStep {
                work,
                output: transition.output,
            })
            .collect())
    }

    /// Completes one request and drops its backend session.
    pub fn finish_request(&mut self, request: RequestId) -> Result<(), RealtimeError<B::Error>> {
        self.scheduler.finish(request)?;
        Ok(())
    }
    /// Cancels one request and discards queued frames.
    pub fn cancel_request(&mut self, request: RequestId) -> Result<(), RealtimeError<B::Error>> {
        self.scheduler.cancel(request)?;
        Ok(())
    }
    /// Releases an idle request for persistence or resumption.
    pub fn release_request(
        &mut self,
        request: RequestId,
    ) -> Result<RealtimeSession<B>, RealtimeError<B::Error>> {
        Ok(self.scheduler.release(request)?)
    }
    /// Removes a terminal identity for explicit reuse.
    pub fn forget_terminal_request(
        &mut self,
        request: RequestId,
    ) -> Result<RequestStatus, RealtimeError<B::Error>> {
        Ok(self.scheduler.forget_terminal(request)?)
    }
    /// Lifecycle state for a known request.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.scheduler.request_status(request)
    }
    /// Queued frame count for one request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.scheduler.queued_for_request(request)
    }
    /// Replaces sampling controls for an idle active request.
    pub fn set_request_sampling(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        let queued = self.scheduler.queued_for_request(request);
        if queued != 0 {
            return Err(RealtimeError::SamplingWhileQueued {
                request: request.value(),
                queued,
            });
        }
        let state = self.scheduler.request_state_mut(request)?;
        model
            .backend
            .set_sampling(&mut state.state, sampling)
            .map_err(RealtimeError::Backend)
    }
    /// Generic occupancy and lifecycle telemetry.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }
    /// Configured bounds and observed backend capabilities.
    pub fn capabilities(&self) -> SchedulerCapabilities {
        self.scheduler.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Clone)]
    struct MockSession {
        step: u32,
        sampling: RealtimeSampling,
    }
    impl SemanticStateTransaction for MockSession {
        type Branch = Self;
        type Error = Infallible;
        fn branch(&self) -> Result<Self, Self::Error> {
            Ok(self.clone())
        }
        fn commit_branch(&mut self, branch: Self) -> Result<(), Self::Error> {
            *self = branch;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Frame(Vec<u32>);
    impl WorkDescriptor for Frame {
        type Error = Infallible;
        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
            output.extend_from_slice(&self.0);
            Ok(())
        }
    }

    struct Done;
    impl Completion for Done {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct MockBackend;
    impl RealtimeBackend for MockBackend {
        type Model = u64;
        type ModelIdentity = u64;
        type Input = Frame;
        type Output = u32;
        type Session = MockSession;
        type Completion = Done;
        type Error = Infallible;

        fn name(&self) -> &str {
            "mock-realtime"
        }
        fn model_identity(&self, model: &u64) -> u64 {
            *model
        }
        fn speech_config(&self, _: &u64) -> RealtimeSpeechConfig {
            RealtimeSpeechConfig::new(2, 1, 1, 1, 0, 0, vec![0, 1]).unwrap()
        }
        fn create_session(
            &self,
            _: &u64,
            sampling: RealtimeSampling,
        ) -> Result<MockSession, Infallible> {
            Ok(MockSession { step: 0, sampling })
        }
        fn validate_session(&self, _: &u64, _: &MockSession) -> Result<(), Infallible> {
            Ok(())
        }
        fn validate_input(&self, _: &u64, _: &Frame) -> Result<(), Infallible> {
            Ok(())
        }
        fn input_batch_size(&self, input: &Frame) -> usize {
            input.0.len()
        }
        fn set_sampling(
            &self,
            session: &mut MockSession,
            sampling: RealtimeSampling,
        ) -> Result<(), Infallible> {
            session.sampling = sampling;
            Ok(())
        }
        fn submit_step(
            &self,
            model: &mut u64,
            session: &mut MockSession,
            input: &Frame,
        ) -> Result<Submission<u32, Done>, Infallible> {
            session.step += 1;
            Ok(Submission {
                output: *model as u32 + session.step + input.0.iter().sum::<u32>(),
                completion: Done,
            })
        }
    }

    impl RealtimeModelLoadingBackend for MockBackend {
        type LoadOptions = u64;

        fn prepare_realtime_model(
            &self,
            _: &Path,
            identity: Self::LoadOptions,
        ) -> Result<Self::Model, Self::Error> {
            Ok(identity)
        }
    }

    #[test]
    fn selected_backend_owns_realtime_model_preparation() {
        let model = load_realtime_model_with_options(MockBackend, "unused", 37).unwrap();
        assert_eq!(*model.model(), 37);
        assert_eq!(model.backend().name(), "mock-realtime");
    }

    #[test]
    fn mock_backend_runs_fair_realtime_sessions_without_accelerator_types() {
        let mut model = RealtimeModel::new(MockBackend, 10);
        let limits = SchedulerLimits::with_execution_bounds(2, 4, 2, 2, 1, usize::MAX).unwrap();
        let mut scheduler = RealtimeScheduler::new(&model, limits).unwrap();
        let first = RequestId::new(1);
        let second = RequestId::new(2);
        scheduler
            .register_request(&model, first, RealtimeSampling::greedy())
            .unwrap();
        scheduler
            .register_request(&model, second, RealtimeSampling::greedy())
            .unwrap();
        scheduler.enqueue(&model, first, Frame(vec![1])).unwrap();
        scheduler.enqueue(&model, second, Frame(vec![2])).unwrap();
        assert!(matches!(
            scheduler.set_request_sampling(
                &model,
                first,
                RealtimeSampling::new(0.5, 0.5, 7).unwrap()
            ),
            Err(RealtimeError::SamplingWhileQueued { .. })
        ));
        assert_eq!(
            scheduler
                .run_queued(&mut model)
                .unwrap()
                .into_iter()
                .map(|step| step.into_parts().1)
                .collect::<Vec<_>>(),
            vec![12, 13]
        );
        let updated = RealtimeSampling::new(0.5, 0.5, 7).unwrap();
        scheduler
            .set_request_sampling(&model, first, updated)
            .unwrap();
        assert_eq!(
            scheduler.release_request(first).unwrap().state().sampling,
            updated
        );
    }

    #[test]
    fn sampling_and_speech_config_validate_portably() {
        assert!(RealtimeSampling::new(f32::NAN, 0.0, 0).is_err());
        let config = RealtimeSpeechConfig::new(4, 2, 2, 3, 11, 12, vec![0, 1, 2, 3]).unwrap();
        assert_eq!(config.max_audio_delay(), 3);
        assert_eq!(config.generated_audio_codebooks(), 2);
        assert_eq!(
            serde_json::from_str::<RealtimeSpeechConfig>(&serde_json::to_string(&config).unwrap())
                .unwrap(),
            config
        );
        let sampling = RealtimeSampling::new(0.7, 0.9, 42).unwrap();
        assert_eq!(
            serde_json::from_str::<RealtimeSampling>(&serde_json::to_string(&sampling).unwrap())
                .unwrap(),
            sampling
        );
        assert!(RealtimeSpeechConfig::new(4, 1, 1, 1, 0, 0, vec![0; 4]).is_err());
    }
}
