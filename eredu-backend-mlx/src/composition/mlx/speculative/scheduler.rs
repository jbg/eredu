//! MLX speculative scheduling over the portable executor contract.

use std::{cell::Cell, marker::PhantomData, rc::Rc, time::Duration};

#[cfg(test)]
use eredu_core::generation::MtpRequestPhase;
#[cfg(test)]
use eredu_core::{
    resolve_optimistic_branch, SpeculativeDraftBlock, SpeculativeExecutionTopology,
    SpeculativeOptimisticBranch, SpeculativeProposal,
};
use eredu_core::{
    MtpSchedulerStats, MtpStats, SpeculativeCallbackPublisher, SpeculativeDriverError,
    SpeculativeExecutor, SpeculativeOutputError, SpeculativeOutputRuntime, SpeculativeRequestTable,
    SpeculativeSampling, SpeculativeSemanticConstraint, SpeculativeSemanticState,
    SpeculativeTelemetry,
};
use safemlx::{error::Exception, Array};
#[cfg(test)]
use safemlx::{ops::indexing::TryIndexOp, transforms::async_eval_with_event, Stream};

use crate::{
    backend::runtime::generation::sampler::SpeculativeSampler,
    backend::runtime::media::input::{InputPayload, Modality, ModelInput},
    composition::mlx::{
        speculative::{MlxSpeculativeCompletion, MlxSpeculativeSampling, MtpExecutionStreams},
        MlxModelInput,
    },
};
use eredu_core::generation::{
    FinishReason, GenerationCancellationToken, GenerationSequence, MtpConfig, MtpRequestId,
    MtpSchedulerOptions, SemanticEvent,
};

/// Component timings accumulated by an architecture-specific MTP backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpComponentTimings {
    /// Committed-context encoding and assistant K/V projection.
    pub draft_context: Duration,
    /// Assistant proposal-block execution.
    pub draft_assistant: Duration,
    /// Raw vocabulary-head projection.
    pub draft_head: Duration,
    /// Target verification execution.
    pub target_verification: Duration,
}

thread_local! {
    static COMPONENT_TIMING_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in for device-timeline MTP component profiling.
///
/// Schedulers created while this guard is alive collect architecture-level
/// timings. Timestamp boundaries add profiling overhead, so normal generation
/// leaves them disabled. The guard is intentionally bound to its creating
/// thread and restores the previous setting when dropped.
pub struct MtpComponentTimingGuard {
    previous: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl MtpComponentTimingGuard {
    /// Enables component profiling for schedulers created on this thread.
    pub fn enable() -> Self {
        let previous = COMPONENT_TIMING_ENABLED.replace(true);
        Self {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for MtpComponentTimingGuard {
    fn drop(&mut self) {
        COMPONENT_TIMING_ENABLED.set(self.previous);
    }
}

fn component_timing_enabled() -> bool {
    COMPONENT_TIMING_ENABLED.get()
}

impl MtpComponentTimings {
    fn add_to(self, stats: &mut MtpStats) {
        stats.draft_context_time += self.draft_context;
        stats.draft_assistant_time += self.draft_assistant;
        stats.draft_head_time += self.draft_head;
        stats.target_verification_time += self.target_verification;
    }
}

impl SpeculativeTelemetry for MtpComponentTimings {
    fn record(self, stats: &mut MtpStats) {
        self.add_to(stats);
    }
}

/// MLX-associated types required by the facade's sampling loop.
///
/// Model execution is expressed by the portable core contract; only logits
/// sampling and stream placement remain MLX-specific in this module.
pub trait MlxSpeculativeRuntime<'a>:
    SpeculativeExecutor<
        Input = MlxModelInput,
        Logits = Array,
        Context<'a> = MtpExecutionStreams<'a>,
        Completion = MlxSpeculativeCompletion,
        Telemetry = MtpComponentTimings,
        Error = Exception,
    > + 'a
{
}

impl<'a, T> MlxSpeculativeRuntime<'a> for T where
    T: SpeculativeExecutor<
            Input = MlxModelInput,
            Logits = Array,
            Context<'a> = MtpExecutionStreams<'a>,
            Completion = MlxSpeculativeCompletion,
            Telemetry = MtpComponentTimings,
            Error = Exception,
        > + 'a
{
}

type CommittedOutputRuntime<'a, S> = SpeculativeOutputRuntime<
    MlxSpeculativeSampling<S>,
    SpeculativeSemanticConstraint,
    SpeculativeCallbackPublisher<'a>,
>;

fn plain_runtime<'a, S, F>(
    sampler: S,
    config: &MtpConfig,
    mut on_token: F,
) -> CommittedOutputRuntime<'a, S>
where
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception> + 'a,
{
    SpeculativeOutputRuntime::new(
        MlxSpeculativeSampling::new(sampler),
        GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied()),
        SpeculativeSemanticConstraint::plain(),
        SpeculativeCallbackPublisher::tokens(move |token| {
            on_token(token).map_err(|error| SpeculativeOutputError::publication(error.to_string()))
        }),
        GenerationCancellationToken::new(),
    )
}

fn semantic_runtime<'a, S, F>(
    sampler: S,
    config: &MtpConfig,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: F,
) -> CommittedOutputRuntime<'a, S>
where
    S: SpeculativeSampler + Clone,
    F: FnMut(SemanticEvent) + 'a,
{
    SpeculativeOutputRuntime::new(
        MlxSpeculativeSampling::new(sampler),
        GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied()),
        SpeculativeSemanticConstraint::semantic(semantic),
        SpeculativeCallbackPublisher::semantic(on_event),
        cancellation,
    )
}

type MlxRequestTable<'a, B, S> = SpeculativeRequestTable<
    'a,
    B,
    MlxSpeculativeSampling<S>,
    SpeculativeSemanticConstraint,
    SpeculativeCallbackPublisher<'a>,
>;

pub struct MtpRequestOutput<S> {
    pub id: MtpRequestId,
    pub token_ids: Vec<u32>,
    pub stats: MtpStats,
    pub sampler: S,
    pub finish_reason: Option<FinishReason>,
    #[cfg(test)]
    pub cancelled: bool,
}

pub struct MtpScheduleOutput<S> {
    pub requests: Vec<MtpRequestOutput<S>>,
    pub scheduler: MtpSchedulerStats,
}

/// Single-threaded fair MTP request scheduler.
///
/// MLX streams already provide asynchronous device queues. The scheduler stays
/// deliberately single-threaded: it submits lazy target graphs, performs CPU
/// draft work, and synchronizes only when a verification result is resolved.
/// Model parameters are shared through the backend; every request owns its
/// cache, target state, sampler, PRNG substreams, output, and statistics.
pub struct MlxMtpScheduler<'a, B, S>
where
    B: MlxSpeculativeRuntime<'a>,
    S: SpeculativeSampler + Clone + 'a,
{
    backend: &'a mut B,
    streams: MtpExecutionStreams<'a>,
    component_timing: bool,
    requests: MlxRequestTable<'a, B, S>,
}

impl<'a, B, S> MlxMtpScheduler<'a, B, S>
where
    B: MlxSpeculativeRuntime<'a>,
    S: SpeculativeSampler + Clone + 'a,
{
    /// Creates a scheduler over shared model parameters and explicit streams.
    pub fn new(
        backend: &'a mut B,
        streams: MtpExecutionStreams<'a>,
        options: MtpSchedulerOptions,
    ) -> Result<Self, Exception> {
        let requests = SpeculativeRequestTable::new(options, streams.topology())
            .map_err(|error| Exception::custom(error.to_string()))?;
        let component_timing = component_timing_enabled() && backend.supports_telemetry();
        backend.set_telemetry_enabled(component_timing);
        Ok(Self {
            backend,
            streams,
            component_timing,
            requests,
        })
    }

    /// Submits one independent request.
    #[allow(clippy::too_many_arguments)]
    pub fn submit<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: MtpConfig,
        prng_key: Option<Array>,
        sampler: S,
        on_token: F,
    ) -> Result<MtpRequestId, Exception>
    where
        F: FnMut(u32) -> Result<(), Exception> + 'a,
    {
        let runtime = plain_runtime(sampler, &config, on_token);
        self.submit_runtime(cache, input, config, prng_key, runtime)
    }

    /// Submits one request with transactional decoded semantic output.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn submit_with_semantics<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: MtpConfig,
        prng_key: Option<Array>,
        sampler: S,
        semantic: Box<dyn SpeculativeSemanticState>,
        on_event: F,
    ) -> Result<MtpRequestId, Exception>
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        self.submit_with_semantics_cancellable(
            cache,
            input,
            config,
            prng_key,
            sampler,
            semantic,
            GenerationCancellationToken::new(),
            on_event,
        )
    }

    /// Submits one semantic request controlled by a public cancellation token.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_with_semantics_cancellable<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: MtpConfig,
        prng_key: Option<Array>,
        sampler: S,
        semantic: Box<dyn SpeculativeSemanticState>,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<MtpRequestId, Exception>
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        let runtime = semantic_runtime(sampler, &config, semantic, cancellation, on_event);
        self.submit_runtime(cache, input, config, prng_key, runtime)
    }

    fn submit_runtime(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: MtpConfig,
        prng_key: Option<Array>,
        runtime: CommittedOutputRuntime<'a, S>,
    ) -> Result<MtpRequestId, Exception> {
        config
            .validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        validate_input(input)?;
        let input = MlxModelInput::from(input);
        let randomness = <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
            prng_key,
            config.temperature,
            self.streams,
        )?;
        self.requests
            .submit(
                self.backend,
                cache,
                input,
                config,
                runtime,
                randomness,
                self.component_timing,
                self.streams,
            )
            .map_err(speculative_driver_error)
    }

    /// Returns the current phase for a submitted request.
    #[cfg(test)]
    pub fn phase(&self, id: MtpRequestId) -> Option<MtpRequestPhase> {
        self.requests.phase(id)
    }

    /// Requests independent cancellation.
    ///
    /// An in-flight target transaction is resolved to a safe cache boundary
    /// before the request enters `Cancelled`; other requests continue.
    #[cfg(test)]
    pub fn cancel(&mut self, id: MtpRequestId) -> Result<(), Exception> {
        self.requests.cancel(id).map_err(speculative_driver_error)
    }

    /// Returns whether every request is completed or cancelled.
    #[cfg(test)]
    pub fn is_finished(&self) -> bool {
        self.requests.is_finished()
    }

    /// Performs one fair scheduler operation.
    ///
    /// Returns `false` when every request is terminal.
    pub fn step(&mut self) -> Result<bool, Exception> {
        self.requests
            .step(self.backend, self.streams.is_split(), self.streams)
            .map_err(speculative_driver_error)
    }

    /// Drives all submitted requests to completion.
    pub fn run(&mut self) -> Result<(), Exception> {
        while self.step()? {}
        Ok(())
    }

    /// Consumes a finished scheduler and returns results in submission order.
    pub fn finish(self) -> Result<MtpScheduleOutput<S>, Exception> {
        let output = self.requests.finish().map_err(speculative_driver_error)?;
        Ok(MtpScheduleOutput {
            requests: output
                .requests
                .into_iter()
                .map(|request| MtpRequestOutput {
                    id: request.id,
                    token_ids: request.token_ids,
                    stats: request.stats,
                    sampler: request.sampler.into_inner(),
                    finish_reason: request.finish_reason,
                    #[cfg(test)]
                    cancelled: request.phase == MtpRequestPhase::Cancelled,
                })
                .collect(),
            scheduler: output.scheduler,
        })
    }
}

fn speculative_driver_error(error: SpeculativeDriverError<Exception>) -> Exception {
    Exception::custom(error.to_string())
}

#[cfg(test)]
fn generate<'runtime, B, S>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &'runtime Stream,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler + Clone + 'runtime,
{
    generate_with_streams(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        MtpExecutionStreams::single(stream),
    )
}

#[cfg(test)]
fn generate_with_streams<'runtime, B, S>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: MtpExecutionStreams<'runtime>,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler + Clone + 'runtime,
{
    generate_tokens(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        streams,
        MtpSchedulerOptions::default(),
        |_| Ok(()),
    )
}

/// Runs one scheduled request with explicit streams, scheduler options, and a
/// commit callback.
#[allow(clippy::too_many_arguments)]
pub fn generate_tokens<'runtime, B, S, F>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: MtpExecutionStreams<'runtime>,
    options: MtpSchedulerOptions,
    on_token: F,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler + Clone + 'runtime,
    F: FnMut(u32) -> Result<(), Exception> + 'runtime,
{
    let final_sampler;
    let token_ids;
    let stats;
    {
        let mut scheduler = MlxMtpScheduler::new(backend, streams, options)?;
        scheduler.submit(
            cache,
            input,
            config.clone(),
            prng_key,
            sampler.clone(),
            on_token,
        )?;
        scheduler.run()?;
        let mut output = scheduler.finish()?.requests;
        let request = output.pop().expect("one request was submitted");
        final_sampler = request.sampler;
        token_ids = request.token_ids;
        stats = request.stats;
    }
    *sampler = final_sampler;
    Ok((token_ids, stats))
}

/// Runs one request with transactional semantic output.
#[allow(clippy::too_many_arguments)]
pub fn generate_semantic<'runtime, B, S, F>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    streams: MtpExecutionStreams<'runtime>,
    options: MtpSchedulerOptions,
    on_event: F,
) -> Result<(Vec<u32>, MtpStats, FinishReason), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler + Clone + 'runtime,
    F: FnMut(SemanticEvent) + 'runtime,
{
    let final_sampler;
    let token_ids;
    let stats;
    let finish_reason;
    {
        let mut scheduler = MlxMtpScheduler::new(backend, streams, options)?;
        scheduler.submit_with_semantics_cancellable(
            cache,
            input,
            config.clone(),
            prng_key,
            sampler.clone(),
            semantic,
            cancellation,
            on_event,
        )?;
        scheduler.run()?;
        let mut output = scheduler.finish()?.requests;
        let request = output.pop().expect("one request was submitted");
        final_sampler = request.sampler;
        token_ids = request.token_ids;
        stats = request.stats;
        finish_reason = request.finish_reason.ok_or_else(|| {
            Exception::custom("completed semantic MTP request has no finish reason")
        })?;
    }
    *sampler = final_sampler;
    Ok((token_ids, stats, finish_reason))
}

fn validate_input(input: ModelInput<'_>) -> Result<(), Exception> {
    if input.parts.is_empty() {
        return Err(Exception::custom(
            "MTP input must contain at least one part",
        ));
    }
    if input
        .parts
        .iter()
        .all(|part| part.modality == Modality::Text)
    {
        let mut tokens = 0i32;
        for part in input.parts {
            let InputPayload::TokenIds(ids) = part.payload else {
                return Err(Exception::custom(
                    "MTP text input must contain token-id payloads",
                ));
            };
            if ids.ndim() != 2 {
                return Err(Exception::custom(format!(
                    "MTP text token ids must have rank 2, got {:?}",
                    ids.shape()
                )));
            }
            tokens = tokens.saturating_add(ids.dim(1));
        }
        if tokens == 0 {
            return Err(Exception::custom("MTP text input contains no tokens"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc, sync::Arc};

    use eredu_core::{SpeculativeCommit, SpeculativePrefill, Submission};
    use safemlx::{Device, DeviceType, ExecutionContext};

    use super::*;
    use crate::backend::runtime::media::input::InputPart;
    use eredu_runtime::{DefaultSampler, GenerationSampler, MirostatV2Sampler};

    #[derive(Clone, Default)]
    struct CountingSampler {
        process_calls: usize,
        histories: Vec<Vec<u32>>,
        committed: Vec<u32>,
    }

    impl SpeculativeSampler for CountingSampler {
        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn process_logits(
            &mut self,
            logits: &Array,
            _temperature: f32,
            history: &[u32],
            _stream: &Stream,
        ) -> Result<Array, Exception> {
            self.process_calls += 1;
            self.histories.push(history.to_vec());
            Ok(logits.clone())
        }

        fn commit_token(
            &mut self,
            _processed_logits: &Array,
            token: u32,
            _stream: &Stream,
        ) -> Result<(), Exception> {
            self.committed.push(token);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct GrammarCountingSampler {
        inner: CountingSampler,
        complete_after: usize,
    }

    impl SpeculativeSampler for GrammarCountingSampler {
        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn process_logits(
            &mut self,
            logits: &Array,
            temperature: f32,
            history: &[u32],
            stream: &Stream,
        ) -> Result<Array, Exception> {
            self.inner
                .process_logits(logits, temperature, history, stream)
        }

        fn commit_token(
            &mut self,
            processed_logits: &Array,
            token: u32,
            stream: &Stream,
        ) -> Result<(), Exception> {
            self.inner.commit_token(processed_logits, token, stream)
        }

        fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
            Ok(self.inner.committed.len() >= self.complete_after)
        }

        fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
            Ok(history.len() >= self.complete_after)
        }
    }

    #[derive(Clone, Default)]
    struct TestSemanticState {
        tokens: Vec<u32>,
        stop: Vec<u32>,
        events: Vec<SemanticEvent>,
    }

    impl SpeculativeSemanticState for TestSemanticState {
        fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
            let mut fork = self.clone();
            fork.events.clear();
            Ok(Box::new(fork))
        }

        fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
            self.tokens.push(token);
            self.events
                .push(SemanticEvent::TextDelta(token.to_string()));
            Ok(!self.stop.is_empty() && self.tokens.ends_with(&self.stop))
        }

        fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
            self.events.push(SemanticEvent::Finished { reason });
            Ok(())
        }

        fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
            self.events.push(SemanticEvent::Finished {
                reason: FinishReason::Cancelled,
            });
            Ok(())
        }

        fn take_events(&mut self) -> Vec<SemanticEvent> {
            std::mem::take(&mut self.events)
        }
    }

    #[derive(Clone, Copy, Default)]
    struct UniformSampler;

    impl SpeculativeSampler for UniformSampler {
        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn process_logits(
            &mut self,
            logits: &Array,
            _temperature: f32,
            _history: &[u32],
            stream: &Stream,
        ) -> Result<Array, Exception> {
            logits.multiply(Array::from_f32(0.0), stream)
        }
    }

    struct ScriptedBackend {
        first_token: u32,
        rejection_token: u32,
        reject_first: bool,
        accept_second: bool,
        bonus_token: u32,
        routes: Vec<(&'static str, DeviceType)>,
        draft_storage: Vec<usize>,
        draft_capacities: Vec<usize>,
    }

    #[derive(Clone)]
    struct ScriptedDraftState {
        step: usize,
        storage: Arc<()>,
    }

    impl ScriptedBackend {
        fn record(&mut self, operation: &'static str, stream: &Stream) -> Result<(), Exception> {
            self.routes
                .push((operation, stream.get_device()?.get_type()?));
            Ok(())
        }
    }

    impl SpeculativeExecutor for ScriptedBackend {
        type Input = MlxModelInput;
        type Cache = usize;
        type TargetState = ();
        type DraftState = ScriptedDraftState;
        type CacheCheckpoint = usize;
        type Verification = Array;
        type Logits = Array;
        type Context<'a> = MtpExecutionStreams<'a>;
        type Completion = MlxSpeculativeCompletion;
        type Telemetry = MtpComponentTimings;
        type Error = Exception;

        fn max_proposals(&self) -> usize {
            2
        }

        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn prefill<'context>(
            &mut self,
            _input: MlxModelInput,
            cache: &mut Self::Cache,
            streams: MtpExecutionStreams<'context>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Exception>
        where
            Self: 'context,
        {
            let stream = streams.target();
            self.record("prefill", stream)?;
            *cache = 1;
            let mut first = [0.0f32; 3];
            first[self.first_token as usize] = 10.0;
            Ok(SpeculativePrefill {
                logits: Array::from_slice(&first, &[1, 3]),
                state: (),
                evaluated_tokens: 1,
            })
        }

        fn begin_proposal(
            &mut self,
            state: &Self::TargetState,
            last_token: u32,
            proposal_capacity: usize,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Self::DraftState, Exception> {
            self.draft_capacities.push(proposal_capacity);
            let _ = (state, last_token);
            self.record("begin_target", streams.target())?;
            self.record("begin_draft", streams.draft())?;
            Ok(ScriptedDraftState {
                step: 0,
                storage: Arc::new(()),
            })
        }

        fn proposal_logits(
            &mut self,
            state: &mut Self::DraftState,
            _last_token: u32,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Array, Exception> {
            let stream = streams.draft();
            self.record("draft", stream)?;
            self.draft_storage
                .push(Arc::as_ptr(&state.storage) as usize);
            let logits = match state.step {
                0 => Array::from_slice(&[0.0f32, 0.0, 10.0], &[1, 1, 3]),
                1 | 2 => Array::from_slice(&[10.0f32, 0.0, 0.0], &[1, 1, 3]),
                _ => Array::from_slice(&[0.0f32, 10.0, 0.0], &[1, 1, 3]),
            };
            state.step += 1;
            Ok(logits)
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            *cache
        }

        fn submit_verification(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Exception> {
            let input_len = input_tokens.len();
            let stream = streams.target();
            self.record(
                match input_len {
                    2 => "verify_input_2",
                    3 => "verify_input_3",
                    _ => "verify_input_other",
                },
                stream,
            )?;
            self.record("verify", stream)?;
            *cache += input_len;
            let first = if self.reject_first {
                let mut logits = [0.0f32; 3];
                logits[self.rejection_token as usize] = 10.0;
                logits
            } else {
                [0.0f32, 0.0, 10.0]
            };
            let second = if self.accept_second {
                [10.0f32, 0.0, 0.0]
            } else {
                [0.0f32, 10.0, 0.0]
            };
            let mut bonus = [0.0f32; 3];
            bonus[self.bonus_token as usize] = 10.0;
            let output = Array::from_slice(
                &[
                    first[0], first[1], first[2], second[0], second[1], second[2], bonus[0],
                    bonus[1], bonus[2],
                ],
                &[1, 3, 3],
            );
            let completion = MlxSpeculativeCompletion::submit([&output])?;
            Ok(Submission { output, completion })
        }

        fn verification_logits<'a>(
            output: &Self::Verification,
            index: usize,
            streams: MtpExecutionStreams<'a>,
        ) -> Result<Array, Exception>
        where
            Self: 'a,
        {
            output.try_index_device((.., index as i32, ..), streams.target())
        }

        fn commit_verification(
            &mut self,
            output: Self::Verification,
            _draft_state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: Self::CacheCheckpoint,
            verified_inputs: usize,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
            if verified_inputs != output.dim(1) as usize {
                self.record("cache_truncate", streams.target())?;
            }
            self.record("commit_target", streams.target())?;
            self.record("commit_draft", streams.draft())?;
            *cache = checkpoint + verified_inputs;
            Ok(SpeculativeCommit {
                state: (),
                replayed_tokens: 0,
            })
        }
    }

    struct CommitFailBackend {
        inner: ScriptedBackend,
    }

    impl SpeculativeExecutor for CommitFailBackend {
        type Input = MlxModelInput;
        type Cache = usize;
        type TargetState = ();
        type DraftState = ScriptedDraftState;
        type CacheCheckpoint = usize;
        type Verification = Array;
        type Logits = Array;
        type Context<'a> = MtpExecutionStreams<'a>;
        type Completion = MlxSpeculativeCompletion;
        type Telemetry = MtpComponentTimings;
        type Error = Exception;

        fn max_proposals(&self) -> usize {
            self.inner.max_proposals()
        }

        fn prefill<'context>(
            &mut self,
            input: MlxModelInput,
            cache: &mut Self::Cache,
            streams: MtpExecutionStreams<'context>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Exception>
        where
            Self: 'context,
        {
            self.inner.prefill(input, cache, streams)
        }

        fn begin_proposal(
            &mut self,
            state: &Self::TargetState,
            last_token: u32,
            proposal_capacity: usize,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Self::DraftState, Exception> {
            self.inner
                .begin_proposal(state, last_token, proposal_capacity, streams)
        }

        fn proposal_logits(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Array, Exception> {
            self.inner.proposal_logits(state, last_token, streams)
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            *cache
        }

        fn submit_verification(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Exception> {
            self.inner.submit_verification(input_tokens, cache, streams)
        }

        fn verification_logits<'a>(
            output: &Self::Verification,
            index: usize,
            streams: MtpExecutionStreams<'a>,
        ) -> Result<Array, Exception>
        where
            Self: 'a,
        {
            output.try_index_device((.., index as i32, ..), streams.target())
        }

        fn commit_verification(
            &mut self,
            _output: Self::Verification,
            _draft_state: Self::DraftState,
            _cache: &mut Self::Cache,
            _checkpoint: Self::CacheCheckpoint,
            _verified_inputs: usize,
            _streams: MtpExecutionStreams<'_>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
            Err(Exception::custom("injected commit failure"))
        }
    }

    fn scripted_backend() -> ScriptedBackend {
        ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 1,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        }
    }

    #[test]
    fn scheduler_sizes_new_draft_rounds_to_the_available_proposals() {
        fn capacities(max_tokens: usize) -> Vec<usize> {
            let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut cache = 0;
            let mut backend = scripted_backend();
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::single(context.stream()),
                MtpSchedulerOptions::default().with_lookahead(false),
            )
            .unwrap();
            scheduler
                .submit_with_semantics(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    CountingSampler::default(),
                    Box::new(TestSemanticState::default()),
                    |_| {},
                )
                .unwrap();
            scheduler.run().unwrap();
            scheduler.finish().unwrap();
            backend.draft_capacities
        }

        assert_eq!(capacities(2), [1]);
        assert_eq!(capacities(3), [2]);
    }

    #[test]
    fn ordinary_canonical_mtp_and_lookahead_mtp_have_identical_semantics() {
        fn run_mtp(lookahead: bool) -> (Vec<u32>, Vec<SemanticEvent>, FinishReason) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let events = Rc::new(RefCell::new(Vec::new()));
            let callback = Rc::clone(&events);
            let mut cache = 0;
            let mut backend = scripted_backend();
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                MtpSchedulerOptions::default().with_lookahead(lookahead),
            )
            .unwrap();
            scheduler
                .submit_with_semantics(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    CountingSampler::default(),
                    Box::new(TestSemanticState::default()),
                    move |event| callback.borrow_mut().push(event),
                )
                .unwrap();
            scheduler.run().unwrap();
            let request = scheduler.finish().unwrap().requests.pop().unwrap();
            let finish_reason = request.finish_reason.unwrap();
            let events = events.borrow().clone();
            (request.token_ids, events, finish_reason)
        }

        let canonical = run_mtp(false);
        let lookahead = run_mtp(true);
        assert_eq!(lookahead, canonical);

        let mut ordinary = TestSemanticState::default();
        let mut ordinary_events = Vec::new();
        for &token in &canonical.0 {
            assert!(!ordinary.push_token(token).unwrap());
            ordinary_events.extend(ordinary.take_events());
        }
        ordinary.finish(canonical.2).unwrap();
        ordinary_events.extend(ordinary.take_events());
        assert_eq!(ordinary_events, canonical.1);
        assert_eq!(
            canonical.1.last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::MaxTokens
            })
        );
    }

    #[test]
    fn optimistic_semantic_stop_truncates_an_accepted_block_transactionally() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = target.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let events = Rc::new(RefCell::new(Vec::new()));
        let callback_events = Rc::clone(&events);
        let mut backend = scripted_backend();
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(stream, draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                Box::new(TestSemanticState {
                    stop: vec![1, 2],
                    ..TestSemanticState::default()
                }),
                move |event| callback_events.borrow_mut().push(event),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 2]);
        assert_eq!(request.finish_reason, Some(FinishReason::StopSequence));
        assert_eq!(request.stats.accept_lens, vec![1]);
        assert_eq!(request.sampler.process_calls, 2);
        assert_eq!(request.stats.optimistic_draft_blocks, 1);
        assert_eq!(request.stats.discarded_optimistic_blocks, 1);
        assert_eq!(cache, 2);
        assert_eq!(
            events.borrow().as_slice(),
            [
                SemanticEvent::TextDelta("1".into()),
                SemanticEvent::TextDelta("2".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence
                }
            ],
            "draft and optimistic tokens must never publish semantic events"
        );
        assert_eq!(
            events.borrow().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            })
        );
    }

    #[test]
    fn grammar_completion_mid_block_matches_the_committed_prefix() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::single(stream),
            MtpSchedulerOptions::default().with_lookahead(false),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                GrammarCountingSampler {
                    inner: CountingSampler::default(),
                    complete_after: 2,
                },
                Box::new(TestSemanticState::default()),
                |_| {},
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 2]);
        assert_eq!(request.finish_reason, Some(FinishReason::GrammarComplete));
        assert_eq!(request.sampler.inner.process_calls, 2);
        assert_eq!(request.sampler.inner.committed, request.token_ids);
        assert_eq!(request.stats.draft_tokens, 1);
        assert_eq!(
            backend.draft_storage.len(),
            1,
            "drafting must stop as soon as the logical prefix completes the grammar"
        );
        assert_eq!(cache, 2);

        let mut ordinary = GrammarCountingSampler {
            inner: CountingSampler::default(),
            complete_after: 2,
        };
        let mut ordinary_tokens = Vec::new();
        for expected in [1u32, 2, 0] {
            let mut values = [0.0f32; 3];
            values[expected as usize] = 10.0;
            let logits = Array::from_slice(&values, &[1, 3]);
            let processed = ordinary
                .process_logits(&logits, 0.0, &ordinary_tokens, stream)
                .unwrap();
            let chosen = ordinary
                .sample_processed(&processed, 0.0, None, stream)
                .unwrap()
                .item::<u32>(stream);
            ordinary.commit_token(&processed, chosen, stream).unwrap();
            ordinary_tokens.push(chosen);
            if ordinary.grammar_is_complete().unwrap() {
                break;
            }
        }
        assert_eq!(request.token_ids, ordinary_tokens);
    }

    #[test]
    fn terminal_grammar_bonus_discards_matching_optimistic_work() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let mut backend = scripted_backend();
        backend.bonus_token = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 8,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                GrammarCountingSampler {
                    inner: CountingSampler::default(),
                    complete_after: 4,
                },
                Box::new(TestSemanticState::default()),
                |_| {},
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 2, 0, 0]);
        assert_eq!(request.finish_reason, Some(FinishReason::GrammarComplete));
        assert_eq!(request.stats.optimistic_target_bonus_tokens, 1);
        assert_eq!(request.stats.discarded_optimistic_tokens, 1);
        assert_eq!(request.stats.reused_optimistic_tokens, 0);
        assert_eq!(request.stats.consumed_optimistic_tokens, 0);
        assert_eq!(request.sampler.inner.committed, request.token_ids);
    }

    #[test]
    fn cancellation_discards_only_the_affected_request_runtime() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let events_a = Rc::new(RefCell::new(Vec::new()));
        let events_b = Rc::new(RefCell::new(Vec::new()));
        let callback_a = Rc::clone(&events_a);
        let callback_b = Rc::clone(&events_b);
        let cancellation_a = GenerationCancellationToken::new();
        let cancellation_b = GenerationCancellationToken::new();
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let config = MtpConfig {
            max_tokens: 5,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let first = scheduler
            .submit_with_semantics_cancellable(
                &mut cache_a,
                ModelInput::new(&parts_a),
                config.clone(),
                None,
                GrammarCountingSampler {
                    inner: CountingSampler::default(),
                    complete_after: usize::MAX,
                },
                Box::new(TestSemanticState::default()),
                cancellation_a.clone(),
                move |event| callback_a.borrow_mut().push(event),
            )
            .unwrap();
        scheduler
            .submit_with_semantics_cancellable(
                &mut cache_b,
                ModelInput::new(&parts_b),
                config,
                None,
                GrammarCountingSampler {
                    inner: CountingSampler::default(),
                    complete_after: 3,
                },
                Box::new(TestSemanticState::default()),
                cancellation_b.clone(),
                move |event| callback_b.borrow_mut().push(event),
            )
            .unwrap();

        scheduler.step().unwrap();
        scheduler.step().unwrap();
        assert!(scheduler
            .requests
            .request(first)
            .unwrap()
            .has_pending_verification());
        cancellation_a.cancel();
        assert!(!cancellation_b.is_cancelled());
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();

        assert!(output.requests[0].cancelled);
        assert_eq!(output.requests[0].token_ids, vec![1]);
        assert_eq!(
            output.requests[0].finish_reason,
            Some(FinishReason::Cancelled)
        );
        assert_eq!(output.requests[0].sampler.inner.committed, vec![1]);
        assert_eq!(
            events_a.borrow().as_slice(),
            &[
                SemanticEvent::TextDelta("1".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::Cancelled
                }
            ]
        );
        assert!(!output.requests[1].cancelled);
        assert_eq!(output.requests[1].token_ids, vec![1, 2, 0]);
        assert_eq!(
            output.requests[1].finish_reason,
            Some(FinishReason::GrammarComplete)
        );
        assert_eq!(
            output.requests[1].sampler.inner.committed,
            output.requests[1].token_ids
        );
        assert_eq!(
            events_b.borrow().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::GrammarComplete
            })
        );
    }

    #[test]
    fn semantic_callback_token_cancels_at_the_committed_prefill_boundary() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let events = Rc::new(RefCell::new(Vec::new()));
        let callback_events = Rc::clone(&events);
        let cancellation = GenerationCancellationToken::new();
        let callback_cancellation = cancellation.clone();
        let mut cache = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::single(context.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();

        scheduler
            .submit_with_semantics_cancellable(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                GrammarCountingSampler {
                    inner: CountingSampler::default(),
                    complete_after: usize::MAX,
                },
                Box::new(TestSemanticState::default()),
                cancellation,
                move |event| {
                    callback_events.borrow_mut().push(event.clone());
                    if matches!(event, SemanticEvent::TextDelta(_)) {
                        callback_cancellation.cancel();
                    }
                },
            )
            .unwrap();

        assert!(scheduler.is_finished());
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        let request = &output.requests[0];
        assert!(request.cancelled);
        assert_eq!(request.token_ids, vec![1]);
        assert_eq!(request.sampler.inner.committed, request.token_ids);
        assert_eq!(request.finish_reason, Some(FinishReason::Cancelled));
        assert_eq!(cache, 1);
        assert!(backend.draft_storage.is_empty());
        assert_eq!(
            events.borrow().as_slice(),
            &[
                SemanticEvent::TextDelta("1".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::Cancelled
                }
            ]
        );
    }

    #[test]
    fn no_lookahead_full_acceptance_bonus_eos_and_max_tokens_use_safe_boundaries() {
        fn run(
            max_tokens: usize,
            eos_token_ids: Vec<u32>,
        ) -> (Vec<u32>, FinishReason, MtpStats, usize) {
            let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let stream = context.stream();
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut cache = 0;
            let mut backend = scripted_backend();
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::single(stream),
                MtpSchedulerOptions::default().with_lookahead(false),
            )
            .unwrap();
            scheduler
                .submit_with_semantics(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids,
                    },
                    None,
                    CountingSampler::default(),
                    Box::new(TestSemanticState::default()),
                    |_| {},
                )
                .unwrap();
            scheduler.run().unwrap();
            let request = scheduler.finish().unwrap().requests.pop().unwrap();
            (
                request.token_ids,
                request.finish_reason.unwrap(),
                request.stats,
                cache,
            )
        }

        let eos = run(6, vec![2]);
        assert_eq!(eos.0, vec![1, 2]);
        assert_eq!(eos.1, FinishReason::Eos);
        assert_eq!(eos.2.accept_lens, vec![1]);
        assert_eq!(eos.3, 2);

        let max = run(2, Vec::new());
        assert_eq!(max.0, vec![1, 2]);
        assert_eq!(max.1, FinishReason::MaxTokens);
        assert_eq!(max.2.accept_lens, vec![1]);
        assert_eq!(max.3, 2);

        let bonus = run(4, Vec::new());
        assert_eq!(bonus.0, vec![1, 2, 0, 1]);
        assert_eq!(bonus.1, FinishReason::MaxTokens);
        assert_eq!(bonus.2.accept_lens, vec![2]);
        assert_eq!(bonus.2.accepted_tokens, 2);
        assert_eq!(bonus.3, 4);
    }

    #[test]
    fn no_lookahead_residual_replacement_commits_only_the_replacement() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let mut backend = scripted_backend();
        backend.reject_first = true;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::single(stream),
            MtpSchedulerOptions::default().with_lookahead(false),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 2,
                    max_draft_tokens: 2,
                    temperature: 1.0,
                    eos_token_ids: Vec::new(),
                },
                Some(safemlx::random::key(11).unwrap()),
                CountingSampler::default(),
                Box::new(TestSemanticState::default()),
                |_| {},
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 1]);
        assert_eq!(request.finish_reason, Some(FinishReason::MaxTokens));
        assert_eq!(request.stats.accept_lens, vec![0]);
        assert_eq!(request.sampler.committed, request.token_ids);
        assert_eq!(cache, 2);
    }

    #[test]
    fn commit_failure_does_not_publish_or_advance_canonical_output_state() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let events = Rc::new(RefCell::new(Vec::new()));
        let callback_events = Rc::clone(&events);
        let mut backend = CommitFailBackend {
            inner: scripted_backend(),
        };
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::single(stream),
            MtpSchedulerOptions::default().with_lookahead(false),
        )
        .unwrap();
        let id = scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                Box::new(TestSemanticState::default()),
                move |event| callback_events.borrow_mut().push(event),
            )
            .unwrap();
        let published_after_prefill = events.borrow().len();
        assert!(scheduler.run().is_err());
        let request = scheduler.requests.request(id).unwrap();

        assert_eq!(events.borrow().len(), published_after_prefill);
        assert_eq!(request.sequence().tokens(), &[1]);
        assert_eq!(request.sampler().inner().committed, vec![1]);
        assert_eq!(request.sampler().inner().process_calls, 1);
    }

    #[test]
    fn greedy_engine_commits_only_the_accepted_prefix() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut emitted = Vec::new();
        let (tokens, stats) = generate_tokens(
            &mut ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: false,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            },
            &mut cache,
            input,
            &config,
            None,
            &mut DefaultSampler,
            MtpExecutionStreams::single(stream),
            MtpSchedulerOptions::default(),
            |token| {
                emitted.push(token);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 1]);
        assert_eq!(emitted, tokens);
        assert_eq!(stats.accept_lens, vec![1]);
        assert_eq!(stats.accepted_tokens, 1);
        assert_eq!(cache, 3);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn cpu_draft_gpu_target_split_stream_routes_and_commits() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;

        let (tokens, _) = generate_with_streams(
            &mut backend,
            &mut cache,
            input,
            &config,
            None,
            &mut DefaultSampler,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 1]);
        for (operation, device) in backend.routes {
            let expected = if operation == "begin_draft"
                || operation == "draft"
                || operation == "commit_draft"
            {
                DeviceType::Cpu
            } else {
                DeviceType::Gpu
            };
            assert_eq!(device, expected, "{operation}");
        }
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn split_stream_engine_preserves_stochastic_acceptance() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 4,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(7).unwrap();

        let (tokens, stats) = generate_with_streams(
            &mut backend,
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 0, 0]);
        assert_eq!(stats.accepted_tokens, 2);
        assert_eq!(sampler.generated_tokens(), tokens);
    }

    #[test]
    fn mirostat_v2_mtp_commits_accepted_target_distributions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 4,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(7).unwrap();

        let (tokens, stats) = generate(
            &mut ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            },
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            stream,
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 0, 0]);
        assert_eq!(stats.draft_tokens, 2);
        assert_eq!(stats.accepted_tokens, 2);
        assert_eq!(sampler.generated_tokens(), tokens);
        assert!((sampler.mu() - 12.0).abs() < 1e-4);
    }

    #[test]
    fn mirostat_v2_mtp_commits_replacement_not_rejected_draft() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 2,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(11).unwrap();

        let (tokens, stats) = generate(
            &mut ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: true,
                accept_second: false,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            },
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            stream,
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 1]);
        assert_eq!(stats.draft_tokens, 1);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(sampler.generated_tokens(), tokens);
        assert!((sampler.mu() - 11.0).abs() < 1e-4);
    }

    #[test]
    fn execution_stream_topologies_classify_on_cpu_devices() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let second_stream = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let second_device = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));

        assert_eq!(
            MtpExecutionStreams::single(target.stream()).topology(),
            SpeculativeExecutionTopology::Single
        );
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), second_stream.stream())
                .unwrap()
                .topology(),
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), second_device.stream())
                .unwrap()
                .topology(),
            SpeculativeExecutionTopology::CrossDeviceSplit
        );
    }

    #[test]
    fn same_device_event_handoffs_order_both_cpu_stream_directions() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let streams = MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap();
        assert_eq!(
            streams.topology(),
            SpeculativeExecutionTopology::SameDeviceSplit
        );

        let lhs = Array::ones::<f32>(&[1024, 1024], target.stream()).unwrap();
        let rhs = Array::ones::<f32>(&[1024, 1024], target.stream()).unwrap();
        let target_output = lhs.matmul(&rhs, target.stream()).unwrap();
        let target_handoff = streams.wait_for_target_outputs([&target_output]).unwrap();
        assert!(
            !target_handoff.is_complete().unwrap(),
            "the MTP target-to-draft handoff blocked the host"
        );

        let draft_output = target_output
            .add(Array::from_f32(1.0), draft.stream())
            .unwrap();
        let draft_handoff = streams.wait_for_draft_outputs([&draft_output]).unwrap();
        let consumed = draft_output
            .add(Array::from_f32(1.0), target.stream())
            .unwrap();
        let consumed_completion = async_eval_with_event([&consumed]).unwrap();

        // Queued waits retain both handoffs after their public handles drop.
        drop(target_handoff);
        drop(draft_handoff);
        consumed_completion.synchronize().unwrap();
        assert_eq!(
            consumed
                .try_index_device((0, 0), target.stream())
                .unwrap()
                .item::<f32>(target.stream()),
            1026.0
        );
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn execution_streams_classify_single_same_device_and_cross_device_topologies() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let second_gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));

        let single = MtpExecutionStreams::single(target.stream());
        let same_device = MtpExecutionStreams::new(target.stream(), second_gpu.stream()).unwrap();
        let cross_device = MtpExecutionStreams::new(target.stream(), cpu.stream()).unwrap();

        assert_eq!(single.topology(), SpeculativeExecutionTopology::Single);
        assert!(!single.is_split());
        assert!(!single.crosses_devices());
        assert_eq!(
            same_device.topology(),
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert!(same_device.is_split());
        assert!(!same_device.crosses_devices());
        assert_eq!(
            cross_device.topology(),
            SpeculativeExecutionTopology::CrossDeviceSplit
        );
        assert!(cross_device.is_split());
        assert!(cross_device.crosses_devices());
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn same_gpu_split_stream_runs_exact_optimistic_lookahead() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        assert_ne!(
            target.stream().get_index().unwrap(),
            draft.stream().get_index().unwrap()
        );
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        let stats = &output.requests[0].stats;

        assert_eq!(
            output.scheduler.execution_topology,
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert_eq!(
            stats.execution_topology,
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert!(stats.optimistic_draft_blocks > 0);
        assert!(stats.optimistic_target_bonus_tokens > 0);
        assert_eq!(
            stats.optimistic_bonus_matches + stats.optimistic_bonus_mismatches,
            stats.optimistic_target_bonus_tokens
        );
        assert!(backend
            .routes
            .iter()
            .all(|(_, device)| *device == DeviceType::Gpu));
        let verify = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "verify")
            .unwrap();
        let optimistic_draft = backend.routes[verify + 1..]
            .iter()
            .position(|(operation, _)| *operation == "draft")
            .map(|offset| verify + 1 + offset)
            .unwrap();
        let resolve = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "commit_target")
            .unwrap();
        assert!(verify < optimistic_draft && optimistic_draft < resolve);
    }

    #[test]
    #[ignore = "explicit Metal MTP event handoff test; run on a Metal host"]
    fn same_gpu_mtp_handoff_does_not_synchronize_the_producer_stream() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let streams = MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap();
        assert_eq!(
            streams.topology(),
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert_ne!(
            target.stream().get_index().unwrap(),
            draft.stream().get_index().unwrap()
        );

        let lhs = Array::ones::<f32>(&[4096, 4096], target.stream()).unwrap();
        let rhs = Array::ones::<f32>(&[4096, 4096], target.stream()).unwrap();
        let produced = lhs.matmul(&rhs, target.stream()).unwrap();
        let handoff = streams.wait_for_target_outputs([&produced]).unwrap();
        assert!(
            !handoff.is_complete().unwrap(),
            "the same-device MTP handoff waited for target completion on the host"
        );

        let consumed = produced.square(draft.stream()).unwrap();
        let completion = async_eval_with_event([&consumed]).unwrap();
        drop(handoff);
        completion.synchronize().unwrap();
    }

    #[test]
    fn same_device_split_preserves_stochastic_lookahead() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let run = |lookahead| {
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache = 0;
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                MtpSchedulerOptions::default().with_lookahead(lookahead),
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 8,
                        max_draft_tokens: 2,
                        temperature: 1.0,
                        eos_token_ids: Vec::new(),
                    },
                    Some(safemlx::random::key(7).unwrap()),
                    UniformSampler,
                    |_| Ok(()),
                )
                .unwrap();
            scheduler.run().unwrap();
            scheduler.finish().unwrap().requests.pop().unwrap()
        };
        let request = run(true);
        let canonical = run(false);

        assert_eq!(
            request.stats.execution_topology,
            SpeculativeExecutionTopology::SameDeviceSplit
        );
        assert_eq!(request.token_ids, canonical.token_ids);
        assert_eq!(request.token_ids.len(), 8);
        assert_eq!(request.stats.emitted_tokens, 8);
        assert!(request.stats.optimistic_draft_blocks > 0);
        assert_eq!(canonical.stats.optimistic_draft_blocks, 0);
    }

    #[test]
    fn full_acceptance_promotes_shared_optimistic_branch() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        let stats = &output.requests[0].stats;

        assert_eq!(stats.optimistic_draft_blocks, 1);
        assert_eq!(stats.reused_optimistic_blocks, 1);
        assert_eq!(stats.reused_optimistic_tokens, 1);
        assert_eq!(stats.consumed_optimistic_tokens, 1);
        assert_eq!(stats.optimistic_target_bonus_tokens, 1);
        assert_eq!(stats.optimistic_bonus_matches, 1);
        assert_eq!(output.scheduler.peak_optimistic_branches, 1);
        assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
        let first_commit = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "commit_target")
            .unwrap();
        assert!(
            backend.routes[..first_commit]
                .iter()
                .all(|(operation, _)| *operation != "cache_truncate"),
            "a fully accepted bonus-emitting verification must not truncate"
        );
    }

    #[test]
    fn matching_bonus_is_emitted_and_consumes_one_paired_proposal() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let id = scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();

        for _ in 0..4 {
            scheduler.step().unwrap();
        }

        let request = scheduler.requests.request(id).unwrap();
        assert_eq!(request.sequence().tokens(), &[1, 2, 0, 0]);
        assert_eq!(request.phase(), MtpRequestPhase::ReadyToDraft);
        let retained = request.block().unwrap();
        assert_eq!(retained.proposals.len(), 1);
        assert_eq!(retained.proposals[0].token, 1);
        assert_eq!(retained.state.step, 4);
        assert_eq!(
            retained.proposals[0]
                .distribution
                .try_index_device((0, 0, 1), draft.stream())
                .unwrap()
                .item::<f32>(draft.stream()),
            10.0
        );
        assert_eq!(request.stats().consumed_optimistic_tokens, 1);
        assert_eq!(request.stats().reused_optimistic_tokens, 1);
        assert_eq!(request.stats().optimistic_bonus_matches, 1);
        scheduler.step().unwrap();
        assert_eq!(
            scheduler
                .requests
                .request(id)
                .unwrap()
                .block()
                .unwrap()
                .proposals
                .len(),
            2,
            "the consumed optimistic token must be topped back up before submission"
        );
        scheduler.step().unwrap();
        assert_eq!(
            scheduler
                .backend
                .routes
                .iter()
                .filter(|(operation, _)| *operation == "verify_input_3")
                .count(),
            2
        );

        scheduler.cancel(id).unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0, 0]);
        assert_eq!(backend.draft_storage.len(), 5);
    }

    #[test]
    fn mismatching_bonus_restores_exact_non_lookahead_state() {
        #[allow(clippy::type_complexity)]
        fn run(
            options: MtpSchedulerOptions,
        ) -> (Vec<u32>, Vec<u32>, usize, MtpStats, CountingSampler) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 1,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache = 0;
            let mut callback_tokens = Vec::new();
            let request;
            {
                let mut scheduler = MlxMtpScheduler::new(
                    &mut backend,
                    MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                    options,
                )
                .unwrap();
                scheduler
                    .submit(
                        &mut cache,
                        ModelInput::new(&parts),
                        MtpConfig {
                            max_tokens: 7,
                            max_draft_tokens: 2,
                            temperature: 0.0,
                            eos_token_ids: Vec::new(),
                        },
                        None,
                        CountingSampler::default(),
                        |token| {
                            callback_tokens.push(token);
                            Ok(())
                        },
                    )
                    .unwrap();
                scheduler.run().unwrap();
                request = scheduler.finish().unwrap().requests.pop().unwrap();
            }
            (
                request.token_ids,
                callback_tokens,
                cache,
                request.stats,
                request.sampler,
            )
        }

        let without = run(MtpSchedulerOptions {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 0,
            lookahead_blocks: 0,
            ..MtpSchedulerOptions::default()
        });
        let with = run(MtpSchedulerOptions::default());

        assert_eq!(with.0, without.0);
        assert_eq!(with.1, without.1);
        assert_eq!(with.2, without.2);
        assert_eq!(with.3.target_tokens, without.3.target_tokens);
        assert_eq!(with.3.draft_tokens, without.3.draft_tokens);
        assert_eq!(with.3.accepted_tokens, without.3.accepted_tokens);
        assert_eq!(with.3.accept_lens, without.3.accept_lens);
        assert_eq!(with.3.emitted_tokens, without.3.emitted_tokens);
        assert_eq!(with.4.process_calls, without.4.process_calls);
        assert_eq!(with.4.histories, without.4.histories);
        assert_eq!(with.4.committed, without.4.committed);
        assert!(with.3.optimistic_bonus_mismatches > 0);
        assert!(with.3.discarded_optimistic_tokens > 0);
        assert_eq!(with.3.consumed_optimistic_tokens, 0);
    }

    #[test]
    fn history_derived_sampler_matches_non_lookahead_execution() {
        fn run(options: MtpSchedulerOptions) -> (Vec<u32>, Vec<usize>) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 1,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache = 0;
            let sampler = GenerationSampler::new()
                .top_k(0)
                .top_p(1.0)
                .min_p(0.0)
                .penalties(1.2, -1, 0.1, 0.1);
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                options,
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 7,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    sampler,
                    |_| Ok(()),
                )
                .unwrap();
            scheduler.run().unwrap();
            let request = scheduler.finish().unwrap().requests.pop().unwrap();
            (request.token_ids, request.stats.accept_lens)
        }

        let without = run(MtpSchedulerOptions {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 0,
            lookahead_blocks: 0,
            ..MtpSchedulerOptions::default()
        });
        let with = run(MtpSchedulerOptions::default());
        assert_eq!(with, without);
    }

    #[test]
    fn stochastic_match_and_mismatch_ignore_interleaving_and_branch_slots() {
        fn run(
            seed: u64,
            options: MtpSchedulerOptions,
            with_peer: bool,
        ) -> (Vec<u32>, Vec<usize>, MtpStats) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
            let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
            let parts_a = [InputPart::text_token_ids(&prompt_a)];
            let parts_b = [InputPart::text_token_ids(&prompt_b)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache_a = 0;
            let mut cache_b = 0;
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                options,
            )
            .unwrap();
            let config = MtpConfig {
                max_tokens: 8,
                max_draft_tokens: 2,
                temperature: 1.0,
                eos_token_ids: Vec::new(),
            };
            scheduler
                .submit(
                    &mut cache_a,
                    ModelInput::new(&parts_a),
                    config.clone(),
                    Some(safemlx::random::key(seed).unwrap()),
                    UniformSampler,
                    |_| Ok(()),
                )
                .unwrap();
            if with_peer {
                scheduler
                    .submit(
                        &mut cache_b,
                        ModelInput::new(&parts_b),
                        config,
                        Some(safemlx::random::key(seed + 1000).unwrap()),
                        UniformSampler,
                        |_| Ok(()),
                    )
                    .unwrap();
            }
            scheduler.run().unwrap();
            let request = scheduler.finish().unwrap().requests.remove(0);
            (
                request.token_ids,
                request.stats.accept_lens.clone(),
                request.stats,
            )
        }

        let no_lookahead = MtpSchedulerOptions {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 0,
            lookahead_blocks: 0,
            ..MtpSchedulerOptions::default()
        };
        let mut saw_match = false;
        let mut saw_mismatch = false;
        for seed in 0..64 {
            let with = run(seed, MtpSchedulerOptions::default(), false);
            if with.2.optimistic_bonus_matches == 0 && with.2.optimistic_bonus_mismatches == 0 {
                continue;
            }
            let without = run(seed, no_lookahead, false);
            let interleaved = run(seed, MtpSchedulerOptions::default(), true);
            assert_eq!((&with.0, &with.1), (&without.0, &without.1));
            assert_eq!((&with.0, &with.1), (&interleaved.0, &interleaved.1));
            saw_match |= with.2.optimistic_bonus_matches > 0;
            saw_mismatch |= with.2.optimistic_bonus_mismatches > 0;
            if saw_match && saw_mismatch {
                break;
            }
        }
        assert!(saw_match, "scripted seeds did not exercise a bonus match");
        assert!(
            saw_mismatch,
            "scripted seeds did not exercise a bonus mismatch"
        );
    }

    #[test]
    fn promoted_round_leaves_last_emitted_token_out_of_target_cache() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let id = scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();

        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        assert_eq!(scheduler.phase(id), Some(MtpRequestPhase::ReadyToDraft));
        scheduler.cancel(id).unwrap();
        let output = scheduler.finish().unwrap();

        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0, 0]);
        // Prefill retained one token. The fully accepted verification evaluated
        // `[first, proposal_1, proposal_2]`. The matching target bonus is
        // emitted immediately but remains outside the target cache.
        assert_eq!(cache, 4);
    }

    #[test]
    fn rejection_discards_branch_sampler_prng_history_and_cache_state() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: true,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let id = scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                |_| Ok(()),
            )
            .unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        assert_eq!(
            scheduler.phase(id),
            Some(MtpRequestPhase::OptimisticDraftReady)
        );
        scheduler.step().unwrap();
        scheduler.cancel(id).unwrap();
        let output = scheduler.finish().unwrap();
        let request = &output.requests[0];

        assert_eq!(request.token_ids, vec![1, 1]);
        assert_eq!(request.sampler.process_calls, 2);
        assert_eq!(request.stats.discarded_optimistic_blocks, 1);
        assert_eq!(request.stats.discarded_optimistic_tokens, 2);
        assert_eq!(cache, 2);
        assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
    }

    #[test]
    fn rejection_matches_execution_with_lookahead_disabled() {
        fn run(options: MtpSchedulerOptions) -> (Vec<u32>, usize, MtpStats) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: true,
                accept_second: false,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache = 0;
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                options,
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |_| Ok(()),
                )
                .unwrap();
            scheduler.run().unwrap();
            let mut requests = scheduler.finish().unwrap().requests;
            let output = requests.pop().unwrap();
            (output.token_ids, cache, output.stats)
        }

        let without = run(MtpSchedulerOptions {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 0,
            lookahead_blocks: 0,
            ..MtpSchedulerOptions::default()
        });
        let with = run(MtpSchedulerOptions::default());
        assert_eq!(with.0, without.0);
        assert_eq!(with.1, without.1);
        assert_eq!(with.2.accept_lens, without.2.accept_lens);
        assert!(with.2.discarded_optimistic_tokens > 0);
        assert_eq!(without.2.optimistic_draft_tokens, 0);
    }

    #[test]
    fn target_eos_discards_in_flight_lookahead() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 0,
            reject_first: true,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: vec![0],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        assert_eq!(request.token_ids, vec![1, 0]);
        assert_eq!(request.stats.optimistic_draft_tokens, 1);
        assert_eq!(request.stats.discarded_optimistic_tokens, 1);
        assert_eq!(request.stats.reused_optimistic_tokens, 0);
    }

    #[test]
    fn bonus_eos_completes_only_its_request_and_discards_continuation() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut callback_a = Vec::new();
        let mut callback_b = Vec::new();
        let output;
        {
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                MtpSchedulerOptions::default(),
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache_a,
                    ModelInput::new(&parts_a),
                    MtpConfig {
                        max_tokens: 6,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: vec![0],
                    },
                    None,
                    DefaultSampler,
                    |token| {
                        callback_a.push(token);
                        Ok(())
                    },
                )
                .unwrap();
            scheduler
                .submit(
                    &mut cache_b,
                    ModelInput::new(&parts_b),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |token| {
                        callback_b.push(token);
                        Ok(())
                    },
                )
                .unwrap();
            scheduler.run().unwrap();
            output = scheduler.finish().unwrap();
        }

        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
        assert_eq!(callback_a, output.requests[0].token_ids);
        assert_eq!(output.requests[0].stats.optimistic_target_bonus_tokens, 1);
        assert_eq!(output.requests[0].stats.optimistic_bonus_matches, 0);
        assert_eq!(output.requests[0].stats.discarded_optimistic_tokens, 1);
        assert_eq!(output.requests[1].token_ids.len(), 5);
        assert_eq!(callback_b, output.requests[1].token_ids);
        assert!(output.requests[1].stats.rounds > output.requests[0].stats.rounds);
        assert!(output.scheduler.cross_request_draft_opportunities > 0);
    }

    #[test]
    fn consumed_one_token_branch_never_submits_empty_verification() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 2, 0, 2, 0]);
        assert_eq!(request.stats.consumed_optimistic_tokens, 1);
        assert_eq!(request.stats.reused_optimistic_tokens, 0);
        assert_eq!(request.stats.reused_optimistic_blocks, 0);
        assert_eq!(
            backend
                .routes
                .iter()
                .filter(|(operation, _)| *operation == "verify")
                .count(),
            2
        );
    }

    #[test]
    fn max_token_boundary_does_not_draft_unusable_lookahead() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 3,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();

        assert_eq!(request.token_ids, vec![1, 2, 0]);
        assert_eq!(request.stats.optimistic_draft_tokens, 0);
        assert_eq!(request.stats.optimistic_draft_blocks, 0);
        assert_eq!(backend.draft_storage.len(), 1);
    }

    #[test]
    fn adaptive_sampler_does_not_claim_exact_optimistic_promotion() {
        assert!(GenerationSampler::new().supports_exact_optimistic_promotion());
        assert!(!MirostatV2Sampler::default().supports_exact_optimistic_promotion());
    }

    #[test]
    fn stale_optimistic_prefix_is_an_error_not_a_fallback() {
        let branch = SpeculativeOptimisticBranch {
            block: SpeculativeDraftBlock {
                state: ScriptedDraftState {
                    step: 1,
                    storage: Arc::new(()),
                },
                proposals: vec![SpeculativeProposal {
                    token: 0,
                    distribution: Array::from_slice(&[1.0f32, 0.0, 0.0], &[1, 1, 3]),
                }],
            },
            assumed_prefix: vec![1, 2],
        };
        let mut stats = MtpStats::default();
        let error = resolve_optimistic_branch(Some(branch), &[1, 0], Some(0), false, &mut stats)
            .err()
            .unwrap();

        assert!(error
            .to_string()
            .contains("diverged from the canonical committed prefix"));
        assert_eq!(stats.optimistic_target_bonus_tokens, 0);
        assert_eq!(stats.optimistic_bonus_matches, 0);
        assert_eq!(stats.optimistic_bonus_mismatches, 0);
        assert_eq!(stats.consumed_optimistic_tokens, 0);
        assert_eq!(stats.discarded_optimistic_tokens, 0);
    }

    #[test]
    fn independent_requests_progress_fairly_and_preserve_output_order() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache_a,
                ModelInput::new(&parts_a),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: vec![0],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler
            .submit(
                &mut cache_b,
                ModelInput::new(&parts_b),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: vec![2],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();

        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
        assert_eq!(output.requests[1].token_ids, vec![1, 2]);
        assert!(output.scheduler.cross_request_draft_opportunities > 0);
        let verify = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "verify")
            .unwrap();
        let cross_draft = backend.routes[verify + 1..]
            .iter()
            .position(|(operation, _)| *operation == "draft")
            .map(|offset| verify + 1 + offset)
            .unwrap();
        let resolve = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "commit_target")
            .unwrap();
        assert!(verify < cross_draft && cross_draft < resolve);
    }

    #[test]
    fn scheduler_limits_bound_retained_transactions_and_branches() {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut scheduler = MlxMtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
            MtpSchedulerOptions {
                max_in_flight_verifications: 2,
                max_optimistic_branches: 1,
                lookahead_blocks: 1,
                ..MtpSchedulerOptions::default()
            },
        )
        .unwrap();
        for (cache, parts) in [(&mut cache_a, &parts_a), (&mut cache_b, &parts_b)] {
            scheduler
                .submit(
                    cache,
                    ModelInput::new(parts),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |_| Ok(()),
                )
                .unwrap();
        }
        scheduler.run().unwrap();
        let stats = scheduler.finish().unwrap().scheduler;
        assert!(stats.peak_in_flight_verifications <= 2);
        assert!(stats.peak_optimistic_branches <= 1);
        assert_eq!(stats.peak_optimistic_branches, 1);
    }

    #[test]
    fn stochastic_request_is_reproducible_across_scheduler_interleavings() {
        fn run(
            with_peer: bool,
        ) -> (
            Vec<u32>,
            Vec<usize>,
            Vec<SemanticEvent>,
            Option<FinishReason>,
        ) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
            let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
            let parts_a = [InputPart::text_token_ids(&prompt_a)];
            let parts_b = [InputPart::text_token_ids(&prompt_b)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache_a = 0;
            let mut cache_b = 0;
            let events = Rc::new(RefCell::new(Vec::new()));
            let callback = Rc::clone(&events);
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                MtpSchedulerOptions::default(),
            )
            .unwrap();
            let config = MtpConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 1.0,
                eos_token_ids: Vec::new(),
            };
            scheduler
                .submit_with_semantics(
                    &mut cache_a,
                    ModelInput::new(&parts_a),
                    config.clone(),
                    Some(safemlx::random::key(7).unwrap()),
                    MirostatV2Sampler::default(),
                    Box::new(TestSemanticState::default()),
                    move |event| callback.borrow_mut().push(event),
                )
                .unwrap();
            if with_peer {
                scheduler
                    .submit(
                        &mut cache_b,
                        ModelInput::new(&parts_b),
                        config,
                        Some(safemlx::random::key(99).unwrap()),
                        MirostatV2Sampler::default(),
                        |_| Ok(()),
                    )
                    .unwrap();
            }
            scheduler.run().unwrap();
            let output = scheduler.finish().unwrap();
            let events = events.borrow().clone();
            (
                output.requests[0].token_ids.clone(),
                output.requests[0].stats.accept_lens.clone(),
                events,
                output.requests[0].finish_reason,
            )
        }

        assert_eq!(run(false), run(true));
    }

    #[test]
    fn adaptive_disabling_stops_unproductive_branches_without_changing_output() {
        fn run(options: MtpSchedulerOptions) -> (Vec<u32>, usize, MtpStats) {
            let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: true,
                accept_second: false,
                bonus_token: 0,
                routes: Vec::new(),
                draft_storage: Vec::new(),
                draft_capacities: Vec::new(),
            };
            let mut cache = 0;
            let mut scheduler = MlxMtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
                options,
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 10,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |_| Ok(()),
                )
                .unwrap();
            scheduler.run().unwrap();
            let request = scheduler.finish().unwrap().requests.pop().unwrap();
            (request.token_ids, cache, request.stats)
        }

        let adaptive = run(MtpSchedulerOptions {
            adaptive_lookahead_min_blocks: 2,
            ..MtpSchedulerOptions::default()
        });
        let disabled = run(MtpSchedulerOptions::default().with_lookahead(false));

        assert_eq!(adaptive.0, disabled.0);
        assert_eq!(adaptive.1, disabled.1);
        assert_eq!(adaptive.2.accept_lens, disabled.2.accept_lens);
        assert_eq!(adaptive.2.optimistic_draft_blocks, 2);
        assert!(adaptive.2.adaptive_lookahead_disabled);
        assert_eq!(disabled.2.optimistic_draft_blocks, 0);
    }

    #[test]
    fn empty_stats_have_zero_acceptance_rate() {
        assert_eq!(MtpStats::default().accept_rate(), 0.0);
    }

    #[test]
    fn component_timings_accumulate_without_overwriting_scheduler_stats() {
        let mut stats = MtpStats {
            rounds: 7,
            draft_context_time: Duration::from_millis(2),
            ..MtpStats::default()
        };
        MtpComponentTimings {
            draft_context: Duration::from_millis(3),
            draft_assistant: Duration::from_millis(5),
            draft_head: Duration::from_millis(7),
            target_verification: Duration::from_millis(11),
        }
        .add_to(&mut stats);

        assert_eq!(stats.rounds, 7);
        assert_eq!(stats.draft_context_time, Duration::from_millis(5));
        assert_eq!(stats.draft_assistant_time, Duration::from_millis(5));
        assert_eq!(stats.draft_head_time, Duration::from_millis(7));
        assert_eq!(stats.target_verification_time, Duration::from_millis(11));
    }

    #[test]
    fn component_timing_guard_is_scoped_and_nested() {
        assert!(!component_timing_enabled());
        {
            let _outer = MtpComponentTimingGuard::enable();
            assert!(component_timing_enabled());
            {
                let _inner = MtpComponentTimingGuard::enable();
                assert!(component_timing_enabled());
            }
            assert!(component_timing_enabled());
        }
        assert!(!component_timing_enabled());
    }

    #[test]
    fn adaptive_lookahead_uses_deterministic_reuse_accounting() {
        let options = MtpSchedulerOptions {
            adaptive_lookahead_min_blocks: 4,
            ..MtpSchedulerOptions::default()
        };
        let mut profitable = MtpStats {
            optimistic_draft_blocks: 4,
            reused_optimistic_tokens: 3,
            discarded_optimistic_tokens: 2,
            ..MtpStats::default()
        };
        profitable.update_adaptive_lookahead(options);
        assert!(!profitable.adaptive_lookahead_disabled);

        let mut unprofitable = MtpStats {
            optimistic_draft_blocks: 4,
            reused_optimistic_tokens: 1,
            discarded_optimistic_tokens: 2,
            ..MtpStats::default()
        };
        unprofitable.update_adaptive_lookahead(options);
        assert!(unprofitable.adaptive_lookahead_disabled);

        let mut no_reuse = MtpStats {
            optimistic_draft_blocks: 4,
            ..MtpStats::default()
        };
        no_reuse.update_adaptive_lookahead(options);
        assert!(no_reuse.adaptive_lookahead_disabled);

        let mut disabled_policy = unprofitable.clone();
        disabled_policy.adaptive_lookahead_disabled = false;
        disabled_policy.update_adaptive_lookahead(MtpSchedulerOptions {
            adaptive_lookahead: false,
            ..options
        });
        assert!(!disabled_policy.adaptive_lookahead_disabled);
    }
}
